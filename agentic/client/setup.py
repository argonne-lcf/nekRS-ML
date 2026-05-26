"""One-shot laptop-side setup: write endpoints.json, register functions, ping.

After bootstrapping the endpoint on Aurora (see agentic/globus_endpoints/setup_aurora.sh)
you'll have an endpoint UUID and an Aurora-side path to the repo. Plug those in
here and this script does the rest:

    python -m agentic.client.setup \\
        --system aurora \\
        --uuid <ENDPOINT_UUID> \\
        --repo-root /lus/flare/projects/<proj>/<user>/nekRS-ML

By default it:
  1. Writes/updates ~/.config/nekrs-ml-agentic/endpoints.json
  2. Ensures a valid Globus Compute auth session (opens browser if needed)
  3. Registers every function in agentic.functions with Globus Compute
  4. Runs ping() against the endpoint to confirm the round-trip

Pass --skip-register to keep existing UUIDs, --skip-ping to defer the smoke
check, --force-register to re-register everything (use after changing a
function body), --reauth to clear cached Globus Compute tokens and force a
fresh login.
"""

from __future__ import annotations

import argparse
import sys

from agentic.client._config import load_endpoints, save_endpoints


def _write_endpoint(system: str, uuid: str, repo_root: str, extras: dict | None = None) -> dict:
    endpoints = load_endpoints()
    entry = {"uuid": uuid, "repo_root": repo_root}
    if extras:
        entry.update(extras)
    if system in endpoints:
        print(f"  updating existing endpoint {system!r}")
    else:
        print(f"  adding new endpoint {system!r}")
    endpoints[system] = entry
    save_endpoints(endpoints)
    return endpoints


def _prepare_auth(reauth: bool) -> int:
    """Optional token cleanup + clear messaging about what's about to happen.

    The Globus Compute SDK handles auth lazily on first API call. It prints
    a URL and prompts for the auth code via terminal input -- which is the
    behaviour we want (works fine over SSH, no browser-open guessing). All
    we do here is:

      (a) if --reauth, delete the SDK's cached token files so the next API
          call goes through a fresh login flow (matches the alcf reference's
          `rm -r ~/.globus_compute` recovery, but narrower -- we only touch
          token storage, not per-endpoint config dirs that sit alongside it),
      (b) print up-front messaging so the user isn't surprised when the SDK
          interrupts the script with a URL and a prompt.
    """
    from pathlib import Path

    token_dir = Path.home() / ".globus_compute"
    if reauth and token_dir.exists():
        cleared = 0
        for pattern in ("storage.db*", "tokens.json", "*.tokens.json"):
            for p in token_dir.glob(pattern):
                print(f"  --reauth: removing {p}")
                p.unlink()
                cleared += 1
        if cleared == 0:
            print("  --reauth: nothing to clear (no cached tokens found).")

    print("\n[setup] Globus Compute auth note:")
    print("        On first run (or after --reauth), the SDK will print an")
    print("        authentication URL and then pause for input. Open the URL")
    print("        in your browser, complete the ALCF login, and paste the")
    print("        returned auth code back into this terminal. Subsequent")
    print("        runs reuse the cached token under ~/.globus_compute/.")
    return 0


def _register(force: bool) -> int:
    from agentic.client import register as _register_mod

    # Re-use the registration logic by calling main() with reconstructed argv
    saved_argv = sys.argv
    try:
        sys.argv = ["register"] + (["--force"] if force else [])
        return _register_mod.main()
    finally:
        sys.argv = saved_argv


def _ping(system: str) -> int:
    import re

    from agentic.client.system import System

    print(f"\n[setup] Pinging endpoint for {system!r} ...")
    try:
        hpc_cm = System(system)
    except RuntimeError as e:
        print(f"  setup error: {e}", file=sys.stderr)
        return 2
    with hpc_cm as hpc:
        result = hpc.ping(message="setup-smoke")
        if not result.get("ok"):
            print(f"  ping failed: {result.get('error')}", file=sys.stderr)
            return 1
        py_str = result.get("python", "")
        print(f"  hostname : {result.get('hostname')}")
        print(f"  user     : {result.get('user')}")
        print(f"  python   : {py_str.splitlines()[0] if py_str else ''}")

    # Globus Compute serialises functions/results between laptop and endpoint via pickle. 
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", py_str)
    if m:
        endpoint_full = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        local_full = sys.version_info[:3]
        if endpoint_full[:2] != local_full[:2]:
            print(
                f"\n  WARNING: Python MAJOR.MINOR mismatch.\n"
                f"    laptop  : {local_full[0]}.{local_full[1]}.{local_full[2]}\n"
                f"    endpoint: {endpoint_full[0]}.{endpoint_full[1]}.{endpoint_full[2]}\n"
                f"  Globus Compute pickling is sensitive to MAJOR.MINOR drift; "
                f"function calls may fail at runtime. Recreate the laptop venv "
                f"with python{endpoint_full[0]}.{endpoint_full[1]}.{endpoint_full[2]} "
                f"(see README Part 2).",
                file=sys.stderr,
            )
        elif endpoint_full != local_full:
            print(
                f"\n  Note: Python PATCH mismatch ({local_full[0]}.{local_full[1]}.{local_full[2]} "
                f"laptop vs {endpoint_full[0]}.{endpoint_full[1]}.{endpoint_full[2]} endpoint).\n"
                f"  Calls will still work, but the Globus Compute SDK will print a\n"
                f"  'UserWarning: Environment differences detected' on every call.\n"
                f"  To silence it, recreate the laptop venv pinned to {endpoint_full[0]}.{endpoint_full[1]}.{endpoint_full[2]}\n"
                f"  (e.g., `conda create -n nekrs-ml-agentic python={endpoint_full[0]}.{endpoint_full[1]}.{endpoint_full[2]} -y`).",
                file=sys.stderr,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="aurora", help="HPC system label (default: aurora).")
    parser.add_argument("--uuid", required=True, help="Globus Compute endpoint UUID from `globus-compute-endpoint list` on the HPC.")
    parser.add_argument("--repo-root", required=True, help="Absolute path to nekRS-ML on the HPC (the endpoint's view, not the laptop's).")
    parser.add_argument("--nekrs-home", default=None, help="Optional: default NEKRS_HOME on the HPC, stored alongside the endpoint entry.")
    parser.add_argument("--skip-register", action="store_true", help="Don't (re)register functions; keep whatever's already in functions.json.")
    parser.add_argument("--force-register", action="store_true", help="Re-register every function, replacing existing UUIDs.")
    parser.add_argument("--reauth", action="store_true", help="Clear cached Globus Compute tokens and force a fresh browser login.")
    parser.add_argument("--skip-ping", action="store_true", help="Don't run the round-trip ping at the end.")
    args = parser.parse_args()

    print(f"[setup] Writing endpoint entry for {args.system!r} ...")
    extras = {"nekrs_home": args.nekrs_home} if args.nekrs_home else None
    _write_endpoint(args.system, args.uuid, args.repo_root, extras)

    if args.skip_register:
        # Even with --skip-register the ping call needs auth, so do it unless
        # the user also skips the ping.
        if not args.skip_ping:
            rc = _prepare_auth(reauth=args.reauth)
            if rc != 0:
                return rc
        print("[setup] --skip-register: leaving functions.json untouched.")
    else:
        rc = _prepare_auth(reauth=args.reauth)
        if rc != 0:
            return rc
        print(f"\n[setup] Registering functions (force={args.force_register}) ...")
        rc = _register(force=args.force_register)
        if rc != 0:
            print(f"  registration failed (exit {rc})", file=sys.stderr)
            return rc

    if args.skip_ping:
        print("\n[setup] --skip-ping: deferring smoke test.")
        return 0
    return _ping(args.system)


if __name__ == "__main__":
    raise SystemExit(main())
