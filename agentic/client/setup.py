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
  2. Registers every function in agentic.functions with Globus Compute
  3. Runs ping() against the endpoint to confirm the round-trip

Pass --skip-register to keep existing UUIDs, --skip-ping to defer the smoke
check, --force-register to re-register everything (use after changing a
function body).
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
    from agentic.client.system import System

    print(f"\n[setup] Pinging endpoint for {system!r} ...")
    try:
        hpc = System(system)
    except RuntimeError as e:
        print(f"  setup error: {e}", file=sys.stderr)
        return 2
    result = hpc.ping(message="setup-smoke")
    if not result.get("ok"):
        print(f"  ping failed: {result.get('error')}", file=sys.stderr)
        return 1
    print(f"  hostname : {result.get('hostname')}")
    print(f"  user     : {result.get('user')}")
    print(f"  python   : {result.get('python', '').splitlines()[0]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="aurora", help="HPC system label (default: aurora).")
    parser.add_argument("--uuid", required=True, help="Globus Compute endpoint UUID from `globus-compute-endpoint list` on the HPC.")
    parser.add_argument("--repo-root", required=True, help="Absolute path to nekRS-ML on the HPC (the endpoint's view, not the laptop's).")
    parser.add_argument("--nekrs-home", default=None, help="Optional: default NEKRS_HOME on the HPC, stored alongside the endpoint entry.")
    parser.add_argument("--skip-register", action="store_true", help="Don't (re)register functions; keep whatever's already in functions.json.")
    parser.add_argument("--force-register", action="store_true", help="Re-register every function, replacing existing UUIDs.")
    parser.add_argument("--skip-ping", action="store_true", help="Don't run the round-trip ping at the end.")
    args = parser.parse_args()

    print(f"[setup] Writing endpoint entry for {args.system!r} ...")
    extras = {"nekrs_home": args.nekrs_home} if args.nekrs_home else None
    _write_endpoint(args.system, args.uuid, args.repo_root, extras)

    if args.skip_register:
        print("[setup] --skip-register: leaving functions.json untouched.")
    else:
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
