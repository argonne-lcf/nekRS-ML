"""Manual end-to-end check of the laptop -> endpoint pipeline.

Run from the laptop after `python -m agentic.client.setup ...` has succeeded.
The script exercises the agentic functions without involving Claude, so you
can tell "endpoint is healthy" apart from "the agent picked the right tool"
when something misbehaves.

What it does:

  1. Constructs `System(<system>)` and prints the endpoint UUID / repo root.
  2. Calls `ping()` and prints hostname / user / Python version.
  3. If `--case-dir` is given, calls `list_results(case_dir)` and prints the
     five most recently modified output files.
  4. If `--case-dir` is given AND there's at least one matching file, tails
     the most recent one (default 20 lines).

Examples:

    # minimum: just confirm the pipeline works
    python -m agentic.tests.smoke_remote

    # also exercise list_results + tail_log against a real case
    python -m agentic.tests.smoke_remote \
        --case-dir /lus/flare/projects/myproj/me/nekRS-ML/examples/tgv_gnn_offline

Exit codes: 0 success, 1 a remote call returned ok=False, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import sys


def _print(label: str, value) -> None:
    print(f"  {label:18} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="aurora", help="HPC system label (default: aurora).")
    parser.add_argument("--case-dir", default=None, help="Absolute path on the HPC to a case directory. If set, exercise list_results + tail_log.")
    parser.add_argument("--tail-lines", type=int, default=20, help="Lines to tail from the most recent file (default: 20).")
    parser.add_argument("--json", action="store_true", help="Print full JSON results in addition to the pretty summary.")
    args = parser.parse_args()

    from agentic.client.system import System

    print(f"[smoke] Connecting to {args.system!r} ...")
    try:
        hpc = System(args.system)
    except RuntimeError as e:
        print(f"  setup error: {e}", file=sys.stderr)
        print(
            "  Hint: run `python -m agentic.client.setup --uuid <UUID> --repo-root <PATH>` first.",
            file=sys.stderr,
        )
        return 2
    _print("endpoint UUID:", hpc.endpoint_uuid)
    _print("repo_root:", hpc.repo_root)
    _print("nekrs_home:", hpc.nekrs_home or "<unset; pass --nekrs-home to setup if you want a default>")

    # ---- 1. ping -----------------------------------------------------------
    print("\n[smoke] ping() ...")
    ping = hpc.ping(message="smoke")
    if args.json:
        print(json.dumps(ping, indent=2, default=str))
    if not ping.get("ok"):
        print(f"  FAILED: {ping.get('error')}", file=sys.stderr)
        return 1
    _print("hostname:", ping.get("hostname"))
    _print("user:", ping.get("user"))
    _print("python:", (ping.get("python", "").splitlines() or [""])[0])
    _print("duration_s:", f"{ping.get('duration_s', 0):.3f}")

    if not args.case_dir:
        print("\n[smoke] OK. (Pass --case-dir to also exercise list_results + tail_log.)")
        return 0

    # ---- 2. list_results ---------------------------------------------------
    print(f"\n[smoke] list_results(case_dir={args.case_dir!r}) ...")
    lr = hpc.list_results(case_dir=args.case_dir)
    if args.json:
        print(json.dumps(lr, indent=2, default=str))
    if not lr.get("ok"):
        print(f"  FAILED: {lr.get('error')}", file=sys.stderr)
        return 1
    files = lr.get("files", [])
    _print("files found:", len(files))
    for f in files[:5]:
        _print("  ->", f"{f['size']:>10}  {f['mtime']:.0f}  {f['path']}")
    if not files:
        print("  (no files matched; nothing to tail.)")
        return 0

    # ---- 3. tail_log on the most recent file -------------------------------
    target = files[0]["path"]
    print(f"\n[smoke] tail_log(path={target!r}, n_lines={args.tail_lines}) ...")
    tl = hpc.tail_log(path=target, n_lines=args.tail_lines)
    if args.json:
        print(json.dumps(tl, indent=2, default=str))
    if not tl.get("ok"):
        print(f"  FAILED: {tl.get('error')}", file=sys.stderr)
        return 1
    _print("file size:", tl.get("file_size_bytes"))
    _print("lines returned:", tl.get("n_lines_returned"))
    print("  ---- tail ----")
    for line in (tl.get("stdout") or "").splitlines():
        print(f"    {line}")
    print("  --------------")

    print("\n[smoke] OK. ping + list_results + tail_log all succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
