"""Manual end-to-end check of the laptop -> endpoint pipeline.

Run from the laptop after `python -m agentic.client.setup ...` has succeeded.
The script exercises the agentic functions without involving Claude, so you
can tell "endpoint is healthy" apart from "the agent picked the right tool"
when something misbehaves.

What it does:

  1. Constructs `System(<system>)` and prints the endpoint UUID / repo root.
  2. Calls `ping()` and prints hostname / user / Python version.
  3. If `--case-dir` is given, calls `list_results(case_dir)` with a broad
     pattern set covering both input files (*.par, *.usr, ...) and post-run
     output (*.log, *.o<jobid>, *.fld*, ...). Prints the five most recent.
     If nothing matches, falls back to `*` so you can see whether the
     directory is empty vs the patterns are too narrow vs the path is wrong.
  4. If a file was found, tails the most recent one (default 20 lines).

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
    parser.add_argument(
        "--patterns",
        default=None,
        help=(
            "Comma-separated glob patterns for list_results. "
            "Defaults to a broad set that covers BOTH nekRS input files "
            "(*.par, *.usr, *.udf, *.re2, *.box, *.oudf, *.sh) and post-run "
            "output (*.log, *.o*, *.e*, *.fld*, out*, logfile*) -- so the "
            "test reports something whether or not the case has been run yet."
        ),
    )
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
    # Default to a broad pattern set: covers both fresh case dirs (input files
    # like *.par, *.usr) and post-run dirs (*.log, *.o<jobid>). The defaults in
    # list_results itself target post-run output only -- which is the right
    # default for the agent's normal use (finding fresh results), but makes
    # this smoke test useless on an unrun case directory. Override via --patterns.
    DEFAULT_SMOKE_PATTERNS = [
        "*.par", "*.usr", "*.udf", "*.re2", "*.box", "*.oudf", "*.sh",
        "*.log", "*.o*", "*.e*", "*.fld*", "out*", "logfile*",
    ]
    patterns = (
        [p.strip() for p in args.patterns.split(",") if p.strip()]
        if args.patterns else DEFAULT_SMOKE_PATTERNS
    )
    print(f"\n[smoke] list_results(case_dir={args.case_dir!r}) ...")
    print(f"        patterns={patterns}")
    lr = hpc.list_results(case_dir=args.case_dir, patterns=patterns)
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
        # Last-resort: glob "*" to confirm the directory actually has SOMETHING,
        # so the user can tell "patterns don't match" apart from "wrong path".
        print("  (no files matched the patterns; trying '*' to show what's there)")
        lr_all = hpc.list_results(case_dir=args.case_dir, patterns=["*"])
        all_files = (lr_all.get("files") or [])[:10]
        if not all_files:
            print("  directory appears empty (or path is wrong / unreadable)")
        else:
            print(f"  directory contains {len(lr_all.get('files', []))} entries; first 10:")
            for f in all_files:
                _print("  ->", f"{f['size']:>10}  {f['mtime']:.0f}  {f['path']}")
        print("  Pass --patterns to narrow, or point --case-dir at a post-run case.")
        return 0
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
