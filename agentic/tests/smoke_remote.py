"""Manual end-to-end smoke against a real Aurora endpoint.

Not a pytest. Run from the laptop after the endpoint is up and functions are
registered:

    python -m agentic.tests.smoke_remote

It calls ping() on the configured Aurora endpoint and prints what came back.
A successful run means the laptop -> Globus Compute -> Aurora-login-node path
is working end to end. After this, the same System handle can be used to drive
build_nekrs, setup_case, etc.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    from agentic.client.system import System

    try:
        hpc = System("aurora")
    except RuntimeError as e:
        print(f"setup error: {e}", file=sys.stderr)
        print(
            "Hint: ~/.config/nekrs-ml-agentic/endpoints.json must contain an 'aurora' entry "
            "with at least 'uuid'. Run `python -m agentic.client.setup --uuid <UUID> --repo-root <PATH>` "
            "to set it up.",
            file=sys.stderr,
        )
        return 2

    print(f"endpoint UUID: {hpc.endpoint_uuid}")
    print(f"repo_root    : {hpc.repo_root}")
    print("calling ping() ...")
    result = hpc.ping(message="smoke")
    print(json.dumps(result, indent=2, default=str))
    if not result.get("ok"):
        return 1
    print("\nOK: round-trip succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
