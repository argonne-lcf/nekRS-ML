"""Register all agentic.functions with Globus Compute and persist their UUIDs.

Run once on the laptop after you have the endpoint UUID:

    python -m agentic.client.register

This walks agentic.functions.REGISTERED_FUNCTIONS, registers each with the
Globus Compute service, and writes the resulting {name: function_uuid} map to
~/.config/nekrs-ml-agentic/functions.json. The System class reads from that file.

Re-running is safe: existing UUIDs are kept unless --force is passed, which
re-registers every function (use this after you change a function's signature
or body).
"""

from __future__ import annotations

import argparse
import importlib

from agentic.client._config import load_functions, save_functions
from agentic.functions import REGISTERED_FUNCTIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-register every function, replacing existing UUIDs.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Register only the named function(s). May be repeated.",
    )
    args = parser.parse_args()

    from globus_compute_sdk import Client

    client = Client()
    existing = load_functions()
    fn_module = importlib.import_module("agentic.functions")

    targets = args.only or list(REGISTERED_FUNCTIONS)
    for name in targets:
        if name not in REGISTERED_FUNCTIONS:
            print(f"  skip {name}: not in REGISTERED_FUNCTIONS")
            continue
        if name in existing and not args.force:
            print(f"  keep {name}: {existing[name]}")
            continue
        fn = getattr(fn_module, name)
        uuid = client.register_function(fn, function_name=f"nekrs_ml_{name}")
        existing[name] = uuid
        action = "re-registered" if name in existing else "registered"
        print(f"  {action} {name}: {uuid}")

    save_functions(existing)
    print(f"\nWrote {len(existing)} function UUIDs to ~/.config/nekrs-ml-agentic/functions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
