"""Local config locations.

We keep endpoint UUIDs and function UUIDs out of the repo. They live under
~/.config/nekrs-ml-agentic/ on the laptop. The format is JSON and the schema
is intentionally minimal so the agent can read/write them with a single
import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path(
    os.environ.get(
        "NEKRS_ML_AGENTIC_CONFIG_DIR",
        Path.home() / ".config" / "nekrs-ml-agentic",
    )
)
ENDPOINTS_FILE = CONFIG_DIR / "endpoints.json"
FUNCTIONS_FILE = CONFIG_DIR / "functions.json"


def load_endpoints() -> dict:
    if not ENDPOINTS_FILE.exists():
        return {}
    return json.loads(ENDPOINTS_FILE.read_text())


def load_functions() -> dict:
    if not FUNCTIONS_FILE.exists():
        return {}
    return json.loads(FUNCTIONS_FILE.read_text())


def save_functions(mapping: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    FUNCTIONS_FILE.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def save_endpoints(mapping: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ENDPOINTS_FILE.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def require_endpoint(system: str) -> dict:
    eps = load_endpoints()
    if system not in eps:
        raise RuntimeError(
            f"No endpoint configured for {system!r}. "
            f"Add it to {ENDPOINTS_FILE} with at least a 'uuid' and 'repo_root'."
        )
    ep = eps[system]
    if "uuid" not in ep:
        raise RuntimeError(f"{ENDPOINTS_FILE} entry for {system!r} is missing 'uuid'.")
    return ep
