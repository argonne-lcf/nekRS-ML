"""Guards that keep the login-node endpoint within policy.

Two invariants are enforced here:
  1. No compute launches on the login node (no mpiexec/mpirun/srun/aprun, etc.).
  2. Scheduler queries are throttled so the agent cannot accidentally DOS qstat.

These are belt-and-suspenders. The function surface in agentic.functions is
already narrow enough that compute commands should never appear; this module
exists so that any future careless addition fails loudly instead of silently
running on the login node.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from pathlib import Path

FORBIDDEN_LOGIN_NODE_COMMANDS = (
    "mpiexec",
    "mpirun",
    "srun",
    "aprun",
    "jsrun",
    "ibrun",
)

DEFAULT_QUERY_MIN_INTERVAL_S = 15.0

_query_state: dict[str, float] = {}
_query_lock = threading.Lock()


class LoginNodePolicyError(RuntimeError):
    """Raised when a command would violate login-node policy."""


def assert_login_node_safe(cmd: str | list[str]) -> None:
    """Refuse commands that would launch compute work on the login node.

    Accepts either a string (interpreted via shlex) or a token list.
    """
    tokens = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    for tok in tokens:
        base = os.path.basename(tok)
        if base in FORBIDDEN_LOGIN_NODE_COMMANDS:
            raise LoginNodePolicyError(
                f"Refusing to run {base!r} on a login node. "
                "Compute launches must go through the scheduler "
                "(submit_job)."
            )


def throttle_query(key: str, min_interval_s: float = DEFAULT_QUERY_MIN_INTERVAL_S) -> float:
    """Sleep until at least min_interval_s has passed since the last call for key.

    Returns the number of seconds slept (0.0 if no wait was needed). The throttle
    is per-key so concurrent queries against different job IDs don't block each
    other, but repeated polling of the same job ID is naturally rate-limited.

    Implemented with sleep-on-call so callers don't have to reason about retry
    or caching semantics: the call simply takes a bit longer when over-quota.
    """
    now = time.monotonic()
    with _query_lock:
        last = _query_state.get(key)
        if last is None:
            _query_state[key] = now
            return 0.0
        wait = (last + min_interval_s) - now
        if wait <= 0:
            _query_state[key] = now
            return 0.0
        # release the lock while we sleep so other keys aren't blocked
    time.sleep(wait)
    with _query_lock:
        _query_state[key] = time.monotonic()
    return wait


def validate_path_within(path: str | Path, roots: list[str | Path]) -> Path:
    """Resolve path and require it to live inside at least one allowed root.

    Prevents the agent from operating on files outside the project area
    (e.g., if it hallucinates an absolute path). The roots list is the
    set of directories the user has whitelisted in the endpoint config.
    """
    p = Path(path).expanduser().resolve()
    for root in roots:
        r = Path(root).expanduser().resolve()
        try:
            p.relative_to(r)
            return p
        except ValueError:
            continue
    raise PermissionError(
        f"Path {p} is outside the allowed roots: {[str(Path(r).expanduser().resolve()) for r in roots]}"
    )
