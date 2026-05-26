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
import re
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

# Shells that take a script path as their first non-flag argument. When we see
# `bash <script>`, the immediate command is `bash` (allowed) but the script
# itself may contain `mpiexec`. The guard peeks inside those scripts.
_SHELL_INTERPRETERS = ("bash", "sh", "zsh", "ksh", "dash")

DEFAULT_QUERY_MIN_INTERVAL_S = 15.0

_query_state: dict[str, float] = {}
_query_lock = threading.Lock()


class LoginNodePolicyError(RuntimeError):
    """Raised when a command would violate login-node policy."""


def assert_login_node_safe(cmd: str | list[str]) -> None:
    """Refuse commands that would launch compute work on the login node.

    Two layers of inspection:
      1. The immediate command tokens (`mpiexec`, `srun`, ...) -- direct
         attempts to launch compute on the login node.
      2. If the immediate command is a shell interpreter (bash/sh/zsh) running
         a script, the script contents are scanned for the same forbidden
         commands. Catches `bash run.sh` where run.sh contains `mpiexec`.

    Accepts either a string (interpreted via shlex) or a token list.
    """
    tokens = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)

    # Layer 1: direct command
    for tok in tokens:
        base = os.path.basename(tok)
        if base in FORBIDDEN_LOGIN_NODE_COMMANDS:
            raise LoginNodePolicyError(
                f"Refusing to run {base!r} on a login node. "
                "Compute launches must go through the scheduler "
                "(submit_job)."
            )

    # Layer 2: shell-interpreted script
    if not tokens:
        return
    interp = os.path.basename(tokens[0])
    if interp not in _SHELL_INTERPRETERS:
        return
    # Find the first positional arg after the shell -- skip option flags so
    # `bash -c "..."` and `bash -x script.sh` both resolve correctly.
    script_path: str | None = None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-c":
            # `bash -c "<inline>"` -- scan the inline string directly
            if i + 1 < len(tokens):
                _scan_text_for_forbidden(tokens[i + 1], origin=f"bash -c (inline)")
            return
        if tok.startswith("-"):
            i += 1
            continue
        script_path = tok
        break
    if script_path is None:
        return
    try:
        text = Path(script_path).expanduser().read_text(errors="replace")
    except OSError:
        # Script doesn't exist yet, isn't readable, etc. Defer to Layer 1 only.
        return
    _scan_text_for_forbidden(text, origin=script_path)


def _scan_text_for_forbidden(text: str, *, origin: str) -> None:
    """Refuse if `text` contains any FORBIDDEN_LOGIN_NODE_COMMANDS as a command
    invocation. Skips comment-only lines (first non-whitespace char is `#`) so
    `# this script does not call mpiexec` is not a false positive."""
    # Strip whole-line comments. Doesn't try to handle in-line trailing comments
    # (`mpiexec ... # comment`) -- if mpiexec appears in code AND a trailing
    # comment, it should still trigger.
    non_comment_lines = []
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        non_comment_lines.append(ln)
    scrubbed = "\n".join(non_comment_lines)
    for forbidden in FORBIDDEN_LOGIN_NODE_COMMANDS:
        # Match the command at start of line, after whitespace, or after a
        # pipe/&&/; -- but not as part of a larger word.
        pat = rf"(?:^|[\s|&;])(?:[\w./]*/)?{re.escape(forbidden)}\b"
        if re.search(pat, scrubbed, re.MULTILINE):
            raise LoginNodePolicyError(
                f"Refusing to bash-execute {origin!r} on a login node: it "
                f"contains {forbidden!r}. Compute scripts must go through "
                f"submit_job (which qsubs onto compute nodes)."
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
