"""Laptop-side handle for invoking registered functions on one HPC system.

A `System(name)` instance represents a connection to one HPC site (Aurora,
Polaris, ...) through its Globus Compute endpoint. Any function name listed
in agentic.functions.REGISTERED_FUNCTIONS is callable as a method on the
instance; the method is resolved dynamically by __getattr__, the function's
signature is inspected, and `system` / `repo_root` are auto-injected from
the endpoints.json entry when the underlying function accepts them.

That means adding a new remote function only touches functions.py + its
addition to REGISTERED_FUNCTIONS; no edits here are needed.

Example:

    from agentic.client.system import System

    hpc = System("aurora")
    print(hpc.ping(message="from-laptop")["hostname"])
    out = hpc.build_nekrs(nekrs_home="/home/me/.local/nekrs")
    setup = hpc.setup_case(case_dir="/lus/.../examples/tgv_gnn_offline",
                           nekrs_home="/home/me/.local/nekrs",
                           options={"nodes": 2, "time": "01:00",
                                    "proj_id": "myproj", "model": "dist-gnn",
                                    "deployment": "offline"})
    job = hpc.submit_job(script_path=setup["generated_scripts"][-1])
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from agentic.client._config import load_functions, require_endpoint
from agentic.functions import REGISTERED_FUNCTIONS


class System:
    """Handle for one HPC system's Globus Compute endpoint."""

    def __init__(self, name: str, timeout_s: float = 300.0):
        self.name = name
        self.timeout_s = timeout_s
        self._endpoint = require_endpoint(name)
        self._functions = load_functions()
        if not self._functions:
            raise RuntimeError(
                "No registered functions found. Run "
                "`python -m agentic.client.setup --uuid <UUID> --repo-root <PATH>` "
                "(or `python -m agentic.client.register` if you only need to refresh "
                "function UUIDs)."
            )
        from globus_compute_sdk import Executor  # noqa: F401, deferred to keep module import light
        self._Executor = Executor
        # Cached, reused across calls. Globus Compute's recommended pattern is
        # one Executor per (endpoint, process) -- opening a fresh one per call
        # not only churns AMQP connections but can itself trigger the 409
        # RESOURCE_CONFLICT we used to retry around.
        self._executor = None

    # ---- read-only views of the endpoint config ---------------------------

    @property
    def repo_root(self) -> str | None:
        return self._endpoint.get("repo_root")

    @property
    def nekrs_home(self) -> str | None:
        return self._endpoint.get("nekrs_home")

    @property
    def endpoint_uuid(self) -> str:
        return self._endpoint["uuid"]

    # ---- dynamic method dispatch ------------------------------------------

    def __getattr__(self, attr: str) -> Callable[..., dict]:
        # __getattr__ is only called for misses, so dunders and real attrs
        # don't reach here. Guard anyway to avoid surprises during pickling
        # or repr.
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr not in REGISTERED_FUNCTIONS:
            raise AttributeError(
                f"{type(self).__name__!r} has no remote function {attr!r}. "
                f"Available: {sorted(REGISTERED_FUNCTIONS)}"
            )

        # Inspect the local copy of the function to decide what to auto-inject.
        # The function actually runs on the endpoint, but its signature is the
        # same on both sides because the agentic package is installed on each.
        from agentic import functions as _fn

        fn = getattr(_fn, attr)
        params = inspect.signature(fn).parameters

        def call(**kwargs: Any) -> dict:
            if "system" in params and "system" not in kwargs:
                kwargs["system"] = self.name
            if "repo_root" in params and "repo_root" not in kwargs and self.repo_root:
                kwargs["repo_root"] = self.repo_root
            if "nekrs_home" in params and "nekrs_home" not in kwargs and self.nekrs_home:
                kwargs["nekrs_home"] = self.nekrs_home
            return self._call(attr, **kwargs)

        call.__name__ = attr
        call.__doc__ = fn.__doc__
        return call

    def __dir__(self) -> list[str]:
        # Make tab-completion in REPLs reveal the dynamic remote functions.
        return sorted(set(super().__dir__()) | set(REGISTERED_FUNCTIONS))

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Shut down the cached Executor and release its AMQP connection.

        Called automatically by __exit__ / __del__. Safe to call multiple
        times. After close(), the next _call will lazily build a fresh
        Executor -- which is what the 409-retry path relies on.
        """
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except Exception:
                pass  # best-effort; we're about to drop the reference anyway
            finally:
                self._executor = None

    def __enter__(self) -> "System":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):
        # Best-effort cleanup if the user forgets to call close() / use a
        # context manager. Wrapped in try/except because __del__ runs during
        # interpreter teardown when imports may be gone.
        try:
            self.close()
        except Exception:
            pass

    # ---- the one place that touches Globus Compute ------------------------

    # The Globus Compute service occasionally returns 409 RESOURCE_CONFLICT
    # ("endpoint is already in use: possibly due to concurrent requests --
    # please try again"). The recommended fix is to drop our Executor (which
    # may itself be the source of the conflict if its AMQP connection is in a
    # bad state) and retry with a fresh one. Bounded so a wedged endpoint
    # doesn't hang the agent indefinitely.
    _MAX_CONFLICT_RETRIES = 4
    _INITIAL_CONFLICT_BACKOFF_S = 1.0

    def _get_executor(self):
        if self._executor is None:
            self._executor = self._Executor(endpoint_id=self.endpoint_uuid)
        return self._executor

    def _call(self, name: str, /, **kwargs: Any) -> dict:
        import time

        try:
            fn_uuid = self._functions[name]
        except KeyError as e:
            raise RuntimeError(
                f"Function {name!r} not registered with Globus Compute. Run "
                f"`python -m agentic.client.register --only {name}`."
            ) from e

        backoff = self._INITIAL_CONFLICT_BACKOFF_S
        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_CONFLICT_RETRIES + 1):
            try:
                ex = self._get_executor()
                fut = ex.submit_to_registered_function(fn_uuid, kwargs=kwargs)
                return fut.result(timeout=self.timeout_s)
            except Exception as e:
                msg = str(e)
                is_conflict = "RESOURCE_CONFLICT" in msg or ("409" in msg and "in use" in msg)
                if is_conflict:
                    last_exc = e
                    # The cached Executor may itself be the source of the
                    # conflict (stale AMQP claim). Drop it so the next attempt
                    # builds a fresh one.
                    self.close()
                    if attempt < self._MAX_CONFLICT_RETRIES:
                        print(
                            f"  endpoint busy (409 RESOURCE_CONFLICT); "
                            f"dropped Executor, retrying in {backoff:.1f}s "
                            f"(attempt {attempt}/{self._MAX_CONFLICT_RETRIES})"
                        )
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                # Either non-retryable or out of attempts: surface the original
                # with actionable guidance for the persistent case.
                if is_conflict:
                    import sys as _sys
                    print(
                        f"\nGlobus Compute keeps returning 409 RESOURCE_CONFLICT for endpoint\n"
                        f"  {self.endpoint_uuid}\n"
                        f"Persistent (not transient) 409 almost always means one of:\n"
                        f"\n"
                        f"  1. UUID MISMATCH (most common): your endpoints.json points at an OLD\n"
                        f"     endpoint UUID that no longer matches the live one on the HPC. On the\n"
                        f"     HPC, run\n"
                        f"         globus-compute-endpoint list\n"
                        f"     and note the *Running* UUID. If it differs from the one above, fix it:\n"
                        f"         python -m agentic.client.setup --uuid <NEW UUID> \\\n"
                        f"             --repo-root <PATH> --force-register\n"
                        f"     If a stale UUID is also listed (Disconnected), purge it:\n"
                        f"         globus-compute-endpoint delete <stale_name>\n"
                        f"         rm -rf ~/.globus_compute/<stale_name>\n"
                        f"\n"
                        f"  2. Wedged endpoint: restart it on the HPC.\n"
                        f"         globus-compute-endpoint stop <name>\n"
                        f"         globus-compute-endpoint start <name> --detach\n"
                        f"\n"
                        f"  3. Check the endpoint log for clues:\n"
                        f"         tail -100 ~/.globus_compute/<name>/endpoint.log",
                        file=_sys.stderr,
                    )
                raise
        assert last_exc is not None
        raise last_exc
