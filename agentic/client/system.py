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

    # ---- the one place that touches Globus Compute ------------------------

    def _call(self, name: str, /, **kwargs: Any) -> dict:
        try:
            fn_uuid = self._functions[name]
        except KeyError as e:
            raise RuntimeError(
                f"Function {name!r} not registered with Globus Compute. Run "
                f"`python -m agentic.client.register --only {name}`."
            ) from e
        with self._Executor(endpoint_id=self.endpoint_uuid) as ex:
            fut = ex.submit_to_registered_function(fn_uuid, kwargs=kwargs)
            return fut.result(timeout=self.timeout_s)
