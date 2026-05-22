"""Per-system scheduler dispatch.

v1 covers Aurora (PBS Pro). Polaris will reuse the PBS adapter when added;
Frontier (Slurm) will get its own adapter. The interface is kept small so
the agentic.functions layer never branches on system internals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# PBS qsub prints the full job ID on stdout, e.g. "12345.aurora-pbs-0001.hostmgmt2000.cm.americas.sgi.com"
_PBS_JOBID_RE = re.compile(r"^\s*(\d+\.[A-Za-z0-9._-]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Scheduler:
    name: str
    submit_cmd: Sequence[str]
    status_cmd_template: Sequence[str]
    cancel_cmd: Sequence[str]

    def parse_submit_output(self, stdout: str) -> str:
        raise NotImplementedError

    def parse_status_output(self, stdout: str) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class PBSScheduler(Scheduler):
    def parse_submit_output(self, stdout: str) -> str:
        # qsub may emit the job ID alone on its line, often the last non-empty line
        m = _PBS_JOBID_RE.search(stdout)
        if m:
            return m.group(1)
        # Fallback: last non-empty line
        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"could not parse PBS job id from qsub output: {stdout!r}")
        return lines[-1]

    def parse_status_output(self, stdout: str) -> dict:
        """Parse `qstat -f -F json` output if available, else `qstat -f` text.

        We keep this tolerant: callers should treat the dict as best-effort and
        fall back to inspecting raw output when fields are missing.
        """
        import json
        try:
            data = json.loads(stdout)
        except (ValueError, json.JSONDecodeError):
            # Fall back to crude text parsing
            fields = {}
            for ln in stdout.splitlines():
                if "=" in ln:
                    k, _, v = ln.partition("=")
                    fields[k.strip()] = v.strip()
            return {"raw": stdout, "fields": fields}
        jobs = data.get("Jobs") or {}
        if not jobs:
            return {"raw": stdout, "fields": {}}
        # Single job lookup: return the one entry
        (job_id, attrs), = jobs.items()
        return {
            "job_id": job_id,
            "state": attrs.get("job_state"),
            "exit_status": attrs.get("Exit_status"),
            "queue": attrs.get("queue"),
            "stime": attrs.get("stime"),
            "mtime": attrs.get("mtime"),
            "fields": attrs,
        }


_SCHEDULERS = {
    "aurora": PBSScheduler(
        name="pbs",
        submit_cmd=("qsub",),
        status_cmd_template=("qstat", "-f", "-F", "json"),
        cancel_cmd=("qdel",),
    ),
}


def get_scheduler(system: str) -> Scheduler:
    sys_key = system.lower()
    try:
        return _SCHEDULERS[sys_key]
    except KeyError:
        raise ValueError(
            f"Unsupported system {system!r}. Supported: {sorted(_SCHEDULERS)}"
        )
