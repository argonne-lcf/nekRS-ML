"""Local-only unit tests that don't need an HPC endpoint.

Run with: pytest agentic/tests/test_local.py
"""

from __future__ import annotations

import time

import pytest

from agentic import safety
from agentic.schedulers import PBSScheduler, get_scheduler


def test_assert_login_node_safe_allows_normal_commands():
    safety.assert_login_node_safe(["bash", "BuildMeOnAurora", "/home/me/.local/nekrs"])
    safety.assert_login_node_safe(["qsub", "submit.sh"])
    safety.assert_login_node_safe("cp run.sh /tmp/")


@pytest.mark.parametrize(
    "cmd",
    [
        ["mpiexec", "-n", "12", "./nekrs"],
        ["mpirun", "-np", "4", "a.out"],
        ["srun", "--ntasks=4", "true"],
        "aprun -n 4 ./bin",
    ],
)
def test_assert_login_node_safe_rejects_compute_launches(cmd):
    with pytest.raises(safety.LoginNodePolicyError):
        safety.assert_login_node_safe(cmd)


def test_throttle_query_first_call_does_not_wait():
    safety._query_state.clear()  # reset module state
    waited = safety.throttle_query("k1", min_interval_s=0.5)
    assert waited == 0.0


def test_throttle_query_repeat_call_sleeps():
    safety._query_state.clear()
    safety.throttle_query("k2", min_interval_s=0.2)
    start = time.monotonic()
    waited = safety.throttle_query("k2", min_interval_s=0.2)
    elapsed = time.monotonic() - start
    assert waited > 0
    assert elapsed >= 0.15  # allow slack on slow CI


def test_throttle_query_different_keys_independent():
    safety._query_state.clear()
    safety.throttle_query("a", min_interval_s=0.5)
    waited = safety.throttle_query("b", min_interval_s=0.5)
    assert waited == 0.0


def test_pbs_scheduler_parses_simple_job_id():
    sched = get_scheduler("aurora")
    assert isinstance(sched, PBSScheduler)
    parsed = sched.parse_submit_output("12345.aurora-pbs-0001.hostmgmt2000.cm.americas.sgi.com\n")
    assert parsed.startswith("12345.aurora-pbs-")


def test_get_scheduler_rejects_unknown_system():
    with pytest.raises(ValueError):
        get_scheduler("frontier")  # not yet wired in v1
