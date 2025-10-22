import os
import re

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.schedulers.pbs import PbsJobScheduler


def _query_exit_code_fixed(self, job):
    """Try to retrieve the exit code of a past job."""

    # With PBS Pro we can obtain the exit status of a past job
    extended_info = osext.run_command(f"qstat -xf {job.jobid}")
    exit_status_match = re.search(
        r"^ *Exit_status *= *(?P<exit_status>-?\d+)",
        extended_info.stdout,
        flags=re.MULTILINE,
    )
    if exit_status_match:
        return int(exit_status_match.group("exit_status"))

    return None


class CompileOnlyTest(rfm.CompileOnlyRegressionTest):
    project = variable(str, value="")
    queue = variable(str, value="")
    walltime = variable(str, value="00:30:00")
    filesystems = variable(str, value="home")

    def __init__(self, walltime="1:00:00"):
        super().__init__()
        self.maintainers = ["tratnayaka@anl.gov"]
        self.valid_systems = ["*"]
        self.valid_prog_environs = ["*"]
        self.sourcesdir = None
        self.build_locally = True
        self.build_system = None

        # FIXME This is a ReFrame bug. Remove once it is fixed upstream.
        # https://github.com/reframe-hpc/reframe/pull/3571
        PbsJobScheduler._query_exit_code = _query_exit_code_fixed


class RunOnlyTest(rfm.RunOnlyRegressionTest):
    project = variable(str, value="")
    queue = variable(str, value="")
    walltime = variable(str, value="00:30:00")
    filesystems = variable(str, value="home")

    def __init__(self, num_nodes, walltime=None):
        super().__init__()
        self.maintainers = ["tratnayaka@anl.gov"]
        self.valid_systems = ["*"]
        self.valid_prog_environs = ["*"]

        self.num_nodes = num_nodes
        if walltime is not None:
            self.walltime = walltime

    @run_before("run")
    def set_scheduler_options(self):
        self.num_tasks_per_node = self.current_partition.extras[
            "ranks_per_node"
        ]
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = 1

        self.job.options = [
            f"-A {self.project}",
            f"-q {self.queue}",
            f"-l walltime={self.walltime}",
            f"-l filesystems={self.filesystems}",
        ]

    # https://github.com/reframe-hpc/reframe/pull/2993
    def get_job_exit_code(self):
        return self._current_partition.scheduler._query_exit_code(self.job)
