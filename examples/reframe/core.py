import os

import reframe as rfm
import reframe.utility.sanity as sn


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
        self.build_system = "Autotools"


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
            "max_local_jobs"
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
