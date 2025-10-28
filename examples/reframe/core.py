import os
import re
import functools
import itertools
import time

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.schedulers.pbs import PbsJobScheduler

# ReFrame Docs
# ============
# ReFrame test stages/pipeline: https://reframe-hpc.readthedocs.io/en/stable/pipeline.html#the-regression-test-pipeline
# ReFrame pipeline hooks: https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html#pipeline-hooks
# ReFrame test decorators: https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html#test-decorators
# Builtins can be used to define essential test elements, such as variables, parameters, fixtures, pipeline hooks:
# https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html#builtins
# Test attributes to scheduler map:
# https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html#mapping-of-test-attributes-to-job-scheduler-backends


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
# FIXME: Consider making this a configuration parameter
PBS_OUTPUT_WRITEBACK_WAIT = 3

# Minimum amount of time between its submission and its cancellation. If you
# immediately cancel a PBS job after submission, its output files may never
# appear in the output causing the wait() to hang.
# FIXME: Consider making this a configuration parameter
PBS_CANCEL_DELAY = 3


_run_strict = functools.partial(osext.run_command, check=True)

# FIXME: adding "F" for completed since thats the way PBS does it
# (https://2021.help.altair.com/2021.1.2/PBS%20Professional/PBSHooks2021.1.2.pdf pg. 135).
JOB_STATES = {
    "Q": "QUEUED",
    "H": "HELD",
    "R": "RUNNING",
    "E": "EXITING",
    "T": "MOVED",
    "W": "WAITING",
    "S": "SUSPENDED",
    "C": "COMPLETED",
    "F": "COMPLETED",
}


def poll_fixed(self, *jobs):
    def output_ready(job):
        # We report a job as finished only when its stdout/stderr are
        # written back to the working directory
        stdout = os.path.join(job.workdir, job.stdout)
        stderr = os.path.join(job.workdir, job.stderr)
        return os.path.exists(stdout) and os.path.exists(stderr)

    if jobs:
        # Filter out non-jobs
        jobs = [job for job in jobs if job is not None]

    if not jobs:
        return

    completed = osext.run_command(
        f"qstat -f {' '.join(job.jobid for job in jobs)}"
    )
    # Depending on the configuration, completed jobs will remain on the job
    # list for a limited time, or be removed upon completion.
    # If qstat cannot find any of the job IDs, it will return 153.
    # Otherwise, it will return with return code 0 and print information
    # only for the jobs it could find.
    if completed.returncode == 153:
        self.log(f"Return code is {completed.returncode}")
        for job in jobs:
            job._state = "COMPLETED"
            if job.cancelled or output_ready(job):
                self.log(f"Assuming job {job.jobid} completed")
                job._completed = True
                job._exitcode = self._query_exit_code(job)

        return

    # Depending on the configuration, completed jobs will remain on the job
    # list for a limited time, or be removed upon completion.
    # If qstat cannot find any of the job IDs, it will return 153.
    # Otherwise, it will return with return code 0 and print information
    # only for the jobs it could find.
    if completed.returncode == 35:
        self.log(f"Return code is {completed.returncode}")
        for job in jobs:
            # FIXME: this is the only line modified. it is changed since
            # output_ready could be true when the job isn't actually done since
            # it just checks that stdout and stderr exist, but those exist once
            # the job starts running, not when it's done.
            if job.cancelled or (
                output_ready(job)
                and f"{job.jobid} Job has finished" in completed.stderr
            ):
                job._state = "COMPLETED"
                self.log(f"Assuming job {job.jobid} completed")
                job._completed = True
                job._exitcode = self._query_exit_code(job)

        return

    if completed.returncode != 0:
        raise JobSchedulerError(
            f"qstat failed with exit code {completed.returncode} "
            f"(standard error follows):\n{completed.stderr}"
        )

    # Store information for each job separately
    jobinfo = {}
    for job_raw_info in completed.stdout.split("\n\n"):
        jobid_match = re.search(
            r"^Job Id:\s*(?P<jobid>\S+)", job_raw_info, re.MULTILINE
        )
        if jobid_match:
            jobid = jobid_match.group("jobid")
            jobinfo[jobid] = job_raw_info

    for job in jobs:
        if job.jobid not in jobinfo:
            self.log(f"Job {job.jobid} not known to scheduler")
            job._state = "COMPLETED"
            if job.cancelled or output_ready(job):
                self.log(f"Assuming job {job.jobid} completed")
                job._completed = True

            continue

        info = jobinfo[job.jobid]
        state_match = re.search(
            r"^\s*job_state = (?P<state>[A-Z])", info, re.MULTILINE
        )
        if not state_match:
            self.log(f"Job state not found (job info follows):\n{info}")
            continue

        state = state_match.group("state")
        job._state = JOB_STATES[state]
        nodelist_match = re.search(
            r"exec_host = (?P<nodespec>[\S\t\n]+)", info, re.MULTILINE
        )
        self.log(f"jobs: {job.state}")
        if nodelist_match:
            nodespec = nodelist_match.group("nodespec")
            nodespec = re.sub(r"[\n\t]*", "", nodespec)
            self._update_nodelist(job, nodespec)
        # FIXME: will likely never get here since qstat -f is used
        if job.state == "COMPLETED":
            exitcode_match = re.search(
                r"^\s*exit_status = (?P<code>\d+)",
                info,
                re.MULTILINE,
            )
            if exitcode_match:
                job._exitcode = int(exitcode_match.group("code"))

            # We report a job as finished only when its stdout/stderr are
            # written back to the working directory
            done = job.cancelled or output_ready(job)
            if done:
                job._completed = True
        elif (
            job.state in ["QUEUED", "HELD", "WAITING"] and job.max_pending_time
        ):
            if time.time() - job.submit_time >= job.max_pending_time:
                self.cancel(job)
                job._exception = JobError(
                    "maximum pending time exceeded", job.jobid
                )


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


def wait_fixed(self, job):
    intervals = itertools.cycle([5, 6, 7])
    while not self.finished(job):
        self.poll(job)
        time.sleep(next(intervals))


class CompileOnlyTest(rfm.CompileOnlyRegressionTest):
    project = variable(str, value="")
    queue = variable(str, value="")
    filesystems = variable(str, value="")
    walltime = variable(str, value="01:00:00")

    def __init__(self):
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
        # FIXME: We are increasing wait times to avoid polling too often.
        PbsJobScheduler.wait = wait_fixed
        # FIXME: Colleen testing the poll
        PbsJobScheduler.poll = poll_fixed


class RunOnlyTest(rfm.RunOnlyRegressionTest):
    project = variable(str, value="")
    queue = variable(str, value="")
    filesystems = variable(str, value="")
    walltime = variable(str, value="01:00:00")

    def __init__(self, num_nodes, ranks_per_node):
        super().__init__()
        self.maintainers = ["tratnayaka@anl.gov"]
        self.valid_systems = ["*"]
        self.valid_prog_environs = ["*"]

        self.num_nodes = num_nodes
        self.num_tasks_per_node = ranks_per_node

        # FIXME This is a ReFrame bug. Remove once it is fixed upstream.
        # https://github.com/reframe-hpc/reframe/pull/3571
        PbsJobScheduler._query_exit_code = _query_exit_code_fixed
        # FIXME: We are increasing wait times to avoid polling too often.
        PbsJobScheduler.wait = wait_fixed
        # FIXME: Colleen testing the poll
        PbsJobScheduler.poll = poll_fixed

    @run_before("run")
    def set_scheduler_options(self):
        max_rpn = self.current_partition.extras["ranks_per_node"]
        if self.num_tasks_per_node > max_rpn:
            import warnings

            warnings.warn(
                (
                    f"Requested ranks per node ({self.num_tasks_per_node}) is larger "
                    f"than the maximum value of the system({max_rpn})"
                ),
                RuntimeWarning,
            )

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
