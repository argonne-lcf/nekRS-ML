# Right now this config is based on the default config of Aurora found under
# "/soft/tools/alcf_reframe/config".

from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher


@register_launcher("alcf_mpiexec")
class ALCF_MpiexecLauncher(JobLauncher):
    def command(self, job):
        return ["mpiexec"]


site_configuration = {
    "systems": [
        {
            "name": "aurora",
            "descr": "Aurora at ALCF",
            "modules_system": "lmod",
            "modules": ["frameworks"],
            "hostnames": [
                "^aurora-uan*",
                "^aurora-gateway-[0-9]{4}.*",
                "^x4[1-7]*",
            ],
            "partitions": [
                {
                    "name": "login",
                    "descr": "Login nodes",
                    "scheduler": "local",
                    "launcher": "alcf_mpiexec",
                    "environs": [
                        "PrgEnv-intel",
                    ],
                    "extras": {
                        "max_local_jobs": 12,
                        "cpu_bind_list": "list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99",
                    },
                },
                {
                    "name": "compute",
                    "descr": "Compute nodes",
                    "scheduler": "pbs",
                    "launcher": "alcf_mpiexec",
                    "max_jobs": 128,
                    "environs": [
                        "PrgEnv-intel",
                    ],
                    "extras": {
                        "max_local_jobs": 12,
                    },
                },
            ],
        },
    ],
    "environments": [
        {
            "name": "PrgEnv-intel",
            "modules": ["oneapi"],
            "prepare_cmds": ["module restore", "module list"],
            "target_systems": ["aurora"],
            "cc": "icx",
            "cxx": "icpx",
            "ftn": "ifx",
        },
    ],
    "logging": [
        {
            "perflog_compat": True,
            "handlers": [
                {
                    "type": "file",
                    "name": "reframe.log",
                    "level": "debug2",
                    "append": False,
                },
                {
                    "type": "stream",
                    "name": "stdout",
                    "level": "info",
                    "format": "%(message)s",
                },
                {
                    "type": "file",
                    "name": "reframe.out",
                    "level": "info",
                    "format": "%(message)s",
                    "append": False,
                },
            ],
            "handlers_perflog": [
                {
                    "type": "filelog",
                    "prefix": "%(check_system)s/%(check_partition)s",
                    "level": "info",
                    "format": (
                        "%(check_job_completion_time)s|"
                        "%(check_job_exitcode)s|"
                        "%(check_result)s|"
                        "%(check_info)s|"
                        "num_tasks=%(check_num_tasks)s|"
                        "jobid=%(check_jobid)s|"
                        "output=%(check_outputdir)s|"
                        "stage=%(check_stagedir)s|"
                        "%(check_perf_var)s=%(check_perf_value)s|"
                        "ref=%(check_perf_ref)s "
                        "(l=%(check_perf_lower_thres)s, "
                        "u=%(check_perf_upper_thres)s)|"
                        "%(check_perf_unit)s"
                    ),  # noqa: E501
                    "datefmt": "%FT%T%:z",
                    "append": True,
                },
                {
                    "type": "stream",
                    "name": "stdout",
                    "prefix": "%(check_system)s/%(check_partition)s",
                    "level": "info",
                    "format": (
                        "%(check_job_completion_time)s|reframe %(version)s|"
                        "%(check_info)s|jobid=%(check_jobid)s|"
                        "%(check_perf_var)s=%(check_perf_value)s|"
                        "ref=%(check_perf_ref)s "
                        "(l=%(check_perf_lower_thres)s, "
                        "u=%(check_perf_upper_thres)s)|"
                        "%(check_perf_unit)s"
                    ),
                    "append": True,
                },
            ],
        }
    ],
}
