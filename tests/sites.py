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
                        "PrgEnv-Aurora",
                    ],
                    "extras": {
                        "ranks_per_node": 16,
                    },
                },
                {
                    "name": "compute",
                    "descr": "Compute nodes",
                    "scheduler": "pbs",
                    "launcher": "alcf_mpiexec",
                    "max_jobs": 128,
                    "environs": [
                        "PrgEnv-Aurora",
                    ],
                    "env_vars": [
                        ["TZ", "/usr/share/zoneinfo/US/Central"],
                        ["ZE_FLAT_DEVICE_HIERARCHY", "FLAT"],
                        ["FI_CXI_RX_MATCH_MODE", "hybrid"],
                        ["UR_L0_USE_COPY_ENGINE", 0],
                        ["CCL_ALLTOALLV_MONOLITHIC_KERNEL", 0],
                    ],
                    "extras": {
                        "ranks_per_node": 12,
                        "cpu_bind_list": "1:8:16:24:32:40:53:60:68:76:84:92",
                        "db_bind_list": "100,101,102,103",
                        "gpu_bind_list": "0:1:2:3:4:5:6:7:8:9:10:11",
                        "backend": "DPCPP",
                    },
                },
            ],
        },
        {
            "name": "polaris",
            "descr": "Polaris at ALCF",
            "modules_system": "lmod",
            "modules": ["conda"],
            "hostnames": ["^polaris-login-[0-9]{2}.*", "^x3[1-7]{4}.*"],
            "partitions": [
                {
                    "name": "login",
                    "descr": "Login nodes",
                    "scheduler": "local",
                    "launcher": "alcf_mpiexec",
                    "environs": [
                        "PrgEnv-Polaris",
                    ],
                    "extras": {
                        # The NekRS build on Polaris fails with >= 12 parallel
                        # threads on Polaris. Maybe running out of memory?
                        "ranks_per_node": 10,
                    },
                },
                {
                    "name": "gateway",
                    "descr": "Gateway nodes",
                    "scheduler": "ssh",
                    "sched_options": {
                        "ssh_hosts": [
                            *(
                                f"polaris-gateway-{i:02}.hostmgmt.cm.polaris.alcf.anl.gov"
                                for i in range(1, 51)
                            )
                        ]
                    },
                    "launcher": "mpiexec",
                    "environs": [
                        "PrgEnv-Polaris",
                    ],
                },
                {
                    "name": "compute",
                    "descr": "Compute nodes",
                    "scheduler": "pbs",
                    "launcher": "alcf_mpiexec",
                    "max_jobs": 128,
                    "environs": ["PrgEnv-Polaris"],
                    "env_vars": [
                        ["TZ", "/usr/share/zoneinfo/US/Central"],
                        ["NEKRS_CACHE_BCAST", "0"],
                        ["NEKRS_LOCAL_TMP_DIR", "/local/scratch"],
                        ["NEKRS_GPU_MPI", "0"],
                        ["MPICH_MPIIO_STATS", "0"],
                        ["MPICH_GPU_SUPPORT_ENABLED", "0"],
                        ["MPICH_OFI_NIC_POLICY", "NUMA"],
                    ],
                    "extras": {
                        "ranks_per_node": 4,
                        "cpu_bind_list": "24:16:8:1",
                        "db_bind_list": "100,101,102,103",
                        "backend": "CUDA",
                    },
                },
            ],
        },
    ],
    "environments": [
        {
            "name": "PrgEnv-Aurora",
            "prepare_cmds": [
                "module restore",
                "module load frameworks",
                "module list",
            ],
            "env_vars": [
                ["OCCA_CXX", "icpx"],
                [
                    "OCCA_CXXFLAGS",
                    '"-O3 -g -fdebug-info-for-profiling -gline-tables-only"',
                ],
                [
                    "OCCA_DPCPP_COMPILER_FLAGS",
                    '"-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"',
                ],
            ],
            "target_systems": ["aurora"],
            "cc": "mpicc",
            "cxx": "mpicxx",
            "ftn": "mpif77",
        },
        {
            "name": "PrgEnv-Polaris",
            "prepare_cmds": [
                "module restore",
                "module load libfabric",
                "module load PrgEnv-gnu",
                "module use /soft/modulefiles/",
                "module load spack-pe-base cmake",
                "module load conda",
                "conda activate",
                "module list",
            ],
            "env_vars": [],
            "target_systems": ["polaris"],
            "cc": "cc",
            "cxx": "CC",
            "ftn": "ftn",
        },
    ],
    "logging": [
        {
            "perflog_compat": True,
            "handlers": [
                {
                    "type": "file",
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
