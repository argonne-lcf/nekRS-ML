import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
from reframe.core.backends import getlauncher
from pathlib import Path
from core import CompileOnlyTest, RunOnlyTest
import os.path


class NekRSBuild(CompileOnlyTest):
    version = variable(str, value="2024-11-22")

    def __init__(self):
        super().__init__()
        self.descr = "nekRS build"
        self.maintainers = ["kris.rowe@anl.gov", "tratnayaka@anl.gov"]
        self.tags = {"build"}

    @run_before("compile")
    def configure_build(self):
        self.sourcesdir = "https://github.com/argonne-lcf/nekRS-ML.git"
        self.build_system = "CMake"
        self.build_system.flags_from_environ = False
        self.build_system.builddir = "build"
        self.build_system.max_concurrency = 16
        self.build_system.make_opts = ["install"]
        self.install_path = os.path.join(f"{self.stagedir}", "install")
        self.binary_path = os.path.join(self.install_path, "bin")
        self.build_system.config_opts = [
            f"-DCMAKE_INSTALL_PREFIX={self.install_path}",
            "-DENABLE_ADIOS=OFF",
            "-DENABLE_SMARTREDIS=OFF",
        ]

    @sanity_function
    def validate_build(self):
        nekrs_binary = os.path.join(self.binary_path, "nekrs")
        return sn.assert_true(
            os.path.isfile(nekrs_binary),
            f"nekRS binary could not be found in path {nekrs_binary}",
        )


# Encapsulate case-specific info
class NekRSCase:
    def __init__(self, name):
        self._name = name
        self._directory = os.path.join(Path(os.getcwd()).parent, name)

    @property
    def name(self):
        return self._name

    @property
    def directory(self):
        return self._directory


class NekRSTest(RunOnlyTest):
    nekrs_build = fixture(NekRSBuild, scope="environment")

    def __init__(self, nekrs_case):
        super().__init__(num_nodes=1)
        self.descr = "nekRS test"
        self.maintainers = ["kris.rowe@anl.gov"]
        self.tags = {"nekrs"}
        self.case_name = nekrs_case.name
        self.sourcesdir = nekrs_case.directory
        self.readonly_files = [f"{nekrs_case.name}.re2"]

    @run_after("setup")
    def set_paths_exec(self):
        self.nekrs_home = os.path.realpath(self.nekrs_build.install_path)
        self.nekrs_binary = os.path.join(self.nekrs_build.binary_path, "nekrs")
        self.executable = f"gpu_tile_compact.sh {self.nekrs_binary}"

    def set_environment(self):
        self.env_vars |= {
            "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{self.nekrs_build.install_path}/lib",
            "NEKRS_HOME": self.nekrs_home,
        }

        if "OCCA_DPCPP_COMPILER_FLAGS" in self.current_partition.extras:
            self.env_vars |= {
                "OCCA_DPCPP_COMPILER_FLAGS": f'"{self.current_partition.extras["OCCA_DPCPP_COMPILER_FLAGS"]}"'
            }
        if "OCCA_CXX" in self.current_partition.extras:
            self.env_vars |= {
                "OCCA_CXX": f'"{self.current_partition.extras["OCCA_CXX"]}"'
            }
        if "OCCA_CXXFLAGS" in self.current_partition.extras:
            self.env_vars |= {
                "OCCA_CXXFLAGS": f'"{self.current_partition.extras["OCCA_CXXFLAGS"]}"'
            }

    def set_launcher_options(self):
        self.cpu_bind = self.current_partition.extras["cpu_bind_list"]
        self.ranks_per_node = self.current_partition.extras["max_local_jobs"]
        self.total_ranks = self.num_nodes * self.ranks_per_node
        self.job.launcher.options += [
            f"-np {self.total_ranks}",
            f"-ppn {self.ranks_per_node}",
            f"--cpu-bind={self.cpu_bind}",
        ]

    def set_executable_options(self):
        self.device_id = 0
        self.backend = self.current_partition.extras["backend"]
        self.executable_opts += [
            f"--setup {self.case_name}",
            f"--backend {self.backend}",
            f"--device-id {self.device_id}",
        ]

    @run_before("run")
    def setup_run(self):
        self.set_environment()
        self.set_launcher_options()
        self.set_executable_options()

    @sanity_function
    def check_exit_code(self):
        return sn.assert_found(
            r"finished with exit code 0",
            self.stdout,
            msg="finished with non-zero exit code.",
        )
