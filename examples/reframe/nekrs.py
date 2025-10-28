import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
from reframe.core.backends import getlauncher
from pathlib import Path
from core import CompileOnlyTest, RunOnlyTest
import os.path


def list_to_cmd(l):
    return " ".join(l)


class SmartRedisBuild(CompileOnlyTest):
    def __init__(self):
        super().__init__()
        self.descr = "SmartRedis build"
        self.maintainers = ["tratnayaka@anl.gov"]

    @run_before("compile")
    def configure_buld(self):
        self.sourcesdir = "https://github.com/rickybalin/SmartRedis.git"
        self.build_system = "Make"
        self.build_system.cc = "mpicc"
        self.build_system.cxx = "mpicxx"
        self.build_system.ftn = "mpif77"
        self.build_system.flags_from_environ = False
        self.build_system.max_concurrency = 16
        self.build_system.options = ["lib"]
        self.install_path = os.path.join(f"{self.stagedir}", "install")

    def get_install_path(self):
        return self.install_path


class NekRSBuild(CompileOnlyTest):
    version = variable(str, value="2024-11-22")
    smartredis_build = fixture(SmartRedisBuild, scope="environment")

    def __init__(self):
        super().__init__()
        self.descr = "nekRS-ML build"
        self.maintainers = ["kris.rowe@anl.gov", "tratnayaka@anl.gov"]

    @run_before("compile")
    def configure_build(self):
        self.sourcesdir = "https://github.com/argonne-lcf/nekRS-ML.git"
        self.build_system = "CMake"
        self.build_system.cc = "mpicc"
        self.build_system.cxx = "mpicxx"
        self.build_system.ftn = "mpif77"
        self.build_system.flags_from_environ = False
        self.build_system.builddir = "build"
        self.build_system.max_concurrency = 16
        self.build_system.make_opts = ["install"]
        self.install_path = os.path.join(f"{self.stagedir}", "install")
        self.binary_path = os.path.join(self.install_path, "bin")
        self.build_system.config_opts = [
            f"-DCMAKE_INSTALL_PREFIX={self.install_path}",
            "-DENABLE_ADIOS=OFF",
            "-DENABLE_SMARTREDIS=ON",
            f"-DSMARTREDIS_INSTALL_DIR={self.smartredis_build.get_install_path()}",
        ]

    @sanity_function
    def validate_build(self):
        nekrs_binary = os.path.join(self.binary_path, "nekrs")
        return sn.assert_true(
            os.path.isfile(nekrs_binary),
            f"nekrs binary could not be found in path {nekrs_binary}",
        )


# Encapsulate case-specific info
class NekRSCase:
    def __init__(self, name, directory=None):
        self._name = name
        if directory is None:
            directory = name
        self._directory = os.path.join(Path(os.getcwd()).parent, directory)

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
        self.descr = "nekRS-ML test"
        self.maintainers = ["kris.rowe@anl.gov"]
        self.tags = {"all"}
        self.case_name = nekrs_case.name
        self.sourcesdir = nekrs_case.directory
        self.readonly_files = [f"{nekrs_case.name}.re2"]

    @run_after("setup")
    def set_paths_exec(self):
        self.nekrs_home = os.path.realpath(self.nekrs_build.install_path)
        self.nekrs_binary = os.path.join(self.nekrs_build.binary_path, "nekrs")

    def set_environment(self):
        self.env_vars |= {
            "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{self.nekrs_build.install_path}/lib",
            "NEKRS_HOME": self.nekrs_home,
        }

    def set_launcher_options(self):
        cpu_bind_list = self.current_partition.extras["cpu_bind_list"]
        ranks_per_node = self.current_partition.extras["ranks_per_node"]
        total_ranks = self.num_nodes * ranks_per_node
        self.job.launcher.options = [
            f"-np {total_ranks}",
            f"-ppn {ranks_per_node}",
            f"--cpu-bind={cpu_bind_list}",
        ]

    def set_executable_options(self):
        self.executable = f"{self.nekrs_binary}"
        backend = self.current_partition.extras["backend"]
        self.executable_opts = [
            f"--setup {self.case_name}",
            f"--backend {backend}",
        ]

    @run_before("run")
    def setup_run(self):
        self.set_environment()
        self.set_executable_options()
        self.set_launcher_options()

    @sanity_function
    def check_exit_code(self):
        return sn.assert_found(
            r"finished with exit code 0",
            self.stdout,
            msg="NekRS finished with non-zero exit code.",
        )


class NekRSMLTest(NekRSTest):
    def __init__(self, nekrs_case):
        super().__init__(nekrs_case)
        self.tags |= {"ml"}

    def set_prerun_cmds(self):
        # Get mpiexec command and options:
        mpiexec = (
            self.job.launcher.command(self.job) + self.job.launcher.options
        )

        # Run nekrs:
        super().set_executable_options()
        nekrs = [self.executable] + self.executable_opts
        run_nekrs = list_to_cmd(mpiexec + nekrs)

        # Setup the gnn case:
        venv_path = os.path.join(self.stagedir, "_env")
        run_setup_case = list_to_cmd([
            os.path.join(Path(self.nekrs_home), "examples", "setup_case.sh"),
            self.current_system.name,
            self.nekrs_home,
            "--venv_path",
            venv_path,
        ])

        run_source_venv = list_to_cmd([
            "source",
            os.path.join(venv_path, "bin", "activate"),
        ])

        # halo_info
        self.gnn_output_dir = os.path.join(self.stagedir, "gnn_outputs_poly_3")
        halo_info = [
            "python",
            os.path.join(
                self.nekrs_home,
                "3rd_party",
                "dist-gnn",
                "create_halo_info_par.py",
            ),
            "--POLY",
            "3",
            "--PATH",
            self.gnn_output_dir,
        ]
        run_halo_info = list_to_cmd(mpiexec + halo_info)

        # check GNN input files
        run_check_input = list_to_cmd([
            "python",
            os.path.join(
                self.nekrs_home, "3rd_party", "dist-gnn", "check_input_files.py"
            ),
            "--REF",
            os.path.join(self.sourcesdir, "ref"),
            "--PATH",
            self.gnn_output_dir,
        ])

        # run all the pre-training steps
        self.prerun_cmds = [
            run_nekrs,
            run_setup_case,
            run_source_venv,
            run_halo_info,
            run_check_input,
        ]

    def set_executable_options(self):
        main = [
            "python",
            os.path.join(self.nekrs_home, "3rd_party", "dist-gnn", "main.py"),
        ]
        self.executable = list_to_cmd(main)
        # FIXME: master_addr=$head_node
        self.executable_opts = [
            # FIXME: backend should be calculated.
            "backend=xccl",
            "halo_swap_mode=all_to_all_opt",
            "layer_norm=True",
            f"gnn_outputs_path={self.gnn_output_dir}",
            "target_loss=1.6206e-04",
        ]

    @run_before("run")
    def setup_run(self):
        super().set_environment()
        # sets launcher and options which are used in the subsequent steps.
        super().set_launcher_options()
        # sets self.gnn_output_dir
        self.set_prerun_cmds()
        self.set_executable_options()

    @sanity_function
    def check_exit_code(self):
        nekrs_ok = super().check_exit_code()
        gnn_ok = sn.assert_found(
            r"SUCCESS! GNN training validated!",
            self.stdout,
            msg="GNN validation failed.",
        )

        return nekrs_ok and gnn_ok
