import subprocess
import reframe as rfm
import reframe.utility.sanity as sn
from pathlib import Path
from functools import cache
from core import CompileOnlyTest, RunOnlyTest
import os.path


def list_to_cmd(l):
    return " ".join(l)


def grep(pattern, file):
    return subprocess.run(
        [
            "grep",
            "-i",
            pattern,
            file,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def check_args(user_args, required_args):
    for arg in required_args:
        if arg not in user_args:
            raise KeyError(f"Required kwarg {arg} was not found.")


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


class NekRSTest(RunOnlyTest):
    nekrs_build = fixture(NekRSBuild, scope="environment")

    def __init__(self, case, directory, nn, rpn):
        self.nekrs_case_name = case
        self.nekrs_case_dir = directory

        super().__init__(num_nodes=nn, ranks_per_node=rpn)
        self.descr = "NekRS test"
        self.maintainers = ["kris.rowe@anl.gov"]
        self.readonly_files = [f"{self.nekrs_case_name}.re2"]

    @run_after("setup")
    def set_paths_exec(self):
        self.nekrs_home = os.path.realpath(self.nekrs_build.install_path)
        self.nekrs_binary = os.path.join(self.nekrs_build.binary_path, "nekrs")
        self.sourcesdir = os.path.join(
            self.nekrs_build.install_path, "examples", self.nekrs_case_dir
        )

    def set_environment(self):
        self.env_vars |= {
            "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{self.nekrs_build.install_path}/lib",
            "NEKRS_HOME": self.nekrs_home,
        }

    def set_launcher_options(self):
        cpu_bind_list = self.current_partition.extras["cpu_bind_list"]
        ranks_per_node = self.num_tasks_per_node
        total_ranks = self.num_nodes * ranks_per_node
        self.job.launcher.options = [
            f"-np {total_ranks}",
            f"-ppn {ranks_per_node}",
            f"--cpu-bind={cpu_bind_list}",
        ]

    def set_executable_options(self):
        self.executable = f"{self.current_partition.extras['nrs_affinity']} {self.nekrs_binary}"
        backend = self.current_partition.extras["backend"]
        self.executable_opts = [
            f"--setup {self.nekrs_case_name}",
            f"--backend {backend}",
            "--device-id 0",
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
    def __init__(self, **kwargs):
        # Check if the required arguments are in kwargs.
        check_args(
            kwargs, ["case", "directory", "nn", "rpn", "time_dependency"]
        )

        self.ml_kwargs = kwargs.copy()
        super().__init__(
            self.ml_kwargs["case"],
            self.ml_kwargs["directory"],
            self.ml_kwargs["nn"],
            self.ml_kwargs["rpn"],
        )
        self.descr = "NekRS-ML test"

    @cache
    def get_mpiexec(self):
        return self.job.launcher.command(self.job) + self.job.launcher.options

    @cache
    def get_order(self):
        par_file = os.path.join(self.sourcesdir, f"{self.nekrs_case_name}.par")
        result = grep("gnnPolynomialOrder", par_file)
        if result.stdout is None:
            result = grep("polynomialOrder", par_file)
        return int(result.stdout.split()[2])

    @cache
    def get_venv_path(self):
        return os.path.join(self.stagedir, "_env")

    @cache
    def get_gnn_output_dir(self):
        order = self.get_order()
        return os.path.join(self.stagedir, f"gnn_outputs_poly_{order}")

    @cache
    def get_traj_root(self):
        order = self.get_order()
        return os.path.join(f"traj_poly_{order}", "tinit_0.000000_dtfactor_10")

    @cache
    def get_traj_dir(self):
        return os.path.join(self.stagedir, self.get_traj_root())

    @cache
    def get_check_input_files(self):
        return os.path.join(
            self.nekrs_home, "3rd_party", "dist-gnn", "check_input_files.py"
        )

    @cache
    def get_ranks(self):
        return self.ml_kwargs["nn"] * self.ml_kwargs["rpn"]

    def nekrs_cmd(self, extra_args=[]):
        # Set nekrs executable options used in NekRSTest class.
        super().set_executable_options()
        return list_to_cmd(
            self.get_mpiexec()
            + [self.executable]
            + self.executable_opts
            + extra_args
        )

    def setup_case_cmd(self, extra_args=[]):
        return list_to_cmd([
            os.path.join(Path(self.nekrs_home), "bin", "setup_case"),
            self.current_system.name,
            self.nekrs_home,
            "--venv_path",
            self.get_venv_path(),
            *extra_args,
        ])

    def source_cmd(self):
        return list_to_cmd([
            "source",
            os.path.join(self.get_venv_path(), "bin", "activate"),
        ])

    def check_halo_info_cmd(self):
        halo_info = [
            "python",
            os.path.join(
                self.nekrs_home,
                "3rd_party",
                "dist-gnn",
                "create_halo_info_par.py",
            ),
            "--POLY",
            str(self.get_order()),
            "--PATH",
            self.get_gnn_output_dir(),
        ]
        return list_to_cmd(self.get_mpiexec() + halo_info)

    def check_input_files_cmd(self):
        return list_to_cmd([
            "python",
            self.get_check_input_files(),
            "--REF",
            os.path.join(self.sourcesdir, "ref"),
            "--PATH",
            self.get_gnn_output_dir(),
        ])

    def check_traj_cmd(self):
        # Check the GNN traj if the case is of `traj` type.
        if self.ml_kwargs["time_dependency"] != "time_dependent":
            return []

        ranks = self.get_ranks()
        cmds = []
        for rank in range(ranks):
            suffix = f"data_rank_{rank}_size_{ranks}"
            cmd = list_to_cmd([
                "python",
                self.get_check_input_files(),
                "--REF",
                os.path.join(
                    self.sourcesdir, "ref", self.get_traj_root(), suffix
                ),
                "--PATH",
                os.path.join(self.get_traj_dir(), suffix),
            ])
            cmds.append(cmd)
        return cmds

    def set_environment(self):
        super().set_environment()

    def set_launcher_options(self):
        super().set_launcher_options()


class NekRSMLOfflineTest(NekRSMLTest):
    def __init__(self, **kwargs):
        # Check if the required arguments are in kwargs.
        check_args(kwargs, ["target_loss"])

        super().__init__(**kwargs)
        self.descr = "NekRS-ML offline test"

    def set_executable_options(self):
        self.executable = list_to_cmd([
            "python",
            os.path.join(self.nekrs_home, "3rd_party", "dist-gnn", "main.py"),
        ])

        # FIXME: master_addr=$head_node
        self.executable_opts = [
            # FIXME: backend should be calculated.
            "backend=xccl",
            "halo_swap_mode=all_to_all_opt",
            "layer_norm=True",
            f"target_loss={self.ml_kwargs['target_loss']}",
            f"time_dependency={self.ml_kwargs['time_dependency']}",
            f"gnn_outputs_path={self.get_gnn_output_dir()}",
            f"traj_data_path={self.get_traj_dir()}",
        ]

    def set_prerun_cmds(self):
        # Run all the pre-training steps
        self.prerun_cmds += [
            self.nekrs_cmd(),
            self.setup_case_cmd(),
            self.source_cmd(),
            self.check_halo_info_cmd(),
            self.check_input_files_cmd(),
            *self.check_traj_cmd(),
        ]

    @run_before("run")
    def setup_run(self):
        super().set_environment()
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
