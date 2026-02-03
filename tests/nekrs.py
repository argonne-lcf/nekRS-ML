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


def init_missing_args(args):
    def init_value(key, dval):
        if key not in args:
            args[key] = dval

    init_value("time", "01:00")

    init_value("model", "dist-gnn")
    if args["model"] == "sr-gnn":
        init_value("epochs", 1)
        init_value("n_element_neighbors", 0)
        init_value("n_messagePassing_layers", 2)

    init_value(
        "deployment",
        "offline" if args["test_type"] == "offline" else "colocated",
    )
    init_value("client", "posix")
    init_value("db_nodes", 1)
    init_value("sim_nodes", 1)
    init_value("train_nodes", 1)

    return args


def validate_args(args):
    def validate_value(key, valid_values, allow_empty=False):
        if allow_empty and key not in args:
            return
        if args[key] not in valid_values:
            raise ValueError(
                f"Input '{key}' has an invalid value: {args[key]}, "
                f"valid values are: {valid_values}"
            )

    if args["test_type"] == "online":
        validate_value("deployment", ["colocated", "clustered"])
    else:
        validate_value("deployment", ["clustered", "colocated", "offline"])
    validate_value("model", ["dist-gnn", "sr-gnn"])
    validate_value("ml_task", ["train", "inference"], allow_empty=True)
    validate_value("client", ["smartredis", "adios", "posix"])

    return args


class SmartRedisBuild(CompileOnlyTest):
    def __init__(self):
        super().__init__()
        self.descr = "SmartRedis build"
        self.maintainers = ["tratnayaka@anl.gov"]

    @run_before("compile")
    def configure_buld(self):
        self.sourcesdir = "https://github.com/rickybalin/SmartRedis.git"
        self.build_system = "Make"
        self.build_system.cc = self.current_environ.cc
        self.build_system.cxx = self.current_environ.cxx
        self.build_system.ftn = self.current_environ.ftn
        self.build_system.flags_from_environ = False
        # For SmartRedis, actual build parallelization is set with `NRPOC` environment variable.
        self.build_system.max_concurrency = 1
        self.build_system.options = ["lib"]
        self.install_path = os.path.join(f"{self.stagedir}", "install")

        self.prebuild_cmds += [
            f"export NPROC={self.current_partition.extras['ranks_per_node']}"
        ]

    def get_install_path(self):
        return self.install_path


class NekRSBuild(CompileOnlyTest):
    commit = variable(str, value="main")
    smartredis_build = fixture(SmartRedisBuild, scope="environment")

    def __init__(self):
        super().__init__()
        self.descr = "nekRS-ML build"
        self.maintainers = ["kris.rowe@anl.gov", "tratnayaka@anl.gov"]

    @run_before("compile")
    def configure_build(self):
        self.sourcesdir = "https://github.com/argonne-lcf/nekRS-ML.git"
        self.build_system = "CMake"
        self.build_system.cc = self.current_environ.cc
        self.build_system.cxx = self.current_environ.cxx
        self.build_system.ftn = self.current_environ.ftn
        self.build_system.flags_from_environ = False
        self.build_system.builddir = "build"
        self.build_system.max_concurrency = self.current_partition.extras[
            "ranks_per_node"
        ]
        self.build_system.make_opts = ["install"]
        self.install_path = os.path.join(f"{self.stagedir}", "install")
        self.binary_path = os.path.join(self.install_path, "bin")
        self.build_system.config_opts = [
            f"-DCMAKE_INSTALL_PREFIX={self.install_path}",
            "-DENABLE_ADIOS=OFF",
            "-DENABLE_SMARTREDIS=ON",
            f"-DSMARTREDIS_INSTALL_DIR={self.smartredis_build.get_install_path()}",
        ]

        self.prebuild_cmds += [
            "git fetch",
            f"git checkout {self.commit}",
            f"export CC={self.build_system.cc}",
            f"export CXX={self.build_system.cxx}",
            f"export FC={self.build_system.ftn}",
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

        super().__init__(nn, rpn)
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
            f"--cpu-bind=list:{cpu_bind_list}",
        ]

        if "gpu_bind_list" in self.current_partition.extras:
            gpu_bind_list = self.current_partition.extras["gpu_bind_list"]
            self.job.launcher.options += [f"--gpu-bind=list:{gpu_bind_list}"]

    def get_nekrs_executable_options(self):
        backend = self.current_partition.extras["occa_backend"]
        exec_opts = [
            f"--setup {self.nekrs_case_name}",
            f"--backend {backend}",
        ]

        if "gpu_bind_list" in self.current_partition.extras:
            exec_opts += ["--device-id 0"]

        return exec_opts

    def set_executable_options(self):
        self.executable = f"{self.nekrs_binary}"
        self.executable_opts = self.get_nekrs_executable_options()

    def get_gnn_dir(self):
        return os.path.join(
            self.nekrs_home, "3rd_party", "gnn", self.ml_args["model"]
        )

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
            msg="NekRS finished with non-zero exit code.",
        )


class NekRSMLTest(NekRSTest):
    def __init__(self, **kwargs):
        required_args = ["case", "directory", "nn", "rpn", "time_dependency"]
        for arg in required_args:
            if arg not in kwargs:
                raise KeyError(f"Required kwarg {arg} was not found.")

        super().__init__(
            kwargs["case"], kwargs["directory"], kwargs["nn"], kwargs["rpn"]
        )

        # Initialize missing arguments with default values from setup_case script.
        self.ml_args = init_missing_args(kwargs)
        self.descr = f"NekRS-ML {self.ml_args['test_type']} test"
        self.tags = {"all", self.ml_args["model"]}

    @cache
    def get_mpiexec(self):
        return self.job.launcher.command(self.job) + self.job.launcher.options

    def get_order(self, pattern):
        pf = os.path.join(self.sourcesdir, f"{self.nekrs_case_name}.par")
        txt = grep(pattern, pf)
        if txt is None:
            raise ValueError(f"Expected pattern '{pattern}' not found in {pf}")
        return int(txt.stdout.split()[2])

    @cache
    def get_gnn_order(self):
        return self.get_order("gnnPolynomialOrder")

    @cache
    def get_sim_order(self):
        return self.get_order("polynomialOrder")

    @cache
    def get_venv_path(self):
        return os.path.join(self.stagedir, "_env")

    @cache
    def get_sim_ranks(self):
        rpn = self.ml_args["rpn"]
        # FIXME: colocated vs clustered
        if self.ml_args["deployment"] == "colocated":
            rpn = int(rpn / 2)
        return self.ml_args["nn"] * rpn

    def nekrs_cmd(self, extra_args=[]):
        # Set nekrs executable options used in NekRSTest class.
        super().set_executable_options()
        super().set_launcher_options()
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
            "--nodes",
            str(self.ml_args["nn"]),
            "--model",
            str(self.ml_args["model"]),
            *extra_args,
        ])

    def source_cmd(self):
        return list_to_cmd([
            "source",
            os.path.join(self.get_venv_path(), "bin", "activate"),
        ])

    def set_environment(self):
        super().set_environment()

    def set_launcher_options(self):
        super().set_launcher_options()


class NekRSMLOfflineTest(NekRSMLTest):
    def __init__(self, **kwargs):
        kwargs["test_type"] = "offline"
        super().__init__(**kwargs)

    @cache
    def get_gnn_output_dir(self):
        order = self.get_gnn_order()
        return os.path.join(self.stagedir, f"gnn_outputs_poly_{order}")

    def check_halo_info_cmd(self):
        halo_info = [
            "python",
            os.path.join(self.get_gnn_dir(), "create_halo_info_par.py"),
            "--POLY",
            str(self.get_gnn_order()),
            "--PATH",
            self.get_gnn_output_dir(),
        ]
        return list_to_cmd(self.get_mpiexec() + halo_info)

    @cache
    def get_check_input_files_path(self):
        return os.path.join(self.get_gnn_dir(), "check_input_files.py")

    def check_input_files_cmd(self):
        return list_to_cmd([
            "python",
            self.get_check_input_files_path(),
            "--REF",
            os.path.join(self.sourcesdir, "ref"),
            "--PATH",
            self.get_gnn_output_dir(),
        ])

    @cache
    def get_traj_root(self):
        order = self.get_gnn_order()
        return os.path.join(f"traj_poly_{order}", "tinit_0.000000_dtfactor_10")

    @cache
    def get_traj_dir(self):
        return os.path.join(self.stagedir, self.get_traj_root())

    def check_traj_cmd(self):
        # Return if the case is not a `traj` case.
        if self.ml_args["time_dependency"] != "time_dependent":
            return []

        ranks = self.get_sim_ranks()
        cmds = []
        for rank in range(ranks):
            suffix = f"data_rank_{rank}_size_{ranks}"
            cmd = list_to_cmd([
                "python",
                self.get_check_input_files_path(),
                "--REF",
                os.path.join(
                    self.sourcesdir, "ref", self.get_traj_root(), suffix
                ),
                "--PATH",
                os.path.join(self.get_traj_dir(), suffix),
            ])
            cmds.append(cmd)
        return cmds

    def set_sr_gnn_target_and_input_list(self):
        tlist = f"{self.nekrs_case_name}_p{self.get_sim_order() * 10}*"
        ilist = f"{self.nekrs_case_name}_p{self.get_gnn_order() * 10}*"
        return list_to_cmd([
            f"target_list=`ls {tlist}`; input_list=`ls {ilist}`"
        ])

    def generate_sr_gnn_data_cmd(self):
        train_sr = [
            "python",
            os.path.join(self.get_gnn_dir(), "nek_to_pt.py"),
            f"--case_path {self.stagedir}",
            "--target_snap_list ${target_list}",
            "--input_snap_list ${input_list}",
            f"--target_poly_order {self.get_sim_order()}",
            f"--input_poly_order {self.get_gnn_order()}",
            f"--n_element_neighbors {self.ml_args['n_element_neighbors']}",
        ]
        return list_to_cmd(train_sr)

    def set_prerun_cmds(self):
        self.prerun_cmds += [
            self.setup_case_cmd(),
            self.source_cmd(),
            self.nekrs_cmd(extra_args=[f"--build-only {self.get_sim_ranks()}"]),
            self.nekrs_cmd(),
        ]

        if self.ml_args["model"] == "dist-gnn":
            self.prerun_cmds += [
                self.check_halo_info_cmd(),
                self.check_input_files_cmd(),
                *self.check_traj_cmd(),
            ]
        elif self.ml_args["model"] == "sr-gnn":
            self.prerun_cmds += [
                self.set_sr_gnn_target_and_input_list(),
                self.generate_sr_gnn_data_cmd(),
            ]

    def set_executable_options(self):
        main_path = self.get_gnn_dir()
        self.executable = list_to_cmd([
            "python",
            os.path.join(self.get_gnn_dir(), "main.py"),
        ])

        if self.ml_args["model"] == "dist-gnn":
            self.executable_opts = [
                "halo_swap_mode=all_to_all_opt",
                "layer_norm=True",
                f"gnn_outputs_path={self.get_gnn_output_dir()}",
                f"traj_data_path={self.get_traj_dir()}",
                f"target_loss={self.ml_args['target_loss']}",
                f"time_dependency={self.ml_args['time_dependency']}",
            ]
        elif self.ml_args["model"] == "sr-gnn":
            self.executable_opts = [
                f"epochs={self.ml_args['epochs']}",
                f"n_element_neighbors={self.ml_args['n_element_neighbors']}",
                f"n_messagePassing_layers={self.ml_args['n_messagePassing_layers']}",
                f"data_dir={os.path.join(self.stagedir, 'pt_datasets')}",
                f"model_dir={os.path.join(self.stagedir, 'saved_models')}",
            ]

    def set_postrun_cmds(self):
        if self.ml_args["model"] != "sr-gnn":
            return

        self.postrun_cmds += [
            "export model=${PWD}/`ls saved_models/*.tar`",
            list_to_cmd([
                "python",
                os.path.join(self.get_gnn_dir(), "postprocess.py"),
                "--model_path ${model}",
                f"--case_path {self.stagedir}",
                f"--output_name {self.nekrs_case_name}",
                f"--target_snap_list",
                f"{self.nekrs_case_name}_p{self.get_sim_order() * 10}.f00000",
                f"--input_snap_list",
                f"{self.nekrs_case_name}_p{self.get_gnn_order() * 10}.f00000",
                f"--target_poly_order {self.get_sim_order()}",
                f"--input_poly_order {self.get_gnn_order()}",
                f"--n_element_neighbors {self.ml_args['n_element_neighbors']}",
            ]),
        ]

    @run_before("run")
    def setup_run(self):
        super().set_environment()
        super().set_launcher_options()
        self.set_prerun_cmds()
        self.set_executable_options()
        self.set_postrun_cmds()

    @sanity_function
    def check_run(self):
        nekrs_ok = super().check_exit_code()

        pattern = (
            r"Total training time: \S+ seconds"
            if self.ml_args["model"] == "sr-gnn"
            else r"SUCCESS! GNN training validated!"
        )
        gnn_ok = sn.assert_found(
            pattern,
            self.stdout,
            msg="GNN validation failed.",
        )

        inference_ok = (
            sn.assert_found(
                "Done with inference!",
                self.stdout,
                msg="GNN validation failed (inference).",
            )
            if self.ml_args["model"] == "sr-gnn"
            else True
        )

        return nekrs_ok and gnn_ok and inference_ok


class NekRSMLOnlineTest(NekRSMLTest):
    def __init__(self, **kwargs):
        # deployment must be colocated or clustered for online cases.
        kwargs["test_type"] = "online"
        super().__init__(**kwargs)
        self.nekrs_ml_experiment = f"NekRS-ML-{self.nekrs_case_name}"

    def setup_torch_env_vars(self):
        return [
            "export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )",
            "export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH",
            "export SR_SOCKET_TIMEOUT=10000",
        ]

    def create_ssim_config(self):
        args = self.ml_args

        # FIXME: This only works for colocated, not clustered.
        case, rpn = args["case"], int(args["rpn"])
        ml_rpn, sim_rpn = int(rpn / 2), rpn - int(rpn / 2)

        db_bind_list = self.current_partition.extras["db_bind_list"]
        db_rpn = len(db_bind_list.split(","))

        ids = self.current_partition.extras["cpu_bind_list"].split(":")
        sim_ids, ml_ids = ids[:sim_rpn], ids[sim_rpn:]

        config_yaml = os.path.join(self.stagedir, "ssim_config.yaml.reframe")
        with open(config_yaml, "w") as f:
            f.write("# Database config\n")
            f.write("database:\n")
            f.write("    launch: True\n")
            # FIXME: This should be the `--client` value in the ml_args.
            f.write('    backend: "redis"\n')
            f.write(f'    deployment: "{args["deployment"]}"\n')
            f.write(f'    exp_name: "{self.nekrs_ml_experiment}"\n')
            # FIXME: The following should be machine-dependent:
            f.write("    port: 6782\n")
            f.write('    network_interface: "uds"\n')
            f.write('    launcher: "pals"\n')
            f.write("\n")

            f.write("# Run config\n")
            f.write("run_args:\n")
            f.write(f"    nodes: {args['nn']}\n")
            f.write(f"    db_nodes: {args['db_nodes']}\n")
            f.write(f"    sim_nodes: {args['sim_nodes']}\n")
            f.write(f"    ml_nodes: {args['sim_nodes']}\n")
            f.write(f"    simprocs: {args['sim_nodes'] * sim_rpn}\n")
            f.write(f"    simprocs_pn: {sim_rpn}\n")
            f.write(f"    mlprocs: {args['train_nodes'] * ml_rpn}\n")
            f.write(f"    mlprocs_pn: {ml_rpn}\n")
            f.write(f"    dbprocs_pn: {db_rpn}\n")
            f.write(f'    sim_cpu_bind: "list:{":".join(sim_ids)}"\n')
            f.write(f'    ml_cpu_bind: "list:{":".join(ml_ids)}"\n')
            f.write(f"    db_cpu_bind: [{db_bind_list}]\n")
            f.write("\n")

            f.write("# Simulation config\n")
            f.write("sim:\n")
            f.write(f'    executable: "{self.nekrs_binary}"\n')
            f.write(
                f'    arguments: "{list_to_cmd(self.get_nekrs_executable_options())}"\n'
            )
            f.write(f'    affinity: "./affinity_nrs.sh"\n')
            f.write(
                f'    copy_files: ["./{case}.usr","./{case}.par","./{case}.udf","./{case}.re2"]\n'
            )
            f.write('    link_files: ["./affinity_nrs.sh", ".cache"]\n')
            f.write("\n")

            f.write("# Trainer config\n")
            f.write("train:\n")
            f.write(
                f'    executable: "{os.path.join(self.get_gnn_dir(), "main.py")}"\n'
            )
            f.write('    affinity: ""\n')
            f.write(
                (
                    "    arguments: "
                    '"halo_swap_mode=all_to_all_opt layer_norm=True online=True verbose=True '
                    f"consistency=True client.db_nodes={args['db_nodes']} target_loss={args['target_loss']} "
                    f'device_skip={sim_rpn} time_dependency={args["time_dependency"]}"\n'
                )
            )
            f.write("    copy_files: []\n")
            f.write('    link_files: ["./affinity_ml.sh"]\n')

    def set_prerun_cmds(self):
        self.prerun_cmds += [
            self.setup_case_cmd(
                extra_args=[
                    f"--client {self.ml_args['client']}",
                    f"--deployment {self.ml_args['deployment']}",
                ]
            ),
            self.source_cmd(),
            *self.setup_torch_env_vars(),
            # FIXME: Temporary workaround.
            list_to_cmd(["mv", "ssim_config.yaml.reframe", "ssim_config.yaml"]),
            self.nekrs_cmd(extra_args=[f"--build-only {self.get_sim_ranks()}"]),
        ]

    def set_executable_options(self):
        self.executable_opts = []
        self.executable = list_to_cmd([
            "python",
            os.path.join(f"{self.stagedir}", "ssim_driver.py"),
        ])

    def set_launcher_options(self):
        self.job.launcher.options = ["-np 1", "-ppn 1"]

    @run_before("run")
    def setup_run(self):
        super().set_environment()
        self.create_ssim_config()
        self.set_prerun_cmds()
        self.set_launcher_options()
        self.set_executable_options()

    @sanity_function
    def check_run(self):
        nekrs_ok = super().check_exit_code()

        train_out = os.path.join(
            self.stagedir, self.nekrs_ml_experiment, "train", "train.out"
        )
        train_out_present = sn.assert_true(
            os.path.isfile(train_out),
            f"train.out could not be found in path {train_out}",
        )

        gnn_ok = sn.assert_found(
            r"SUCCESS! GNN training validated!",
            train_out,
            msg="GNN validation failed.",
        )

        return nekrs_ok and train_out_present and gnn_ok
