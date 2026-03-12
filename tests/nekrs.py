import subprocess
from pathlib import Path
import os.path

import reframe as rfm
import reframe.core.config as config
import reframe.utility.sanity as sn
from core import CompileOnlyTest, RunOnlyTest


def reframe_dir():
    return Path(__file__).parent.resolve()


def lst2cmd(l):
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
        # deployment must be colocated or clustered for online cases.
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

        self.prebuild_cmds += [
            f"export NPROC={self.current_partition.extras['ranks_per_node']}"
        ]

    @property
    def install_path(self):
        return os.path.join(f"{self.stagedir}", "install")


class NekRSBuild(CompileOnlyTest):
    commit = variable(str, value="main")
    smartredis_build = fixture(SmartRedisBuild, scope="environment")

    def __init__(self):
        super().__init__()
        self.descr = "nekRS-ML build"
        self.maintainers = ["kris.rowe@anl.gov", "tratnayaka@anl.gov"]

    @property
    def install_path(self):
        return os.path.join(f"{self.stagedir}", "install")

    @property
    def binary_path(self):
        return os.path.join(self.install_path, "bin")

    @run_before("compile")
    def configure_build(self):
        self.sourcesdir = "https://github.com/argonne-lcf/nekRS-ML.git"
        self.build_system = "CMake"
        self.build_system.cc = self.current_environ.cc
        self.build_system.cxx = self.current_environ.cxx
        self.build_system.ftn = self.current_environ.ftn
        self.build_system.flags_from_environ = False
        self.build_system.builddir = "build"
        self.build_system.make_opts = ["install"]
        self.build_system.config_opts = [
            f"-DCMAKE_INSTALL_PREFIX={self.install_path}",
            "-DENABLE_SMARTREDIS=ON",
            f"-DSMARTREDIS_INSTALL_DIR={self.smartredis_build.install_path}",
            "-DENABLE_ADIOS=ON",
            "-DPython_ROOT=${python_root}",
            "-DADIOS2_INSTALL_DIR=${adios2_root}",
        ]

        self.build_system.max_concurrency = self.current_partition.extras[
            "ranks_per_node"
        ]
        # Update the concurrency from login partition if that exists.
        system = self.current_partition.fullname.split(":")[0]
        cfg = config.load_config(os.path.join(reframe_dir(), "sites.py"))
        for sys in cfg["systems"]:
            if sys["name"] == system:
                for part in sys["partitions"]:
                    if part["name"] == "login":
                        self.build_system.max_concurrency = part["extras"][
                            "ranks_per_node"
                        ]

        self.prebuild_cmds += [
            "git fetch",
            f"git checkout {self.commit}",
            f"export CC={self.build_system.cc}",
            f"export CXX={self.build_system.cxx}",
            f"export FC={self.build_system.ftn}",
            "export python_root=${PYTHON_ROOT:-$(dirname $(dirname $(realpath `which python3`)))}",
            'export adios2_root=${ADIOS2_ROOT:-""}',
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
        self.case_name = case
        self.case_dir = directory

        super().__init__(nn, rpn)
        self.descr = "NekRS test"
        self.maintainers = ["kris.rowe@anl.gov"]
        self.readonly_files = [f"{self.case_name}.re2"]

    @run_after("setup")
    def set_paths_exec(self):
        self.nekrs_home = os.path.realpath(self.nekrs_build.install_path)
        self.nekrs_binary = os.path.join(self.nekrs_build.binary_path, "nekrs")
        self.sourcesdir = os.path.join(
            self.nekrs_build.install_path, "examples", self.case_dir
        )

    def set_environment(self):
        self.env_vars |= {
            "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{self.nekrs_build.install_path}/lib",
            "NEKRS_HOME": self.nekrs_home,
        }

    def set_launcher_options(self, nn=None, rpn=None):
        cpu_bind_list = self.current_partition.extras["cpu_bind_list"]
        rpn_ = rpn if rpn is not None else self.num_tasks_per_node
        nn_ = nn if nn is not None else self.num_nodes
        self.job.launcher.options = [
            f"-np {nn_ * rpn_}",
            f"-ppn {rpn_}",
            f"--cpu-bind=list:{cpu_bind_list}",
        ]

    @property
    def nekrs_exec_opts(self):
        backend = self.current_partition.extras["occa_backend"]
        return [
            f"--setup {self.case_name}",
            f"--backend {backend}",
            "--device-id 0",
        ]

    @property
    def nekrs_exec_cmd(self):
        return [f"{self.nekrs_binary}"]

    def set_executable_options(self):
        self.executable = self.nekrs_exec_cmd
        self.executable_opts = self.nekrs_exec_opts

    def nekrs_cmd(self, extra_args=[]):
        return lst2cmd(
            self.mpiexec
            + self.nekrs_exec_cmd
            + self.nekrs_exec_opts
            + extra_args
        )

    @property
    def gnn_dir(self):
        return os.path.join(
            self.nekrs_home, "3rd_party", "gnn", self.ml_args["model"]
        )

    @run_before("run")
    def setup_run(self):
        self.set_environment()
        self.set_launcher_options()
        self.set_executable_options()

    @sanity_function
    def check_nekrs_exit_code(self):
        return sn.assert_found(
            r"finished with exit code 0",
            self.stdout,
            msg="NekRS finished with non-zero exit code.",
        )


class NekRSMLTest(NekRSTest):
    def __init__(self, **args):
        for arg in ["case", "directory", "nn", "rpn", "time_dependency"]:
            if arg not in args:
                raise KeyError(f"Required kwarg {arg} was not found.")

        super().__init__(
            args["case"], args["directory"], args["nn"], args["rpn"]
        )

        # Init missing arguments with default values from setup_case script.
        self.ml_args = init_missing_args(args)
        self.descr = f"NekRS-ML {self.ml_args['test_type']} test"
        self.tags = {
            "all",
            self.ml_args["model"],
            self.ml_args["case"],
            self.ml_args["time_dependency"],
        }

    @property
    def time_dependency(self):
        return self.ml_args["time_dependency"]

    @property
    def target_loss(self):
        return self.ml_args["target_loss"]

    @property
    def model(self):
        return self.ml_args["model"]

    @property
    def deployment(self):
        return self.ml_args["deployment"]

    @property
    def nn(self):
        return self.ml_args["nn"]

    @property
    def rpn(self):
        return self.num_tasks_per_node

    @property
    def mpiexec(self):
        return (
            self.job.launcher.command(self.job)
            + self.job.launcher.options
            + ["--"]
        )

    def order(self, p):
        pf = f"{self.case_name}.par"
        txt = grep(p, os.path.join(self.sourcesdir, pf))
        if txt is None:
            raise ValueError(f"Expected pattern '{p}' not found in {pf}")
        return int(txt.stdout.split()[2])

    @property
    def gnn_order(self):
        return self.order("gnnPolynomialOrder")

    @property
    def sim_order(self):
        return self.order("polynomialOrder")

    @property
    def ml_rpn(self):
        return int(self.rpn / 2) if self.deployment == "colocated" else self.rpn

    @property
    def sim_rpn(self):
        return self.rpn - self.ml_rpn

    @property
    def db_rpn(self):
        return len(self.db_cpu_ids.split(","))

    @property
    def ml_nn(self):
        return self.nn if self.deployment == "colocated" else int(self.nn / 2)

    @property
    def sim_nn(self):
        return (
            self.nn
            if self.deployment == "colocated"
            else (self.nn - self.ml_nn)
        )

    @property
    def db_nn(self):
        return self.ml_args["db_nodes"]

    @property
    def sim_ranks(self):
        return self.sim_nn * self.sim_rpn

    @property
    def ml_ranks(self):
        return self.ml_nn * self.ml_rpn

    @property
    def sim_cpu_ids(self):
        return self.current_partition.extras["cpu_bind_list"].split(":")[
            : self.sim_rpn
        ]

    @property
    def ml_cpu_ids(self):
        return self.current_partition.extras["cpu_bind_list"].split(":")[
            self.sim_rpn :
        ]

    @property
    def db_cpu_ids(self):
        return self.current_partition.extras["db_bind_list"]

    @property
    def venv_path(self):
        return os.path.join(self.stagedir, "_env")

    @property
    def system(self):
        return self.current_system.name

    def setup_case_cmd(self, extra_args=[]):
        return lst2cmd([
            os.path.join(Path(self.nekrs_home), "bin", "setup_case"),
            self.current_system.name,
            self.nekrs_home,
            "--venv_path",
            self.venv_path,
            "--nodes",
            str(self.nn),
            "--model",
            str(self.model),
            *extra_args,
        ])

    def source_cmd(self):
        return lst2cmd([
            "source",
            os.path.join(self.venv_path, "bin", "activate"),
        ])

    @run_before("run")
    def setup_run(self):
        super().setup_run()
        self.set_scheduler_options()


class NekRSMLOfflineTest(NekRSMLTest):
    def __init__(self, **kwargs):
        kwargs["test_type"] = "offline"
        super().__init__(**kwargs)

    @property
    def gnn_output_dir(self):
        return os.path.join(self.stagedir, f"gnn_outputs_poly_{self.gnn_order}")

    @property
    def check_input_files_py(self):
        return os.path.join(self.gnn_dir, "check_input_files.py")

    @property
    def traj_root(self):
        return os.path.join(
            f"traj_poly_{self.gnn_order}", "tinit_0.000000_dtfactor_10"
        )

    @property
    def traj_dir(self):
        return os.path.join(self.stagedir, self.traj_root)

    def set_sr_gnn_target_and_input_list(self):
        tlist = f"{self.case_name}_p{self.sim_order * 10}*"
        ilist = f"{self.case_name}_p{self.gnn_order * 10}*"
        return lst2cmd([f"target_list=`ls {tlist}`; input_list=`ls {ilist}`"])

    def check_halo_info_cmd(self):
        return lst2cmd(
            self.mpiexec
            + [
                "python",
                os.path.join(self.gnn_dir, "create_halo_info_par.py"),
                "--POLY",
                str(self.gnn_order),
                "--PATH",
                self.gnn_output_dir,
            ]
        )

    def check_input_files_cmd(self):
        return lst2cmd([
            "python",
            self.check_input_files_py,
            "--REF",
            os.path.join(self.sourcesdir, "ref"),
            "--PATH",
            self.gnn_output_dir,
        ])

    def check_traj_cmd(self):
        cmds = []
        if self.time_dependency == "time_dependent":
            for rank in range(self.sim_ranks):
                suffix = f"data_rank_{rank}_size_{self.sim_ranks}"
                cmd = lst2cmd([
                    "python",
                    self.check_input_files_py,
                    "--REF",
                    os.path.join(
                        self.sourcesdir, "ref", self.traj_root, suffix
                    ),
                    "--PATH",
                    os.path.join(self.traj_dir, suffix),
                ])
                cmds.append(cmd)
        return cmds

    def generate_sr_gnn_data_cmd(self):
        return lst2cmd([
            "python",
            os.path.join(self.gnn_dir, "nek_to_pt.py"),
            f"--case_path {self.stagedir}",
            "--target_snap_list ${target_list}",
            "--input_snap_list ${input_list}",
            f"--target_poly_order {self.sim_order}",
            f"--input_poly_order {self.gnn_order}",
            f"--n_element_neighbors {self.ml_args['n_element_neighbors']}",
        ])

    def set_prerun_cmds(self):
        self.prerun_cmds += [
            self.setup_case_cmd(),
            self.source_cmd(),
            self.nekrs_cmd(extra_args=[f"--build-only {self.sim_ranks}"]),
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
        self.executable = lst2cmd([
            "python",
            os.path.join(self.gnn_dir, "main.py"),
        ])

        args = self.ml_args
        if args["model"] == "dist-gnn":
            self.executable_opts = [
                "halo_swap_mode=all_to_all_opt",
                "layer_norm=True",
                f"gnn_outputs_path={self.gnn_output_dir}",
                f"traj_data_path={self.traj_dir}",
                f"target_loss={args['target_loss']}",
                f"time_dependency={args['time_dependency']}",
            ]
        elif args["model"] == "sr-gnn":
            self.executable_opts = [
                f"epochs={args['epochs']}",
                f"n_element_neighbors={args['n_element_neighbors']}",
                f"n_messagePassing_layers={args['n_messagePassing_layers']}",
                f"data_dir={os.path.join(self.stagedir, 'pt_datasets')}",
                f"model_dir={os.path.join(self.stagedir, 'saved_models')}",
            ]

    def set_postrun_cmds(self):
        if self.ml_args["model"] != "sr-gnn":
            return

        self.postrun_cmds += [
            "export model=${PWD}/`ls saved_models/*.tar`",
            lst2cmd([
                "python",
                os.path.join(self.gnn_dir, "postprocess.py"),
                "--model_path ${model}",
                f"--case_path {self.stagedir}",
                f"--output_name {self.case_name}",
                f"--target_snap_list",
                f"{self.case_name}_p{self.sim_order * 10}.f00000",
                f"--input_snap_list",
                f"{self.case_name}_p{self.gnn_order * 10}.f00000",
                f"--target_poly_order {self.sim_order}",
                f"--input_poly_order {self.gnn_order}",
                f"--n_element_neighbors {self.ml_args['n_element_neighbors']}",
            ]),
        ]

    @run_before("run")
    def setup_run(self):
        super().setup_run()
        self.set_prerun_cmds()
        self.set_executable_options()
        self.set_postrun_cmds()

    @sanity_function
    def check_run(self):
        nekrs_ok = self.check_nekrs_exit_code()

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
        kwargs["test_type"] = "online"
        super().__init__(**kwargs)

    @property
    def experiment_name(self):
        return f"NekRS-ML-{self.case_name}"

    @property
    def client(self):
        return self.ml_args["client"]

    @property
    def client_prefix(self):
        return "ssim_" if self.client == "smartredis" else ""

    @property
    def driver(self):
        return os.path.join(self.stagedir, self.client_prefix + "driver.py")

    @property
    def config_yaml(self):
        return os.path.join(self.stagedir, self.client_prefix + "config.yaml")

    def setup_torch_env_vars(self):
        return [
            "export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )",
            "export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH",
            "export SR_SOCKET_TIMEOUT=10000",
        ]

    def setup_adios_env_vars(self):
        return [
            "py_version=`python --version`",
            "parsed_version=$(echo ${py_version#Python } | awk -F. '{print $1\".\"$2}')",
            "export PYTHONPATH=$PYTHONPATH:${NEKRS_HOME}/lib/python${parsed_version}/site-packages",
            "export OMP_PROC_BIND=spread",
            "export OMP_PLACES=threads",
        ]

    def create_traj_config(self):
        with open(f"{self.config_yaml}.reframe", "w") as f:
            if self.client == "smartredis":
                f.write("###################\n")
                f.write("# Database config #\n")
                f.write("###################\n")
                f.write("database:\n")
                f.write("    launch: True\n")
                f.write('    backend: "redis"\n')
                f.write(f'    deployment: "{self.deployment}"\n')
                f.write(f'    exp_name: "{self.experiment_name}"\n')
                # FIXME: The following should be machine-dependent:
                f.write("    port: 6782\n")
                f.write('    network_interface: "uds"\n')
                f.write('    launcher: "pals"\n')
            elif self.client == "adios":
                f.write("###################\n")
                f.write("# Workflow config #\n")
                f.write("###################\n")
                f.write(
                    f"scheduler: {self.current_partition.scheduler.registered_name}\n"
                )
                f.write(f'deployment: "{self.deployment}"\n')
            f.write("\n")

            f.write("##############\n")
            f.write("# Run config #\n")
            f.write("##############\n")
            f.write("run_args:\n")
            f.write(f"    nodes: {self.nn}\n")
            f.write(f"    sim_nodes: {self.sim_nn}\n")
            f.write(f"    simprocs: {self.sim_ranks}\n")
            f.write(f"    simprocs_pn: {self.sim_rpn}\n")
            f.write(f'    sim_cpu_bind: "list:{":".join(self.sim_cpu_ids)}"\n')
            f.write(f"    ml_nodes: {self.ml_nn}\n")
            f.write(f"    mlprocs: {self.ml_ranks}\n")
            f.write(f"    mlprocs_pn: {self.ml_rpn}\n")
            f.write(f'    ml_cpu_bind: "list:{":".join(self.ml_cpu_ids)}"\n')
            if self.client == "smartredis":
                f.write(f"    db_nodes: {self.db_nn}\n")
                f.write(f"    dbprocs_pn: {self.db_rpn}\n")
                f.write(f"    db_cpu_bind: [{self.db_cpu_ids}]\n")
            f.write("\n")

            f.write("#####################\n")
            f.write("# Simulation config #\n")
            f.write("#####################\n")
            f.write("sim:\n")
            f.write(f'    executable: "{self.nekrs_binary}"\n')
            f.write(f'    arguments: "{lst2cmd(self.nekrs_exec_opts)}"\n')
            f.write(f'    affinity: ""\n')
            if self.client == "smartredis":
                f.write(
                    f'    copy_files: ["./{self.case_name}.usr","./{self.case_name}.par","./{self.case_name}.udf","./{self.case_name}.re2"]\n'
                )
                f.write('    link_files: [".cache"]\n')
            f.write("\n")

            f.write("##################\n")
            f.write("# Trainer config #\n")
            f.write("##################\n")
            f.write("train:\n")
            f.write(
                f'    executable: "{os.path.join(self.gnn_dir, "main.py")}"\n'
            )
            f.write('    affinity: ""\n')

            arg_str = (
                "    arguments: "
                '"halo_swap_mode=all_to_all_opt layer_norm=True online=True verbose=True '
                f"consistency=True target_loss={self.target_loss} "
                f"device_skip={self.sim_rpn} time_dependency={self.time_dependency} "
            )
            if self.client == "smartredis":
                arg_str += f'client.db_nodes={self.db_nn}" '
            elif self.client == "adios":
                arg_str += (
                    f"client.backend=adios client.adios_transport={self.current_partition.extras['adios_transport']} "
                    'online_update_freq=500 hidden_channels=32 n_mlp_hidden_layers=5 n_messagePassing_layers=4" '
                )
            f.write(arg_str + "\n")

            if self.client == "smartredis":
                f.write("    copy_files: []\n")
                f.write("    link_files: []\n")

    def set_prerun_cmds(self):
        self.prerun_cmds += [
            self.setup_case_cmd(
                extra_args=[
                    f"--client {self.client}",
                    f"--deployment {self.deployment}",
                ]
            ),
            self.source_cmd(),
            *self.setup_torch_env_vars(),
            *self.setup_adios_env_vars(),
            lst2cmd([
                "cp",
                f"{self.config_yaml}.reframe",
                self.config_yaml,
            ]),
            self.nekrs_cmd(extra_args=[f"--build-only {self.sim_ranks}"]),
        ]

    def set_executable_options(self):
        self.executable_opts = []
        self.executable = lst2cmd(["python", self.driver])

    @run_before("run")
    def setup_run(self):
        super().setup_run()
        self.create_traj_config()
        self.set_prerun_cmds()
        self.set_launcher_options(nn=1, rpn=1)
        self.set_executable_options()

    @sanity_function
    def check_run(self):
        nekrs_ok = self.check_nekrs_exit_code()

        if self.client == "smartredis":
            train_out = os.path.join(
                self.stagedir, self.experiment_name, "train", "train.out"
            )
        else:
            train_out = os.path.join(self.stagedir, "logs", "train_0.out")

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
