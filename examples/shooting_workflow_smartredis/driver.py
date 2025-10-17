# general imports
import os
import sys 
import socket
from omegaconf import DictConfig, OmegaConf
import hydra
import psutil
from typing import Optional, Tuple
from statistics import harmonic_mean

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings


class ShootingWorkflow():
    """Class for the solution shooting workflow alternating between 
    fine-tuning a surrogate from an ongoing simulation and deploying
    the surrogate to shoot the solution forward
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.nodelist = []
        self.num_nodes = 1
        self.sim_nodes = ''
        self.db_nodes = ''
        self.dbNodes_list = []
        self.train_nodes = ''
        self.inference_nodes = ''
        self.fine_tune_iter = -1
        self.inference_iter = -1
        self.run_dir = os.getcwd()
        self.log_dir = os.path.join(self.run_dir,'nekRS-ML')
        self.host = socket.gethostname()

        # Parse the node list from the scheduler
        self.parseNodeList()

        # Split the nodes between the components
        self.assignNodes()

        # Initialize the SmartSim experiment
        self.db = None
        self.nekrs_model = None
        self.train_model = None
        self.infer_model = None
        self.port = cfg.database.port
        self.exp = Experiment(self.cfg.database.exp_name, launcher=self.cfg.database.launcher)

    def parseNodeList(self) -> None:
        """Parse the nodelist from the scheduler
        """
        if (self.cfg.database.launcher=='pals'):
            hostfile = os.getenv('PBS_NODEFILE')
            with open(hostfile) as file:
                self.nodelist = file.readlines()
                self.nodelist = [line.rstrip() for line in self.nodelist]
                self.nodelist = [line.split('.')[0] for line in self.nodelist]
        else:
            sys.exit('Only the pals launcher is implemented')
        self.num_nodes = len(self.nodelist)

    def assignNodes(self) -> None:
        """Assign the total nodes of the job to the different components
        """
        if (self.cfg.database.deployment=='clustered'):
            self.sim_nodes = ','.join(self.nodelist[0: self.cfg.run_args.sim_nodes])
            self.db_nodes = ','.join(self.nodelist[self.cfg.run_args.sim_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes])
            self.dbNodes_list = self.nodelist[self.cfg.run_args.sim_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes]
            self.train_nodes = ','.join(self.nodelist[self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes + \
                                self.cfg.run_args.ml_nodes])
            self.inference_nodes = str(self.train_nodes)
            print(f"Database running on {self.cfg.run_args.db_nodes} nodes:")
            print(self.db_nodes)
            print(f"nekRS running on {self.cfg.run_args.sim_nodes} nodes:")
            print(self.sim_nodes)
            print(f"Training running on {self.cfg.run_args.ml_nodes} nodes:")
            print(self.train_nodes)
            print(f"Inference running on {self.cfg.run_args.ml_nodes} nodes:")
            print(self.inference_nodes,'\n',flush=True)
        else:
            self.sim_nodes = ','.join(self.nodelist)
            self.db_nodes = str(self.sim_nodes)
            self.train_nodes = str(self.sim_nodes)
            self.inference_nodes = str(self.sim_nodes)
            print(f"Database, nekRS, training and inference running on {self.cfg.run_args.sim_nodes} nodes:")
            print(self.sim_nodes,'\n',flush=True)

    def launchClusteredDB(self) -> None:
        """Launch the clustered SmartSim Orchestrator (DB)
        """
        runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
        kwargs = {
            'maxclients': 100000,
            'threads_per_queue': 4, # set to 4 for improved performance
            'inter_op_parallelism': 1,
            'intra_op_parallelism': 4,
            'cluster-node-timeout': 30000,
        }
        if (self.cfg.database.launcher=='pals'): run_command = 'mpiexec'
        network = self.cfg.database.network_interface if type(self.cfg.database.network_interface)==str \
                                             else OmegaConf.to_object(self.cfg.database.network_interface)
        self.db = self.exp.create_database(port=self.port,
                                      batch=False,
                                      db_nodes=self.cfg.run_args.db_nodes,
                                      run_command=run_command,
                                      interface=network,
                                      hosts=self.dbNodes_list,
                                      run_args=runArgs,
                                      single_cmd=True,
                                      **kwargs
        )
        self.exp.generate(self.db)
        print("\nStarting database ...")
        self.exp.start(self.db)
        print("Done\n", flush=True)

    def stopClusteredDB(self) -> None:
        """Stop the clustered SmartSim Orchestrator (DB)
        """
        print("Stopping the clustered DB ...")
        self.exp.stop(self.db)
        print("Done\n", flush=True)

    def launchPersistentCoDB(self) -> None:
        """Launch a persistent colocated DB
        """
        cmd = "from time import sleep\n" + \
              "from smartredis import Client\n" + \
              "client = Client(cluster=False)\n" + \
              "while True:\n" + \
              "    if client.key_exists('stop-coDB'):\n" + \
              "        print('Found stop-coDB',flush=True)\n" + \
              "        break\n" + \
              "    else:\n" + \
              "        sleep(5)\n"
        fname = self.run_dir + '/launch_db.py'
        with open(fname,'w') as f:
            f.write(cmd)

        coDB_manager_settings = PalsMpiexecSettings('python',
                                           exe_args=fname,
                                           run_args=None,
                                           env_vars=None
        )
        coDB_manager_settings.set_tasks(self.num_nodes)
        coDB_manager_settings.set_tasks_per_node(1)
        coDB_manager_settings.set_hostlist(self.db_nodes)
        db_bind = None if self.cfg.run_args.db_cpu_bind=='None' else self.cfg.run_args.db_cpu_bind
        db_bind_cores = 0 if db_bind is None else len(db_bind)
        coDB_manager_settings.set_cpu_binding_type(f"list:{psutil.cpu_count(logical=False)-db_bind_cores-1}")

        kwargs = {
            'maxclients': 100000,
            'threads_per_queue': 4, # set to 4 for improved performance
            'inter_op_parallelism': 1,
            'intra_op_parallelism': 1,
            'cluster-node-timeout': 30000,
        }
        self.coDB_model = self.exp.create_model(f"coDB", coDB_manager_settings)
        db_bind = None if self.cfg.run_args.db_cpu_bind=='None' else self.cfg.run_args.db_cpu_bind
        if (self.cfg.database.network_interface=='udf'):
            self.coDB_model.colocate_db_uds(
                    db_cpus=self.cfg.run_args.dbprocs_pn,
                    custom_pinning=db_bind,
                    debug=False,
                    **kwargs
            )
        else:
            self.coDB_model.colocate_db_tcp(
                    port=self.port,
                    ifname=self.cfg.database.network_interface,
                    db_cpus=self.cfg.run_args.dbprocs_pn,
                    custom_pinning=db_bind,
                    debug=False,
                    **kwargs
            )

        print("Launching colocated DB ... ")
        self.exp.generate(self.coDB_model, overwrite=True)
        self.exp.start(self.coDB_model, block=False, summary=False, monitor=False)
        print("Done\n", flush=True)

    def stopPersistentCoDB(self) -> None:
        """Stop a persistent colocated DB
        """
        cmd = "from smartredis import Client\n" + \
              "import os\n" + \
              "import numpy as np\n" + \
              "SSDB = os.getenv('SSDB')\n" + \
              "client = Client(address=SSDB,cluster=False)\n" + \
              "client.put_tensor('stop-coDB',np.array([1]))\n"
        fname = self.run_dir + '/stop_db.py'
        with open(fname,'w') as f:
            f.write(cmd)

        SSDB = self.nekrs_model.run_settings.env_vars['SSDB']
        env_vars = {'SSDB': SSDB}
        stop_settings = PalsMpiexecSettings('python',
                                           exe_args=fname,
                                           run_args=None,
                                           env_vars=env_vars
        )
        stop_settings.set_tasks(self.num_nodes)
        stop_settings.set_tasks_per_node(1)
        stop_settings.set_hostlist(self.db_nodes)

        print("Stopping the colocated DB ...")
        stop_model = self.exp.create_model(f"stop_coDB", stop_settings)
        self.exp.generate(stop_model, overwrite=True)
        self.exp.start(stop_model, block=True, summary=False)
        print("Done\n", flush=True)

    def launchDatabase(self) -> None:
        """Launch the database
        """
        if self.cfg.database.deployment == 'colocated':
            self.launchPersistentCoDB()
        elif self.cfg.database.deployment == 'clustered':
            self.launchClusteredDB()

    def stopDatabase(self) -> None:
        """Stop the database
        """
        if self.cfg.database.deployment == 'colocated':
            self.stopPersistentCoDB()
        elif self.cfg.database.deployment == 'clustered':
            self.stopClusteredDB()

    def launchNekRS(self) -> None:
        """Launch the nekRS simulation
        """
        env_vars = None
        if (self.cfg.database.deployment=='colocated'):
            SSDB = self.coDB_model.run_settings.env_vars['SSDB']
            env_vars = {'SSDB': SSDB}
        client_exe = self.cfg.sim.executable
        nrs_settings = PalsMpiexecSettings(client_exe,
                                           exe_args=None,
                                           run_args=None,
                                           env_vars=env_vars
        )
        nrs_settings.set_tasks(self.cfg.run_args.simprocs)
        nrs_settings.set_tasks_per_node(self.cfg.run_args.simprocs_pn)
        nrs_settings.set_hostlist(self.sim_nodes)
        nrs_settings.set_cpu_binding_type(self.cfg.run_args.sim_cpu_bind)
        nrs_settings.add_exe_args(self.cfg.sim.arguments)
        if (self.cfg.sim.affinity):
            nrs_settings.set_gpu_affinity_script(self.cfg.sim.affinity,
                                                 self.cfg.run_args.simprocs_pn)
        
        print("Launching nekRS ...")
        self.nekrs_model = self.exp.create_model(f"nekrs_{self.fine_tune_iter}", nrs_settings)
        if len(self.cfg.sim.copy_files)>0 or len(self.cfg.sim.link_files)>0:
            self.nekrs_model.attach_generator_files(
                to_copy=list(self.cfg.sim.copy_files), 
                to_symlink=list(self.cfg.sim.link_files)
            )
        self.exp.generate(self.nekrs_model, overwrite=True)
        self.exp.start(self.nekrs_model, block=False, summary=False)
        print("Done\n", flush=True)

    def launchTrainer(self) -> None:
        """Launch the GNN trainer
        """
        env_vars = None
        if (self.cfg.database.deployment=='colocated'):
            SSDB = self.nekrs_model.run_settings.env_vars['SSDB']
            env_vars = {'SSDB': SSDB}
        ml_exe = self.cfg.train.executable
        ml_exe = ml_exe + ' ' + self.cfg.train.arguments + f' master_addr={self.train_nodes.split(',')[0]}'
        ml_settings = PalsMpiexecSettings(
                           'python',
                           exe_args=ml_exe,
                           run_args=None,
                           env_vars=env_vars,
        )
        ml_settings.set_tasks(self.cfg.run_args.mlprocs)
        ml_settings.set_tasks_per_node(self.cfg.run_args.mlprocs_pn)
        ml_settings.set_hostlist(self.train_nodes)
        ml_settings.set_cpu_binding_type(self.cfg.run_args.ml_cpu_bind)
        if (self.cfg.train.affinity):
            skip = 0 if self.cfg.database.deployment=='clustered' else self.cfg.run_args.simprocs_pn
            ml_settings.set_gpu_affinity_script(self.cfg.train.affinity,
                                                self.cfg.run_args.mlprocs_pn,
                                                skip
            )

        print("Launching GNN training ... ")
        self.train_model = self.exp.create_model(f"train_{self.fine_tune_iter}", ml_settings)
        if len(self.cfg.train.copy_files)>0 or len(self.cfg.train.link_files)>0:
            self.train_model.attach_generator_files(to_copy=list(self.cfg.train.copy_files), 
                                            to_symlink=list(self.cfg.train.link_files)
            )
        self.exp.generate(self.train_model, overwrite=True)
        self.exp.start(self.train_model, block=True, summary=False)
        print("Done\n", flush=True)

    def launchInference(self) -> None:
        """Launch the GNN model for inference
        """
        env_vars = None
        if (self.cfg.database.deployment=='colocated'):
            SSDB = self.nekrs_model.run_settings.env_vars['SSDB']
            env_vars = {'SSDB': SSDB}
        ml_exe = self.cfg.inference.executable
        ml_exe = ml_exe + ' ' + self.cfg.inference.arguments + f' master_addr={self.host}'
        ml_exe = ml_exe + f' model_dir={os.getcwd()}/nekRS-ML/train_{self.fine_tune_iter}/saved_models/'
        ml_settings = PalsMpiexecSettings(
                           'python',
                           exe_args=ml_exe,
                           run_args=None,
                           env_vars=env_vars,
        )
        ml_settings.set_tasks(self.cfg.run_args.mlprocs)
        ml_settings.set_tasks_per_node(self.cfg.run_args.mlprocs_pn)
        ml_settings.set_hostlist(self.train_nodes)
        ml_settings.set_cpu_binding_type(self.cfg.run_args.ml_cpu_bind)
        if (self.cfg.train.affinity):
            skip = 0 if self.cfg.database.deployment=='clustered' else self.cfg.run_args.simprocs_pn
            ml_settings.set_gpu_affinity_script(self.cfg.train.affinity,
                                                self.cfg.run_args.mlprocs_pn,
                                                skip
            )

        print("Launching GNN inference ... ")
        self.infer_model = self.exp.create_model(f"infer_{self.inference_iter}", ml_settings)
        if len(self.cfg.inference.copy_files)>0 or len(self.cfg.inference.link_files)>0:
            self.infer_model.attach_generator_files(to_copy=list(self.cfg.inference.copy_files), 
                                            to_symlink=list(self.cfg.inference.link_files)
            )
        self.exp.generate(self.infer_model, overwrite=True)
        self.exp.start(self.infer_model, block=True, summary=False)
        print("Done\n", flush=True)

    def fineTune(self) -> None:
        """Fine-tune the GNN model from the nekRS simulation
        """
        self.fine_tune_iter += 1
        self.launchNekRS()
        self.launchTrainer() # blocks code progress

    def rollout(self) -> None:
        """Roll-out the surrogate model and advance the solution
        """
        self.inference_iter += 1
        self.launchInference()

    def runner(self) -> None:
        """Runner function for the workflow responsible for alternating
        between fine-tuning and inference and deploying the components
        """
        # Launch the DB
        self.launchDatabase()

        # Start the workflow loop
        #while True:
        # Fine-tune model
        self.fineTune()

        # Roll-out model
        self.rollout()
        
        # Stop DB
        self.stopDatabase()

    def compute_fom_nekrs(self) -> float:
        """Compute the nekRS FOM from reading input and log files
        """
        with open(f'{self.log_dir}/nekrs_0/nekrs_0.out','r') as fh:
            for l in fh:
                if 'runtime statistics' in l:
                    nekrs_steps = int(l.split('(')[-1].split(' ')[0].split('=')[-1])
                if ' solve ' in l:
                    nekrs_time = float(l.split('solve')[-1].split('s')[0].strip())
                if ' udfExecuteStep ' in l:
                    udf_time = float(l.split('udfExecuteStep')[-1].split('s')[0].strip())
        with open(f'{self.run_dir}/turbChannel.box','r') as fh:
            for l in fh:
                if 'nelx' in l:
                    elms = l.split()
        elms = elms[:3]
        elms = [int(item)*-1 for item in elms]
        with open(f'{self.run_dir}/turbChannel.par','r') as fh:
            for l in fh:
                if 'polynomialOrder' in l:
                    p = int(l.split()[-1].strip())
        num_nodes = elms[0] * elms[1] * elms[2] * (p+1)**3 / 1.0e6
        return num_nodes * nekrs_steps / (nekrs_time - udf_time)
    
    def compute_fom_train(self) -> Tuple[float,float]:
        """Compute the triaing and transfer FOM from reading log files
        """
        with open(f'{self.log_dir}/train_0/train_0.out','r') as fh:
            for l in fh:
                if 'FOM_train' in l:
                    fom_train = float(l.split(']:')[-1].split(',')[-1].split('=')[-1])
                if 'FOM_transfer' in l:
                    fom_transfer = float(l.split(']:')[-1].split(',')[-1].split('=')[-1])
        return fom_train, fom_transfer
    
    def compute_fom_inference(self) -> float:
        """Compute the inference FOM from reading log files
        """
        with open(f'{self.log_dir}/infer_0/infer_0.out','r') as fh:
            for l in fh:
                if 'FOM_inference' in l:
                    fom_inference = float(l.split(']:')[-1].split(',')[-1].split('=')[-1])
        return fom_inference
    
    def compute_fom(self) -> None:
        """Compute the workflow FOM for the fine tuning and shooting stages
        """
        fom_nekrs = self.compute_fom_nekrs()
        fom_train, fom_transfer = self.compute_fom_train()
        fom_inference = self.compute_fom_inference()
        print('\n\nWorkflow FOM:')
        print(f'\tFOM_nekrs [million mesh nodes x nekRS steps / nekRS time] = {fom_nekrs:.4g}')
        print(f'\tFOM_train [million graph nodes x train steps / train time] = {fom_train:.4g}')
        print(f'\tFOM_transfer [GB / transfer time] = {fom_transfer:.4g}')
        print(f'\tFOM_inference [million graph nodes x inference steps / inference time] = {fom_inference:.4g}')
        fom_finetune = harmonic_mean([fom_nekrs,fom_train,fom_transfer])
        print(f'\tFOM_finetune = {fom_finetune:.4g}')
        fom_shoot = fom_inference / fom_nekrs
        print(f'\tFOM_shoot = {fom_shoot:.4g}')
        print('\n',flush=True)


## Main function
@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: DictConfig):
    # Initialize workflow class
    workflow = ShootingWorkflow(cfg)

    # Run the workflow
    workflow.runner()

    # Compute and print the FOM
    workflow.compute_fom()

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
