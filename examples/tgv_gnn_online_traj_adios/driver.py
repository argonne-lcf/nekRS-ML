# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra
import subprocess
import socket
from time import sleep
from typing import Optional, Tuple
from statistics import harmonic_mean


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
        self.train_nodes = ''
        self.inference_nodes = ''
        self.fine_tune_iter = -1
        self.inference_iter = -1
        self.nekrs_proc = {'name': 'nekRS', 
                           'process': None,
                           'status': 'not running'}
        self.train_proc = {'name': 'GNN training', 
                           'process': None,
                           'status': 'not running'}
        self.infer_proc = {'name': 'GNN inference', 
                           'process': None,
                           'status': 'not running'}
        self.run_dir = os.getcwd()
        self.log_dir = os.path.join(self.run_dir,'logs')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        # Parse the node list from the scheduler
        self.parseNodeList()

        # Split the nodes between the components
        self.assignNodes()

    def parseNodeList(self) -> None:
        """Parse the nodelist from the scheduler
        """
        if (self.cfg.scheduler == 'pbs'):
            hostfile = os.getenv('PBS_NODEFILE')
            with open(hostfile) as file:
                self.nodelist = file.readlines()
                self.nodelist = [line.rstrip() for line in self.nodelist]
                self.nodelist = [line.split('.')[0] for line in self.nodelist]
        else:
            sys.exit('Only the PBS scheduler is implemented for now')
        self.num_nodes = len(self.nodelist)

    def assignNodes(self) -> None:
        """Assign the total nodes of the job to the different components
        """
        if (self.cfg.deployment == 'clustered'):
            self.sim_nodes = ','.join(self.nodelist[0: self.cfg.run_args.sim_nodes])
            self.train_nodes = ','.join(self.nodelist[self.cfg.run_args.sim_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.ml_nodes])
            self.inference_nodes = str(self.train_nodes)
            print(f"nekRS running on {self.cfg.run_args.sim_nodes} nodes:")
            print(self.sim_nodes)
            print(f"Training running on {self.cfg.run_args.ml_nodes} nodes:")
            print(self.train_nodes)
            print(f"Inference running on {self.cfg.run_args.ml_nodes} nodes:")
            print(self.inference_nodes,'\n',flush=True)
        else:
            self.sim_nodes = ','.join(self.nodelist)
            self.train_nodes = str(self.sim_nodes)
            self.inference_nodes = str(self.sim_nodes)
            print(f"nekRS, training and inference running on {self.cfg.run_args.sim_nodes} nodes:")
            print(self.sim_nodes,'\n',flush=True)

    def launchNekRS(self) -> None:
        """Launch the nekRS simulation
        """
        cmd = f"mpiexec " + \
              f"-n {self.cfg.run_args.simprocs} " + \
              f"--ppn {self.cfg.run_args.simprocs_pn} " + \
              f"--cpu-bind {self.cfg.run_args.sim_cpu_bind} " + \
              f"--hosts {self.sim_nodes} "
        if self.cfg.sim.affinity:
            cmd += f"{self.cfg.sim.affinity} {self.cfg.run_args.simprocs_pn} "
        cmd += f"{self.cfg.sim.executable} {self.cfg.sim.arguments}"
        print("Launching nekRS ...")
        self.nekrs_proc['process'] = subprocess.Popen(cmd,
                                executable="/bin/bash",
                                shell=True,
                                stdout=open(os.path.join(self.log_dir,f'nekrs_{self.fine_tune_iter}.out'),'wb'),
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                cwd=self.run_dir,
                                env=os.environ.copy()
        )
        self.nekrs_proc['status'] = 'running'
        print("Done\n", flush=True)

    def launchTrainer(self) -> None:
        """Launch the GNN trainer
        """
        skip = 0 if self.cfg.deployment=='clustered' else self.cfg.run_args.simprocs_pn
        cmd = f"mpiexec " + \
              f"-n {self.cfg.run_args.mlprocs} " + \
              f"--ppn {self.cfg.run_args.mlprocs_pn} " + \
              f"--cpu-bind {self.cfg.run_args.ml_cpu_bind} " + \
              f"--hosts {self.train_nodes} "
        if self.cfg.train.affinity:
            cmd += f"{self.cfg.train.affinity} {self.cfg.run_args.simprocs_pn} {skip} "
        cmd += f"python {self.cfg.train.executable} {self.cfg.train.arguments}"
        cmd += f" master_addr={socket.gethostname()}"
        print("Launching GNN training ...")
        self.train_proc['process'] = subprocess.Popen(cmd,
                                executable="/bin/bash",
                                shell=True,
                                stdout=open(os.path.join(self.log_dir,f'train_{self.fine_tune_iter}.out'),'wb'),
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                cwd=self.run_dir,
                                env=os.environ.copy()
        )
        self.train_proc['status'] = 'running'
        print("Done\n", flush=True)

    def launchInference(self) -> None:
        """Launch the GNN model for inference
        """
        skip = 0 if self.cfg.deployment=='clustered' else self.cfg.run_args.simprocs_pn
        cmd = f"mpiexec " + \
              f"-n {self.cfg.run_args.mlprocs} " + \
              f"--ppn {self.cfg.run_args.mlprocs_pn} " + \
              f"--cpu-bind {self.cfg.run_args.ml_cpu_bind} " + \
              f"--hosts {self.train_nodes} "
        if self.cfg.train.affinity:
            cmd += f"{self.cfg.train.affinity} {self.cfg.run_args.simprocs_pn} {skip} "
        cmd += f"python {self.cfg.inference.executable} " + \
               f"{self.cfg.inference.arguments} model_dir={self.run_dir}/saved_models/" + \
               f" master_addr={socket.gethostname()}"
        print("\nLaunching GNN inference ...")
        self.infer_proc['process'] = subprocess.Popen(cmd,
                                executable="/bin/bash",
                                shell=True,
                                stdout=open(os.path.join(self.log_dir,f'infer_{self.inference_iter}.out'),'wb'),
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                cwd=self.run_dir,
                                env=os.environ.copy()
        )
        self.infer_proc['status'] = 'running'
        print("Done\n", flush=True)

    def kill_processes(self, processes: list) -> None:
        """Kill processes
        """
        for proc in processes:
            if proc['process'] is not None:
                proc['process'].terminate()
                proc['process'].wait()
                print(f'Killed process {proc["name"]}', flush=True)

    def poll_processes(self, processes: list, interval: Optional[int] = 5) -> None:
        """Poll the list of processes passed to the function and return
        boolean if all processes are done
        """
        all_finished = False
        failure = False
        finished = 0
        try:
            while not all_finished:
                sleep(interval)
                for proc in processes:
                    if proc['process'] is not None:
                        status = proc['process'].poll()
                        if status is not None:
                            if proc['process'].returncode == 0:
                                proc['status'] = "finished"
                                finished += 1
                                proc['process'] = None 
                            else:
                                proc['status'] = "failed"
                                failure = True
                        print(f"Process {proc['name']} status: {proc['status']}",flush=True)
                if finished == len(processes): all_finished = True
                if failure:
                   self.kill_processes(processes)
                   sys.exit(0)
        except KeyboardInterrupt:
            print('\nCtrl+C detected!', flush=True)
            self.kill_processes(processes)
            sys.exit(0)

    def fineTune(self) -> None:
        """Fine-tune the GNN model from the nekRS simulation
        """
        self.fine_tune_iter += 1
        self.launchNekRS()
        self.launchTrainer()
        self.poll_processes([self.nekrs_proc, self.train_proc])

    def rollout(self) -> None:
        """Roll-out the surrogate model and advance the solution
        """
        self.inference_iter += 1
        self.launchInference()
        self.poll_processes([self.infer_proc])

    def runner(self) -> None:
        """Runner function for the workflow responsible for alternating
        between fine-tuning and inference and deploying the components
        """
        # Fine-tune model
        self.fineTune()
                   

## Main function
@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: DictConfig):
    # Initialize workflow class
    workflow = ShootingWorkflow(cfg)

    # Run the workflow
    workflow.runner()

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
