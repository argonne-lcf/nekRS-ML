# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra
import subprocess
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
        jobid = os.getenv("PBS_JOBID").split('.')[0]
        self.log_dir = os.path.join(self.run_dir,f'logs_{jobid}')
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
               f"{self.cfg.inference.arguments} model_dir={self.run_dir}/saved_models/"
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
        # Start the workflow loop
        #while True:
        # Fine-tune model
        self.fineTune()

        # Roll-out model and shoot solution forward
        self.rollout()

    def compute_fom_nekrs(self) -> float:
        """Compute the nekRS FOM from reading input and log files
        """
        with open(f'{self.log_dir}/nekrs_0.out','r') as fh:
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
        with open(f'{self.log_dir}/train_0.out','r') as fh:
            for l in fh:
                if 'FOM_train' in l:
                    fom_train = float(l.split(']:')[-1].split(',')[-1].split('=')[-1])
                if 'FOM_transfer' in l:
                    fom_transfer = float(l.split(']:')[-1].split(',')[-1].split('=')[-1])
        return fom_train, fom_transfer
    
    def compute_fom_inference(self) -> float:
        """Compute the inference FOM from reading log files
        """
        with open(f'{self.log_dir}/infer_0.out','r') as fh:
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
        print(f'\tFOM_transfer [TB / transfer time] = {fom_transfer:.4g}')
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
