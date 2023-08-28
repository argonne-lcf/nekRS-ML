# general imports
import os
import sys 
import argparse

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings


## Co-located DB launch
def launch_coDB(args):
    # Initialize the SmartSim Experiment
    PORT = 6780
    exp = Experiment("NekRS_SmartSim", launcher="local")

    # Set the run settings, including the client executable and how to run it
    client_exe = args.nrs_exe
    nrs_settings = RunSettings(client_exe,
                   exe_args=args.nrs_args,
                   run_command='mpirun',
                   run_args={"-n" : args.nrs_nprocs},
                   env_vars=None
                   )

    # Create the co-located database model
    colo_model = exp.create_model("nekrs", nrs_settings)
    colo_model.colocate_db(
                port=PORT,
                db_cpus=1,
                debug=True,
                limit_app_cpus=False,
                ifname="lo",
                )

    # Start the co-located model
    print("Launching NekRS and SmartSim co-located DB ... ")
    exp.start(colo_model, block=False, summary=False)
    print("Done\n")

    # Setup and launch the training script
    ml_exe = args.ml_exe + ' ' + args.ml_args
    ml_settings = RunSettings('python',
                   exe_args=ml_exe,
                   run_command='mpirun',
                   run_args={"-n" : args.ml_nprocs},
                   env_vars=colo_model.run_settings.env_vars
                   )
    ml_model = exp.create_model("train_model", ml_settings)
    print("Launching training script ... ")
    exp.start(ml_model, block=True, summary=False)
    print("Done\n")


## Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nrs_exe',default='/Users/rbalin/.local/nekrs/bin/nekrs',help='Executable path to NekRS')
    parser.add_argument('--nrs_nprocs',default=1,type=int,help='Number of processes for NekRS')
    parser.add_argument('--nrs_args',default='--setup channel.par',help='Arguments to NekRS executable')
    parser.add_argument('--ml_exe',default='./trainer.py',help='Path to training script')
    parser.add_argument('--ml_nprocs',default=1,type=int,help='Number of processes for ML training')
    parser.add_argument('--ml_args',default=' ',help='Arguments to ML training script')
    args = parser.parse_args()

    # Call co-located DB launcher
    print(f"\nRunning with co-located DB\n")
    launch_coDB(args)

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
