#!/bin/bash

if [ ! -d "./.venv" ]; then 
    python3 -m venv ./.venv --system-site-packages
fi

source ./.venv/bin/activate

if [ ! -d "ensemble_launcher" ]; then
    git clone https://github.com/argonne-lcf/ensemble_launcher.git
fi

if [ ! -d "./.venv" ]; then 
    cd ensemble_launcher
    python3 -m pip install ./
    cd ../
fi

##build genbox to get .re2 files
git clone https://github.com/rickybalin/Nek5000.git
cd Nek5000/tools
./BuildGenboxAurora
cd ../../

##Create the input files
if [ "$#" -ne 3 ]; then
    echo "Error: Exactly 3 arguments are required: Re_tau, Lx, and Lz."
    echo "usage: run_ensembles.sh" "min,max,nsteps for Re_tau" "min,max,nsteps for Lx" "min,max,nsteps for Lz"
    exit 1
fi

if [ -z "$NEKRS_HOME" ]; then
    echo "Error: NEKRS_HOME environment variable is not set."
    exit 1
fi

python3 gen_ensemble_inputs.py --Re_tau "$1" --Lx "$2" --Lz "$3"

el start ./run_dir/config.json --system-config-file ./run_dir/system_config.json --launcher-config-file ./run_dir/launcher_config.json


