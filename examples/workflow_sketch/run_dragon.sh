#!/bin/bash

module use /soft/modulefiles
module load conda/2024-04-29
conda activate base
source /eagle/projects/datascience/balin/Nek/venv/conda_2024-04-29_dragon/_ssim_env/bin/activate

export SMARTSIM_LOG_LEVEL=debug
export SR_LOG_LEVEL=debug
export SMARTSIM_DRAGON_TIMEOUT=120000
export SMARTSIM_DRAGON_TRANSPORT=tcp
export SMARTSIM_DRAGON_STARTUP_TIMEOUT=-1

smart teardown --dragon
rm -r MSR

python test_dragon_launcher.py

smart teardown --dragon


