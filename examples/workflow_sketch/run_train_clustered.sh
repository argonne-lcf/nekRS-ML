#!/bin/bash

export SSIM_VENV=/eagle/projects/datascience/balin/Nek/venv/conda_2024-04-29

module use /soft/modulefiles
module load conda/2024-04-29
conda activate base
source $SSIM_VENV/_ssim_env/bin/activate

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

# Run the driver script
python ssim_driver.py 


