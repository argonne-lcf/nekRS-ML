#!/bin/bash

export SSIM_VENV=/eagle/projects/datascience/balin/Nek/venv/conda_2024-04-29

module use /soft/modulefiles
module load conda/2024-04-29
conda activate base
source $SSIM_VENV/_ssim_env/bin/activate

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

# Run the driver script
# 2 nodes
python ssim_driver.py --sim_nodes=1 --db_nodes=1 --simprocs=1 --simprocs_pn=1
#python ssim_driver.py --sim_nodes=1 --db_nodes=1 --simprocs=2 --simprocs_pn=2
#python ssim_driver.py --sim_nodes=1 --db_nodes=1 --simprocs=4 --simprocs_pn=4


