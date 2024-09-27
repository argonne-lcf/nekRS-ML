#!/bin/bash

export SSIM_VENV=/eagle/projects/datascience/balin/Nek/venv/conda_2024-04-29

module use /soft/modulefiles
module load conda/2024-04-29
conda activate base
source $SSIM_VENV/_ssim_env/bin/activate

BASE_PATH=/eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch

for i in 1 2 3 4
do
    mpiexec -n 1 --ppn 1 -host x3001c0s1b0n0 python $BASE_PATH/test_nrs.py &
done


for i in 1 2 3 4
do
    mpiexec -n 1 --ppn 1 -host x3001c0s1b1n0 python $BASE_PATH/test_nrs.py &
done

wait

