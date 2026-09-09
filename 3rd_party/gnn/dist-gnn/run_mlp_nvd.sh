#!/bin/bash -l
##PBS -S /bin/bash
##PBS -N nsys
##PBS -l select=1
##PBS -l walltime=1:00:00
##PBS -l filesystems=home:eagle
##PBS -A datascience
##PBS -q debug
##PBS -k doe
##PBS -j oe
#cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2025-09-25
conda activate
module list

echo Using `nsys --version`
echo Using `ncu --version`
echo

DATE=$(date +%Y%m%d_%H%M%S)

nsys profile \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx,osrt,cudnn \
  --stats=true \
  -o nsys_report_${DATE} \
  python mlp_prof_nvd.py model_task=train phase1_steps=2 hidden_channels=256 n_mlp_hidden_layers=2 precision=fp32

#--capture-range=nvtx \
#--nvtx-domain-include="training" \

ncu \
  --set roofline \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
  --profile-from-start off \
  --nvtx --nvtx-include "training/" --nvtx-push-pop-scope process \
  --kernel-name "regex:gemm*" \
  -o roofline_report_${DATE} \
  python mlp_prof_nvd.py model_task=train phase1_steps=2 hidden_channels=256 n_mlp_hidden_layers=2 precision=fp32

# --kernel-name "regex:*gemm*"
#--kernel-name regex:ampere --launch-skip 0 --launch-count 1

