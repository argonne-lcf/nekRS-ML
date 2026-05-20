#!/bin/bash
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
source /tegu/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_online/../_env_dist-gnn_smartredis/bin/activate
module list
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export NEKRS_HOME=/home/balin/.local/nekrs/
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

export CCL_ALLTOALLV_MONOLITHIC_KERNEL=0

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
export SR_SOCKET_TIMEOUT=10000

# precompilation
#date
#case_tmp=tgv
#ntasks_tmp=
#mpiexec -n 12 -ppn 12 --cpu-bind list:1:8:16:24:32:40:53:60:68:76:84:92 -- /home/balin/.local/nekrs//bin/nekrs --setup tgv --backend dpcpp  --build-only 12
#if [ $? -ne 0 ]; then
#  exit
#fi
#date

# actual run
python ssim_driver.py
