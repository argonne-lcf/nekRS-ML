#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
module list

export NEKRS_HOME=/home/balin/.local/nekrs/
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

# Run nekRS
mpiexec -n 2 -ppn 2 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- /home/balin/.local/nekrs//bin/nekrs --setup turbChannel --backend dpcpp

# Post-process
