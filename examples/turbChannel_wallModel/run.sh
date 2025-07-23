#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module use /soft/modulefiles/
module load conda
conda activate
module list

export NEKRS_HOME=/home/balin/.local/nekrs/
export NEKRS_CACHE_BCAST=0
export NEKRS_LOCAL_TMP_DIR=/local/scratch
export NEKRS_GPU_MPI=0
export MPICH_MPIIO_HINTS=
export MPICH_MPIIO_STATS=0
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_NIC_POLICY=NUMA

# Run nekRS
mpiexec -n 2 -ppn 2 --cpu-bind=list:24:16:8:1 -- /home/balin/.local/nekrs//bin/nekrs --setup turbChannel --backend CUDA

# Post-process
