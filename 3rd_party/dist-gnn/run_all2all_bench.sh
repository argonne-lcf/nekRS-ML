#!/bin/bash

SYSTEM="aurora"

if [ ${SYSTEM} == "aurora" ]; then
  module load frameworks

  # oneCCL env variables
  export CCL_PROCESS_LAUNCHER=pmix
  export CCL_ATL_TRANSPORT=mpi
  export CCL_KVS_MODE=mpi
  export CCL_ENABLE_SYCL_KERNELS=1
  export CCL_ALLTOALLV=naive
  export CCL_ALLTOALLV_SCALEOUT=naive
  export CCL_ALLTOALLV_MONOLITHIC_KERNEL=0
  export CCL_CONFIGURATION=cpu_gpu_dpcpp
  #export CCL_LOG_LEVEL=debug

  # Other env variables
  export FI_CXI_DEFAULT_CQ_SIZE=1048576

  # Use custom oneCCL
  #unset CCL_ROOT
  #export CCL_CONFIGURATION_PATH=""
  #export CCL_ROOT="/soft/compilers/oneapi/2025.3.0/oneapi/ccl/latest/"
  #export LD_LIBRARY_PATH=${CCL_ROOT}/lib:$LD_LIBRARY_PATH
  #export CPATH=${CCL_ROOT}/include:$CPATH
  #export LIBRARY_PATH=${CCL_ROOT}/lib:$LIBRARY_PATH

  RANKS_PER_NODE=12
  CPU_BINDING=list:1-4:8-11:16-19:24-27:32-35:40-43:53-56:60-63:68-71:76-79:84-87:92-95
fi

EXE=./all2all_bench.py
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind  $CPU_BINDING \
        python $EXE \
        --all_to_all_buff optimized \
        --logging verbose
