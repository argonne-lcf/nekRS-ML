#!/bin/bash

# Set run env
nodes=`wc -l < $PBS_NODEFILE`
striping_unit=16777216
max_striping_factor=128
let striping_factor=$nodes/2
if [ $striping_factor -gt $max_striping_factor ]; then
  striping_factor=$max_striping_factor
fi
if [ $striping_factor -lt 1 ]; then
  striping_factor=1
fi
MPICH_MPIIO_HINTS="*:striping_unit=${striping_unit}:striping_factor=${striping_factor}:romio_cb_write=enable:romio_ds_write=disable:romio_no_indep_rw=true"

ulimit -s unlimited
export NEKRS_GPU_MPI=1
export MPICH_MPIIO_HINTS=$MPICH_MPIIO_HINTS
export MPICH_MPIIO_STATS=1
export NEKRS_CACHE_BCAST=1
export NEKRS_LOCAL_TMP_DIR=/local/scratch
export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_OFI_NIC_POLICY=NUMA
export FI_OFI_RXM_RX_SIZE=32768

# Set the correct .udf file
cp turbChannel_inference.udf turbChannel.udf
cp turbChannel_inference.par turbChannel.par

# Run the driver script
sim_arguments="--setup turbChannel.par --backend CUDA"
python ssim_driver_polaris.py \
  database.deployment=clustered database.network_interface=hsn0 \
  run_args.nodes=2 run_args.db_nodes=1 run_args.sim_nodes=1 \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=4 run_args.simprocs_pn=4 \
  sim.arguments="${sim_arguments}" \
  inference.model_path=./model_jit.pt inference.device=GPU

rm turbChannel.udf
rm turbChannel.par
