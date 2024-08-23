#!/bin/sh
module restore
module use /soft/modulefiles
module load jax/0.4.29-dev

NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=4
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of ML ranks per node: $PROCS_PER_NODE
echo Number of ML total ranks: $PROCS
echo

POLY=1
DATA_PATH=/lus/eagle/projects/datascience/sbarwey/codes/nek/nekRS-ML-GNN/examples/tgv_gnn/gnn_outputs_poly_${POLY}/

echo Running create_halo_info.py 
python create_halo_info.py --SIZE $PROCS --POLY ${POLY} --PATH ${DATA_PATH} 

#HALO_SWAP_MODE=none
#HALO_SWAP_MODE=all_to_all
HALO_SWAP_MODE=all_to_all_opt
EXE=./main.py
ARGS="backend=nccl halo_swap_mode=${HALO_SWAP_MODE} gnn_outputs_path=${DATA_PATH}"

echo Running script $EXE
echo with arguments $ARGS
echo
echo `date`
mpiexec --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:24:16:8:1 python $EXE ${ARGS}
echo `date`
