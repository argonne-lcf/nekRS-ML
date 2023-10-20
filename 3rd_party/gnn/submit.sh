#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=8:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
##PBS -q debug
#PBS -A datascience
#PBS -N GNN_DDP


# Change to working directory
cd ${PBS_O_WORKDIR}

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"


# Load modules: 
source /lus/eagle/projects/datascience/sbarwey/codes/ml/pytorch_geometric/module_config

# Get number of ranks 
NUM_NODES=$(wc -l < "${PBS_NODEFILE}")

# Get number of GPUs per node
NGPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# Get total number of GPUs 
NGPUS="$((${NUM_NODES}*${NGPUS_PER_NODE}))"

# Print 
echo $NUM_NODES $NGPUS_PER_NODE $NGPUS

# run 
mpiexec \
	--verbose \
	--envall \
	-n $NGPUS \
	--ppn $NGPUS_PER_NODE \
	--hostfile="${PBS_NODEFILE}" \
    --cpu-bind none \
	./set_affinity_gpu_polaris.sh python3 main.py
