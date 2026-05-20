#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module use /soft/modulefiles/
module load conda
conda activate
source /eagle/datascience/balin/Nek/nekRS-ML/examples/tgv_gnn_offline/../_env_dist-gnn_posix/bin/activate
module list

export NEKRS_HOME=/home/balin/.local/nekrs/
export NEKRS_CACHE_BCAST=0
export NEKRS_LOCAL_TMP_DIR=/local/scratch
export NEKRS_GPU_MPI=0
export MPICH_MPIIO_HINTS=
export MPICH_MPIIO_STATS=0
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_NIC_POLICY=NUMA

RANKS=1

rm -r gnn_outputs_poly_3_$RANKS gnn_outputs_poly_3

# Run nekRS
mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:24:16:8:1 -- /home/balin/.local/nekrs//bin/nekrs --setup tgv --backend CUDA
mv gnn_outputs_poly_3 gnn_outputs_poly_3_$RANKS

# Generate the halo_info, edge_weights and node_degree files
mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:24:16:8:1 python /home/balin/.local/nekrs//3rd_party/gnn/dist-gnn/create_halo_info_par.py --POLY 3 --PATH ./gnn_outputs_poly_3_$RANKS

# Check the GNN input files
#echo "Checking GNN graph input files ..."
#python /home/balin/.local/nekrs//3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref --PATH ./gnn_outputs_poly_3

# Train the GNN
head_node=`head -1 $PBS_NODEFILE | cut -d'.' -f1`
mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:24:16:8:1 python /home/balin/.local/nekrs//3rd_party/gnn/dist-gnn/main.py master_addr=$head_node halo_swap_mode=all_to_all_opt layer_norm=True gnn_outputs_path=/eagle/datascience/balin/Nek/nekRS-ML/examples/tgv_gnn_offline/gnn_outputs_poly_3_$RANKS target_loss=2.7161e-04 transform_x=true transform_y=true transform_z=true
