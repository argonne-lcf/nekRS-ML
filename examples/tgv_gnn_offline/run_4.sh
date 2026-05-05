#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
source /flare/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_offline/../_env_dist-gnn_posix/bin/activate
module list
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export NEKRS_HOME=/home/balin/.local/nekrs
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

export CCL_ALLTOALLV_MONOLITHIC_KERNEL=1

RANKS=4

# Run nekRS
#mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- /home/balin/.local/nekrs/bin/nekrs --setup tgv --backend dpcpp

# Generate the halo_info, edge_weights and node_degree files
#mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/create_halo_info_par.py --POLY 3 --PATH ./gnn_outputs_poly_3

# Check the GNN input files
#echo "Checking GNN graph input files ..."
#python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref --PATH ./gnn_outputs_poly_3

# Train the GNN
head_node=`head -1 $PBS_NODEFILE | cut -d'.' -f1`
mpiexec -n $RANKS -ppn $RANKS --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python ../../3rd_party/gnn/dist-gnn/main.py master_addr=$head_node halo_swap_mode=all_to_all_opt layer_norm=False gnn_outputs_path=/flare/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_offline/gnn_outputs_poly_3_$RANKS target_loss=4.1699e-03 phase1_steps=100 n_messagePassing_layers=4 transform_x=true transform_y=true transform_z=true
