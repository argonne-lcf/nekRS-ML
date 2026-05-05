#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
source /flare/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_offline_traj/../_env_dist-gnn_posix/bin/activate
module list
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export NEKRS_HOME=/home/balin/.local/nekrs
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

export CCL_ALLTOALLV_MONOLITHIC_KERNEL=0

# Precompile nekRS
#mpiexec -n 12 -ppn 12 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- /home/balin/.local/nekrs/bin/nekrs --setup tgv --backend dpcpp --build-only 12

# Run nekRS
#mpiexec -n 2 -ppn 2 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- /home/balin/.local/nekrs/bin/nekrs --setup tgv --backend dpcpp

# Generate the halo_info, edge_weights and node_degree files
#mpiexec -n 2 -ppn 2 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/create_halo_info_par.py --POLY 7 --PATH ./gnn_outputs_poly_7

# Check the GNN input files
#echo "Checking GNN graph input files ..."
#python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref/gnn_outputs_poly_7 --PATH ./gnn_outputs_poly_7

# Check the GNN trajectory files
#echo "Checking GNN trajectory files ..."
#for RANK in {0..3}; do
#  python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref/traj_poly_7/tinit_0.000000_dtfactor_10/data_rank_${RANK}_size_4 --PATH ./traj_poly_7/tinit_0.000000_dtfactor_10/data_rank_${RANK}_size_4
#done

# Train the GNN
head_node=`head -1 $PBS_NODEFILE | cut -d'.' -f1`
mpiexec -n 2 -ppn 2 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python ../../3rd_party/gnn/dist-gnn/main.py master_addr=$head_node halo_swap_mode=all_to_all_opt layer_norm=True gnn_outputs_path=/flare/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_offline_traj/gnn_outputs_poly_7_2 traj_data_path=/flare/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_offline_traj/traj_poly_7_2/tinit_0.000000_dtfactor_10 time_dependency=time_dependent target_loss=6.9031e-01 transform_x=true transform_y=true transform_z=true
