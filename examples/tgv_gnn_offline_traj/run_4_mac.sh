#!/bin/bash

source ../_env/bin/activate

export NEKRS_HOME=/Users/riccardobalin/.local/nekrs
export GLOO_SOCKET_IFNAME=lo0

RANKS=4

rm -r gnn_outputs_poly_7_$RANKS gnn_outputs_poly_7
rm -r traj_poly_7 traj_poly_7_$RANKS

# Run nekRS
mpirun -n $RANKS $NEKRS_HOME/bin/nekrs --setup tgv
mv gnn_outputs_poly_7 gnn_outputs_poly_7_$RANKS
mv traj_poly_7 traj_poly_7_$RANKS

# Generate the halo_info, edge_weights and node_degree files
mpirun -n $RANKS python $NEKRS_HOME/3rd_party/gnn/dist-gnn/create_halo_info_par.py --POLY 7 --PATH ./gnn_outputs_poly_7_$RANKS

# Check the GNN input files
#echo "Checking GNN graph input files ..."
#python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref --PATH ./gnn_outputs_poly_3

# Train the GNN
mpirun -n $RANKS python ../../3rd_party/gnn/dist-gnn/main.py phase1_steps=10 master_addr=127.0.0.1 halo_swap_mode=all_to_all layer_norm=True gnn_outputs_path=./gnn_outputs_poly_7_$RANKS traj_data_path=./traj_poly_7_${RANKS}/tinit_0.000000_dtfactor_10 time_dependency=time_dependent target_loss=7.893e-01 transform_x=true transform_y=true transform_z=true

