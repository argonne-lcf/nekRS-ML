#!/bin/bash

source ../_env/bin/activate

export NEKRS_HOME=/Users/riccardobalin/.local/nekrs
export GLOO_SOCKET_IFNAME=lo0

RANKS=2

rm -r gnn_outputs_poly_3_$RANKS gnn_outputs_poly_3

# Run nekRS
mpirun -n $RANKS $NEKRS_HOME/bin/nekrs --setup tgv
mv gnn_outputs_poly_3 gnn_outputs_poly_3_$RANKS

# Generate the halo_info, edge_weights and node_degree files
mpirun -n $RANKS python $NEKRS_HOME/3rd_party/gnn/dist-gnn/create_halo_info_par.py --POLY 3 --PATH ./gnn_outputs_poly_3_$RANKS

# Check the GNN input files
#echo "Checking GNN graph input files ..."
#python /home/balin/.local/nekrs/3rd_party/gnn/dist-gnn/check_input_files.py --REF ./ref --PATH ./gnn_outputs_poly_3

# Train the GNN
mpirun -n $RANKS python ../../3rd_party/gnn/dist-gnn/main.py master_addr=127.0.0.1 halo_swap_mode=all_to_all layer_norm=True gnn_outputs_path=./gnn_outputs_poly_3_$RANKS target_loss=2.7161e-04 phase1_steps=100 n_messagePassing_layers=4 transform_x=true transform_y=true transform_z=true
