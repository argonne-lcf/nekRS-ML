#!/bin/bash

module load thapi
module load frameworks
source ../../../examples/_env_dist-gnn_posix/bin/activate
module list

IPROF=/flare/datascience/balin/ALCF-4/nekRS-ML/Sept26/iprof/THAPI/build/ici/bin/iprof
#echo Using iprof from `which iprof`

DATE=$(date +%Y%m%d_%H%M%S)

mpiexec -n 1 --ppn 1 --cpu-bind list:1-4 \
  -- $IPROF \
  python mlp_prof_pvc.py phase1_steps=10 hidden_channels=256 n_mlp_hidden_layers=2 

#iprof -r $PWD/iprof_trace_$DATE
#-- $IPROF -l $PWD/iprof_trace_$DATE/out.pftrace --trace-output $PWD/iprof_trace_$DATE -- \
