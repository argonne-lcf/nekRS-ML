#!/bin/bash

module load thapi
module load frameworks
source ../../examples/_env/bin/activate
module list

echo Using iprof from `which iprof`

DATE=$(date +%Y%m%d_%H%M%S)

mpiexec -n 1 --ppn 1 --cpu-bind list:1-4 \
  -- iprof -l $PWD/iprof_trace_$DATE/out.pftrace --trace-output $PWD/iprof_trace_$DATE -- \
  python mlp_prof.py phase1_steps=10 hidden_channels=256 n_mlp_hidden_layers=2 

iprof -r $PWD/iprof_trace_$DATE
