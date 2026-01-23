#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N advisor
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -q prod
#PBS -k doe
#PBS -j oe
cd $PBS_O_WORKDIR

module load frameworks
source ../../examples/_env/bin/activate
module list

echo Using Intel Advisor from `which advisor`

DATE=$(date +%Y%m%d_%H%M%S)

advisor --start-paused --collect=survey --profile-gpu --project-dir=$PWD/advisor_trace_$DATE --gpu-sampling-interval=1.0 -- \
  python mlp_prof.py model_task=train phase1_steps=2 hidden_channels=256 n_mlp_hidden_layers=2 precision=fp32
advisor --start-paused --collect=tripcounts --profile-gpu --flop --no-trip-counts --project-dir=$PWD/advisor_trace_$DATE --gpu-sampling-interval=1.0 -- \
  python mlp_prof.py model_task=train phase1_steps=2 hidden_channels=256 n_mlp_hidden_layers=2 precision=fp32
advisor --report=all --project-dir=$PWD/advisor_trace_$DATE --report-output=$PWD/advisor_trace_$DATE/roofline_all.html
