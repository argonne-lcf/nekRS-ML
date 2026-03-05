#!/bin/bash
num_gpus=$1
offset=$2
shift
shift
gpu_id=$(( PALS_LOCAL_RANKID % ${num_gpus} + ${offset} ))
export CUDA_VISIBLE_DEVICES=$gpu_id
exec "$@"
