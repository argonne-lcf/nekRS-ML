#!/bin/bash
num_gpus=$1
shift
gpu_id=$((PALS_LOCAL_RANKID % ${num_gpus} ))
export CUDA_VISIBLE_DEVICES=$gpu_id
exec "$@"
