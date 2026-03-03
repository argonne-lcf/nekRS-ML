#!/bin/bash
num_gpus=$1
shift
gpu_id=$((PALS_LOCAL_RANKID % ${num_gpus} ))
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=$gpu_id
exec "$@"
