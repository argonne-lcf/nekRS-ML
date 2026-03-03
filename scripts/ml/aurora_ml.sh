#!/bin/bash
num_gpus=$1
offset=$2
shift
shift
gpu_id=$((PALS_LOCAL_RANKID % ${num_gpus} + ${offset} ))
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=$gpu_id
exec "$@"
