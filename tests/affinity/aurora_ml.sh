#!/bin/bash
offset=$1
shift
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=$(( PALS_LOCAL_RANKID + offset ))
exec "$@"
