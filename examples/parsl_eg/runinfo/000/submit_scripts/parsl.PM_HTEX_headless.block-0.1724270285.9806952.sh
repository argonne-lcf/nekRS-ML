
export JOBNAME=parsl.PM_HTEX_headless.block-0.1724270285.9806952
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=2
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py  --max_workers_per_node=1 -a 10.201.1.241,10.140.57.88,140.221.69.42,10.201.1.233,127.0.0.1 -p 0 -c 32.0 -m None --poll 10 --task_port=54158 --result_port=54215 --cert_dir None --logdir=/home/viralss2/nekRS-ML_MSR/examples/parsl_eg/runinfo/000/PM_HTEX_headless --block_id=0 --hb_period=30  --hb_threshold=120 --drain_period=None --cpu-affinity none  --mpi-launcher=mpiexec --available-accelerators 
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
