source /home/viralss2/ALCF_Hands_on_HPC_Workshop/workflows/parsl/launch_eg/set_run_env;
                            cd /home/viralss2/nekRS-ML_MSR/examples/parsl_MSR_smartsim
export JOBNAME=parsl.HighThroughputExecutor.block-0.1724113243.0351868
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py   -a 10.140.56.174 -p 0 -c 1.0 -m None --poll 10 --task_port=54692 --result_port=54597 --cert_dir None --logdir=/home/viralss2/nekRS-ML_MSR/examples/parsl_MSR_smartsim/runinfo/021/HighThroughputExecutor --block_id=0 --hb_period=30  --hb_threshold=120 --drain_period=None --cpu-affinity block-reverse  --mpi-launcher=mpiexec --available-accelerators 0,1,2,3
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