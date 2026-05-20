#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

db_stdout=$(/tegu/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/_env_dist-gnn_smartredis/bin/python -m smartsim._core.entrypoints.colocated +lockfile smartsim-825cbcf.lock +db_cpus 4 +command taskset -c 100,101,102,103 /lus/tegu/projects/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_online/SmartSim/smartsim/_core/bin/redis-server /lus/tegu/projects/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_online/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/tegu/projects/datascience/balin/Nek/nekRS-ML/nekRS-ML/examples/tgv_gnn_online/SmartSim/smartsim/_core/lib/redisai.so THREADS_PER_QUEUE 4 INTER_OP_PARALLELISM 1 INTRA_OP_PARALLELISM 1 --port 0 --unixsocket /tmp/redis.socket --unixsocketperm 755 --logfile /dev/null --maxclients 100000 --cluster-node-timeout 30000)
DBPID=$(echo $db_stdout | sed -n 's/.*__PID__\([0-9]*\)__PID__.*/\1/p')
$@

