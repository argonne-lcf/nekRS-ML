import time
import sys
import socket
import os
sys.path.append(f'{os.getcwd()}/ensemble_launcher')
from ensemble_launcher import ensemble_launcher

if __name__ == '__main__':
    el = ensemble_launcher("run_dir/config.json",ncores_per_node=12)
    start_time = time.perf_counter()
    print(f'Launching node is {socket.gethostname()}')
    total_poll_time = el.run_tasks()
    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    print(f"{total_run_time=}")