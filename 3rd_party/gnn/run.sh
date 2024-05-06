source polaris_setup
mpiexec -n 2 ./set_affinity_gpu_polaris.sh python3 main.py backend=nccl halo_swap_mode=all_to_all
mpiexec -n 2 ./set_affinity_gpu_polaris.sh python3 main.py backend=gloo halo_swap_mode=all_to_all

mpiexec -n 2 ./set_affinity_gpu_polaris.sh python3 main.py backend=nccl halo_swap_mode=send_recv
mpiexec -n 2 ./set_affinity_gpu_polaris.sh python3 main.py backend=gloo halo_swap_mode=send_recv
