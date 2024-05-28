source polaris_setup

mpiexec -n 2 -ppn 4 -d 8 --cpu-bind depth ./set_affinity_gpu_polaris.sh python3 main_simplified.py backend=nccl halo_swap_mode=all_to_all
mpiexec -n 2 -ppn 4 -d 8 --cpu-bind depth ./set_affinity_gpu_polaris.sh python3 main_simplified.py backend=nccl halo_swap_mode=send_recv
