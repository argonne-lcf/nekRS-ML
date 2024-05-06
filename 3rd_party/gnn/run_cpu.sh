
# make sure WITH_CUDA override is turned on in main.py 
# make sure WITH_CUDA override is turned on in main.py 
# make sure WITH_CUDA override is turned on in main.py 

source polaris_setup

mpiexec -n 2 python3 main.py backend=gloo halo_swap_mode=all_to_all
mpiexec -n 2 python3 main.py backend=gloo halo_swap_mode=send_recv
