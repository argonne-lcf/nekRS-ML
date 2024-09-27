import shlex, subprocess

command_line_1 = 'mpiexec -n 1 --ppn 1 -host x3001c0s1b0n0 python /eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/test_nrs.py'
command_line_2 = 'mpiexec -n 1 --ppn 1 -host x3001c0s1b1n0 python /eagle/datascience/balin/Nek/nekRS-ML_ConvReac/examples/workflow_sketch/test_nrs.py'

args_1 = shlex.split(command_line_1)
args_2 = shlex.split(command_line_2)
print(args_1)
p1 = subprocess.Popen(args_1) 
p2 = subprocess.Popen(args_1) 
p3 = subprocess.Popen(args_1) 
p4 = subprocess.Popen(args_1) 


p5 = subprocess.Popen(args_2) 
p6 = subprocess.Popen(args_2) 
p7 = subprocess.Popen(args_2) 
p8 = subprocess.Popen(args_2) 



