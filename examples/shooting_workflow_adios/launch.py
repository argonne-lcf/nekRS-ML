import subprocess
import os

cmd = f"mpiexec " + \
              f"-n 24 " + \
              f"--ppn 12 "
cmd += f"python test.py"
print("Launching test ...")
p = subprocess.Popen(cmd,
                                executable="/bin/bash",
                                shell=True,
                                stdout=open('test.out','wb'),
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                cwd=os.getcwd(),
                                env=os.environ.copy()
)
