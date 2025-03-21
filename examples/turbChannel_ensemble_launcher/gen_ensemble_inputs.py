"""
This scripts generates input files for ensemble launcher to run an ensemble of nekRS
simulations by varying Re_tau, Lx, and Ly.
The resolution requirements are obtained by scaling up and down the turnChannel example
Number of elements in the y-direction are simply scaled by ratio of input Re_tau and 550
Lx elements are scaled based on input Lx and 6.283185307
Lz elements are scaled based on input Lz and 3.141592653
"""

import argparse
import numpy as np
import json
import os
import math
import shutil

parser = argparse.ArgumentParser(description="Generate input files for ensemble launcher.")
parser.add_argument("--Re_tau", type=str, required=True, help="Target Reynolds number range (min,max,nsteps).")
parser.add_argument("--Lx", type=str, required=True, help="Domain length in the x-direction range (min,max,nsteps) in H units.")
parser.add_argument("--Lz", type=str, required=True, help="Domain length in the z-direction range (min,max,nsteps) in H units.")
parser.add_argument("--outdir", type=str, default="./run_dir/", help="Run directory")
args = parser.parse_args()

# Parse the input ranges and generate lists in one step
Re_tau_list = np.linspace(*map(int, args.Re_tau.split(",")),dtype=float)
Lx_list = np.linspace(*map(int, args.Lx.split(",")),dtype=float)
Lz_list = np.linspace(*map(int, args.Lz.split(",")),dtype=float)

##create input files
outdir = args.outdir
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir,exist_ok=True)
with open("turbChannel_ref.par","r") as f:
    ref_par_lines = f.readlines()
    for l in ref_par_lines:
        if("xLength" in l):
            Lx_ref = float(l.strip().split("=")[-1].strip())
        elif("zLength" in l):
            Lz_ref = float(l.strip().split("=")[-1].strip())
        elif("ReTau" in l):
            Re_tau_ref = float(l.strip().split("=")[-1].strip())
with open("turbChannel_ref.box","r") as f:
    ref_box_lines = f.readlines()
    for l in ref_box_lines:
        if "nelx" in l:
            nelx_ref,nely_ref,nelz_ref = tuple(map(float,list(l.split("  ")[0:3])))
            break

rundirs = []
nodes = []
for Re_tau in Re_tau_list:
    for Lx in Lx_list:
        for Lz in Lz_list:
            ext = f"dir_{Re_tau}_{Lx}_{Lz}"
            os.makedirs(os.path.join(outdir,ext),exist_ok=True)
            rundirs.append(ext)
            ##create new par files
            new_par_lines = []
            for l in ref_par_lines:
                if("xLength" in l):
                    new_par_lines.append(f"xLength = {Lx:.8f}\n")
                elif("zLength" in l):
                    new_par_lines.append(f"zLength = {Lz:.8f}\n")
                elif("ReTau" in l):
                    new_par_lines.append(f"ReTau = {Re_tau:.1f}\n")
                else:
                    new_par_lines.append(l)
            with open(os.path.join(outdir,ext,"turbChannel.par"),"w") as f:
                f.writelines(new_par_lines)
            
            #create new box files
            new_box_lines = []
            for l in ref_box_lines:
                if "nelx" in l:
                    nelx = math.ceil(nelx_ref*Lx/Lx_ref)
                    nely = math.ceil(nely_ref*Re_tau/Re_tau_ref)
                    nelz = math.ceil(nelz_ref*Lz/Lz_ref)
                    new_box_lines.append(f"{nelx} {nely} {nelz}" + " "*10 + "nelx,nely,nelz for Box\n")
                else:
                    new_box_lines.append(l)
            with open(os.path.join(outdir,ext,"turbChannel.box"),"w") as f:
                f.writelines(new_box_lines)

            # Copy additional files from the current directory to the new run directory
            files_to_copy = [ "turbChannel.udf","turbChannel.usr"]
            for file_name in files_to_copy:
                src = os.path.join(".", file_name)
                dst = os.path.join(outdir, ext, file_name)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            ##compute nodes needed
            nodes.append(max(int((nelx*nely*nelz)/(nelx_ref*nely_ref*nelz_ref)),1))

            # Run genbox command
            assert os.path.exists("Nek5000/bin/genbox")
            ret_code = os.system(f"Nek5000/bin/genbox {outdir}/{ext}/turbChannel.box turbChannel")
            if ret_code != 0:
                print(f"Error running genbox in {ext}")
            os.system(f"mv turbChannel.re2 {outdir}/{ext}/")

# Assert that the required environment variable exists
required_env_var = "NEKRS_HOME"
if required_env_var not in os.environ:
    raise EnvironmentError(f"Environment variable '{required_env_var}' is not set.")

ensemble = {}
ensemble["poll_interval"] = 5
ensemble["ensembles"] = {}
ensemble["ensembles"]["nekRS_test"]={}
ensemble["ensembles"]["nekRS_test"] = {
        "num_nodes": nodes,
        "num_processes_per_node": 12,
        "launcher": "mpi",
        "relation": "one-to-one",
        "run_dir": [outdir+"/"+r for r in rundirs],
        "cmd": f"--cpu-bind list:1:8:16:24:32:40:53:60:68:76:84:92 -- {os.getenv('NEKRS_HOME')}/bin/nekrs --setup turbChannel --backend dpcpp --device-id 0",
    }
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(ensemble, f, indent=4)
            

