```
                 __    ____  _____
   ____   ___   / /__ / __ \/ ___/
  / __ \ / _ \ / //_// /_/ /\__ \ 
 / / / //  __// ,<  / _, _/___/ / 
/_/ /_/ \___//_/|_|/_/ |_|/____/  
COPYRIGHT (c) 2019-2023 UCHICAGO ARGONNE, LLC
```

[![Build Status](https://travis-ci.com/Nek5000/nekRS.svg?branch=master)](https://travis-ci.com/Nek5000/nekRS)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7984525.svg)](https://doi.org/10.5281/zenodo.7984525)

This branch of NekRS-ML includes a plugin that enables communication with a SmartSim database through the use of the SmartRedis API. 
[SmartSim](https://github.com/CrayLabs/SmartSim) and [SmartRedis](https://github.com/CrayLabs/SmartRedis) are open-source libraries developed by HPE that can be used for coupling traditional HPC applications with AI/ML functionality in situ. 

Currently, this branch uses the `turbChannel_smartredis` example to demonstrate online training and inference with SmartSim/SmartRedis and NekRS. 
In particular, an MLP which takes the streamwise velocity component at some prescribed locatio off the wall as inputs is trained to predict the wall-shear stress at the corresponding wall node. This can be thought of as a crude example of using ML to train a wall-shear stress model valuable for wall-modeled LES.
The [instructions](https://github.com/argonne-lcf/nekRS-ML/tree/smartredis#polaris-instructions) below detail how to build the code, train the MLP model from a live NekRS simulation on the Polaris GPU, and then perform inference with the trained model from NekRS to compare the ML predictions with the true values. 
Note that the functions defined in the new plugin are called from `UDF_Setup()` and `UDF_ExecuteStep()` in the `.udf` file.

**nekRS** is a fast and scaleable computational fluid dynamics (CFD) solver targeting HPC applications. The code started as an early fork of [libParanumal](https://github.com/paranumal/libparanumal) in 2019.

Capabilities:

* Incompressible and low Mach-number Navier-Stokes + scalar transport 
* High-order curvilinear conformal spectral elements in space 
* Variable time step 2nd/3rd order semi-implicit time integration
* MPI + [OCCA](https://github.com/libocca/occa) (backends: CUDA, HIP, OPENCL, SERIAL/C++)
* LES and RANS turbulence models
* Arbitrary-Lagrangian-Eulerian moving mesh
* Lagrangian phase model
* Overlapping overset grids
* Conjugate fluid-solid heat transfer
* Various boundary conditions
* VisIt & Paraview support for data analysis and visualization
* Legacy interface to [Nek5000](https://github.com/Nek5000/Nek5000) 

## Build Instructions

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* C++17/C99 compatible compiler
* GNU/Intel/NVHPC Fortran compiler
* MPI-3.1 or later
* CMake version 3.18 or later 

Download the latest release available under

```sh
https://github.com/argonne-lcf/nekRS-ML/archive/refs/heads/smartredis.zip
```

or clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```
The `master` branch always points to the latest stable release while `next`
provides an early preview of the next upcoming release (do not use in a production environment).

#
To build and install the code run:

```sh
CC=mpicc CXX=mpic++ FC=mpif77 ./nrsconfig [-DCMAKE_INSTALL_PREFIX=$HOME/.local/nekrs]
```
Build settings can be customized through CMake options passed to `nrsconfig`. 
Please remove the previous build and installation directory in case of an update. 

## Setting the Enviroment

Assuming you run `bash` and your install directory is $HOME/.local/nekrs, 
add the following line to your $HOME/.bash_profile:

```sh
export NEKRS_HOME=$HOME/.local/nekrs
export PATH=$NEKRS_HOME/bin:$PATH
```
then type `source $HOME/.bash_profile` in the current terminal window. 

## Run the Code

We try hard not to break userland but the code is evolving quickly so things might change from one version to another without being backward compatible. Please consult `RELEASE.md` *before* using the code. 

```sh
cd $NEKRS_HOME/examples/turbPipePeriodic
mpirun -np 2 nekrs --setup turbPipe.par
```
For convenience we provide various launch scripts in the `bin` directory.

## Polaris Instructions

From an interactive session on a single node, set the build environment 
```sh
module load conda/2022-09-08
conda activate /eagle/projects/fallwkshp23/SmartSim/ssim
module load cudatoolkit-standalone
module load cmake
export CRAY_ACCEL_TARGET=nvidia80
```

and build the code
```sh
CC=cc CXX=CC FC=ftn ./nrsconfig -DCMAKE_INSTALL_PREFIX=</path/to/install/dir> -DENABLE_SMARTREDIS=1 -DSMARTREDIS_PATH=/eagle/projects/fallwkshp23/SmartSim/SmartRedis
```
where `</path/to/install/dir>` can be a user's home directory or a project space. 
Note that this version of NekRS requires the additional arguments `-DENABLE_SMARTREDIS=1 -DSMARTREDIS_PATH=</path/to/SmartRedis>` to the config script.

Set up the run environment
```sh
export NEKRS_HOME=</path/to/install/dir>
export PATH=$NEKRS_HOME/bin:$PATH
export LD_LIBRARY_PATH=/eagle/projects/fallwkshp23/SmartSim/SmartRedis/install/lib:$LD_LIBRARY_PATH
cd examples/turbChannel_smartredis
```

Run the online training example
```sh
ln -s turbChannel_train.udf turbChannel.udf
python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=2 run_args.simprocs_pn=2 train.executable=./trainer.py run_args.mlprocs=2 run_args.mlprocs_pn=2 train.device=cuda train.affinity=./affinity_ml.sh
```
Currently, this example runs NekRS in parallel on the 2 GPU of a Polaris node and the ML distributed training on the other 2 GPU of the node. Note also that this sets up a co-located database on the node, but the  `ssim_driver_polaris.py` script is set for both co-located and clustered workflows. The example produces the log files `nekrs.out`, `nekrs.err`, `train_model.out`, and `train_model.err` for NekRS and the ML training, respectively, and saves the trained model to file in normal and jitted formats as `model.pt` and `model_jit.pt`, respectively. 

Run the online inference example using the online trained model
```sh
rm turbChannel.udf
ln -s turbChannel_inference.udf turbChannel.udf
python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=3 run_args.simprocs_pn=3 inference.model_path=./model_jit.pt inference.device=GPU:3
```
Currently, this example runs NekRS in parallel on the first 3 GPU and performs ML inference on the fourth GPU. Similarly to the training example above, this a co-located database is launched by default but both deployment options are available. The example produces the log files `nekrs.out` and `nekrs.err`.

Finally, note that the full list of configuration options to set up the training and inference runs can be found in `conf/ssim_config.yaml`.

## Documentation 
For documentation, see our [readthedocs page](https://nekrs.readthedocs.io/en/latest/). For now it's just a dummy. We hope to improve it soon. 

## Discussion Group
Please visit [GitHub Discussions](https://github.com/Nek5000/nekRS/discussions). Here we help, find solutions, share ideas, and follow discussions.

## Contributing
Our project is hosted on [GitHub](https://github.com/Nek5000/nekRS). To learn how to contribute, see `CONTRIBUTING.md`.

## Reporting Bugs
All bugs are reported and tracked through [Issues](https://github.com/Nek5000/nekRS/issues). If you are having trouble installing the code or getting your case to run properly, you should first vist our discussion group.

## License
nekRS is released under the BSD 3-clause license (see `LICENSE` file). 
All new contributions must be made under the BSD 3-clause license.

## Citing nekRS
[NekRS, a GPU-Accelerated Spectral Element Navier-Stokes Solver](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000710) 

## Acknowledgment
This research was supported by the Exascale Computing Project (17-SC-20-SC), 
a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security 
Administration, responsible for delivering a capable exascale ecosystem, including software, 
applications, and hardware technology, to support the nation's exascale computing imperative.
