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

This branch of NekRS-ML combines the GNN and smartredis branches for online (in situ) training of GNN with a direct coupling of the nekRS mesh to the graph being used for the ML model. 
It therefore includes both the gnn and the smartredis plugins, as well as more developments to integrate the two. For more information on the respective features please see the `GNN` and `smartredis` branches of this repo. 

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


## Polaris Build and Run Instructions

Clone the repository and switch to the current `smartredis` branch
```sh
git clone https://github.com/argonne-lcf/nekRS-ML.git
cd nekRS-ML
git checkout onlineGNN
```

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
