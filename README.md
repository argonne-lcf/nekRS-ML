```
███    ██ ███████ ██   ██ ██████  ███████
████   ██ ██      ██  ██  ██   ██ ██     
██ ██  ██ █████   █████   ██████  ███████
██  ██ ██ ██      ██  ██  ██   ██      ██
██   ████ ███████ ██   ██ ██   ██ ███████ 
(c) 2019-2024 UCHICAGO ARGONNE, LLC
```

[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)

nekRS-ML is a fork of the ALCF managed [nekRS v24](https://github.com/argonne-lcf/nekRS) computational fluid dynamics (CFD) solver augmented to provides examples and capabilities for AI-enabled CFD research on HPC systems. 
It is meant to be a sandbox showcasing ways in which ML methods and *in-situ* workflows can be used to integrate AI with traditional CFD simulations on HPC systems.

Some key functionalities of nekRS-ML are:

* Graph neural network (GNN) modeling: 
  * [Dist-GNN](./3rd_party/gnn/dist-gnn/) is a scalable and consistent GNN for mesh-modeling of dynamical systems on very large graphs. It relies on tailored neural message passing layers and loss constructions to guarantee arithmetic consistency on domain-decomposed graphs partitioned similarly to a CFD mesh. It can be used to perform both time dependent modeling (e.g., advance the solution field) and time independent modeling (e.g., predict a flow quantity from another). For detailed information on the Dist-GNN model, please see the following [paper](https://ieeexplore.ieee.org/abstract/document/10820662).
  * [SR-GNN](./3rd_party/gnn/sr-gnn/) is a GNN for mesh-based, three-dimensional super-resolution of fluid flows. The SR-GNN model operates on individual elements (and their small neighborhood if set up as such), but not on the full mesh/graph of the domain, thus unlike the Dist-GNN model this one is local in nature. SR-GNN is comprised of coarse- and fine-scale message passing layers for multi-scale modeling. For detailed information on the SR-GNN model, please see the following [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782525003445).
* [Conversion tools for mesh-based distributed GNN modeling](./src/plugins/gnn.hpp): NekRS-ML provides a GNN plugin capable of extracting the necessary information from nekRS to contruct the partitioned graph needed by Dist-GNN. The same GNN plugin and the [trajectory generation plugin](./src/plugins/trajGen.hpp) can be used to extract the field information from nekRS to produce training data for the Dist-GNN. The GNN and trajectory generation plugins can create graphs and the respective training data from p-coarsened nekRS meshes to enable development of surrogates on coarser discretizations.  
* [Data streaming with ADIOS2](./src/plugins/adiosStreamer.hpp): nekRS v24 comes with ADIOS2 for I/O, thus nekRS-ML expands the usage of ADIOS2 to enable data streaming between nekRS and GNN training, enabling online (or *in-situ*) training/fine-tuning of the ML models.  
* [In-memory data staging with SmartSim](./src/plugins/smartRedis.hpp): nekRS-ML can also be linked to the [SmartRedis](https://github.com/CrayLabs/SmartRedis) library, which when coupled with a [SmartSim](https://github.com/CrayLabs/SmartSim) workflow enables online training and inference with in-memory data-staging. 

### Progression of AI-enabled examples

nekRS-ML hosts a series of AI-enabled examples listed below in order of complexity to provide a smooth learning progression. 
Users can find more details on each of the examples in the  README files contained within the respective directories. 

* [tgv_gnn_offline](./examples/tgv_gnn_offline/): Offline training pipeline to generate data and perform time independent training of the Dist-GNN model.
* [tgv_gnn_offline_coarse_mesh](./examples/tgv_gnn_offline_coarse_mesh/): Offline training pipeline to generate data and perform time independent training of the Dist-GNN model on a p-coarsened grid relative to the one used by the nekRS simulation.
* [tgv_gnn_offline_traj](./examples/tgv_gnn_offline_traj/): Offline training pipeline to generate data and perform time dependent training of the Dist-GNN model.
* [tuurbChannel_srgnn](./examples/turbChannel_srgnn/): Offline training pipeline to generate data, perform training, and evaluate the model through inference with the SR-GNN model. 
* [turbChannel_wallModel_ML](./examples/turbChannel_wallModel_ML/): Online training and inference workflows of a data-driven wall shear stress model for LES applied to a turbulent channel flow at a friction Reynolds number of 950. This example is an extension to [turbChannel_wallModel](./examples/turbChannel_wallModel/), which uses an algebraic equilibrium wall model (no ML).
* [tgv_gnn_online](./examples/tgv_gnn_online/): Online training workflow using SmartSim to cuncurrently generate data and perform time independent training of the Dist-GNN model.
* [tgv_gnn_online_traj](./examples/tgv_gnn_online_traj/): Online training workflow using SmartSim to cuncurrently generate data and perform time dependent training of the Dist-GNN model.
* [tgv_gnn_online_traj_adios](./examples/tgv_gnn_online_traj_adios/): Online training workflow using ADIOS2 to cuncurrently generate data and perform time dependent training of the Dist-GNN model.
* [shooting_workflow_smartredis](./examples/shooting_workflow_smartredis/): Online training workflow using SmartSim to shoot the nekRS solution forward in time leveraging the Dist-GNN model.
* [shooting_workflow_adios](./examples/shooting_workflow_adios/): Online training workflow using ADIOS2 to shoot the nekRS solution forward in time leveraging the Dist-GNN model.


## Build Instructions

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later 

Optional requirements:
* PyTorch and PyTorch Geometric (for the examples using the GNN)
* SmartSim and SmartRedis (for the examples using SmartSim as a workflow driver)

To build nekRS and the required dependencoes, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

The `main` (default) branch always points to the latest stable version of the code. 
Other branches available in the repository should be considered experimental. 

Then, simply execute one of the build scripts contained in the reposotory. 
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) @ Argonne LCF
* [Aurora](https://docs.alcf.anl.gov/aurora/) @ Argonne LCF
* [Crux](https://docs.alcf.anl.gov/crux/) @ Argonne LCF (limited support for ML-enabled exampels)

For example, to build nekRS-ML on Aurora without the SmartRedis client, execute from a compute node

```sh
./BuildMeOnAurora
```

If istead the SmartRedis client is desired, execute

```sh
ENABLE_SMARTREDIS=ON ./BuildMeOnAurora
```

If a build script for a specific HPC system is not available, please submit an issue or feel free to contribute a PR (see below for details on both).


## Running the AI-enabled Examples

To run any of the AI-enabled examples listed above, simply `cd` to the example directory of interest and **from a compute node** execute

```sh
./gen_run_script <system_name> </path/to/nekRS>
```

or

```sh
./gen_run_script <system_name> </path/to/nekRS> -v </path/to/venv/bin/activate>
```

if you have the necessary packages already installed in a Python virtual environment. 
For more information on all the options available to configure the `gen_run_script` scripts, run `./gen_run_script -h`.

The case setup script will produce a `run.sh` script specifically tailored to the desired system and using the desired nekRS install directory. 

Finally, the examples are run **from the compute nodes** executing

```sh
./run.sh
```

## Documentation 
For documentation on the nekRS solver, see the [readthedocs page](https://nekrs.readthedocs.io/en/latest/). Please note these pages are a work in progress. For documentation on the specific nekRS-ML examples, we encourage users to follow the README files within each example directory.

## Discussion Group
For nekRS specific questions, please visit the [GitHub Discussions](https://github.com/Nek5000/nekRS/discussions). Here nekRS developers help, find solutions, share ideas, and follow discussions.

## Contributing
Our project is hosted on [GitHub](https://github.com/argonne-lcf/nekRS-ML). To learn how to contribute, see `CONTRIBUTING.md`.

## Reporting Bugs
All bugs are reported and tracked through [Issues](https://github.com/argonne-lcf/nekRS-ML/issues). If you are having trouble installing the code or getting your case to run properly, please submit an issue.

## License
nekRS is released under the BSD 3-clause license (see `LICENSE` file). 
All new contributions must be made under the BSD 3-clause license.

## Acknowledgment
This research was supported by the Exascale Computing Project (17-SC-20-SC), 
a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security 
Administration, responsible for delivering a capable exascale ecosystem, including software, 
applications, and hardware technology, to support the nation's exascale computing imperative.
