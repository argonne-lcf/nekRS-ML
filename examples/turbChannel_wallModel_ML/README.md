# Online Training and Inference of a Data-Driven Wall Shear Stress Model for LES

This example performs online training and inference of a data-driven wall shear stress model for LES applied to a turbulent channel flow at a friction Reynolds number of 950.
It was designed by Vishal Kumar at the Barcelona Supercomputing Center and adapted for this version of nekRS.


The example uses the `smartRedis` plugin to stage data for training as well as evaluate the ML model every time step from the `.udf` file.


## Building nekRS

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported)
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later
* PyTorch
* SmartSim and SmartRedis

To build nekRS and the required dependencies, first clone our GitHub repository:

```sh
https://github.com/argonne-lcf/nekRS-ML.git
```

Then, simply execute one of the build scripts contained in the repository.
The HPC systems currently supported are:
* [Polaris](https://docs.alcf.anl.gov/polaris/) (Argonne LCF)
* [Aurora](https://docs.alcf.anl.gov/aurora/) (Argonne LCF)

For example, to build nekRS-ML on Aurora, execute from a compute node

```sh
ENABLE_SMARTREDIS=ON ./BuildMeOnAurora
```

## Running the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.

**From a compute node** execute

```sh
./gen_run_script <system_name> </path/to/nekRS> <ML task>
```

or

```sh
./gen_run_script <system_name> </path/to/nekRS> <ML task> </path/to/venv/bin/activate>
```

if you have the necessary packages already installed in a Python virtual environment.

The script will produce a `run.sh` script specifically tailored to the desired system, using the desired nekRS install directory and performing either training or inference of the wall shear stress model.

Finally, simply execute the run script **from the compute nodes** with

```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training or inference.
- The workflow driver `ssim_driver.py` is executed with Python, setting up nekRS, the SmartSim Orchestrator (the database), and either ML training or uploading the model for inference.


The outputs of the nekRS and trainer will be within the `./nekRS` directory created at runtime.

## Known Issues and Tips
- The clustered deployment requires at least 3 nodes for training and 2 nodes for inference since it deploys each component on a distinct set of nodes. Also note that the SmartSim database can be run on a single node or sharded across 3 or more nodes (sharding across 2 nodes is not allowed).
- On Aurora, inference can only be run on the CPU. nekRS will still run on the GPU, but model inference will be executed on the host. This is due to a limitation of RedisAI on Intel hardware.
