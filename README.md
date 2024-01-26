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

This branch of NekRS-ML contains plugins and training routines for NekRS-compatible graph neural networks (GNNs). GNN capability is provided through (a) a NekRS-side independent plugin, which produces a graph interpretation of the mesh, and (b) a Python interface between plugin outputs and PyTorch / PyTorch Geometric to enable training of models.

**NOTE**: The current example demonstrates offline training for a naive distributed GNN. Here, "naive" refers to the fact that although the mesh-based graph is partitioned across multiple ranks, the GNN model does not take into account information exchange across sub-domain (or sub-graph) boundaries corresponding to processor interfaces. A forthcoming update will include a so-called consistent GNN implementation, which will take into account this information exchange step. 

**nekRS** is a fast and scaleable computational fluid dynamics (CFD) solver targeting HPC applications. The code started as an early fork of [libParanumal](https://github.com/paranumal/libparanumal) in 2019.

**Graph neural networks** are a branch of the [geometric deep learning](https://geometricdeeplearning.com/) paradigm, and are considered state-of-the-art for developing accelerated data-based models (e.g., surrogates, sub-grid models, etc.) in mesh-based scientific applications. A recent review can be found [here](https://www.sciencedirect.com/science/article/pii/S2666651021000012).

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

## Building on Polaris
First, clone this repository, and checkout the `GNN` branch:
```
git clone https://github.com/argonne-lcf/nekRS-ML.git
cd nekRS-ML
git checkout GNN
```
**On a compute node**, in the `nekRS-ML` directory, configure the necessary modules:
```sh
source module_config_polaris
```
Then, create the intallation directory, and build and install the solver: 
```sh
mkdir -p install_dir/nekrs
source build_and_install_polaris
```

Build settings can be customized through CMake options passed to `nrsconfig`. 
Please remove the previous build and installation directory in case of an update. 

## Compiling and Running an Example Case
An example case is provided in `examples/periodicHill_gnn`: 
```sh
cd examples/periodicHill_gnn
```
The UDF file, `periodicHill.udf`, invokes the GNN plugin. More specifically, a GNN object is initialized, and member functions are called in `UDF_Setup`. Ultimately, upon running this example, the GNN plugin produces a set of input files (written to disk) in the solver initialization stage. These input files are used to configure GNN training via PyTorch Geometric on the Python side. 

To run this example execute the following command **on a login node**:
```sh
PROJ_ID=[YOUR_PROJECT_ID] QUEUE=debug ./nrsqsub_polaris periodicHill 1 00:30:00
```

The above commmand will queue a job on a single Polaris node and use 4 GPUs to run the periodicHill example. Note the initial execution may take some time, as the case will be JIT-compiled before actually running the solver. Compiled kernels are stored in the `.cache` directory, such that subsequent runs will skip the JIT-compilation step unless modifications are made in the `.par` file or `.udf` file. 

### GNN Plugin Outputs (`gnn_outputs`)
After the run is completed, the GNN plugin will have created the `gnn_outputs` directory. A snippet is provided here: 
```bash
# Seeing the contents of gnn_outputs
$ ls gnn_outputs
edge_index_rank_0_size_4  global_ids_rank_3_size_4         local_unique_mask_rank_2_size_4  x_rank_1_size_4
edge_index_rank_1_size_4  halo_unique_mask_rank_0_size_4   local_unique_mask_rank_3_size_4  x_rank_2_size_4
edge_index_rank_2_size_4  halo_unique_mask_rank_1_size_4   pos_node_rank_0_size_4           x_rank_3_size_4
edge_index_rank_3_size_4  halo_unique_mask_rank_2_size_4   pos_node_rank_1_size_4           y_rank_0_size_4
global_ids_rank_0_size_4  halo_unique_mask_rank_3_size_4   pos_node_rank_2_size_4           y_rank_1_size_4
global_ids_rank_1_size_4  local_unique_mask_rank_0_size_4  pos_node_rank_3_size_4           y_rank_2_size_4
global_ids_rank_2_size_4  local_unique_mask_rank_1_size_4  x_rank_0_size_4                  y_rank_3_size_4
```

Each file is composed with a header, followed by a rank ID and the MPI world size as general identifiers. A description is provided below.
* **`edge_index_rank_0_size_4`** : This is a sparse representation of the adjacency matrix for the sub-graph corresponding to Rank 0, which in turn is 1 out of a total of 4 ranks used to partition the mesh. The shape of this array is `[N_edges, 2]`, where `N_edges` is the number of edges in the graph. For a single row, the first column represents the local ID of a sender graph node, and the second column is the local ID of the receiver graph node. These graph nodes coincide with GLL points; as such, local IDs in the `edge_index` correspond to the processor-local GLL points in the solution array. 
* **`pos_node_rank_0_size_4`** : This contains the GLL node (and graph node) physical space coordinates for the sub-graph corresponding to Rank 0. The shape of this array is `[N_nodes, 3]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each column corresponds to the cartesian coordinate of the node in the x, y, and z directions respectively.
* **`global_ids_rank_0_size_4`** : This is a node index map for the Rank 0 sub-graph. The shape of this array is `[N_nodes, 1]`, where `N_nodes` is the total number of graph nodes, which is equivalent to the total number of GLL points in the Rank 0 sub-domain. Each entry represents the global node ID. Multiple nodes may have the sample global ID, which implies coincident nodes in physical space. 
*  **`local_unique_mask_rank_0_size_4`** : This is a mask array for the nodes in the Rank 0 sub-graph. The shape of this array is `[N_nodes,]`. A value of `1` represents that the corresponding node is an owner, and a value of `0` means either (a) is a non-owner on the local graph, or (b) it is a node that is shared by another rank (a non-local node, or halo node). In other words, this mask does not index nodes that are shared across processor boundaries. 
*  **`halo_unique_mask_rank_0_size_4`** : This is another mask array for the nodes in the Rank 0 sub-graph. The shape of this array is also `[N_nodes,]`. A value of `1` represents that the corresponding node is an owner of a halo node (a node shared by another rank), and a value of `0` means it is either (a) is a non-owner of a halo node, or (b) it is a local node. In other words, this mask does not index nodes that are purely local to the rank.

#### Flow Data for Offline Training Example
As seen in the UDF file via `UDF_ExecuteStep`, when the solver reaches the last simulation time step, the instantaneous flow field is written to disk. The written flow fields are used to train an example GNN offline, where the GNN is tasked to learn an instantaneous velocity to pressure map. The flowfield is written to the `gnn_outputs` folder in the following manner:
* **`x_rank_0_size_4`** : This contains the instantaneous velocity field corresponding to the final time step, for the Rank 0 sub-graph. The shape of this array is `[N_nodes, 3]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each column corresponds to the velocity component stored on the node in the x, y, and z directions respectively.
* **`y_rank_0_size_4`** : This contains the instantaneous pressure field corresponding to the final time step, for the Rank 0 sub-graph. The shape of this array is `[N_nodes, 1]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each value corresponds to the pressure stored on the node.


#### Producing the Files Required for Consistent GNN Training/Inference
The `periodicHill_gnn` case directory includes the file `create_halo_info.py`. Running this script adds additional files to the `gnn_outputs` directory required for consistent distributed GNN training and inference, based on graph node non-local scatter operations. As an example, execute the following on an **intereactive node** in the `periodicHill_gnn` directory to run this script: 
```bash
module load conda/2022-09-08
conda activate /lus/eagle/projects/datascience/sbarwey/codes/conda_envs/base-clone
python create_halo_info.py --SIZE 4 --POLY 7
```
This will create the following files for each rank: `halo_info_*`, `node_degree_*`, and `edge_weights_*`. These files are used to facilitate halo exchanges and consistent GNN operations on the partitioned graph. These are not utilized in the naive GNN implementation. Detailed descriptions of these files will be provided in an upcoming repository update.

## Offline Training Example
The Python component for GNN training is located in `3rd_party/gnn`. To execute the offline training example, run the following command on an **interactive node** :
```
module load conda/2022-09-08
module load cudatoolkit-standalone/11.4.4
conda activate /lus/eagle/projects/datascience/sbarwey/codes/conda_envs/base-clone
cd 3rd_party/gnn
mpiexec -n 4 ./set_affinity_gpu_polaris.sh python3 main.py gnn_outputs_path=[PATH_TO_NEKRS_HOME]/examples/periodicHill_gnn/gnn_outputs/
```
The first three lines configure the Python conda environment, which includes the GPU-compatible PyTorch Geometric library. The last line trains a GNN using 4 GPUs, consistent with the number of ranks used to run the `periodicHill_gnn` example case. Upon executing the script, an initialization step is carried out to parse the files conatined in the `examples/periodicHill_gnn/gnn_outputs` directory. Training commences after this step. During training, a single snapshot is used to learn a velocity-to-pressure mapping. Each rank executes the GNN forward pass on its own local sub-graph (equivalent to its sub-domain mesh), and DDP is utilized during training to synchronize models across ranks. As mentioned above, in this demonstration, a naive training approach is leveraged, in that the ranks do not communicate with eachother during the GNN forward pass. As a result, predictions are expected to be inconsistent across sub-graph boundaries. A forthcoming update will include a so-called **consistent** GNN implementation, which takes into account this information exchange step.

## License
nekRS is released under the BSD 3-clause license (see `LICENSE` file). 
All new contributions must be made under the BSD 3-clause license.

## Citing nekRS
[NekRS, a GPU-Accelerated Spectral Element Navier-Stokes Solver](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000710) 

## Acknowledgment
The development of NekRS was supported by the Exascale Computing Project (17-SC-20-SC), 
a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security 
Administration, responsible for delivering a capable exascale ecosystem, including software, 
applications, and hardware technology, to support the nation's exascale computing imperative. 

