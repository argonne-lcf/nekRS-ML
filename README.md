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

This branch of NekRS-ML contains plugins and training routines for NekRS-compatible graph neural networks (GNNs). GNN capability is provided through (a) a NekRS-side independent plugin, which produces a graph interpretation of the mesh, and (b) a Python interface between plugin outputs and PyTorch / PyTorch Geometric to enable training of models. The code here corresponds to the following article: forthcoming. 

**NOTE**: The current example demonstrates a few training iterations for a distributed+consistent GNN. Here, "consistent" refers to the fact that, through the NekRS-provided mesh-based graph is partitioning across multiple ranks, the GNN takes into account information exchange across sub-domain (or sub-graph) boundaries by means of node-based halo exchanges. This, combined with the utilization of consistent message passing layers, leads to a consistent GNN. A forthcoming update will include a so-called consistent GNN implementation, which will take into account this information exchange step. The demo here is geared towards ALCF's Polaris supercomputer.  

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
**On a single interactive Polaris compute node**, in the `nekRS-ML` directory, configure the necessary modules:
```sh
source module_config_polaris
```
Then, create the intallation directory, and build and install the NekRS solver with the GNN plugin: 
```sh
mkdir -p install_dir/nekrs
source build_and_install_polaris
```

Build settings can be customized through CMake options passed to `nrsconfig`. 
Please remove the previous build and installation directory in case of an update. 

## Compiling and Running an Example Case
An example case is provided in `examples/tgv_gnn`: 
```sh
cd examples/tgv_gnn
```
As can be seen in the `tgv.par` file, the case is set up to not time-advance. Instead, it parses the mesh and produces graph files that can then be leveraged by the GNN models on the PyTorch end. The UDF file, `tgv.udf`, invokes the GNN plugin. More specifically, a GNN object is initialized, and member functions are called in `UDF_Setup` (the routines here are called in an initialization stage in NekRS). Ultimately, upon running this example, the GNN plugin produces a set of input files (written to disk) in this solver initialization stage. These input files are used to configure GNN training via PyTorch Geometric on the Python side. 

To run this example execute the following command in the `examples/tgv_gnn` directory **on an interactive compute node**:
```sh
source s.bin
```

The above commmand will use 4 GPUs to run the `tgv_gnn` example using a p=1 polynomial order. Note the initial execution may take some time, as the case will be JIT-compiled before actually running the solver. Compiled kernels are stored in the `.cache` directory, such that subsequent runs will skip the JIT-compilation step unless modifications are made in the `.par` file or `.udf` file. 

### GNN Plugin Outputs (`gnn_outputs`)
After the run is completed, the GNN plugin will have created the `gnn_outputs_poly_1` directory, per the polynomial order of 1 specified in the `tgv.par` file. If the code is re-run with a polynomial order of 5, new files will be written to a directory called `gnn_outputs_poly_5`, to emphasize the point that graphs constructed with different element polynomial orders are different. In what is described below, without loss of generality, the focus will be placed on the p=1 case. A snippet of the contents of `gnn_outputs_poly_1` is provided here, which contains the sub-graph (along with other) info for each of the four participating ranks: 
```bash
# Seeing the contents of gnn_outputs_poly_1
$ ls gnn_outputs_poly_1
edge_index_element_local_rank_0_size_4         fld_u_time_0.0_rank_2_size_4.bin     node_element_ids_rank_0_size_4.bin
edge_index_element_local_rank_1_size_4         fld_u_time_0.0_rank_3_size_4.bin     node_element_ids_rank_1_size_4.bin
edge_index_element_local_rank_2_size_4         global_ids_rank_0_size_4.bin         node_element_ids_rank_2_size_4.bin
edge_index_element_local_rank_3_size_4         global_ids_rank_1_size_4.bin         node_element_ids_rank_3_size_4.bin
edge_index_element_local_vertex_rank_0_size_4  global_ids_rank_2_size_4.bin         Np_rank_0_size_4
edge_index_element_local_vertex_rank_1_size_4  global_ids_rank_3_size_4.bin         Np_rank_1_size_4
edge_index_element_local_vertex_rank_2_size_4  halo_unique_mask_rank_0_size_4.bin   Np_rank_2_size_4
edge_index_element_local_vertex_rank_3_size_4  halo_unique_mask_rank_1_size_4.bin   Np_rank_3_size_4
edge_index_rank_0_size_4.bin                   halo_unique_mask_rank_2_size_4.bin   N_rank_0_size_4
edge_index_rank_1_size_4.bin                   halo_unique_mask_rank_3_size_4.bin   N_rank_1_size_4
edge_index_rank_2_size_4.bin                   local_unique_mask_rank_0_size_4.bin  N_rank_2_size_4
edge_index_rank_3_size_4.bin                   local_unique_mask_rank_1_size_4.bin  N_rank_3_size_4
fld_p_time_0.0_rank_0_size_4.bin               local_unique_mask_rank_2_size_4.bin  pos_node_rank_0_size_4.bin
fld_p_time_0.0_rank_1_size_4.bin               local_unique_mask_rank_3_size_4.bin  pos_node_rank_1_size_4.bin
fld_p_time_0.0_rank_2_size_4.bin               Nelements_rank_0_size_4              pos_node_rank_2_size_4.bin
fld_p_time_0.0_rank_3_size_4.bin               Nelements_rank_1_size_4              pos_node_rank_3_size_4.bin
fld_u_time_0.0_rank_0_size_4.bin               Nelements_rank_2_size_4
fld_u_time_0.0_rank_1_size_4.bin               Nelements_rank_3_size_4
```

Each file is composed with a header, followed by a rank ID and the MPI world size as general identifiers. A description is provided below.
* **`edge_index_rank_0_size_4.bin`** : This is a sparse representation of the adjacency matrix for the sub-graph corresponding to Rank 0, which in turn is 1 out of a total of 4 ranks used to partition the mesh. The shape of this array is `[N_edges, 2]`, where `N_edges` is the number of edges in the graph. For a single row, the first column represents the local ID of a sender graph node, and the second column is the local ID of the receiver graph node. These graph nodes coincide with GLL points; as such, local IDs in the `edge_index` correspond to the processor-local GLL points in the solution array. 
* **`pos_node_rank_0_size_4.bin`** : This contains the GLL node (and graph node) physical space coordinates for the sub-graph corresponding to Rank 0. The shape of this array is `[N_nodes, 3]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each column corresponds to the cartesian coordinate of the node in the x, y, and z directions respectively.
* **`global_ids_rank_0_size_4.bin`** : This is a node index map for the Rank 0 sub-graph. The shape of this array is `[N_nodes, 1]`, where `N_nodes` is the total number of graph nodes, which is equivalent to the total number of GLL points in the Rank 0 sub-domain. Each entry represents the global node ID. Multiple nodes may have the sample global ID, which implies coincident nodes in physical space. 
*  **`local_unique_mask_rank_0_size_4.bin`** : This is a mask array for the nodes in the Rank 0 sub-graph. The shape of this array is `[N_nodes,]`. A value of `1` represents that the corresponding node is an owner, and a value of `0` means either (a) is a non-owner on the local graph, or (b) it is a node that is shared by another rank (a non-local node, or halo node). In other words, this mask does not index nodes that are shared across processor boundaries. 
*  **`halo_unique_mask_rank_0_size_4.bin`** : This is another mask array for the nodes in the Rank 0 sub-graph. The shape of this array is also `[N_nodes,]`. A value of `1` represents that the corresponding node is an owner of a halo node (a node shared by another rank), and a value of `0` means it is either (a) is a non-owner of a halo node, or (b) it is a local node. In other words, this mask does not index nodes that are purely local to the rank.

#### Flow Data for Offline Training Example
As seen in the UDF file via `UDF_ExecuteStep`, when the solver reaches the last simulation time step, the instantaneous flow field is written to disk. The written flow fields are used to train an example GNN offline, where the GNN is tasked to learn an instantaneous velocity to pressure map. The flowfield is written to the `gnn_outputs_poly_1` folder in the following manner:
* **`fld_u_time_0.0_rank_0_size_4.bin`** : This contains the instantaneous velocity field corresponding to the final time step, for the Rank 0 sub-graph (since the simulation here is run in intitialization-only mode for this demo, the time at which these are written is 0). The shape of this array is `[N_nodes, 3]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each column corresponds to the velocity component stored on the node in the x, y, and z directions respectively.
* **`fld_p_time_0.0_rank_0_size_4.bin`** : This contains the instantaneous pressure field corresponding to the same time step, for the Rank 0 sub-graph. The shape of this array is `[N_nodes, 1]`, where `N_nodes` is the number of nodes in the graph, equivalent to the number of GLL points. Each value corresponds to the pressure stored on the node.


## Offline Training Example
#### Producing the Files Required for Consistent GNN Training/Inference
The Python component for GNN training is located in `3rd_party/gnn`. The main code containing the setup and training loop is found in `main.py`, and the GNN model is defined in `gnn.py`. The config file is located in `conf/config.yaml` -- the GNN in this demo is configured for the "large" model setting described in the paper. 

Before executing training iterations, a preprocessing step is required, which parses the graphs described above and appends to the `gnn_outputs_poly_1` folder an additional set of files (for each rank) containing the halo node information. These additional files define the halo nodes and facilitate the halo exchanges across sub-graph boundaries. To produce these files, run the following on an interactive node:
```
cd ~/3rd_party/gnn
./create_halo_info.sh
```
Upon inspection of `create_halo_info.sh`, it can be seen that this script calls `create_halo_info.py`, and writes additional files to the same `gnn_outputs_poly_1` folder. In the `gnn_outputs_poly_1` folder, the following files for each rank are created: `halo_info_*`, `node_degree_*`, and `edge_weights_*`. These files are used to facilitate halo exchanges and consistent GNN operations on the partitioned graph via the consistent message passing layers. 


#### Running distributed GNN training iterations
Again, on the same interactive node, iterations of GNN training (configured in this demo for a trivial autoencoding task) can be executed as follows:
```
./run_gnn_polaris.sh
```
This is set up to run 50 training iterations for the GNN using all 4 GPUs on the Polaris node, consistent with the number of ranks used to run the `tgv_gnn` example case. Upon executing the script, an initialization step is carried out to parse the files contained in the `examples/tgv_gnn/gnn_outputs_poly_1` directory. Training commences after this step. Each rank executes the GNN forward pass on its own local sub-graph (equivalent to its sub-domain mesh). The halo exchange setting is specified using the `halo_swap_mode` input. When looking at `run_gnn_polaris.sh`, it can be seen that the `all_to_all_opt` setting is utilized (refer to the paper for details on the various options. Additional options (commented-out) for the halo exchange include `none` (resulting in an inconsistent GNN formulation) and `all_to_all` (the standard but sub-optimal buffer configuration for the all-to-all operation).   

## License
nekRS is released under the BSD 3-clause license (see `LICENSE` file). 
All new contributions must be made under the BSD 3-clause license.

## Citing this work
Consistent GNN paper, describing the methodology and scaling: FORTHCOMING 
[NekRS, a GPU-Accelerated Spectral Element Navier-Stokes Solver](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000710) 

## Acknowledgment
The development of NekRS was supported by the Exascale Computing Project (17-SC-20-SC), 
a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security 
Administration, responsible for delivering a capable exascale ecosystem, including software, 
applications, and hardware technology, to support the nation's exascale computing imperative. 

