#ifdef ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include "gnn.hpp"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
wallModel_data *wm = new wallModel_data;
SmartRedis::Client *client_ptr;

// Initialize the SmartRedis client and the smartredis struct
void smartredis::init_client(nrs_t *nrs)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  // Replace this with variable in .par file
  sr->npts_per_tensor = nrs->fieldOffset;
  sr->num_tot_tensors = size;
  sr->num_db_tensors = size;
  sr->head_rank = 0;
  sr->db_nodes = 1;

  // Initialize SR client
  if (rank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode;
  if (sr->db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  std::string logger_name("Client");
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n");
}

// Check value of check-run variable to know when to quit
int smartredis::check_run()
{
  int rank = platform->comm.mpiRank;
  int exit_val;
  std::string run_key = "check-run";
  int *check_run = new int[2]();

  // Check value of check-run tensor in DB from head rank
  if (rank%sr->num_db_tensors == 0) {
    client_ptr->unpack_tensor(run_key, check_run, {2},
                       SRTensorTypeInt32, SRMemLayoutContiguous);
    exit_val = check_run[0];
  }

  // Broadcast exit value and return it
  MPI_Bcast(&exit_val, 1, MPI_INT, 0, platform->comm.mpiComm);
  if (exit_val==0 && platform->comm.mpiRank==0) {
    printf("\nML training says time to quit ...\n");
  }
  return exit_val;
}

// Initialize the check-run tensor on DB
void smartredis::init_check_run()
{
  int rank = platform->comm.mpiRank;
  std::vector<int> check_run(2,0);
  check_run[0] = 1; check_run[1] = 1;
  std::string run_key = "check-run";

  if (rank%sr->num_db_tensors == 0) {
    client_ptr->put_tensor(run_key, check_run.data(), {2},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Put check-run in DB\n\n");
}

// Put step number in DB
void smartredis::put_step_num(int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  std::string key = "step";
  std::vector<double> step_num(1,0);
  step_num[0] = tstep;

  // Send time step to DB
  if (rank == 0)
    printf("\nSending time step number ...\n");
  client_ptr->put_tensor(key, step_num.data(), {1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// --------------------------------------------------- //
// ----- Online GNN ---------------------------------- //
// --------------------------------------------------- //
// Initialize training with a GNN
void smartredis::init_train_gnn(gnn_t *graph)
{
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;
  int num_nodes = graph->get_num_nodes();
  int num_edges = graph->get_num_edges();

  if (rank == 0) 
    printf("\nSending mesh data for GNN setup ...\n");

  std::string irank = "_rank_" + std::to_string(rank);
  std::string nranks = "_size_" + std::to_string(size);
  std::string pos_key = "pos_node" + irank + nranks;
  dlong *pos = graph->get_pos();
  client_ptr->put_tensor(pos_key, pos, {num_nodes,3},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  std::string edge_key = "edge_index" + irank + nranks;
  dlong *edge_index = graph->get_edges();
  client_ptr->put_tensor(edge_key, edge_index, {num_edges,2},
                    SRTensorTypeInt64, SRMemLayoutContiguous);


   MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

}



// --------------------------------------------------- //
// ----- Wall-Shear Stress Model --------------------- //
// --------------------------------------------------- //
// Initialize training for the wall shear stress model
void smartredis::init_wallModel_train(nrs_t *nrs)
{
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;
  std::vector<int> tensor_info(6,0);
  auto mesh = nrs->meshV;

  if (rank == 0) 
    printf("\nSending training metadata for wall shear stress model ...\n");

  // Loop over mesh coordinates to find wall and off-wall node indices
  std::vector<int> ind_owall_nodes_raw;
  for (int i=0; i<mesh->Nlocal; i++) {
    const auto y = mesh->y[i];
    if (y <= wm->wall_height+wm->eps) {
      wm->ind_wall_nodes.push_back(i);
    }
    else if (y >= wm->off_wall_height-wm->eps && y <= wm->off_wall_height+wm->eps) {
      ind_owall_nodes_raw.push_back(i);
    }
  }
  int wall_node_ct = wm->ind_wall_nodes.size();
  int owall_node_ct = ind_owall_nodes_raw.size();
  printf("Found %d wall nodes and %d off-wall nodes\n",wall_node_ct, owall_node_ct);
  assert(wall_node_ct == owall_node_ct);
  sr->npts_per_tensor = wall_node_ct;
  wm->num_samples = wall_node_ct;

  // Pair the wall nodes with the off-wall nodes based on the x and z coordinate
  // to pair the inputs and outputs of the model
  std::vector<int> ind_owall_nodes_tmp(wall_node_ct,0);
  ind_owall_nodes_tmp = ind_owall_nodes_raw;
  int pairs = 0;
  for (int i=0; i<wall_node_ct; i++) {
    int ind_w = wm->ind_wall_nodes[i];
    const auto x_wall = mesh->x[ind_w];
    const auto z_wall = mesh->z[ind_w];
    for (int j=0; j<ind_owall_nodes_tmp.size(); j++) {
      int ind_ow = ind_owall_nodes_tmp[j];
      const auto x_owall = mesh->x[ind_ow];
      const auto z_owall = mesh->z[ind_ow];
      if ((x_owall>=x_wall-wm->eps && x_owall<=x_wall+wm->eps) && 
          (z_owall>=z_wall-wm->eps && z_owall<=z_wall+wm->eps)) {
        wm->ind_owall_nodes_matched.push_back(ind_ow);
        ind_owall_nodes_tmp.erase(ind_owall_nodes_tmp.begin() + j);
        pairs++;
      }
    }
  }
  printf("Found %d paris of wall nodes and off-wall nodes\n",pairs);

  // Create and send metadata for training
  tensor_info[0] = sr->npts_per_tensor;
  tensor_info[1] = sr->num_tot_tensors;
  tensor_info[2] = sr->num_db_tensors;
  tensor_info[3] = sr->head_rank;
  tensor_info[4] = wm->num_inputs;
  tensor_info[5] = wm->num_outputs;
  std::string info_key = "tensorInfo";
  if (rank%sr->num_db_tensors == 0) {
    client_ptr->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);

  if (rank == 0)
    printf("Done\n\n");
}

// Put training data for wall shear stress model in DB
void smartredis::put_wallModel_data(nrs_t *nrs, int tstep)
{
  int rank = platform->comm.mpiRank;
  int num_samples = wm->num_samples;
  int num_cols = wm->num_inputs+wm->num_outputs;
  dfloat mue;
  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *train_data = new dfloat[num_samples*num_cols]();
  dfloat *vel_data = new dfloat[num_samples]();
  dfloat *shear_data = new dfloat[num_samples]();

  // Extract velocity at off-wall nodes (inputs)
  for (int i=0; i<num_samples; i++) {
    int ind = wm->ind_owall_nodes_matched[i];
    vel_data[i] = nrs->U[ind+0*nrs->fieldOffset];
  }

  // Extract strain rate at wall and multiply by viscosity to obtain stress
  platform->options.getArgs("VISCOSITY",mue);
  for (int i=0; i<num_samples; i++) {
    int ind = wm->ind_wall_nodes[i];
    //shear_data[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
    shear_data[i] = 0.055;
  }

  // Concatenate inputs and outputs
  for (int i=0; i<num_samples; i++) {
    train_data[i*num_cols] = vel_data[i];
    train_data[i*num_cols+1] = shear_data[i];
  }

  // Send training data to DB
  if (rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {num_samples,num_cols},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// Run ML model for inference
void smartredis::run_wallModel(nrs_t *nrs, int tstep)
{
  int rank = platform->comm.mpiRank;
  int num_samples = wm->num_samples;
  dfloat mue;
  std::string in_key = "x." + std::to_string(rank);
  std::string out_key = "y." + std::to_string(rank);
  dfloat *outputs = new dfloat[num_samples]();
  dfloat *inputs = new dfloat[num_samples]();
  dfloat *targets = new dfloat[num_samples]();

  // Extract velocity at off-wall nodes (inputs)
  for (int i=0; i<num_samples; i++) {
    int ind = wm->ind_owall_nodes_matched[i];
    inputs[i] = nrs->U[ind+0*nrs->fieldOffset];
  }

  // Extract strain rate at wall and multiply by viscosity to obtain stress
  platform->options.getArgs("VISCOSITY",mue);
  for (int i=0; i<num_samples; i++) {
    int ind = wm->ind_wall_nodes[i];
    //targets[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
    targets[i] = 0.055;
  }
  
  // Send input data
  if (rank == 0)
    printf("\nSending field with key %s \n",in_key.c_str());
  client_ptr->put_tensor(in_key, inputs, {num_samples,1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Run ML model on input data
  if (rank == 0)
    printf("\nRunning ML model ...\n");
  client_ptr->run_model("model", {in_key}, {out_key});
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Retrieve model pedictions
  if (rank == 0)
    printf("\nRetrieving field with key %s \n",out_key.c_str());
  client_ptr->unpack_tensor(out_key, outputs, {num_samples},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
  
  // Compute error in prediction
  double error = 0.0;
  for (int i=0; i<num_samples; i++) {
    error = error + (outputs[i] - targets[i])*(outputs[i] - targets[i]);
    //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
  }
  error = error / num_samples;
  printf("[%d]: Mean Squared Error in wall shear stress field = %E\n\n",rank,error);
  fflush(stdout);
}

// --------------------------------------------------- //
// ----- Pressure Model ------------------------------ //
// --------------------------------------------------- //
// Initialize training for the velocity-pressure model
void smartredis::init_velNpres_train(nrs_t *nrs)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  if (rank == 0) 
    printf("\nSending training metadata for velocity-pressure model ...\n");

  // Create and send metadata for training
  std::vector<int> tensor_info(6,0);
  tensor_info[0] = sr->npts_per_tensor;
  tensor_info[1] = sr->num_tot_tensors;
  tensor_info[2] = sr->num_db_tensors;
  tensor_info[3] = sr->head_rank;
  tensor_info[4] = 3; // number of model inputs
  tensor_info[5] = 1; // number of model outputs
  std::string info_key = "tensorInfo";
  if (rank%sr->num_db_tensors == 0) {
    client_ptr->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);

  if (rank == 0)
    printf("Done\n\n");
}

// Put velocity and pressure data in DB
void smartredis::put_velNpres_data(nrs_t *nrs, int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *train_data = new dfloat[nrs->fieldOffset * 4]();
  int size_U = nrs->fieldOffset * 3;
  int size_P = nrs->fieldOffset;

  // Concatenate velocity (inputs) and pressure (output)
  std::copy(nrs->U,nrs->U+size_U,train_data);
  std::copy(nrs->P,nrs->P+size_P,train_data+size_U);

  // Send training data to DB
  if (rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {nrs->fieldOffset,4},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// Run a ML model for inference
void smartredis::run_pressure_model(nrs_t *nrs, int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int npts = nrs->fieldOffset;
  std::string in_key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  std::string out_key = "y." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *outputs = new dfloat[npts]();
  
  // Send input data
  if (rank == 0)
    printf("\nSending field with key %s \n",in_key.c_str());
  client_ptr->put_tensor(in_key, nrs->U, {npts,3},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Run ML model on input data
  if (rank == 0)
    printf("\nRunning ML model ...\n");
  client_ptr->run_model("model", {in_key}, {out_key});
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Retrieve model pedictions
  if (rank == 0)
    printf("\nRetrieving field with key %s \n",out_key.c_str());
  client_ptr->unpack_tensor(out_key, outputs, {npts},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
  
  // Compute error in prediction
  double error = 0.0;
  for (int n=0; n<npts; n++) {
    error = error + (outputs[n] - nrs->P[n])*(outputs[n] - nrs->P[n]);
    //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
  }
  error = error / npts;
  printf("[%d]: Mean Squared Error in pressure field = %f\n\n",rank,error);
  fflush(stdout);
}

#endif
