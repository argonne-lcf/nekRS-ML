#ifdef NEKRS_ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

// Initialize the SmartRedis client
smartredis_client_t::smartredis_client_t(nrs_t *nrs)
{
  // MPI
  _rank = platform->comm.mpiRank;
  _size = platform->comm.mpiCommSize;

  // nekRS object
  _nrs = nrs;

  // Set up SmartRedis parameters
  platform->options.getArgs("SSIM DB DEPLOYMENT",_deployment);
  _num_tot_tensors = _size;
  if (_deployment == "colocated") {
    _num_db_tensors = std::stoi(getenv("PALS_LOCAL_SIZE"));
    if (_rank%_num_db_tensors == 0)
      _head_rank = _rank;
    _db_nodes = 1;
  } else if (_deployment == "clustered") {
    _num_db_tensors = _size;
    _head_rank = 0;
    platform->options.getArgs("SSIM DB NODES",_db_nodes);
  }

  // Initialize SR client
  if (_rank == 0)
    printf("\nInitializing SmartRedis client ...\n");
  bool cluster_mode;
  if (_db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  std::string logger_name("Client");
  _client = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("All done\n");
  fflush(stdout);
}
 // destructor
smartredis_client_t::~smartredis_client_t()
{
    if (_rank == 0) printf("Taking down smartredis_client_t\n");
    delete[] _client;
}

// Initialize the check-run tensor on DB
void smartredis_client_t::init_check_run()
{
  int *check_run = new int[1]();
  check_run[0] = 1;
  std::string run_key = "check-run";

  if (_rank % _num_db_tensors == 0) {
    _client->put_tensor(run_key, check_run, {1},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Put check-run in DB\n\n");
  fflush(stdout);
}

// Check value of check-run variable to know when to quit
int smartredis_client_t::check_run()
{
  int exit_val;
  std::string run_key = "check-run";
  int *check_run = new int[1]();

  // Check value of check-run tensor in DB from head rank
  if (_rank%_num_db_tensors == 0) {
    _client->unpack_tensor(run_key, check_run, {1},
                       SRTensorTypeInt32, SRMemLayoutContiguous);
    exit_val = check_run[0];
  }

  // Broadcast exit value and return it
  MPI_Bcast(&exit_val, 1, MPI_INT, 0, platform->comm.mpiComm);
  if (exit_val==0 && _rank==0) {
    printf("\nML training says time to quit ...\n");
  }
  fflush(stdout);
  return exit_val;
}

// Put step number in DB
void smartredis_client_t::put_step_num(int tstep)
{
  // Initialize local variables
  std::string key = "step";
  std::vector<long> step_num(1,0);
  step_num[0] = tstep;

  // Send time step to DB
  if (_rank == 0)
    printf("\nSending time step number ...\n");
  if (_rank%_num_db_tensors == 0) {
    _client->put_tensor(key, step_num.data(), {1},
                    SRTensorTypeInt64, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");
  fflush(stdout);
}

// Append a new DataSet to a list and put in DB
void smartredis_client_t::append_dataset_to_list(const std::string& dataset_name,
  const std::string& tensor_name,
  const std::string& list_name,
  dfloat* data,
  unsigned long int num_rows,
  unsigned long int num_cols) 
{
  if (_rank == 0)
    printf("\nAdding dataset to list ...\n");
  SmartRedis::DataSet dataset(dataset_name);
  dataset.add_tensor(tensor_name,  data, {num_rows,num_cols}, 
                     SRTensorTypeDouble, SRMemLayoutContiguous);
  _client->put_dataset(dataset);
  _client->append_to_list(list_name,dataset);
  if (_rank == 0)
    printf("Done\n");
}

// Checkpoint the solution
void smartredis_client_t::checkpoint()
{
  if (_rank == 0)
    printf("\nWriting checkpoint to DB ...\n");
  MPI_Comm &comm = platform->comm.mpiComm;
  unsigned long int num_dim = _nrs->mesh->dim;
  unsigned long int field_offset = _nrs->fieldOffset;
  std::string irank = "_rank_" + std::to_string(_rank);
  std::string nranks = "_size_" + std::to_string(_size);

  dfloat *U = new dfloat[num_dim * field_offset]();
  dfloat *P = new dfloat[field_offset]();
  _nrs->o_U.copyTo(U, num_dim * field_offset);
  _nrs->o_P.copyTo(U, field_offset);

  _client->put_tensor("checkpt_u"+irank+nranks, U, {field_offset,num_dim},
                  SRTensorTypeDouble, SRMemLayoutContiguous);
  _client->put_tensor("checkpt_p"+irank+nranks, P, {field_offset,1},
                  SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(comm);
}

// Initialize training for the wall shear stress model
void smartredis_client_t::init_wallModel_train(int num_wall_points)
{
  std::vector<int> tensor_info(6,0);

  if (_size <= 64)
    printf("Found %d wall nodes and %d off-wall nodes on rank %d\n",num_wall_points,num_wall_points,_rank);
    fflush(stdout);

  if (_rank == 0) 
    printf("\nSending training metadata for wall shear stress model ...\n");
    fflush(stdout);

  // Create and send metadata for training
  _npts_per_tensor = num_wall_points;
  _num_samples = num_wall_points;
  _num_inputs = 1;
  _num_outputs = 1;
  tensor_info[0] = _npts_per_tensor;
  tensor_info[1] = _num_tot_tensors;
  tensor_info[2] = _num_db_tensors;
  tensor_info[3] = _head_rank;
  tensor_info[4] = _num_inputs;
  tensor_info[5] = _num_outputs;
  std::string info_key = "tensorInfo";
  if (_rank%_num_db_tensors == 0) {
    _client->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);

  if (_rank == 0)
    printf("Done\n");
    fflush(stdout);
}

// Put training data for wall shear stress model in DB
void smartredis_client_t::put_wallModel_data(
        std::vector<dfloat> wall_shear_stress, 
        std::vector<dlong> BdryToV, 
        std::vector<dfloat> Upart,
        int tstep)
{
  unsigned long int num_cols = _num_inputs+_num_outputs;
  std::string key = "x." + std::to_string(_rank) + "." + std::to_string(tstep);
  std::vector<dfloat> train_data(_num_samples*num_cols);
  std::vector<dfloat> vel_data(_num_samples);
  std::vector<dfloat> shear_data(_num_samples);

  // Extract velocity at off-wall nodes (inputs) and shear at wall nodes (outputs)
  dfloat avg_utau = 0.0;
  dfloat avg_ut   = 0.0;
  for (int n = 0; n < _num_samples; ++n) {
    const int v = BdryToV[n];
    vel_data[n] = Upart[0*_num_samples + n];
    avg_ut += vel_data[n];
    shear_data[n] = wall_shear_stress[v+0*_nrs->fieldOffset];
    avg_utau += shear_data[n];
  }
  if (_rank == 0)
    printf("\nTrain :: AVG -- UTAU UT :: %g %g \n",sqrt(abs(avg_utau/_num_samples)),avg_ut/_num_samples);

  // Concatenate inputs and outputs
  for (int i=0; i<_num_samples; i++) {
    train_data[i*num_cols] = vel_data[i];
    train_data[i*num_cols+1] = shear_data[i];
  }

  // Send training data to DB
  if (_rank == 0)
    printf("Sending field with key %s \n",key.c_str());
  _client->put_tensor(key, train_data.data(), {_num_samples,num_cols},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n");
  fflush(stdout);
}

// Run ML model for inference
void smartredis_client_t::run_wallModel(
        std::vector<dfloat> wall_shear_stress, 
        std::vector<dlong> BdryToV, 
        std::vector<dfloat> Upart, 
        int num_wall_points)
{
  _num_samples = num_wall_points;
  std::string in_key = "x." + std::to_string(_rank);
  std::string out_key = "y." + std::to_string(_rank);
  std::vector<dfloat> vel_data(num_wall_points);
  std::vector<dfloat> shear_data(num_wall_points);

  // Extract velocity at off-wall nodes (inputs)
  for (int n = 0; n < _num_samples; ++n) {
    const int v = BdryToV[n];
    vel_data[n] = Upart[0*_num_samples + n];
  }

  if (_rank == 0)
    printf("\nPerforming inference with SmartSim ...\n");

  // Send input data
  _client->put_tensor(in_key, vel_data.data(), {_num_samples,1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);

  // Run ML model on input data
  _client->run_model("model", {in_key}, {out_key});

  // Retrieve model pedictions
  _client->unpack_tensor(out_key, shear_data.data(), {_num_samples},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  
  // Compute the average predicted stress and fill in the shear stress array
  dfloat avg_utau = 0.0;
  for (int n = 0; n < _num_samples; ++n) {
    const int v = BdryToV[n];
    wall_shear_stress[v+0*_nrs->fieldOffset] = shear_data[n];
    avg_utau += shear_data[n];
  }
  if (_rank == 0)
    printf("\nINFERENCE :: AVG -- UTAU :: %g \n",sqrt(abs(avg_utau/_num_samples)));
  MPI_Barrier(platform->comm.mpiComm);
}
#endif
