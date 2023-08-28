#ifdef ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
SmartRedis::Client *client_ptr;

// Initialize the SmartRedis client and the smartredis struct
void smartredis::init_client(nrs_t *nrs)
{
  // Initialize some local variables
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

// Initialize the training
void smartredis::init_train(nrs_t *nrs)
{
  // Initialize some local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  // Create and send tensor metadata
  std::vector<int> tensor_info(6,0);
  tensor_info[0] = sr->npts_per_tensor;
  tensor_info[1] = sr->num_tot_tensors;
  tensor_info[2] = sr->num_db_tensors;
  tensor_info[3] = sr->head_rank;
  tensor_info[4] = 3; // number of model inputs
  tensor_info[5] = 1; // number of model outputs
  std::string info_key = "tensorInfo";
  if (rank%sr->num_db_tensors == 0) {
    printf("\nSending metadata from rank %d ...\n",rank);
    client_ptr->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// Put velocity data in DB
void smartredis::put_velNpres_data(nrs_t *nrs, dfloat time, int tstep)
{
  int rank = platform->comm.mpiRank;
  std::string key = "y." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *train_data = new dfloat[nrs->fieldOffset * 4]();
  int size_U = nrs->fieldOffset * 3;
  int size_P = nrs->fieldOffset;
  std::copy(nrs->U,nrs->U+size_U,train_data);
  std::copy(nrs->P,nrs->P+size_P,train_data);
  if (rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {nrs->fieldOffset,4},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  /*
  dfloat *u = new dfloat[nrs->fieldOffset * 3]();
  client_ptr->unpack_tensor(key, u, {nrs->fieldOffset * 3},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  double error = 0.0;
  for (int n=0; n<nrs->fieldOffset*3; n++) {
    error = error + (u[n] - nrs->U[n])*(u[n] - nrs->U[n]);
  }
  printf("[%d]: Error in fields = %f\n",rank,error);
  fflush(stdout);
  */
}

// Put step number in DB
void smartredis::put_step_num(int tstep)
{
  int rank = platform->comm.mpiRank;
  std::string key = "step";
  std::vector<double> step_num(1,0);
  step_num[0] = tstep;
  if (rank == 0)
    printf("\nSending time step number ...\n");
  client_ptr->put_tensor(key, step_num.data(), {1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

#endif
