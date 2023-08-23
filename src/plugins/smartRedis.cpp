#ifdef ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "SmartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
SmartRedis::Client *client_ptr;

void smartredis::init_client()
{
  // Replace this with variable in .par file
  sr->ranks_per_db = 1;
  sr-> db_nodes = 1;

  // Initialize SR client
  if(platform->comm.mpiRank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode;
  if (sr->db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  //SmartRedis::Client client(cluster_mode, logger_name); // allocates on stack, goes out of scope outside this function
  //client_ptr = &client; // produces dangling reference
  std::string logger_name("Client");
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  if(platform->comm.mpiRank == 0)
    printf("Done\n");
}

// Put velocity data to DB and retrieve it
void smartredis::put_vel_data(nrs_t *nrs, dfloat time, int tstep)
{
  int rank = platform->comm.mpiRank;
  std::string key = "u_" + std::to_string(rank);
  if(rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, nrs->U, {nrs->fieldOffset,3},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  if(rank == 0)
    printf("Done\n\n");

  if(rank == 0)
    printf("Checking array ...\n");
  dfloat *u = new dfloat[nrs->fieldOffset * 3]();
  client_ptr->unpack_tensor(key, u, {nrs->fieldOffset * 3},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  double error = 0.0;
  for (int n=0; n<nrs->fieldOffset*3; n++) {
    error = error + (u[n] - nrs->U[n])*(u[n] - nrs->U[n]);
  }
  if(rank == 0)
    printf("Error in fields = %f\n",error);
    printf("Done\n\n");

}

#endif
