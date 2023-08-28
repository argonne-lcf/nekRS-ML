#if !defined(nekrs_smartredis_hpp_)
#define nekrs_smartredis_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

struct smartredis_data {
  int npts_per_tensor; // number of points (samples) per tensor being sent to DB
  int num_tot_tensors; // number of total tensors being sent to all DB
  int num_db_tensors; // number of tensors being sent to each DB
  int db_nodes; // number of DB nodes (always 1 for co-located DB)
  int head_rank; // rank ID of the head rank on each node (metadata transfer with co-DB)
};

namespace smartredis
{
  void init_client(nrs_t *nrs);
  void init_train(nrs_t *nrs);
  void put_velNpres_data(nrs_t *nrs, dfloat time, int tstep);
  void put_step_num(int tstep);
}

#endif