#if !defined(nekrs_smartredis_hpp_)
#define nekrs_smartredis_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <vector>

struct smartredis_data {
  int npts_per_tensor; // number of points (samples) per tensor being sent to DB
  int num_tot_tensors; // number of total tensors being sent to all DB
  int num_db_tensors; // number of tensors being sent to each DB
  int db_nodes; // number of DB nodes (always 1 for co-located DB)
  int head_rank; // rank ID of the head rank on each node (metadata transfer with co-DB)
};

struct wallModel_data {
  const int num_inputs = 1; // number of input features of model
  const int num_outputs = 1; // number of outputs of model
  const dfloat wall_height=-1.0; // y coordinate of the wall nodes
  //const dfloat off_wall_height = -0.982609; // y coordinate of the off-wall node (ktauChannel)
  const dfloat off_wall_height = -0.948726; // y coordinate of the off-wall node (turbChannel)
  const dfloat eps = 1.0e-6; // epsilon for finding and matching node coordinate
  int num_samples; // number of samples for training (i.e., node and off-node pairs)
  std::vector<int> ind_wall_nodes; // indices of the wall-nodes
  std::vector<int> ind_owall_nodes_matched; // indices of the paired off-wall nodes
};

namespace smartredis
{
  void init_client(nrs_t *nrs);
  int check_run();
  void init_check_run();
  void put_step_num(int tstep);
  void init_wallModel_train(nrs_t *nrs);
  void put_wallModel_data(nrs_t *nrs, int tstep);
  void run_wallModel(nrs_t *nrs, int tstep);
  void init_velNpres_train(nrs_t *nrs);
  void put_velNpres_data(nrs_t *nrs, int tstep);
  void run_pressure_model(nrs_t *nrs, int tstep);
}

#endif
