#if !defined(nekrs_smartredis_hpp_)
#define nekrs_smartredis_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "client.h"
#include <vector>

class smartredis_client_t
{
public:
  smartredis_client_t(nrs_t *nrs);
  ~smartredis_client_t();

  // Client pointer
  SmartRedis::Client *_client;

  // member data
  int _num_db_tensors; // number of tensors being sent to each DB

  // member functions
  void init_check_run();
  int check_run();
  void put_step_num(int tstep);
  void append_dataset_to_list(
    const std::string& dataset_name,
    const std::string& tensor_name,
    const std::string& list_name,
    dfloat* data,
    unsigned long int num_rows,
    unsigned long int num_cols);
  void checkpoint();
  void init_wallModel_train(int num_wall_points);
  void put_wallModel_data(
    std::vector<dfloat> wall_shear_stress, 
    std::vector<dlong> BdryToV, 
    std::vector<dfloat> Upart,
    int tstep);
  void run_wallModel(
    std::vector<dfloat> wall_shear_stress, 
    std::vector<dlong> BdryToV, 
    std::vector<dfloat> Upart,
    int num_wall_points);

private:
  // SmartRedis parameters
  std::string _deployment; // deployment type for DB
  int _npts_per_tensor; // number of points (samples) per tensor being sent to DB
  int _num_tot_tensors; // number of total tensors being sent to all DB
  int _db_nodes; // number of DB nodes (always 1 for co-located DB)
  int _head_rank; // rank ID of the head rank on each node (metadata transfer with co-DB)

  // nekrs objects 
  nrs_t *_nrs;

  // MPI stuff
  int _rank, _size;

  // General model parameters
  int _num_inputs = 1; // number of input features of model
  int _num_outputs = 1; // number of outputs of model
  unsigned long int _num_samples; // number of samples for training (i.e., node and off-node pairs)
 
  // Wall model parameters
  //const dfloat _wall_height= -1.0; // y coordinate of the wall nodes
  //const dfloat _off_wall_height = -0.949428; // y coordinate of the off-wall node (turbChannel)
  //const dfloat _eps = 1.0e-6; // epsilon for finding and matching node coordinate
  //std::vector<int> _ind_wall_nodes; // indices of the wall-nodes
  //std::vector<int> _ind_owall_nodes_matched; // indices of the paired off-wall nodes
};

#endif
