#if !defined(nekrs_gnn_hpp_)
#define nekrs_gnn_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

typedef struct {
    dlong localId; 
    hlong baseId;
    std::vector<dlong> nbrIds;
} graphNode_t;

typedef struct {
    dlong localId;    // local node id
    hlong baseId;     // original global index
    dlong newId;         // new global id
    int owned;        // owner node flag 
} parallelNode_t;

int compareBaseId(const void *a, const void *b);
int compareLocalId(const void *a, const void *b);  

class gnn_t 
{
public:
    gnn_t(nrs_t *nrs);
    ~gnn_t(); 

    // member functions 
    void gnnSetup();
    void gnnWrite();

private:
    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // allocated in constructor 
    dfloat *pos_node; 
    dlong *local_unique_mask;
    dlong *halo_unique_mask;

    // node objects 
    parallelNode_t *localNodes;
    parallelNode_t *haloNodes;
    graphNode_t *graphNodes; 

    // MPI stuff 
    int rank;
    int size;

    // Graph attributes
    dlong num_edges; 

    // member functions 
    void get_graph_nodes();
    void get_global_node_index();
    void get_node_positions();
    void get_node_masks();
    void get_edge_index();
    void write_edge_index(const std::string& filename);

    // for prints 
    bool verbose = true; 
};

#endif
