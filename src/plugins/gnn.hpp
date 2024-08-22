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

template <typename T> 
void writeToFile(const std::string& filename, T* data, int nRows, int nCols); 

template <typename T> 
void writeToFileBinary(const std::string& filename, T* data, int nRows, int nCols); 

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

    // where gnn output files are written, if "write=True". 
    std::string writePath;

private:
    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // allocated in constructor 
    dfloat *pos_node; 
    dlong *node_element_ids;
    dlong *local_unique_mask;
    dlong *halo_unique_mask;

    // node objects 
    parallelNode_t *localNodes;
    parallelNode_t *haloNodes;
    graphNode_t *graphNodes; 
    graphNode_t *graphNodes_element;

    // MPI stuff 
    int rank;
    int size;

    // Graph attributes
    hlong num_edges; 

    // member functions 
    void get_graph_nodes();
    void get_graph_nodes_element();
    void get_global_node_index();
    void get_node_positions();
    void get_node_element_ids();
    void get_node_masks();
    void get_edge_index();
    void write_edge_index(const std::string& filename);
    void write_edge_index_element_local(const std::string& filename);
    void write_edge_index_element_local_vertex(const std::string& filename);

    // binary write functions 
    void write_edge_index_binary(const std::string& filename);
    void write_edge_index_element_local_vertex_binary(const std::string& filename);

    // for prints 
    bool verbose = true; 

    // for writing 
    bool write = true;
};

#endif
