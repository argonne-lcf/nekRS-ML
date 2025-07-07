
#if !defined(nekrs_gnn_hpp_)
#define nekrs_gnn_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#ifdef NEKRS_ENABLE_SMARTREDIS
#include "smartRedis.hpp"
#endif
#include "adiosStreamer.hpp"

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

void writeToFileF(const std::string& filename, dfloat* data, int nRows, int nCols); 
void writeToFileBinaryF(const std::string& filename, dfloat* data, int nRows, int nCols); 

int compareBaseId(const void *a, const void *b);
int compareLocalId(const void *a, const void *b);  

class gnn_t 
{
public:
    gnn_t(nrs_t *nrs);
    ~gnn_t(); 

    std::string writePath;

    // member functions 
    void gnnSetup();
    void gnnWrite();
#ifdef NEKRS_ENABLE_SMARTREDIS
    void gnnWriteDB(smartredis_client_t* client);
#endif
    void gnnWriteADIOS(adios_client_t* client);

private:
    // MPI stuff 
    int rank;
    int size;

    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // Graph attributes
    int gnnMeshPOrder;
    int nekMeshPOrder;
    dlong N;
    hlong num_edges;
    int num_edges_local;
    int num_vertices_local;

    // allocated in constructor 
    dfloat *pos_node; 
    dlong *node_element_ids;
    dlong *local_unique_mask;
    dlong *halo_unique_mask;
    dlong *edge_index;
    dlong *edge_index_local;
    dlong *edge_index_local_vertex;

    // node objects 
    parallelNode_t *localNodes;
    parallelNode_t *haloNodes;
    graphNode_t *graphNodes; 
    graphNode_t *graphNodes_element;

    // member functions 
    void get_graph_nodes();
    void get_graph_nodes_element();
    void add_p1_neighbors();
    void get_global_node_index();
    void get_node_positions();
    void get_node_element_ids();
    void get_node_masks();
    void get_edge_index();
    void get_edge_index_element_local();
    void get_edge_index_element_local_vertex();
    
    void write_edge_index(const std::string& filename);
    void write_edge_index_element_local(const std::string& filename);
    void write_edge_index_element_local_vertex(const std::string& filename);

    // binary write functions 
    void write_edge_index_binary(const std::string& filename);
    void write_edge_index_element_local_vertex_binary(const std::string& filename);

    // for prints 
    bool verbose = false; 

    // model features
    bool multiscale = false;
};

#endif

