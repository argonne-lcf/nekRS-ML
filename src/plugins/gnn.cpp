
#include "nrs.hpp"
#include "mesh.h"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "meshNekReader.hpp"
#include "ogsInterface.h"
#include "gnn.hpp"
#include <cstdlib>
#include <filesystem>

template <typename T>
void writeToFile(const std::string& filename, T* data, int nRows, int nCols)
{
    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }

    // Write to file:
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            file_cpu << data[j * nRows + i] << '\t';
        }
        file_cpu << '\n';
    }
}

void writeToFileF(const std::string& filename, dfloat* data, int nRows, int nCols)
{
    writeToFile(filename, data, nRows, nCols);
}

template <typename T>
void writeToFileBinary(const std::string& filename, T* data, int nRows, int nCols)
{
    std::cout << "Writing file (binary): " << filename << std::endl;
    std::ofstream file_cpu(filename, std::ios::binary);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }

    // Write to file:
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            int index = j * nRows + i;
            file_cpu.write(reinterpret_cast<const char*>(&data[index]), sizeof(T));
        }
    }
}

void writeToFileBinaryF(const std::string& filename, dfloat* data, int nRows, int nCols)
{
    writeToFileBinary(filename, data, nRows, nCols);
}

gnn_t::gnn_t(nrs_t *nrs_, int poly_order, bool log_verbose)
{
    // set MPI rank and size 
    MPI_Comm &comm = platform->comm.mpiComm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    verbose = log_verbose;

    // parse poly_order value
    if (poly_order <= 0) {
        platform->options.getArgs("GNN POLY ORDER", gnnMeshPOrder);
    } else {
        gnnMeshPOrder = poly_order;
    }
    nekMeshPOrder = nrs_->mesh->N;

    // create GNN mesh if needed
    if (gnnMeshPOrder == nekMeshPOrder){
        mesh = nrs_->mesh;
    } else if (gnnMeshPOrder < nekMeshPOrder) {
        if (rank == 0) std::cout << "Generating GNN mesh with polynomial degree ..." << gnnMeshPOrder << std::endl;
        mesh = new mesh_t();
        mesh->Nelements = nrs_->mesh->Nelements;
        mesh->dim = nrs_->mesh->dim;
        mesh->Nverts = nrs_->mesh->Nverts;
        mesh->Nfaces = nrs_->mesh->Nfaces;
        mesh->NfaceVertices = nrs_->mesh->NfaceVertices;
        meshLoadReferenceNodesHex3D(mesh, gnnMeshPOrder, 0);
        mesh->o_x = platform->device.malloc<dfloat>(mesh->Nlocal);
        mesh->o_y = platform->device.malloc<dfloat>(mesh->Nlocal);
        mesh->o_z = platform->device.malloc<dfloat>(mesh->Nlocal);
        nrs_->mesh->interpolate(nrs_->mesh->o_x, mesh, mesh->o_x);
        nrs_->mesh->interpolate(nrs_->mesh->o_y, mesh, mesh->o_y);
        nrs_->mesh->interpolate(nrs_->mesh->o_z, mesh, mesh->o_z);
        meshGlobalIds(mesh);
        meshParallelGatherScatterSetup(mesh,
                                 mesh->Nelements * mesh->Np,
                                 mesh->globalIds,
                                 comm,
                                 OOGS_AUTO,
                                 0);
    } else {
        if (rank == 0) std::cout << "\nError: GNN polynimial degree must be <= nekRS degree\n" << std::endl;
        MPI_Abort(comm, 1);
    }
    ogs = mesh->ogs;
    fieldOffset = mesh->Np * (mesh->Nelements);
    fieldOffset = alignStride<dfloat>(fieldOffset);

    // allocate memory 
    N = mesh->Nelements * mesh->Np; // total number of nodes
    pos_node = new dfloat[N * mesh->dim](); 
    node_element_ids = new dlong[N]();
    local_unique_mask = new dlong[N](); 
    halo_unique_mask = new dlong[N]();

    graphNodes = (graphNode_t*) calloc(N, sizeof(graphNode_t)); // full domain
    graphNodes_element = (graphNode_t*) calloc(mesh->Np, sizeof(graphNode_t)); // a single element

    if (verbose) {
        printf("\n[RANK %d] -- Finished instantiating gnn_t object\n", rank);
        printf("[RANK %d] -- The polynomial degree of the GNN mesh is %d \n", rank, gnnMeshPOrder);
        printf("[RANK %d] -- The number of elements of the GNN mesh is %d \n", rank, mesh->Nelements);
        printf("[RANK %d] -- The number of nodes of the GNN mesh is %d \n", rank, N);
        fflush(stdout);
    }
}

gnn_t::~gnn_t()
{
    if (verbose) std::cout << "[RANK " << rank << "] -- gnn_t destructor\n" << std::endl;
    if (gnnMeshPOrder < nekMeshPOrder) delete mesh;
    delete[] pos_node;
    delete[] node_element_ids;
    delete[] local_unique_mask;
    delete[] halo_unique_mask;
    delete[] edge_index;
    delete[] edge_index_local;
    delete[] edge_index_local_vertex;
    free(localNodes);
    free(haloNodes);
    free(graphNodes);
    free(graphNodes_element);
}

void gnn_t::gnnSetup()
{
    if (verbose) std::cout << "[RANK " << rank << "] -- in gnnSetup()" << std::endl;

    // set multiscale flag
    int poly_order = mesh->Nq - 1;
    if (platform->options.compareArgs("SR GNN MULTISCALE", "TRUE") && poly_order > 1) {
        multiscale = true;
    }
    if (verbose) std::cout << "[RANK " << rank << "] -- using multiscale flag: " << multiscale << std::endl;

    get_graph_nodes(); // populates graphNodes
    if (multiscale) add_p1_neighbors(); // adds additional edges on mesh nodes (p=1)  
    get_graph_nodes_element(); // populates graphNodes_element
    get_node_positions();
    get_node_element_ids(); 
    get_node_masks();
    get_edge_index();
    get_edge_index_element_local();
    get_edge_index_element_local_vertex();
}

void gnn_t::gnnWrite()
{
    if (verbose) printf("[RANK %d] -- in gnnWrite() \n", rank);
    MPI_Comm &comm = platform->comm.mpiComm;

    // output directory
    std::filesystem::path currentPath = std::filesystem::current_path();
    currentPath /= "gnn_outputs";
    writePath = currentPath.string();
    int poly_order = mesh->Nq - 1; 
    writePath = writePath + "_poly_" + std::to_string(poly_order);
    if (multiscale) writePath = writePath + "_multiscale";
    if (rank == 0)
    {
        if (!std::filesystem::exists(writePath))
        {
            std::filesystem::create_directory(writePath);
        }
    }
    MPI_Barrier(comm);
     
    std::string irank = "_rank_" + std::to_string(rank);
    std::string nranks = "_size_" + std::to_string(size);

    // Writing as binary files: 
    //write_edge_index_binary(writePath + "/edge_index" + irank + nranks + ".bin");
    writeToFileBinary(writePath + "/pos_node" + irank + nranks + ".bin", pos_node, N, 3);
    writeToFileBinary(writePath + "/local_unique_mask" + irank + nranks + ".bin", local_unique_mask, N, 1); 
    writeToFileBinary(writePath + "/halo_unique_mask" + irank + nranks + ".bin", halo_unique_mask, N, 1); 
    writeToFileBinary(writePath + "/global_ids" + irank + nranks + ".bin", mesh->globalIds, N, 1);
    writeToFileBinary(writePath + "/edge_index" + irank + nranks + ".bin", edge_index, num_edges, 2);
    writeToFileBinary(writePath + "/node_element_ids" + irank + nranks + ".bin", node_element_ids, N, 1);

    // Writing number of elements, gll points per element, and product of the two  
    if (rank == 0) writeToFile(writePath + "/Np" + irank + nranks, &mesh->Np, 1, 1);

    // Writing element-local edge index as text file (small)
    if (rank == 0) writeToFile(writePath + "/edge_index_element_local", edge_index_local, num_edges_local, 2);
    if (rank == 0) writeToFile(writePath + "/edge_index_element_local_vertex", edge_index_local_vertex, num_vertices_local, 2);
}

#ifdef NEKRS_ENABLE_SMARTREDIS
void gnn_t::gnnWriteDB(smartredis_client_t* client)
{
    if (verbose) printf("[RANK %d] -- in gnnWriteDB() \n", rank);
    MPI_Comm &comm = platform->comm.mpiComm;
    unsigned long int num_nodes = N;
    unsigned long int num_edg = num_edges;
    unsigned long int num_edg_l = num_edges_local;
    unsigned long int num_vert_l = num_vertices_local;
    std::string irank = "_rank_" + std::to_string(rank);
    std::string nranks = "_size_" + std::to_string(size);

    // Writing the graph data
    client->_client->put_tensor("pos_node" + irank + nranks, pos_node, {3,num_nodes},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
    client->_client->put_tensor("local_unique_mask" + irank + nranks, local_unique_mask, {num_nodes},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
    client->_client->put_tensor("halo_unique_mask" + irank + nranks, halo_unique_mask, {num_nodes},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
    client->_client->put_tensor("global_ids" + irank + nranks, mesh->globalIds, {num_nodes,1},
                    SRTensorTypeInt64, SRMemLayoutContiguous);
    
    // Writing edge information
    client->_client->put_tensor("edge_index" + irank + nranks, edge_index, {2,num_edg},
                    SRTensorTypeInt32, SRMemLayoutContiguous);

    // Writing some graph statistics
    client->_client->put_tensor("Np" + irank + nranks, &mesh->Np, {1},
                    SRTensorTypeInt32, SRMemLayoutContiguous);

    MPI_Barrier(comm);
    if (verbose) printf("[RANK %d] -- done sending graph data to DB \n", rank);
}
#endif // NEKRS_ENABLE_SMARTREDIS

void gnn_t::gnnWriteADIOS(adios_client_t* client)
{
    MPI_Comm &comm = platform->comm.mpiComm;
#if defined(NEKRS_ENABLE_ADIOS)
    if (verbose && rank == 0) printf("[RANK %d] -- in gnnWriteADIOS() \n", rank);
    unsigned long _size = size;
    unsigned long _rank = rank;
    client->_num_dim = mesh->dim;

    // Get global size of data
    int global_N, global_num_edges;
    MPI_Allreduce(&N, &global_N, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&num_edges, &global_num_edges, 1, MPI_INT, MPI_SUM, comm);
    client->_N = N;
    client->_global_N = global_N;
    client->_num_edges = num_edges;
    client->_global_num_edges = global_num_edges;

    // Gather size of data
    int* gathered_N = new int[size];
    int* gathered_num_edges = new int[size];
    int offset_N = 0;
    int offset_num_edges = 0;
    MPI_Allgather(&N, 1, MPI_INT, gathered_N, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&num_edges, 1, MPI_INT, gathered_num_edges, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i=0; i<rank; i++) {
        offset_N += gathered_N[i];
        offset_num_edges += gathered_num_edges[i];
    }
    client->_offset_N = offset_N;
    client->_offset_num_edges = offset_num_edges;

    // Define ADIOS2 variables to send
    auto posFloats = client->_write_io.DefineVariable<dfloat>("pos_node", 
                                                            {client->_global_N * client->_num_dim}, 
                                                            {client->_offset_N * client->_num_dim}, 
                                                            {client->_N * client->_num_dim});
    auto locInts = client->_write_io.DefineVariable<dlong>("local_unique_mask", 
                                                            {client->_global_N}, 
                                                            {client->_offset_N}, 
                                                            {client->_N});
    auto haloInts = client->_write_io.DefineVariable<dlong>("halo_unique_mask", 
                                                            {client->_global_N}, 
                                                            {client->_offset_N}, 
                                                            {client->_N});
    auto globInts = client->_write_io.DefineVariable<hlong>("global_ids", 
                                                            {client->_global_N}, 
                                                            {client->_offset_N}, 
                                                            {client->_N});
    auto edgeInts = client->_write_io.DefineVariable<dlong>("edge_index", 
                                                            {client->_global_num_edges * 2}, 
                                                            {client->_offset_num_edges * 2}, 
                                                            {client->_num_edges * 2});
    auto NpInts = client->_write_io.DefineVariable<dlong>("Np", {1}, {1}, {1});
    auto NInts = client->_write_io.DefineVariable<dlong>("N", {_size}, {_rank}, {1});
    auto numedgesInts = client->_write_io.DefineVariable<dlong>("num_edges", {_size}, {_rank}, {1});

    // Write the graph data
    //adios2::Engine graphWriter = client->_stream_io.Open("graphStream", adios2::Mode::Write);
    adios2::Engine graphWriter = client->_write_io.Open("graph.bp", adios2::Mode::Write);
    graphWriter.BeginStep();

    graphWriter.Put<dlong>(NInts, N);
    graphWriter.Put<dlong>(numedgesInts, static_cast<dlong>(num_edges));
    graphWriter.Put<dfloat>(posFloats, pos_node);
    graphWriter.Put<dlong>(locInts, local_unique_mask);
    graphWriter.Put<dlong>(haloInts, halo_unique_mask);
    graphWriter.Put<hlong>(globInts, mesh->globalIds);
    graphWriter.Put<dlong>(edgeInts, edge_index);
    if (rank == 0) {
        graphWriter.Put<dlong>(NpInts, &mesh->Np);
    }

    graphWriter.EndStep();
    graphWriter.Close();
    MPI_Barrier(comm);
    if (verbose and rank == 0) printf("[RANK %d] -- done sending graph data \n", rank);
#else
    if (verbose and rank == 0) printf("[RANK %d] -- ADIOS is not enabled!, Falling back to binary write.\n", rank);
    fflush(stdout);
    gnnWrite();
#endif
}

void gnn_t::get_node_positions()
{
    if (verbose) printf("[RANK %d] -- in get_node_positions() \n", rank);
    auto [x, y, z] = mesh->xyzHost();
    for (int n=0; n < N; n++)
    {
        pos_node[n + 0*N] = x[n];
        pos_node[n + 1*N] = y[n];
        pos_node[n + 2*N] = z[n];
    }
}

void gnn_t::get_node_element_ids()
{
    if (verbose) printf("[RANK %d] -- in get_node_element_ids() \n", rank);
    for (int e = 0; e < mesh->Nelements; e++) // loop through the element 
    {
        for (int i = 0; i < mesh->Np; i++) // loop through the gll nodes 
        {
            node_element_ids[e * mesh->Np + i] = e;
        } 
    }
}

void gnn_t::get_node_masks()
{
	if (verbose) printf("[RANK %d] -- in get_node_masks() \n", rank);

    hlong *ids =  mesh->globalIds; // global node ids 
    MPI_Comm &comm = platform->comm.mpiComm; // mpi comm 
    occa::device device = platform->device.occaDevice(); // occa device 

    //use the host gs to find what nodes are local to this rank
    int *minRank = (int *) calloc(N,sizeof(int));
    int *maxRank = (int *) calloc(N,sizeof(int));
    hlong *flagIds   = (hlong *) calloc(N,sizeof(hlong));

    // Pre-fill the min and max ranks
    for (dlong i=0; i<N; i++)
    {   
        minRank[i] = rank;
        maxRank[i] = rank;
        flagIds[i] = ids[i];
    }   

    ogsHostGatherScatter(minRank, ogsInt, ogsMin, ogs->hostGsh); //minRank[n] contains the smallest rank taking part in the gather of node n
    ogsHostGatherScatter(maxRank, ogsInt, ogsMax, ogs->hostGsh); //maxRank[n] contains the largest rank taking part in the gather of node n
    ogsGsUnique(flagIds, N, comm); //one unique node in each group is 'flagged' kept positive while others are turned negative.

    //count local and halo nodes
    int Nlocal = 0; //  number of local nodes
    int Nhalo = 0; // number of halo nodes
    int NownedHalo = 0; //  number of owned halo nodes
    int NlocalGather = 0; // number of local gathered nodes
    int NhaloGather = 0; // number of halo gathered nodes
    for (dlong i=0;i<N;i++)
    {
        if (ids[i]==0) continue;
        if ((minRank[i]!=rank)||(maxRank[i]!=rank))
        {
            Nhalo++;
            if (flagIds[i]>0)
            {
                NownedHalo++;
            }
        }
        else
        {
            Nlocal++;
        }
    }

    // SB: test 
    if (verbose) {
        printf("[RANK %d] -- \tN: %d \n", rank, N);
        printf("[RANK %d] -- \tNlocal: %d \n", rank, Nlocal);
        printf("[RANK %d] -- \tNhalo: %d \n", rank, Nhalo);
        fflush(stdout);
    }

    // ~~~~ For parsing the local coincident nodes
    if (Nlocal)
    {
        localNodes = (parallelNode_t*) calloc(Nlocal,sizeof(parallelNode_t));
        dlong cnt=0;
        for (dlong i=0;i<N;i++) // loop through all nodes
        {
            if (ids[i]==0) continue; // skip internal (unique) nodes

            if ((minRank[i]==rank)&&(maxRank[i]==rank))
            {
                localNodes[cnt].localId = i; // the local node id
                localNodes[cnt].baseId  = ids[i]; // the global node id
                localNodes[cnt].owned   = 0; // flag
                cnt++;
            }
        }

        // sort based on base ids then local id
        qsort(localNodes, Nlocal, sizeof(parallelNode_t), compareBaseId);

        // get the number of nodes to be gathered
        int freq = 0;
        NlocalGather = 0;
        localNodes[0].newId = 0; // newId is a "new" global node ID, starting at 0 from node 0
        localNodes[0].owned = 1; // a flag specifying that this is the owner node
        for (dlong i=1; i < Nlocal; i++)
        {
            int s = 0;
            // if the global node id of current node is not equal to previous, then assign as new owner
            if (localNodes[i].baseId != localNodes[i-1].baseId)
            {   
                NlocalGather++;
                s = 1;
            }
            localNodes[i].newId = NlocalGather; // interpret as cluster ids
            localNodes[i].owned = s;
        }
        NlocalGather++;
 

        // // SB: testing things out 
        // for (dlong i=0; i < Nlocal; i++)
        // {
        //     if (verbose) printf("[RANK %d] --- Local ID: %d \t Global ID: %d \t New ID: %d \n", 
        //             rank, localNodes[i].localId, localNodes[i].baseId, localNodes[i].newId);
        // }
        if (verbose) std::cout << "[RANK  " << rank << "] -- NlocalGather: " << NlocalGather << std::endl;
   
        // ~~~~ get the mask to move from coincident to non-coincident representation.
        // first, sort based on local ids
        qsort(localNodes, Nlocal, sizeof(parallelNode_t), compareLocalId); 

        // local_unique_mask: [N x 1] array
        // -- 1 if this is a local node we keep
        // -- 0 if this is a local node we discard

        // Loop through all local nodes
        for (dlong i=0;i<N;i++) // loop through all nodes
        {
            if (ids[i]==0) // this indicates an internal node
            {
                local_unique_mask[i] = 1;
            }
            else
            {
                local_unique_mask[i] = 0;
            }
        }

        // Loop through local coincident nodes
        // -- THIS DOES NOT INCLUDE HALO NODES
        for (dlong i=0; i < Nlocal; i++)
        {
            dlong local_id = localNodes[i].localId; // the local node id
            if (localNodes[i].owned == 1)
            {
                local_unique_mask[local_id] = 1;
            }
        }

        // ~~~~ Add additional graph node neighbors for coincident LOCAL nodes 
        // Store the coincident node Ids  
        std::vector<dlong> coincidentOwn[NlocalGather]; // each element is a vector of localIds belonging to the same globalID 
        std::vector<std::vector<dlong>> coincidentNei[NlocalGather]; // each element is a vector of vectors, containing the neighbor IDs of the corresponding localIDs. 
        for (dlong i = 0; i < Nlocal; i++)
        {
            // get the newID:
            dlong nid = localNodes[i].newId;

            // get the localID
            dlong lid = localNodes[i].localId;

            // get the graph node 
            graphNode_t node = graphNodes[lid];
            
            // place owner id in list  
            coincidentOwn[nid].push_back(node.localId); // each element contains an integer

            // place neighbor vector in list 
            coincidentNei[nid].push_back(node.nbrIds); // each element contains a vector 
        }

        // populate hash-table for global-to-local ID lookups 
        std::unordered_map<dlong, std::set<dlong>> globalToLocalMap;
        for (dlong i = 0; i < Nlocal; ++i)
        {
            dlong lid = localNodes[i].localId;
            hlong gid = localNodes[i].baseId; 
            globalToLocalMap[gid].insert(lid);
        }

        // SB: new neighbor modification 
        cnt = num_edges; 
        for (dlong i = 0; i < Nlocal; i++)
        {
            dlong lid = localNodes[i].localId; // localId of node  
            dlong nid = localNodes[i].newId; // newID of node  
            hlong gid = localNodes[i].baseId; // globalID of node 

            graphNode_t node_i = graphNodes[lid]; // graph node 
            
            // printf("node_%d -- localId = %d \t newId = %d \n", i, lid, nid);
            std::vector<dlong> same_ids = coincidentOwn[nid]; 

            for (dlong j = 0; j < same_ids.size(); j++)
            {
                graphNode_t node_j = graphNodes[same_ids[j]]; // graph node that has same local id  
                
                // printf("\t node_%d -- localId = %d \t same_ids[j] = %d \t newId = %d \n", j, node_j.localId, same_ids[j], nid);

                if (node_j.localId != node_i.localId) // if they are different nodes 
                { 
                    for (dlong k = 0; k < node_j.nbrIds.size(); k++) // loop through node j nei
                    {
                        if (std::find(  graphNodes[lid].nbrIds.begin(), 
                                        graphNodes[lid].nbrIds.end(), 
                                        node_j.nbrIds[k]  ) != graphNodes[lid].nbrIds.end() ) 
                        {
                            // node_j.nbrIds[k] is present in nbrIds, so skip 
                            continue; 
                        } 
                        else // node_j.nbrIds[k] is not present in nbrIds, so add 
                        {
                            if (node_i.localId != node_j.nbrIds[k]) // no self-loops 
                            { 
                                graphNodes[lid].nbrIds.push_back( node_j.nbrIds[k] );
                                num_edges++; 
                            }
                        }
                    }
                }
            }

            // Append neighbor list with all other nodes sharing same global Id
            for (dlong j = 0; j < graphNodes[lid].nbrIds.size(); j++)
            {
                dlong added_id_local = graphNodes[lid].nbrIds[j]; // local id of nei 
                hlong added_id_global = graphNodes[added_id_local].baseId; // global id of nei 
                for (dlong additional_id : globalToLocalMap[added_id_global]) // for all local ids with the same global id as "added_id_global" 
                {
                    if (std::find(  graphNodes[lid].nbrIds.begin(), 
                                    graphNodes[lid].nbrIds.end(), 
                                    additional_id  ) != graphNodes[lid].nbrIds.end() ) 
                    {
                        // additional_id is present in nbrIds, so skip 
                        continue; 
                    } 
                    else // additional_id is not present in nbrIds, so add 
                    {
                        if (graphNodes[lid].localId != additional_id) // no self-loops 
                        { 
                            graphNodes[lid].nbrIds.push_back( additional_id );
                            num_edges++; 
                        }
                    }
                }
            }

        }
        num_edges = cnt; 
    }
    else // dummy 
    {
        localNodes = (parallelNode_t*) calloc(1,sizeof(parallelNode_t));
    }

    // ~~~~ For parsing the halo coincident nodes 
    {
        haloNodes = (parallelNode_t*) calloc(Nhalo+1,sizeof(parallelNode_t));
        dlong cnt=0;
        for (dlong i=0;i<N;i++) // loop through all GLL points
        {   
            if (ids[i]==0) continue; // skip unique points
            if ((minRank[i]!=rank)||(maxRank[i]!=rank)) // add if the coinc. node is on another rank
            {   
                haloNodes[cnt].localId = i; // local node ID of the halo node
                haloNodes[cnt].baseId  = flagIds[i]; // global node ID of the halo node
                haloNodes[cnt].owned   = 0; // is this the owner node
                cnt++;
            }   
        }   
            
        if(Nhalo)
        {
            qsort(haloNodes, Nhalo, sizeof(parallelNode_t), compareBaseId);

            //move the flagged node to the lowest local index if present
            cnt = 0;
            NhaloGather=0;
            haloNodes[0].newId = 0;
            haloNodes[0].owned = 1;

            for (dlong i=1;i<Nhalo;i++)
            {
                int s = 0;
                if (abs(haloNodes[i].baseId)!=abs(haloNodes[i-1].baseId))
                { //new gather node
                    s = 1;
                    cnt = i;
                    NhaloGather++;
                }
                haloNodes[i].owned = s;
                haloNodes[i].newId = NhaloGather;
                if (haloNodes[i].baseId>0)
                {
                    haloNodes[i].baseId   = -abs(haloNodes[i].baseId);
                    haloNodes[cnt].baseId =  abs(haloNodes[cnt].baseId);
                }
            }
            NhaloGather++;

            // sort based on local ids
            qsort(haloNodes, Nhalo, sizeof(parallelNode_t), compareLocalId);

            // ~~~~ Gets the mask 
            for (dlong i = 0; i < Nhalo; i++)
            {
                // Fill halo nodes 
                // halo_unique_mask: [N x 1] integer array  
                // -- 1 if this is a halo (nonlocal) node we keep 
                // -- 0 if this is a halo (nonlocal) node we discard 
                dlong local_id = haloNodes[i].localId;
                if (haloNodes[i].owned == 1) // if this is the owner node
                {
                    halo_unique_mask[local_id] = 1;
                }
            }



            // SB -- new neighbor modifications here 
            
            // ~~~~ Add additional graph node neighbors for coincident HALO nodes
            // Store the coincident node Ids  
            std::vector<dlong> coincidentOwnHalo[NhaloGather];
            std::vector<std::vector<dlong>> coincidentNeiHalo[NhaloGather];
            for (dlong i = 0; i < Nhalo; i++)
            {
                // get the newID:
                dlong nid = haloNodes[i].newId;

                // get the localID
                dlong lid = haloNodes[i].localId;

                // get the graph node 
                graphNode_t node = graphNodes[lid];
                
                // place owner id in list  
                coincidentOwnHalo[nid].push_back(node.localId); // each element contains an integer

                // place neighbor vector in list 
                coincidentNeiHalo[nid].push_back(node.nbrIds); // each element contains a vector 
            }
            
            // populate hash-table for global-to-local ID lookups 
            std::unordered_map<dlong, std::set<dlong>> globalToLocalMapHalo;
            for (dlong i = 0; i < Nhalo; ++i)
            {
                dlong lid = haloNodes[i].localId;
                hlong gid = abs(haloNodes[i].baseId); 

                // get the globalId from graphNode (baseIds have some negative signs in "haloNodes") 
                // hlong gid = graphNodes[lid].baseId;

                globalToLocalMapHalo[gid].insert(lid);
            }

            // SB: new neighbor modification 
            cnt = num_edges; 
            for (dlong i = 0; i < Nhalo; i++)
            {
                dlong lid = haloNodes[i].localId; // localId of node  
                dlong nid = haloNodes[i].newId; // newID of node  
                hlong gid = abs(haloNodes[i].baseId); // globalID of node 

                graphNode_t node_i = graphNodes[lid]; // graph node 
                
                // printf("node_%d -- localId = %d \t newId = %d \n", i, lid, nid);
                std::vector<dlong> same_ids = coincidentOwnHalo[nid]; 

                for (dlong j = 0; j < same_ids.size(); j++)
                {
                    graphNode_t node_j = graphNodes[same_ids[j]]; // graph node that has same local id  
                    
                    // if (rank == 0) printf("\t node_%d -- localId = %d \t same_ids[j] = %d \t newId = %d \n", j, node_j.localId, same_ids[j], nid);

                    if (node_j.localId != node_i.localId) // if they are different nodes 
                    { 
                        for (dlong k = 0; k < node_j.nbrIds.size(); k++) // loop through node j nei
                        {
                            if (std::find(  graphNodes[lid].nbrIds.begin(), 
                                            graphNodes[lid].nbrIds.end(), 
                                            node_j.nbrIds[k]  ) != graphNodes[lid].nbrIds.end() ) 
                            {
                                // node_j.nbrIds[k] is present in nbrIds, so skip 
                                continue; 
                            } 
                            else // node_j.nbrIds[k] is not present in nbrIds, so add 
                            {
                                if (node_i.localId != node_j.nbrIds[k]) // no self-loops 
                                { 
                                    graphNodes[lid].nbrIds.push_back( node_j.nbrIds[k] );
                                    num_edges++; 
                                }
                            }
                        }
                    }
                }

                // Append neighbor list with all other nodes sharing same global Id
                for (dlong j = 0; j < graphNodes[lid].nbrIds.size(); j++)
                {
                    dlong added_id_local = graphNodes[lid].nbrIds[j]; // local id of nei 
                    hlong added_id_global = graphNodes[added_id_local].baseId; // global id of nei 
                    for (dlong additional_id : globalToLocalMapHalo[added_id_global]) // for all local ids with the same global id as "added_id_global" 
                    {
                        if (std::find(  graphNodes[lid].nbrIds.begin(), 
                                        graphNodes[lid].nbrIds.end(), 
                                        additional_id  ) != graphNodes[lid].nbrIds.end() ) 
                        {
                            // additional_id is present in nbrIds, so skip 
                            continue; 
                        } 
                        else // additional_id is not present in nbrIds, so add 
                        {
                            if (graphNodes[lid].localId != additional_id) // no self-loops 
                            { 
                                graphNodes[lid].nbrIds.push_back( additional_id );
                                num_edges++; 
                            }
                        }
                    }
                }
            }
            num_edges = cnt; 
        }
    }
    free(minRank); free(maxRank); free(flagIds);
}

void gnn_t::get_edge_index()
{
    if (verbose) printf("[RANK %d] -- in get_edge_index() \n", rank);

    // loop through graph nodes
    num_edges = 0;
    for (dlong i = 0; i < N; i++)
    {               
        dlong num_nbr = graphNodes[i].nbrIds.size();
        num_edges += num_nbr;
    }

    edge_index = new dlong[num_edges * 2]();

    hlong c = 0;
    for (dlong i = 0; i < N; i++)
    {               
        int num_nbr = graphNodes[i].nbrIds.size();
        dlong idx_own = graphNodes[i].localId; 
                    
        for (int j = 0; j < num_nbr; j++)
        {           
            dlong idx_nei = graphNodes[i].nbrIds[j];
            edge_index[c+0*num_edges] = idx_nei;
            edge_index[c+1*num_edges] = idx_own;
            c++;
        }
    }
}

void gnn_t::get_edge_index_element_local()
{
    if (verbose) printf("[RANK %d] -- in get_edge_index_element_local() \n", rank);

    num_edges_local = 0;
    for (int i = 0; i < mesh->Np; i++)
    {               
        int num_nbr = graphNodes_element[i].nbrIds.size();
        num_edges_local += num_nbr;
    }         
    
    edge_index_local = new dlong[num_edges_local * 2]();
    dlong c = 0;
    for (int i = 0; i < mesh->Np; i++)
    {               
        int num_nbr = graphNodes_element[i].nbrIds.size();
        dlong idx_own = graphNodes_element[i].localId; 
        for (int j = 0; j < num_nbr; j++)
        {           
            dlong idx_nei = graphNodes_element[i].nbrIds[j];  
            edge_index_local[c+0*num_edges_local] = idx_nei;
            edge_index_local[c+1*num_edges_local] = idx_own;
            c++;
        }
    }         
}

void gnn_t::get_edge_index_element_local_vertex()
{
    if (verbose) printf("[RANK %d] -- in get_edge_index_element_local_vertex() \n", rank);

    // loop through vertex node indices
    int n_vertex_nodes = 8;
    num_vertices_local = n_vertex_nodes * n_vertex_nodes;
    edge_index_local_vertex = new dlong[num_vertices_local * 2]();
    int c = 0;
    for (int i = 0; i < n_vertex_nodes; i++)
    {
        dlong idx_own = mesh->vertexNodes[i];  
        for (int j = 0; j < n_vertex_nodes; j++)
        {
            dlong idx_nei = mesh->vertexNodes[j];
            edge_index_local_vertex[c+0*num_vertices_local] = idx_nei;
            edge_index_local_vertex[c+1*num_vertices_local] = idx_own;
            c++;
        }
    }
}


void gnn_t::write_edge_index(const std::string& filename)
{
    if (verbose) printf("[RANK %d] -- in write_edge_index() \n", rank);

    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }
                    
    // loop through graph nodes
    for (int i = 0; i < N; i++)
    {               
        int num_nbr = graphNodes[i].nbrIds.size();
        dlong idx_own = graphNodes[i].localId; 
                    
        for (int j = 0; j < num_nbr; j++)
        {           
            dlong idx_nei = graphNodes[i].nbrIds[j];  
            file_cpu << idx_nei << '\t' << idx_own << '\n'; 
        }
    }           
}

void gnn_t::write_edge_index_binary(const std::string& filename)
{
    if (verbose) printf("[RANK %d] -- in write_edge_index() \n", rank);

    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename, std::ios::binary);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }
                    
    // loop through graph nodes
    for (int i = 0; i < N; i++)
    {               
        int num_nbr = graphNodes[i].nbrIds.size();
        dlong idx_own = graphNodes[i].localId; 
                    
        for (int j = 0; j < num_nbr; j++)
        {           
            dlong idx_nei = graphNodes[i].nbrIds[j];  
            // file_cpu << idx_nei << '\t' << idx_own << '\n'; 
            file_cpu.write(reinterpret_cast<const char*>(&idx_nei), sizeof(dlong));
            file_cpu.write(reinterpret_cast<const char*>(&idx_own), sizeof(dlong));
        }
    }           
}

void gnn_t::write_edge_index_element_local(const std::string& filename)
{
    if (verbose) printf("[RANK %d] -- in write_edge_index_element_local() \n", rank);

    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }

    // loop through graph nodes
    for (int i = 0; i < mesh->Np; i++)
    {               
        int num_nbr = graphNodes_element[i].nbrIds.size();
        dlong idx_own = graphNodes_element[i].localId; 
                    
        for (int j = 0; j < num_nbr; j++)
        {           
            dlong idx_nei = graphNodes_element[i].nbrIds[j];  
            file_cpu << idx_nei << '\t' << idx_own << '\n'; 
        }
    }           
} 

void gnn_t::write_edge_index_element_local_vertex(const std::string& filename)
{
    if (verbose) printf("[RANK %d] -- in write_edge_index_element_local_vertex() \n", rank);

    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }

    // loop through vertex node indices 
    int n_vertex_nodes = 8; 
    for (int i = 0; i < n_vertex_nodes; i++)
    {
        dlong idx_own = mesh->vertexNodes[i];  
        for (int j = 0; j < n_vertex_nodes; j++)
        {
            dlong idx_nei = mesh->vertexNodes[j]; 
            file_cpu << idx_nei << '\t' << idx_own << '\n';
        }
    }
}

void gnn_t::write_edge_index_element_local_vertex_binary(const std::string& filename)
{
    if (verbose) printf("[RANK %d] -- in write_edge_index_element_local_vertex_binary() \n", rank);

    std::cout << "Writing file: " << filename << std::endl;
    std::ofstream file_cpu(filename, std::ios::binary);
    if (!file_cpu.is_open())
    {
        std::cout << "Error opening file." << std::endl;
        exit(1);
    }

    // loop through vertex node indices 
    int n_vertex_nodes = 8; 
    for (int i = 0; i < n_vertex_nodes; i++)
    {
        dlong idx_own = mesh->vertexNodes[i];  
        for (int j = 0; j < n_vertex_nodes; j++)
        {
            dlong idx_nei = mesh->vertexNodes[j]; 
            // file_cpu << idx_nei << '\t' << idx_own << '\n';
            file_cpu.write(reinterpret_cast<const char*>(&idx_nei), sizeof(dlong));
            file_cpu.write(reinterpret_cast<const char*>(&idx_own), sizeof(dlong));
        }
    }
}

void gnn_t::interpolateField(nrs_t* nrs, occa::memory& o_field_fine, dfloat* field_coarse, int dim)
{
    if (gnnMeshPOrder == nekMeshPOrder){
        o_field_fine.copyTo(field_coarse, dim * fieldOffset);
    } else if (gnnMeshPOrder < nekMeshPOrder){
        auto o_tmp = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
        for (int i = 0; i < dim; i++) {
            nrs->mesh->interpolate(o_field_fine.slice(i * nrs->fieldOffset, nrs->fieldOffset), mesh, o_tmp);
            o_tmp.copyTo(field_coarse + i * fieldOffset, fieldOffset);
        }
    }
}

