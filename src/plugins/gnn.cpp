#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
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

gnn_t::gnn_t(nrs_t *nrs_)
{
    nrs = nrs_; // set nekrs object
    mesh = nrs->meshV; // set mesh object
    ogs = mesh->ogs; // set ogs object

    // set MPI rank and size 
    MPI_Comm &comm = platform->comm.mpiComm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // allocate memory 
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes
    pos_node = new dfloat[N * 3](); 
    node_element_ids = new dlong[N]();
    local_unique_mask = new dlong[N](); 
    halo_unique_mask = new dlong[N]();
    graphNodes = (graphNode_t*) calloc(N, sizeof(graphNode_t)); // full domain
    graphNodes_element = (graphNode_t*) calloc(mesh->Np, sizeof(graphNode_t)); // a single element

    if (verbose) printf("\n[RANK %d] -- Finished instantiating gnn_t object\n", rank);
    if (verbose) printf("[RANK %d] -- The number of elements is %d \n", rank, mesh->Nelements);
}

gnn_t::~gnn_t()
{
    if (verbose) printf("[RANK %d] -- gnn_t destructor\n", rank);
    delete[] pos_node;
    delete[] node_element_ids;
    delete[] local_unique_mask;
    delete[] halo_unique_mask;
    free(localNodes);
    free(haloNodes);
    free(graphNodes);
    free(graphNodes_element);
}

void gnn_t::gnnSetup()
{
    if (verbose) printf("[RANK %d] -- in gnnSetup() \n", rank);
    get_graph_nodes(); // populates graphNodes
    get_graph_nodes_element(); // populates graphNodes_element
    get_node_positions();
    get_node_element_ids(); // adds neighboring edges to graphNodes
    get_node_masks();

    // output directory 
    if (write)
    {
        std::filesystem::path currentPath = std::filesystem::current_path();
        currentPath /= "gnn_outputs";
        writePath = currentPath.string();
        int poly_order = mesh->Nq - 1; 
        writePath = writePath + "_poly_" + std::to_string(poly_order);
        if (rank == 0)
        {
            if (!std::filesystem::exists(writePath))
            {
                std::filesystem::create_directory(writePath);
            }
        }
        MPI_Comm &comm = platform->comm.mpiComm;
        MPI_Barrier(comm);
    }
}

void gnn_t::gnnWrite()
{
    if (verbose) printf("[RANK %d] -- in gnnWrite() \n", rank);
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes 
    std::string irank = "_rank_" + std::to_string(rank);
    std::string nranks = "_size_" + std::to_string(size);

    // // Writing as text files:
    // write_edge_index(writePath + "/edge_index" + irank + nranks);
    // writeToFile(writePath + "/pos_node" + irank + nranks, pos_node, N, 3);
    // writeToFile(writePath + "/node_element_ids" + irank + nranks, node_element_ids, N, 1); 
    // writeToFile(writePath + "/local_unique_mask" + irank + nranks, local_unique_mask, N, 1); 
    // writeToFile(writePath + "/halo_unique_mask" + irank + nranks, halo_unique_mask, N, 1); 
    // writeToFile(writePath + "/global_ids" + irank + nranks, mesh->globalIds, N, 1);

    // Writing as binary files: 
    write_edge_index_binary(writePath + "/edge_index" + irank + nranks + ".bin");
    writeToFileBinary(writePath + "/pos_node" + irank + nranks + ".bin", pos_node, N, 3);
    writeToFileBinary(writePath + "/node_element_ids" + irank + nranks + ".bin", node_element_ids, N, 1); 
    writeToFileBinary(writePath + "/local_unique_mask" + irank + nranks + ".bin", local_unique_mask, N, 1); 
    writeToFileBinary(writePath + "/halo_unique_mask" + irank + nranks + ".bin", halo_unique_mask, N, 1); 
    writeToFileBinary(writePath + "/global_ids" + irank + nranks + ".bin", mesh->globalIds, N, 1);

    // Writing number of elements, gll points per element, and product of the two  
    writeToFile(writePath + "/Nelements" + irank + nranks, &mesh->Nelements, 1, 1);
    writeToFile(writePath + "/Np" + irank + nranks, &mesh->Np, 1, 1);
    writeToFile(writePath + "/N" + irank + nranks, &N, 1, 1);

    // Writing element-local edge index as text file (small)
    write_edge_index_element_local(writePath + "/edge_index_element_local" + irank + nranks);
    write_edge_index_element_local_vertex(writePath + "/edge_index_element_local_vertex" + irank + nranks);
}

void gnn_t::get_node_positions()
{
    if (verbose) printf("[RANK %d] -- in get_node_positions() \n", rank);
    for (int n=0; n < mesh->Np * mesh->Nelements; n++)
    {
        dfloat x = mesh->x[n]; // location of x GLL point
        dfloat y = mesh->y[n]; // location of y GLL point
        dfloat z = mesh->z[n]; // location of z GLL point
        pos_node[n + 0*(mesh->Np * mesh->Nelements)] = x;
        pos_node[n + 1*(mesh->Np * mesh->Nelements)] = y;
        pos_node[n + 2*(mesh->Np * mesh->Nelements)] = z;
    }
}

void gnn_t::get_node_element_ids()
{
    if (verbose) printf("[RANK %d] -- in get_node_element_ids() \n", rank);
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes 
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

    dlong N = mesh->Nelements * mesh->Np; // total number of nodes 
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
    if (verbose) printf("[RANK %d] -- \tN: %d \n", rank, N);
    if (verbose) printf("[RANK %d] -- \tNlocal: %d \n", rank, Nlocal);
    if (verbose) printf("[RANK %d] -- \tNhalo: %d \n", rank, Nhalo);

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
        std::cout << "[RANK  " << rank << "] -- NlocalGather: " << NlocalGather << std::endl;
   
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

    dlong N = mesh->Nelements * mesh->Np; // total number of nodes
                    
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

    dlong N = mesh->Nelements * mesh->Np; // total number of nodes
                    
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
