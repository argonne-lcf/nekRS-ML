#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "ogsInterface.h"
#include "gnn.hpp"
#include <cstdlib>

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
    local_unique_mask = new dlong[N](); 
    halo_unique_mask = new dlong[N]();
    graphNodes = (graphNode_t*) calloc(N, sizeof(graphNode_t));

    if (verbose) printf("\n[RANK %d] -- Finished instantiating gnn_t object\n", rank);
    if (verbose) printf("[RANK %d] -- The number of elements is %d \n", rank, mesh->Nelements);
}

gnn_t::~gnn_t()
{
    if (verbose) printf("[RANK %d] -- gnn_t destructor\n", rank);
    delete[] pos_node;
    delete[] local_unique_mask;
    delete[] halo_unique_mask;
    free(localNodes);
    free(haloNodes);
    free(graphNodes);
}

void gnn_t::gnnSetup()
{
    if (verbose) printf("[RANK %d] -- in gnnSetup() \n", rank);
    get_graph_nodes();
    get_node_positions();
    get_node_masks();
}

void gnn_t::gnnWrite()
{
    if (verbose) printf("[RANK %d] -- in gnnWrite() \n", rank);
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes 
    std::string proc = "_proc_" + std::to_string(rank);
    write_edge_index("edge_index" + proc);
    writeToFile("pos_node" + proc, pos_node, N, 3);
    writeToFile("local_unique_mask" + proc, local_unique_mask, N, 1); 
    writeToFile("halo_unique_mask" + proc, halo_unique_mask, N, 1); 
    writeToFile("global_ids" + proc, mesh->globalIds, N, 1);
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
        std::vector<dlong> coincidentOwn[NlocalGather];
        std::vector<std::vector<dlong>> coincidentNei[NlocalGather];
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

        
        // loop through NlocalGather  
        cnt = num_edges; 
        for (dlong i = 0; i < NlocalGather; i++)
        {
            // get the owner vector 
            std::vector<dlong> idx_own_vec = coincidentOwn[i];

            // skip if size is 1 
            if (idx_own_vec.size() == 1)
            {
                continue;
            }

            // get the list of neighbor vectors 
            std::vector<std::vector<dlong>> list_nei_vec = coincidentNei[i];

            int n_pairs = idx_own_vec.size();

            // loop through owners:  
            // if (rank == 0) std::cout << "i = " << i << "\tentering loop!" << std::endl;
            for (int j = 0; j < n_pairs; j++)
            {
                // loop through neighbors
                for (int k = 0; k < n_pairs; k++)
                {
                    if (j == k)
                    {
                        continue; // skip redundant pair 
                    }        
                    
                    dlong idx_own = idx_own_vec[j]; // get the owner id  
                    std::vector<dlong> idx_nei_vec = list_nei_vec[k]; // get the neighbor ids 

                    // store the extra edges 
                    for (int l = 0; l < idx_nei_vec.size(); l++)
                    {
                        dlong idx_nei = idx_nei_vec[l];
                        graphNodes[idx_own].nbrIds.push_back(idx_nei);
                        cnt += 1;
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
            
            // loop through NhaloGather  
            cnt = num_edges; 
            for (dlong i = 0; i < NhaloGather; i++)
            {
                // get the owner vector 
                std::vector<dlong> idx_own_vec = coincidentOwnHalo[i];

                // skip if size is 1 
                if (idx_own_vec.size() == 1)
                {
                    continue;
                }

                // get the list of neighbor vectors 
                std::vector<std::vector<dlong>> list_nei_vec = coincidentNeiHalo[i];

                int n_pairs = idx_own_vec.size();

                // loop through owners:  
                for (int j = 0; j < n_pairs; j++)
                {
                    // loop through neighbors
                    for (int k = 0; k < n_pairs; k++)
                    {
                        if (j == k)
                        {
                            continue; // skip redundant pair 
                        }        
                        
                        dlong idx_own = idx_own_vec[j]; // get the owner id  
                        std::vector<dlong> idx_nei_vec = list_nei_vec[k]; // get the neighbor ids 

                        // store the extra edges 
                        for (int l = 0; l < idx_nei_vec.size(); l++)
                        {
                            dlong idx_nei = idx_nei_vec[l];
                            graphNodes[idx_own].nbrIds.push_back(idx_nei);
                            cnt += 1;
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
