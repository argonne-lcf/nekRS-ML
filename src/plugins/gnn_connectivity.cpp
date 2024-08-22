#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "gnn.hpp"


void gnn_t::get_edge_index()
{
    if (verbose) printf("[RANK %d] -- in get_edge_index() \n", rank);
}

// NOTE: this assumes graphNodes is already populated. 
// Call this right after get_graph_nodes() if element-local connectivity is needed.  
void gnn_t::get_graph_nodes_element()
{
    if (verbose) printf("[RANK %d] -- in get_graph_nodes_element() \n", rank);

    // Create data `
    int Nq = mesh->Nq; // (Nq = poly_order + 1)
    int edge_cnt = 0;
    int node_cnt = 0;

    // Copy connectivity for only the first element into graphNodes_element
    for (int i = 0; i < mesh->Np; i++)
    {
        int num_nbr = graphNodes[i].nbrIds.size();
        dlong idx_own = graphNodes[i].localId;
        graphNodes_element[i].localId = idx_own; 
        for (int j = 0; j < num_nbr; j++)
        {               
            dlong idx_nei = graphNodes[i].nbrIds[j];  
            graphNodes_element[i].nbrIds.push_back(idx_nei);
        }   
    }
}

void gnn_t::get_graph_nodes()
{
	if (verbose) printf("[RANK %d] -- in get_graph_nodes() \n", rank);
	int Nq = mesh->Nq; // (Nq = poly_order + 1)
	int edge_cnt = 0;
    int node_cnt = 0;
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes
    hlong *ids =  mesh->globalIds;

    for (int e = 0; e < mesh->Nelements; e++)
    {
        for (int k = 0; k < Nq; k++)
        {
            for (int j = 0; j < Nq; j++)
            {
                for (int i = 0; i < Nq; i++)
                {
                    dlong idx = i + j * Nq + k * Nq * Nq; 
                    dlong idx_nei, idx_own; 
                    dfloat r = mesh->r[idx];
                    dfloat s = mesh->s[idx];
                    dfloat t = mesh->t[idx];
                    idx_own = e * mesh->Np + idx;

                    // populate graph node attributes
                    graphNodes[node_cnt].localId = idx_own;
                    graphNodes[node_cnt].baseId = ids[idx_own];

                    // Internal nodes 
                    if ( (k > 0) and (k < Nq - 1) and (j > 0) and (j < Nq - 1) and (i > 0) and (i < Nq - 1))
                    {
                        // i - 1 
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }

                    // Corners
                    if ( (i == 0) and (j == 0) and (k == 0) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    
                    if ( (i == Nq - 1) and (j == 0) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j == Nq - 1) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j == Nq - 1) and (k == 0) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j == 0) and (k == Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j == 0) and (k == Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j == Nq - 1) and (k == Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j == Nq - 1) and (k == Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    
                    // Edges 
                    if ( (i == 0) and (j == 0) and (k > 0) and (k < Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j == 0) and (k > 0) and (k < Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j == Nq - 1) and (k > 0) and (k < Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j == Nq - 1) and (k > 0) and (k < Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == 0) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == Nq - 1) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == 0) and (k == Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == Nq - 1) and (k == Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j > 0) and (j < Nq - 1) and (k == 0) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == Nq - 1) and (j > 0) and (j < Nq - 1) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i == 0) and (j > 0) and (j < Nq - 1) and (k == Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }                 
                    if ( (i == Nq - 1) and (j > 0) and (j < Nq - 1) and (k == Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }

                    if ( (i == 0) and (j > 0) and (j < Nq - 1) and (k > 0) and (k < Nq - 1) )
                    {
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }

                    if ( (i == Nq - 1) and (j > 0) and (j < Nq - 1) and (k > 0) and (k < Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == 0) and (k > 0) and (k < Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
                    if ( (i > 0) and (i < Nq - 1) and (j == Nq - 1) and (k > 0) and (k < Nq - 1) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k - 1
                        idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }

                    if ( (i > 0) and (i < Nq - 1) and (j > 0) and (j < Nq - 1) and (k == 0) )
                    {
                        // i - 1
                        idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                        
                        // i + 1
                        idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j - 1
                        idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // j + 1
                        idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

                        // k + 1
                        idx_nei = i + j * Nq + (k+1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
                        edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
                    }
				    if ( (i > 0) and (i < Nq - 1) and (j > 0) and (j < Nq - 1) and (k == Nq - 1) )
				    {
					    // i - 1
					    idx_nei = (i-1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
					    edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
					
					    // i + 1
					    idx_nei = (i+1) + j * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
					    edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

					    // j - 1
					    idx_nei = i + (j-1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
					    edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

					    // j + 1
					    idx_nei = i + (j+1) * Nq + k * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
					    edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);

					    // k - 1
					    idx_nei = i + j * Nq + (k-1) * Nq * Nq; 
                        idx_nei = e * mesh->Np + idx_nei;
					    edge_cnt += 1;
                        graphNodes[node_cnt].nbrIds.push_back(idx_nei);
				    }                    

                    // accumulate node_cnt
                    node_cnt += 1;

                }
            }
        }
    }

    num_edges = edge_cnt;
}

