#include <gnn.hpp>
#include <gslib.h>

static void
setup_node_graph(gnnGraph_t *graph, const hlong *const global_ids, unsigned E, unsigned N, MPI_Comm comm)
{
  const uint max_nbrs_per_dof = 27;
  const uint dofs_per_element = (uint)N * (uint)N * (uint)N;
  const uint dofs = (uint)E * dofs_per_element;
  const uint max_nbrs = (uint)E * dofs_per_element * max_nbrs_per_dof;

  hlong *neighbors = tcalloc(hlong, max_nbrs);
  uint index = 0;
  for (uint e = 0; e < E; e++) {
    for (sint i = 0; i < N; i++) {
      for (sint j = 0; j < N; j++) {
        for (sint k = 0; k < N; k++) {
          for (sint di = -1; di <= 1; di++) {
            for (sint dj = -1; dj <= 1; dj++) {
              for (sint dk = -1; dk <= 1; dk++) {
                sint ii = i + di;
                sint jj = j + dj;
                sint kk = k + dk;
                sint src = (di == 0 && dj == 0 && dk == 0);
                sint out_of_range = (ii < 0 || ii >= N || jj < 0 || jj >= N || kk < 0 || kk >= N);
                if (!src && !out_of_range) {
                  neighbors[index++] = global_ids[ii + jj * N + kk * N * N + e * dofs_per_element];
                } else {
                  neighbors[index++] = -1;
                }
              }
            }
          }
        }
      }
    }
  }

  struct comm c;
  comm_init(&c, comm);

  typedef struct {
    ulong src_id;
    ulong nbr_id;
    uint p;
  } nbr_t;

  struct array nbrs;
  array_init(nbr_t, &nbrs, max_nbrs);

  nbr_t nbr = {0, 0, 0};
  index = 0;
  for (uint d = 0; d < dofs; d++) {
    nbr.src_id = global_ids[d];
    nbr.p = nbr.src_id % c.np;
    for (uint i = 0; i < (uint)max_nbrs_per_dof; i++) {
      if (neighbors[index] != -1) {
        nbr.nbr_id = neighbors[index];
        array_cat(nbr_t, &nbrs, &nbr, 1);
      }
      index++;
    }
  }
  free(neighbors);

  struct crystal cr;
  crystal_init(&cr, &c);

  buffer bfr;
  buffer_init(&bfr, 1024);
  sarray_transfer(nbr_t, &nbrs, p, 1, &cr);
  sarray_sort_2(nbr_t, nbrs.ptr, nbrs.n, src_id, 1, nbr_id, 1, &bfr);

  struct array nnzs;
  array_init(nbr_t, &nnzs, max_nbrs);

  const nbr_t *pn = (const nbr_t *)nbrs.ptr;
  uint s = 0;
  while (s < nbrs.n) {
    uint p = pn[s].p;
    uint e = s + 1;
    while (e < nbrs.n && pn[s].src_id == pn[e].src_id) {
      p = (p > pn[e].p) ? pn[e].p : p;
      e++;
    }

    nbr = pn[s], nbr.p = p;
    array_cat(nbr_t, &nnzs, &nbr, 1);

    ulong nbr_id = nbr.nbr_id;
    for (uint i = s + 1; i < e; i++) {
      if (pn[i].nbr_id != nbr_id) {
        nbr.nbr_id = pn[i].nbr_id;
        array_cat(nbr_t, &nnzs, &nbr, 1);
        nbr_id = pn[i].nbr_id;
      }
    }

    s = e;
  }
  array_free(&nbrs);

  sarray_transfer(nbr_t, &nnzs, p, 0, &cr);
  crystal_free(&cr);
  sarray_sort_2(nbr_t, nnzs.ptr, nnzs.n, src_id, 1, nbr_id, 1, &bfr);
  buffer_free(&bfr);

  uint num_nodes = (nnzs.n > 0);
  pn = (const nbr_t *)nnzs.ptr;
  s = 0;
  while (s + 1 < nnzs.n) {
    num_nodes += (pn[s + 1].src_id > pn[s].src_id);
    s++;
  }

  graph->num_nodes = num_nodes;
  graph->nodes = tcalloc(hlong, num_nodes);
  graph->offsets = tcalloc(size_t, num_nodes + 1);
  graph->neighbors = tcalloc(hlong, nnzs.n);

  s = num_nodes = 0;
  uint num_nbrs = 0;
  while (s < nnzs.n) {
    graph->nodes[num_nodes++] = pn[s].src_id;
    uint e = s;
    while (e < nnzs.n && pn[s].src_id == pn[e].src_id) {
      graph->neighbors[num_nbrs++] = pn[s].nbr_id, e++;
    }
    s = graph->offsets[num_nodes] = e;
  }
  assert(graph->num_nodes == num_nodes);
  assert(graph->offsets[num_nodes] == nnzs.n);

  array_free(&nnzs);
  comm_free(&c);
}
