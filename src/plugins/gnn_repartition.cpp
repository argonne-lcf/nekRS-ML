#include <gnn.hpp>
#include <parrsb.h>

typedef struct {
  unsigned num_nodes;
  long long *nodes;
  unsigned *offsets;
  long long *neighbors;
} gnnGraph_t;

static void
setup_node_graph(gnnGraph_t *graph, const hlong *global_ids, hlong E, dlong N, const struct comm *c)
{
  const uint max_nbrs_per_dof = 27;
  const uint dofs_per_element = (uint)N * (uint)N * (uint)N;
  const uint max_nbrs = (uint)E * dofs_per_element * max_nbrs_per_dof;

  typedef struct {
    ulong src_id;
    ulong nbr_id;
    uint p;
  } nbr_t;

  struct array nbrs;
  array_init(nbr_t, &nbrs, max_nbrs);

  nbr_t nbr = {0, 0, 0};
  for (uint e = 0; e < E; e++) {
    for (sint i = 0; i < N; i++) {
      for (sint j = 0; j < N; j++) {
        for (sint k = 0; k < N; k++) {
          uint d = i + j * N + k * N * N + e * dofs_per_element;
          nbr.src_id = global_ids[d];
          nbr.p = nbr.src_id % c->np;
          for (sint di = -1; di <= 1; di++) {
            for (sint dj = -1; dj <= 1; dj++) {
              for (sint dk = -1; dk <= 1; dk++) {
                sint ii = i + di;
                sint jj = j + dj;
                sint kk = k + dk;
                if ((di == 0 && dj == 0 && dk == 0) ||
                    (ii < 0 || ii >= N || jj < 0 || jj >= N || kk < 0 || kk >= N)) {
                  continue;
                }
                nbr.nbr_id = global_ids[ii + jj * N + kk * N * N + e * dofs_per_element];
                array_cat(nbr_t, &nbrs, &nbr, 1);
              }
            }
          }
        }
      }
    }
  }

  struct crystal cr;
  crystal_init(&cr, c);

  buffer bfr;
  buffer_init(&bfr, 1024);

  sarray_transfer(nbr_t, &nbrs, p, 1, &cr);
  sarray_sort_2(nbr_t, nbrs.ptr, nbrs.n, src_id, 1, nbr_id, 1, &bfr);

  struct array nnzs;
  array_init(nbr_t, &nnzs, nbrs.n);

  const nbr_t *pn = (const nbr_t *)nbrs.ptr;
  uint s = 0;
  while (s < nbrs.n) {
    uint p = pn[s].p;
    uint e = s + 1;
    while (e < nbrs.n && pn[s].src_id == pn[e].src_id) {
      p = (p < pn[e].p) ? p : pn[e].p;
      e++;
    }

    array_cat(nbr_t, &nnzs, &pn[s], 1);
    ulong nbr_id = pn[s].nbr_id;
    for (uint i = s + 1; i < e; i++) {
      if (pn[i].nbr_id == nbr_id) {
        continue;
      }
      array_cat(nbr_t, &nnzs, &pn[i], 1);
      nbr_id = pn[i].nbr_id;
    }

    s = e;
  }
  array_free(&nbrs);

  sarray_transfer(nbr_t, &nnzs, p, 0, &cr);
  sarray_sort_2(nbr_t, nnzs.ptr, nnzs.n, src_id, 1, nbr_id, 1, &bfr);

  crystal_free(&cr);
  buffer_free(&bfr);

  uint num_nodes = (nnzs.n > 0);
  pn = (const nbr_t *)nnzs.ptr;
  s = 0;
  while ((s + 1) < nnzs.n) {
    num_nodes += (pn[s + 1].src_id > pn[s].src_id);
    s++;
  }

  graph->num_nodes = num_nodes;
  graph->nodes = tcalloc(long long, num_nodes);
  graph->offsets = tcalloc(unsigned, num_nodes + 1);
  graph->neighbors = tcalloc(long long, nnzs.n);

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
}

void free_node_graph(gnnGraph_t *g)
{
  if (!g) {
    return;
  }
  free(g->nodes), g->nodes = 0;
  free(g->offsets), g->offsets = 0;
  free(g->neighbors), g->neighbors = 0;
}

void repartition_node_graph(int *proc, const hlong *global_ids, hlong E, dlong N, MPI_Comm comm)
{

  struct comm c;
  comm_init(&c, comm);

  gnnGraph_t graph;
  setup_node_graph(&graph, global_ids, E, N, &c);

  int *part = tcalloc(int, graph.num_nodes);

  parrsb_options_t options;
  parrsb_options_get_default(&options);
  parrsb_part_graph(part, graph.num_nodes, graph.nodes, graph.offsets, graph.neighbors, options, comm);
  parrsb_options_free(&options);

  free(part);

  free_node_graph(&graph);
  comm_free(&c);
}
