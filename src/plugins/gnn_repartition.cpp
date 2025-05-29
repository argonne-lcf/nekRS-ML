#include <gnn.hpp>
#include <gslib.h>
#include <parrsb.h>

static uint get_dofs_per_element(uint N, uint dim)
{
  uint dofs = N;
  for (uint i = 1; i < dim; i++) {
    dofs *= N;
  }
  return dofs;
}

/*
 * Unique GNN nodes local to each MPI process.
 */
typedef struct {
  uint n;
  ulong *nodes;
  uint *offsets;
  uint *indices;
} gnn_nodes_t;

static void gnn_nodes_setup(gnn_nodes_t *nodes, const hlong *global_ids, uint E, uint N, uint dim)
{
  buffer bfr;
  buffer_init(&bfr, 1024);

  typedef struct {
    ulong id;
    uint idx;
  } id_t;

  struct array ids;
  array_init(id_t, &ids, E);

  const uint dofs = get_dofs_per_element(N, dim);
  id_t id;
  for (uint e = 0; e < E; e++) {
    for (uint i = 0; i < dofs; i++) {
      id.idx = e * dofs + i;
      id.id = global_ids[id.idx];
      array_cat(id_t, &ids, &id, 1);
    }
  }

  sarray_sort_2(id_t, ids.ptr, ids.n, id, 1, idx, 0, &bfr);

  const id_t *pi = (const id_t *)ids.ptr;
  uint s = 0, n = 0;
  while (s < ids.n) {
    uint e = s + 1;
    while (e < ids.n && (pi[s].id == pi[e].id)) {
      e++;
    }
    s = e, n++;
  }

  nodes->n = n;
  nodes->nodes = tcalloc(ulong, n);
  nodes->offsets = tcalloc(uint, n + 1);
  nodes->indices = tcalloc(uint, ids.n);

  s = n = 0;
  uint ni = 0;
  while (s < ids.n) {
    nodes->nodes[n++] = pi[s].id;
    uint e = s;
    while (e < ids.n && (pi[s].id == pi[e].id)) {
      nodes->indices[ni++] = pi[e++].idx;
    }
    s = nodes->offsets[n] = e;
  }

  array_free(&ids);
  buffer_free(&bfr);
}

static void gnn_nodes_free(gnn_nodes_t *nodes)
{
  if (!nodes) {
    return;
  }
  free(nodes->nodes), nodes->nodes = 0;
  free(nodes->offsets), nodes->offsets = 0;
  free(nodes->indices), nodes->indices = 0;
}

/*
 * Global GNN node graph.
 */
typedef struct {
  uint n;
  long long *nodes;
  uint *offsets;
  long long *neighbors;
} gnn_graph_t;

static void
gnn_graph_setup(gnn_graph_t *graph, const hlong *global_ids, uint E, uint N, uint dim, const struct comm *c)
{
  const uint dofs = get_dofs_per_element(N, dim);
  const uint max_nbrs = (uint)E * dofs * 27;

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
          uint d = i + j * N + k * N * N + e * dofs;
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
                nbr.nbr_id = global_ids[ii + jj * N + kk * N * N + e * dofs];
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

  uint n = (nnzs.n > 0);
  pn = (const nbr_t *)nnzs.ptr;
  s = 0;
  while ((s + 1) < nnzs.n) {
    n += (pn[s + 1].src_id > pn[s].src_id);
    s++;
  }

  graph->n = n;
  graph->nodes = tcalloc(long long, n);
  graph->offsets = tcalloc(uint, n + 1);
  graph->neighbors = tcalloc(long long, nnzs.n);

  s = n = 0;
  uint num_nbrs = 0;
  while (s < nnzs.n) {
    graph->nodes[n++] = pn[s].src_id;
    uint e = s;
    while (e < nnzs.n && pn[s].src_id == pn[e].src_id) {
      graph->neighbors[num_nbrs++] = pn[s].nbr_id, e++;
    }
    s = graph->offsets[n] = e;
  }
  assert(graph->n == n);
  assert(graph->offsets[n] == nnzs.n);

  array_free(&nnzs);
}

void gnn_graph_free(gnn_graph_t *graph)
{
  if (!graph) {
    return;
  }
  free(graph->nodes), graph->nodes = 0;
  free(graph->offsets), graph->offsets = 0;
  free(graph->neighbors), graph->neighbors = 0;
}

void gnn_graph_partition(int *proc, const hlong *global_ids, uint E, uint N, uint dim, MPI_Comm comm)
{

  struct comm c;
  comm_init(&c, comm);

  gnn_graph_t graph;
  gnn_graph_setup(&graph, global_ids, E, N, dim, &c);

  int *part = tcalloc(int, graph.n);

  parrsb_options_t options;
  parrsb_options_get_default(&options);
  parrsb_part_graph(part, graph.n, graph.nodes, graph.offsets, graph.neighbors, options, comm);
  parrsb_options_free(&options);

  free(part);

  gnn_graph_free(&graph);
  comm_free(&c);
}

/*
 * Find GNN edges incident on GNN nodes in the current MPI process.
 */
typedef struct {
  uint n;
  ulong *e1;
  uint *offsets;
  ulong *e2;
} gnn_edges_t;

static void
gnn_edges_setup(gnn_edges_t *edges, const gnn_nodes_t *nodes, const gnn_graph_t *graph, MPI_Comm comm)
{
  struct comm c;
  comm_init(&c, comm);

  buffer bfr;
  buffer_init(&bfr, 1024);

  typedef struct {
    ulong id;
    uint p, dest;
  } request_t;

  struct array reqs;
  array_init(request_t, &reqs, nodes->n + graph->n);

  request_t r = {0, 0, 0};
  for (uint i = 0; i < nodes->n; i++) {
    r.id = nodes->nodes[i];
    r.p = r.id % c.np;
    array_cat(request_t, &reqs, &r, 1);
  }

  // Set `dest` to `c.np` so we can identify the graph vertices from regular GNN nodes.
  r.dest = c.np;
  for (uint i = 0; i < graph->n; i++) {
    r.id = graph->nodes[i];
    r.p = r.id % c.np;
    array_cat(request_t, &reqs, &r, 1);
  }

  struct crystal cr;
  crystal_init(&cr, &c);

  sarray_transfer(request_t, &reqs, p, 1, &cr);
  sarray_sort_2(request_t, reqs.ptr, reqs.n, id, 1, dest, 0, &bfr);

  request_t *pr = (request_t *)reqs.ptr;
  uint s = 0;
  while (s < reqs.n) {
    uint e = s + 1;
    while (e < reqs.n && (pr[s].id == pr[e].id)) {
      e++;
    }
    for (uint i = s; i < e - 1; i++) {
      pr[i].dest = pr[e - 1].p;
    }
    s = e;
  }

  // Remove graph vertices from the requests.
  sarray_sort(request_t, reqs.ptr, reqs.n, dest, 0, &bfr);
  pr = (request_t *)reqs.ptr;
  if (reqs.n > 0) {
    uint i = reqs.n - 1;
    while (i > 0 && pr[i].dest == c.np) {
      i--;
    }
    reqs.n = (i == 0 && pr[i].dest == c.np) ? 0 : (i + 1);
  }

  // Send the edges to the requester.
  typedef struct {
    ulong e1, e2;
    uint p;
  } edge_t;

  struct array edgs;
  array_init(edge_t, &edgs, reqs.n);

  sarray_transfer(request_t, &reqs, dest, 0, &cr);
  sarray_sort(request_t, reqs.ptr, reqs.n, id, 1, &bfr);

  edge_t edge;
  pr = (request_t *)reqs.ptr;
  uint n = 0;
  for (uint i = 0; i < reqs.n; i++) {
    while ((n < graph->n) && (graph->nodes[n] < pr[i].id)) {
      n++;
    }

    edge.e1 = graph->nodes[n];
    edge.p = pr[i].p;

    uint s = graph->offsets[n], e = graph->offsets[n + 1];
    while (s < e) {
      edge.e2 = graph->neighbors[s++];
      array_cat(edge_t, &edgs, &edge, 1);
    }
  }

  sarray_transfer(edge_t, &edgs, p, 0, &cr);
  sarray_sort_2(edge_t, edgs.ptr, edgs.n, e1, 1, e2, 1, &bfr);

  edges->n = 0;
  edges->e1 = tcalloc(ulong, nodes->n);
  edges->offsets = tcalloc(uint, nodes->n + 1);
  edges->e2 = tcalloc(ulong, edgs.n);

  const edge_t *pe = (const edge_t *)edgs.ptr;
  s = 0;
  while (s < edgs.n) {
    uint e = s;
    edges->e1[edges->n++] = pe[s].e1;
    while ((e < edgs.n) && (pe[s].e1 == pe[e].e1)) {
      edges->e2[n++] = pe[e].e2;
      e++;
    }
    s = edges->offsets[edges->n] = e;
  }
  assert(edges->n == nodes->n);

  array_free(&edgs);
  crystal_free(&cr);
  array_free(&reqs);
  buffer_free(&bfr);
  comm_free(&c);
}
