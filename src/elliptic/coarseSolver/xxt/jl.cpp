#include "jl.hpp"

static void write_vec(const double *v, uint un, const char *name,
                      struct comm *c) {
  slong out[2][1], wrk[2][1], in = un;
  comm_scan(out, c, gs_long, gs_add, &in, 1, wrk);
  slong sn = out[0][0], gn = out[1][0];

  struct value_t {
    double v;
    ulong idx;
    uint p;
  };

  struct array vals;
  array_init(struct value_t, &vals, un * c->np);

  struct value_t vt = {.p = 0};
  for (uint i = 0; i < un; i++) {
    vt.v = v[i], vt.idx = sn + i;
    array_cat(struct value_t, &vals, &vt, 1);
  }

  struct crystal cr;
  crystal_init(&cr, c);
  sarray_transfer(struct value_t, &vals, p, 0, &cr);
  crystal_free(&cr);

  buffer bfr;
  buffer_init(&bfr, 1024);
  sarray_sort(struct value_t, vals.ptr, vals.n, idx, 1, &bfr);
  buffer_free(&bfr);

  if (c->id == 0) {
    FILE *fp = fopen(name, "w+");
    struct value_t *vp = (struct value_t *)vals.ptr;
    if (fp) {
      for (uint i = 0; i < vals.n; i++)
        fprintf(fp, "%.15lf\n", vp[i].v);
      fclose(fp);
    }
  }

  array_free(&vals);
}

static int check_alloc_(void *ptr, const char *file, unsigned line) {
  if (ptr == NULL) {
    printf("check_alloc failure: %s:%d\n", file, line);
    return 1;
  }
  return 0;
}
#define check_alloc(ptr) check_alloc_(ptr, __FILE__, __LINE__)

static int gen_crs_basis(dfloat *b, int j_, dfloat *z, int Nq, int Np) {
  dfloat *zr = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *zs = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *zt = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *z0 = (dfloat *)calloc(Nq, sizeof(dfloat));
  dfloat *z1 = (dfloat *)calloc(Nq, sizeof(dfloat));
  if (zr == NULL || zs == NULL || zt == NULL || z0 == NULL || z1 == NULL)
    return 1;

  for (int i = 0; i < Nq; i++) {
    z0[i] = 0.5 * (1 - z[i]);
    z1[i] = 0.5 * (1 + z[i]);
  }

  memcpy(zr, z0, Nq * sizeof(dfloat));
  memcpy(zs, z0, Nq * sizeof(dfloat));
  memcpy(zt, z0, Nq * sizeof(dfloat));

  int jj = j_ + 1;
  if (jj % 2 == 0)
    memcpy(zr, z1, Nq * sizeof(dfloat));
  if (jj == 3 || jj == 4 || jj == 7 || jj == 8)
    memcpy(zs, z1, Nq * sizeof(dfloat));
  if (jj > 4)
    memcpy(zt, z1, Nq * sizeof(dfloat));

  for (int k = 0; k < Nq; k++) {
    for (int j = 0; j < Nq; j++) {
      for (int i = 0; i < Nq; i++) {
        int n = i + Nq * j + Nq * Nq * k + j_ * Np;
        b[n] = zr[i] * zs[j] * zt[k];
      }
    }
  }

  free(zr), free(zs), free(zt), free(z0), free(z1);

  return 0;
}

static int get_local_crs_galerkin(double *a, int nc, mesh_t *mf,
                                  elliptic_t *ef) {
  size_t nelt = mf->Nelements, Np = mf->Np;
  size_t size = nelt * Np;

  dfloat *b = tcalloc(dfloat, nc * Np);
  check_alloc(b);
  for (int j = 0; j < nc; j++)
    gen_crs_basis(b, j, mf->gllz, mf->Nq, mf->Np);

  dfloat *u = tcalloc(dfloat, size), *w = tcalloc(dfloat, size);
  check_alloc(u), check_alloc(w);

  occa::memory o_u = platform->device.malloc<dfloat>(size, u);
  occa::memory o_w = platform->device.malloc<dfloat>(size, w);
  occa::memory o_upf = platform->device.malloc<pfloat>(size);
  occa::memory o_wpf = platform->device.malloc<pfloat>(size);

  int i, j, k, e;
  for (j = 0; j < nc; j++) {
    for (e = 0; e < nelt; e++)
      memcpy(&u[e * Np], &b[j * Np], Np * sizeof(dfloat));

    o_u.copyFrom(u);
    platform->copyDfloatToPfloatKernel(mf->Nlocal, o_u, o_upf);
    ellipticAx(ef, mf->Nelements, mf->o_elementList, o_upf, o_wpf, pfloatString);
    platform->copyPfloatToDfloatKernel(mf->Nlocal, o_wpf, o_w);
    o_w.copyTo(w);

    for (e = 0; e < nelt; e++)
      for (i = 0; i < nc; i++) {
        a[i + j * nc + e * nc * nc] = 0.0;
        for (k = 0; k < Np; k++)
          a[i + j * nc + e * nc * nc] += b[k + i * Np] * w[k + e * Np];
      }
  }

  free(b), free(w), free(u);
  o_u.free(), o_w.free(), o_upf.free(), o_wpf.free();

  return 0;
}

static void set_mat_ij(uint *ia, uint *ja, int nc, int nelt) {
  uint i, j, e;
  for (e = 0; e < nelt; e++) {
    for (j = 0; j < nc; j++) {
      for (i = 0; i < nc; i++) {
        ia[i + j * nc + nc * nc * e] = e * nc + i;
        ja[i + j * nc + nc * nc * e] = e * nc + j;
      }
    }
  }
}

void jl_setup_aux(uint *ntot_, ulong **gids_, uint *nnz_, uint **ia_,
                  uint **ja_, double **a_, elliptic_t *elliptic,
                  elliptic_t *ellipticf) {
  mesh_t *mesh = elliptic->mesh, *meshf = ellipticf->mesh;
  assert(mesh->Nelements == meshf->Nelements);
  uint nelt = meshf->Nelements, nc = mesh->Np;

  // Set global ids: copy and apply the mask
  uint ntot = *ntot_ = nelt * nc;
  ulong *gids = *gids_ = tcalloc(ulong, ntot);

  for (uint j = 0; j < nelt * nc; j++)
    gids[j] = mesh->globalIds[j];

  if (elliptic->Nmasked) {
    dlong *mask_ids = (dlong *)calloc(elliptic->Nmasked, sizeof(dlong));
    elliptic->o_maskIds.copyTo(mask_ids, elliptic->Nmasked);
    for (uint n = 0; n < elliptic->Nmasked; n++)
      gids[mask_ids[n]] = 0;
    free(mask_ids);
  }

  // Set coarse matrix
  uint nnz = *nnz_ = nc * nc * nelt;
  double *a = *a_ = tcalloc(double, nnz);
  check_alloc(a);

  get_local_crs_galerkin(a, nc, meshf, ellipticf);

  uint *ia = *ia_ = tcalloc(uint, nnz), *ja = *ja_ = tcalloc(uint, nnz);
  check_alloc(ia), check_alloc(ja);
  set_mat_ij(ia, ja, nc, nelt);
}
#undef check_alloc

/*
 * nekRS interface to JL solvers.
 */
static struct comm c;
static double *h_x = NULL;
static double *h_b = NULL;
static float *p_x = NULL;
static float *p_b = NULL;
static uint un;

void jl_setup(MPI_Comm comm, uint n, const ulong *id, uint nnz,
             const uint *Ai, const uint *Aj, const double *A, uint null,
             uint verbose) {
  comm_init(&c, comm);

  un = n;
  h_x = tcalloc(double, n);
  h_b = tcalloc(double, n);
  p_x = tcalloc(float, n);
  p_b = tcalloc(float, n);

  xxt_setup(&c, n, id, nnz, Ai, Aj, A, null, verbose);
}

void jl_solve(occa::memory &o_x, occa::memory &o_rhs) {
  o_rhs.copyTo(p_b, un);
  for (uint i = 0; i < un; i++)
    h_b[i] = p_b[i];

  xxt_solve(h_x, h_b);

  for (uint i = 0; i < un; i++)
    p_x[i] = h_x[i];
  o_x.copyFrom(p_x, un);
}

void jl_free() {
  xxt_free();

  free(h_x), h_x = NULL;
  free(h_b), h_b = NULL;
  free(p_x), p_x = NULL;
  free(p_b), p_b = NULL;
  un = 0;

  comm_free(&c);
}
