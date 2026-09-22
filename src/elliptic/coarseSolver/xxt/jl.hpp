#if !defined(_NEKRS_JL_HPP_)
#define _NEKRS_JL_HPP_

#include "elliptic.h"
#include "gslib.h"
#include "platform.hpp"

#include "crs_xxt.hpp"

void jl_setup_aux(uint *ntot, ulong **gids, uint *nnz, uint **ia, uint **ja,
                  double **a, elliptic_t *elliptic, elliptic_t *ellipticf);

void jl_setup(MPI_Comm comm, uint n, const ulong *id, uint nnz,
              const uint *Ai, const uint *Aj, const double *A, uint null,
              uint verbose);

void jl_solve(occa::memory &o_x, occa::memory &o_rhs);

void jl_free();

#endif
