#if !defined(_CRS_XXT_HPP_)
#define _CRS_XXT_HPP_

#include "gslib.h"

void xxt_setup(struct comm *c, uint n, const ulong *id, uint nnz, const uint *Ai,
               const uint *Aj, const double *A, uint null, uint verbose);

void xxt_solve(double *h_x, double *h_b);

void xxt_free();

#endif // _CRS_XXT_HPP_
