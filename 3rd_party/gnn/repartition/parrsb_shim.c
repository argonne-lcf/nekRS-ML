/* Thin C shim exposing parrsb_part_mesh to Python (ctypes).
 *
 * The MPI communicator crosses the language boundary as a Fortran handle
 * (mpi4py comm.py2f() -> MPI_Fint) because MPI_Comm is an opaque C type
 * whose size/representation differs between MPI implementations.
 *
 * Build (see build_parrsb_shim.sh): link against libparRSB.a + libgs.a from
 * a nekRS install (NEKRS_HOME/nek5000/3rd_party/{parRSB,gslib}).
 */

#include <mpi.h>

#include "parRSB.h"

/* Returns 0 on success (parrsb_part_mesh convention). part must hold nel
 * ints; vtx is nel*nv corner-vertex global ids (long long, element-major);
 * xyz is nel*nv*3 corner coordinates (double, element-major, xyz-fastest)
 * or NULL (disables the RCB pre-partition). partitioner: 0 RSB, 1 RCB,
 * 2 RIB. */
int repartition_parrsb_part_mesh(int *part, const long long *vtx,
                                 const double *xyz, int nel, int nv,
                                 int partitioner, int verbose_level,
                                 MPI_Fint fcomm) {
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  parrsb_options opts = parrsb_default_options;
  opts.partitioner = partitioner;
  opts.verbose_level = verbose_level;
  return parrsb_part_mesh(part, vtx, xyz, NULL, nel, nv, &opts, comm);
}
