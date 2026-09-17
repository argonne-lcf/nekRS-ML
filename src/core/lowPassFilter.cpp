#include "platform.hpp"
#include "lowPassFilter.hpp"
#include "nrs.hpp"

namespace
{

struct userFilterContainer
{
  std::string tag;
  int Nc = -1;
  dfloat wght = 0.0;
  occa::memory o_filterRT;

  bool setupCalled = false;
};

std::map<std::string, userFilterContainer> filterMap;

double filterFactorial(int n)
{
  if (n == 0) {
    return 1;
  } else {
    return n * filterFactorial(n - 1);
  }
}

// low Pass
void filterFunctionRelaxation1D(int Nmodes, int Nc, double *A)
{
  // Set all diagonal to 1
  for (int n = 0; n < Nmodes; n++) {
    A[n * Nmodes + n] = 1.0;
  }

  int k0 = Nmodes - Nc;
  for (int k = k0; k < Nmodes; k++) {
    double amp = ((k + 1.0 - k0) * (k + 1.0 - k0)) / (Nc * Nc);
    A[k + Nmodes * k] = 1.0 - amp;
  }
}

// explicit
void filterFunctionRelaxation1Dv2(int Nmodes, int Nc, double wght, double *A)
{
  // zero matrix
  for (int n = 0; n < Nmodes*Nmodes; n++) {
    A[n] = 0.0;
  }

  // Set all diagonal to 1
  for (int n = 0; n < Nmodes; n++) {
    A[n * Nmodes + n] = 1.0;
  }

  int k0 = Nmodes - Nc;
  for (int k = k0; k < Nmodes; k++) {
    double amp = wght * ((k + 1.0 - k0) * (k + 1.0 - k0)) / (Nc * Nc);
    A[k + Nmodes * k] = 1.0 - amp;
  }
}

// jacobi polynomials at [-1,1] for GLL
double filterJacobiP(double a, double alpha, double beta, int N)
{
  double ax = a;

  auto P = (double *)calloc((N + 1), sizeof(double));

  // Zero order
  double gamma0 = pow(2, (alpha + beta + 1)) / (alpha + beta + 1) * filterFactorial(alpha) *
                  filterFactorial(beta) / filterFactorial(alpha + beta);
  double p0 = 1.0 / sqrt(gamma0);

  if (N == 0) {
    free(P);
    return p0;
  }
  P[0] = p0;

  // first order
  double gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0;
  double p1 = ((alpha + beta + 2) * ax / 2 + (alpha - beta) / 2) / sqrt(gamma1);
  if (N == 1) {
    free(P);
    return p1;
  }

  P[1] = p1;

  /// Repeat value in recurrence.
  double aold = 2 / (2 + alpha + beta) * sqrt((alpha + 1.) * (beta + 1.) / (alpha + beta + 3.));
  /// Forward recurrence using the symmetry of the recurrence.
  for (int i = 1; i <= N - 1; ++i) {
    double h1 = 2. * i + alpha + beta;
    double anew =
        2. / (h1 + 2.) *
        sqrt((i + 1.) * (i + 1. + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3));
    double bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2);
    P[i + 1] = 1. / anew * (-aold * P[i - 1] + (ax - bnew) * P[i]);
    aold = anew;
  }

  double pN = P[N];
  free(P);
  return pN;
}

void filterVandermonde1D(int N, int Np, double *r, double *V)
{
  int sk = 0;
  for (int i = 0; i <= N; i++) {
    for (int n = 0; n < Np; n++) {
      V[n * Np + sk] = filterJacobiP(r[n], 0, 0, i);
    }
    sk++;
  }
}

// legendre polynomials at [-1,1] for GLL
void filterLegendreP(double *L, const double x, const int N)
{
   L[0] = 1.0;
   L[1] = x;
   for (int j = 2; j <= N; j++) {
      const double dj = j;
      L[j] = ( (2*dj-1) * x * L[j-1] - (j-1) * L[j-2] ) / dj;
   }
}

void filterBubbleFunc1D(int N, int Np, double *r, double *V)
{
  auto Lj = (double *)malloc(Np * sizeof(double));
  for (int j = 0; j <= N; j++) {
    const double z = r[j];
    filterLegendreP(Lj, z, N);
    V[0 * Np + j] = Lj[0];
    V[1 * Np + j] = Lj[1];
    for (int i = 2; i < Np; i++) {
      V[i * Np + j] = Lj[i] - Lj[i-2];
    }
  }
  free(Lj);
}

// A = V A V^{-1}, both V and A are of size Nmodes^2
void computeFilterMatrix(double *V, double *A, const int Nmodes, const char TRANS)
{
  int INFO;
  int N = Nmodes;
  int LWORK = N * N;
  double *WORK = (double *)calloc(LWORK, sizeof(double));
  int *IPIV = (int *)calloc(Nmodes + 1, sizeof(int));
  double *iV = (double *)calloc(Nmodes * Nmodes, sizeof(double));

  for (int n = 0; n < (Nmodes + 1); n++) {
    IPIV[n] = 1;
  }
  for (int n = 0; n < Nmodes * Nmodes; ++n) {
    iV[n] = V[n];
  }

  dgetrf_(&N, &N, (double *)iV, &N, IPIV, &INFO);
  nekrsCheck(INFO, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "dgetrf failed");

  dgetri_(&N, (double *)iV, &N, IPIV, (double *)WORK, &LWORK, &INFO);
  nekrsCheck(INFO, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "dgetri failed");

  // V*A*V^-1 in row major
  char TRANSA = TRANS;
  char TRANSB = TRANS;
  double ALPHA = 1.0, BETA = 0.0;
  int MD = Nmodes;
  int ND = Nmodes;
  int KD = Nmodes;
  int LDA = Nmodes;
  int LDB = Nmodes;

  double *C = (double *)calloc(Nmodes * Nmodes, sizeof(double));

  int LDC = Nmodes;

  dgemm_(&TRANSA, &TRANSB, &MD, &ND, &KD, &ALPHA, A, &LDA, iV, &LDB, &BETA, C, &LDC);

  TRANSA = TRANS;
  TRANSB = 'N';

  dgemm_(&TRANSA, &TRANSB, &MD, &ND, &KD, &ALPHA, V, &LDA, C, &LDB, &BETA, A, &LDC);

  free(C), free(iV), free(IPIV), free(WORK);
}

} // namespace

occa::memory lowPassFilterSetup(mesh_t *mesh, const dlong filterNc)
{
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");

  nekrsCheck(filterNc < 1,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "number of filter modes must be at least 1, but is set to %d\n",
             filterNc);

  nekrsCheck(filterNc >= mesh->N,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "mumber of filter modes must be < %d\n",
             mesh->N);

  // Construct Filter Function
  const int Nmodes = mesh->N + 1; // N+1, 1D GLL points

  auto V = (double *)calloc(Nmodes * Nmodes, sizeof(double));
  auto A = (double *)calloc(Nmodes * Nmodes, sizeof(double));

  // Construct Filter Function
  filterFunctionRelaxation1D(Nmodes, filterNc, A);

  // Construct Vandermonde Matrix
  {
    auto r = (double *)malloc(mesh->Np * sizeof(double));
    for (int i = 0; i < mesh->Np; i++) {
      r[i] = mesh->r[i];
    }
    filterVandermonde1D(mesh->N, Nmodes, r, V); // this is polyn.^T
    free(r);
  }

  computeFilterMatrix(V, A, Nmodes, 'T');

  auto o_A = platform->device.malloc<dfloat>(Nmodes * Nmodes);
  {
    auto tmp = (dfloat *)calloc(Nmodes * Nmodes, sizeof(dfloat));
    for (int i = 0; i < Nmodes * Nmodes; i++) {
      tmp[i] = A[i]; // cast to dfloat
    }
    o_A.copyFrom(tmp, o_A.length());
    free(tmp);
  }

  if (verbose && platform->comm.mpiRank == 0) {
    for (int j = 0; j < Nmodes; j++) {
      printf("filt mat (rs) %5s:", "hpf");
      for (int i = 0; i < Nmodes; i++) {
        printf("%11.4e", A[i + Nmodes * j]);
      }
      printf("\n");
    }
  }

  free(A);
  free(V);
  return o_A;
}

occa::memory explicitFilterSetup(std::string tag,
                                 mesh_t *mesh,
                                 const dlong filterNc,
                                 const dfloat filterWght)
{
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");

  nekrsCheck(filterNc < 1,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s: number of filter modes must be at least 1, but is set to %d\n",
             tag.c_str(), filterNc);

  nekrsCheck(filterNc >= mesh->N,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s: mumber of filter modes must be < %d\n",
             tag.c_str(), mesh->N);

  nekrsCheck(filterWght < 0 || filterWght > 1,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s: filterWght must in [0,1], but is set to %g\n",
             tag.c_str(), filterWght);

  // Construct Filter Function
  const int Nmodes = mesh->N + 1; // N+1, 1D GLL points

  auto V = (double *)calloc(Nmodes * Nmodes, sizeof(double));
  auto A = (double *)calloc(Nmodes * Nmodes, sizeof(double));

  // Construct Filter Function
  filterFunctionRelaxation1Dv2(Nmodes, filterNc, filterWght, A);

  if (platform->comm.mpiRank == 0) {
    printf("filt trn (rs) %5s:", tag.c_str());
    for (int k = 0; k < Nmodes; k++) {
      printf("%7.4f", A[k + Nmodes * k]);
    }
    printf("\n");
  }

  // Construct Vandermonde Matrix
  {
    auto r = (double *)malloc(mesh->Np * sizeof(double));
    for (int i = 0; i < mesh->Np; i++) {
      r[i] = mesh->r[i];
    }
    filterBubbleFunc1D(mesh->N, Nmodes, r, V); // bubble of Legendre
    free(r);
  }

  computeFilterMatrix(V, A, Nmodes, 'N');

  auto o_A = platform->device.malloc<dfloat>(Nmodes * Nmodes);
  {
    auto tmp = (dfloat *)calloc(Nmodes * Nmodes, sizeof(dfloat));
    for (int i = 0; i < Nmodes * Nmodes; i++) {
      tmp[i] = A[i]; // cast to dfloat
    }
    o_A.copyFrom(tmp, o_A.length());
    free(tmp);
  }

  if (verbose && platform->comm.mpiRank == 0) {
    for (int j = 0; j < Nmodes; j++) {
      printf("filt mat (rs) %5s:", tag.c_str());
      for (int i = 0; i < Nmodes; i++) {
        printf("%11.4e", A[i + Nmodes * j]);
      }
      printf("\n");
    }
  }

  free(A);
  free(V);
  return o_A;
}

void  nrs_t::applyExplicitFilter(std::string tag, mesh_t *mesh, occa::memory &o_fld,
                                 const int filterNc, const dfloat filterWght)
{
  userFilterContainer *m = &filterMap[tag];

  if (!m->setupCalled) {
    m->Nc = filterNc;
    m->wght = filterWght;
    m->o_filterRT = explicitFilterSetup(tag, mesh, m->Nc, m->wght);
    m->setupCalled = true;
  } else {
    nekrsCheck(m->Nc != filterNc,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "explicitFilter %s attempt to use a differnt Nc=%d than setup Nc=%d!\n",
               tag.c_str(), filterNc, m->Nc);
    nekrsCheck(abs(m->wght - filterWght) > 1e-6,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "explicitFilter %s attempt to use a differnt wght=%g than setup wght=%g!\n",
               tag.c_str(), filterWght, m->wght);
  }

  static bool firstCall = true;
  static occa::memory o_offsetScan;
  static occa::memory o_applyFilter;

  if (firstCall) { // set up dummy scalar arrays
    std::vector<dlong> offsetScan = {0};
    o_offsetScan = platform->device.malloc<dlong>(1);
    o_offsetScan.copyFrom(offsetScan.data());

    std::vector<dlong> applyFilter = {1};
    o_applyFilter = platform->device.malloc<dlong>(1);
    o_applyFilter.copyFrom(applyFilter.data());
    firstCall = false;
  }

  this->scalarExplicitFilterKernel(mesh->Nelements,
                                   0, // start idx
                                   1, // nfld
                                   o_offsetScan,
                                   o_applyFilter,
                                   m->o_filterRT,
                                   o_fld);
}
