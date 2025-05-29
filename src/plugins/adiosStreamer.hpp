#if !defined(nekrs_adiosstreamer_hpp_)
#define nekrs_adiosstreamer_hpp_

#include "nrs.hpp"
#if defined(NEKRS_ENABLE_ADIOS)
#include "adios2.h"
#endif

class adios_client_t
{
public:
  adios_client_t(MPI_Comm& comm, nrs_t *nrs);
  ~adios_client_t();

#if defined(NEKRS_ENABLE_ADIOS)
  // adios objects
  adios2::ADIOS *_adios;
  adios2::IO _stream_io;
  adios2::IO _write_io;
  adios2::Engine _solWriter;

  // solution variables and array sizes
  unsigned long _num_dim;
  unsigned long _N, _num_edges;
  unsigned long _global_N, _global_num_edges;
  unsigned long _offset_N, _offset_num_edges;
  unsigned long _field_offset, _global_field_offset, _offset_field_offset;
  // adios objects
  adios2::Variable<dfloat> uIn, uOut;

  // member functions
  int check_run();
  void checkpoint();
  void openStream();
  void closeStream();

private:
  // Streamer parameters
  std::string _engine;
  std::string _transport;
  std::string _stream;

  // adios objects
  adios2::Params _params;

  // nekrs objects 
  nrs_t *_nrs;
#endif

  // MPI stuff
  int _rank, _size;
  MPI_Comm& _comm;
};

#endif
