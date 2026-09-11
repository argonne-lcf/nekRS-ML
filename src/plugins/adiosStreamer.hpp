#if !defined(nekrs_adiosstreamer_hpp_)
#define nekrs_adiosstreamer_hpp_

#include "nrs.hpp"
#if defined(NEKRS_ENABLE_ADIOS)
#include "adios2.h"
#include <map>
#include <string>
#endif

class adios_client_t
{
public:
  adios_client_t(MPI_Comm& comm);
  ~adios_client_t();

#if defined(NEKRS_ENABLE_ADIOS)
  // adios objects
  adios2::ADIOS *_adios;
  adios2::IO _stream_io;
  adios2::IO _write_io;
  adios2::IO _data_io;
  adios2::Engine _solWriter;
  adios2::Engine _dataWriter;

  // solution variables and array sizes
  unsigned long long _num_dim;
  unsigned long long _N, _num_edges;
  unsigned long long _global_N, _global_num_edges;
  unsigned long long _offset_N, _offset_num_edges;
  unsigned long long _field_offset, _global_field_offset, _offset_field_offset;
  // adios objects
  adios2::Variable<dfloat> uIn, uOut;

  // member functions
  int check_run();
  void checkpoint(dfloat *field, int num_dim);
  void openStream();
  void closeStream();

  // Offline training data written to a multi-step BP5 file.
  // NOTE: gnnWriteADIOS() must be called first 
  void openDataFile(const std::string& fname = "trainingData.bp");
  void closeDataFile();
  void beginDataStep();
  void endDataStep();
  void putField(const std::string& name, dfloat *field, int num_dim);
  void putScalar(const std::string& name, int value);
  void putScalar(const std::string& name, dfloat value);

private:
  // Streamer parameters
  std::string _engine;
  std::string _transport;
  std::string _stream;

  // adios objects
  adios2::Params _params;

  // Training data file state.  Variables are defined once on first use and
  // cached, since adios2::IO::DefineVariable throws if a name is redefined.
  bool _dataOpen = false;
  std::map<std::string, adios2::Variable<dfloat>> _dataVars;
  std::map<std::string, adios2::Variable<int>> _dataIntVars;
  std::map<std::string, adios2::Variable<dfloat>> _dataRealVars;
#endif

  // MPI stuff
  int _rank, _size;
  MPI_Comm& _comm;
};

#endif
