#include <filesystem>
#include <thread>
#include <chrono>
#include "adiosStreamer.hpp"

// Initialize the ADIOS2 client
adios_client_t::adios_client_t(MPI_Comm& comm) : _comm(comm)
{
#if defined(NEKRS_ENABLE_ADIOS)
    // Set nekrs object
    //_nrs = nrs;

    // Set MPI comm, rank and size 
    //_comm = platform->comm.mpiComm;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    // Set up adios2 parameters
    platform->options.getArgs("ADIOS ML ENGINE",_engine);
    platform->options.getArgs("ADIOS ML TRANSPORT",_transport);
    platform->options.getArgs("ADIOS ML STREAM",_stream);

    // Initialize adios2
    if (_rank == 0)
        printf("\nInitializing ADIOS2 client ...\n");
    try
    {
        _adios = new adios2::ADIOS(_comm);

        _stream_io = _adios->DeclareIO("streamIO");
        _stream_io.SetEngine(_engine);
        if (_stream == "sync") {
            // block if reader is not ready, sync sim and trainer here
            _params["RendezvousReaderCount"] = "1";
            // block when queue is full
            _params["QueueFullPolicy"] = "Block";
            // number of steps writer allows to be queued before taking action (0 means no limit to queued steps)
            _params["QueueLimit"] = "1";
        } else if (_stream == "async") {
            // block if reader is not ready, sync sim and trainer here
            _params["RendezvousReaderCount"] = "1"; 
            // "Block" or "Discard" when queue is full
            _params["QueueFullPolicy"] = "Discard";
            // number of steps writer allows to be queued before taking action (0 means no limit to queued steps)
            // With QueueFullPolicy=Discard, the newest step is discarded
            // Keep queue small so GNN training sees new data with short lag
            _params["QueueLimit"] = "3";
            // number of steps writer allows to be queued before taking action WHEN NO reader is connected (0 means no limit to queued steps)
            _params["ReserveQueueLimit"] = "0";
        }
        _params["DataTransport"] = _transport;
        _params["OpenTimeoutSecs"] = "600";
        _stream_io.SetParameters(_params);

        _write_io = _adios->DeclareIO("writeIO");
        _write_io.SetEngine("BP5");

        // Separate IO for the offline training data file.
        // Sharing _write_io would fold graph.bp's nine variables into 
        // every training-data step's metadata. Also, check_run() reopens 
        // _write_io in Mode::Read once per timestep.
        _data_io = _adios->DeclareIO("trainingDataIO");
        _data_io.SetEngine("BP5");
    }
    catch (std::exception &e)
    {
        printf("Exception, STOPPING PROGRAM from rank %d\n", _rank);
        std::cout << e.what() << "\n";
        fflush(stdout);
    }
    MPI_Barrier(_comm);
    if (_rank == 0)
        printf("All done\n");
    fflush(stdout);
#endif
}

// destructor
adios_client_t::~adios_client_t()
{
#if defined(NEKRS_ENABLE_ADIOS)
    // Close the stream for transfering the solution data
    closeStream();
    // Close the offline training data file if one was opened
    closeDataFile();
#endif
}

#if defined(NEKRS_ENABLE_ADIOS)
// check if nekRS should quit
int adios_client_t::check_run()
{
    dlong exit_val = 1;
    int exists;
    std::string fname = "check-run.bp";

    // Check if check-run file exists
    if (_rank == 0) {
        if (std::filesystem::exists(fname)) {
            printf("Found check-run file!\n");
            fflush(stdout);
            exists = 1;
        } else {
            exists = 0;
        }
    }
    MPI_Bcast(&exists, 1, MPI_INT, 0, _comm);

    // Read check-run file if exists
    if (exists) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        adios2::Engine reader = _write_io.Open(fname, adios2::Mode::Read);
        reader.BeginStep();
        adios2::Variable<dlong> var = _write_io.InquireVariable<dlong>("check-run");
        if (_rank == 0 and var) {
            reader.Get(var, &exit_val);
        }
        reader.EndStep();
        reader.Close();
        MPI_Bcast(&exit_val, 1, MPI_INT, 0, _comm);
    }

    if (exit_val == 0 && _rank == 0) {
        printf("ML training says time to quit ...\n");
    }
    fflush(stdout);

    return exit_val;
}

// Open the solution transfer stream
void adios_client_t::openStream()
{
    if (_rank == 0) std::cout << "Opening ADIOS2 solutionStream ... " << std::endl;
    try
    {
        _solWriter = _stream_io.Open("solutionStream", adios2::Mode::Write);
    }
    catch (std::exception &e)
    {
        std::cout << "Error opening ADIOS2 solutionStream, STOPPING PROGRAM from rank " << _rank << "\n";
        std::cout << e.what() << "\n";
    }
    MPI_Barrier(_comm);
    if (_rank == 0) std::cout << "All done!" << std::endl;
}

// Close the solution transfer stream.
void adios_client_t::closeStream()
{
    if (!_solWriter) return;
    try
    {
        if (_rank == 0) std::cout << "Closing ADIOS2 solutionStream " << std::endl;
        _solWriter.Close();
    }
    catch (std::exception &e)
    {
        std::cout << "Error closing ADIOS2 solutionStream, STOPPING PROGRAM from rank " << _rank << "\n";
        std::cout << e.what() << "\n";
    }
}

// write checkpoint file
void adios_client_t::checkpoint(dfloat *field, int num_dim)
{
    if (_rank == 0)
        printf("\nWriting checkpoint for GNN inference ...\n");
    std::string fname = "checkpoint.bp";
    unsigned long field_num_dim = num_dim;

    adios2::Variable<dfloat> varField = _write_io.DefineVariable<dfloat>(
        "checkpoint", 
        {_global_field_offset * field_num_dim}, 
        {_offset_field_offset * field_num_dim}, 
        {_field_offset * field_num_dim});
    adios2::Engine writer = _write_io.Open(fname, adios2::Mode::Write);
    writer.BeginStep();
    writer.Put<dfloat>(varField, field);
    writer.EndStep();
    writer.Close();
}

// Open the multi-step BP5 file holding training data for offline training.
// Unlike checkpoint() above, the engine is kept open across steps so that each
// snapshot becomes one ADIOS step of a single file.
void adios_client_t::openDataFile(const std::string& fname)
{
    if (_dataOpen) return;
    if (_rank == 0) std::cout << "Opening ADIOS2 training data file " << fname << " ... " << std::endl;
    try
    {
        _dataWriter = _data_io.Open(fname, adios2::Mode::Write);
        _dataOpen = true;
    }
    catch (std::exception &e)
    {
        std::cout << "Error opening ADIOS2 training data file, STOPPING PROGRAM from rank " << _rank << "\n";
        std::cout << e.what() << "\n";
    }
    MPI_Barrier(_comm);
    if (_rank == 0) std::cout << "All done!" << std::endl;
}

// Close the training data file. Idempotent -- also called from the destructor.
void adios_client_t::closeDataFile()
{
    if (!_dataOpen) return;
    try
    {
        if (_rank == 0) std::cout << "Closing ADIOS2 training data file " << std::endl;
        _dataWriter.Close();
        _dataOpen = false;
    }
    catch (std::exception &e)
    {
        std::cout << "Error closing ADIOS2 training data file, STOPPING PROGRAM from rank " << _rank << "\n";
        std::cout << e.what() << "\n";
    }
}

void adios_client_t::beginDataStep()
{
    _dataWriter.BeginStep();
}

void adios_client_t::endDataStep()
{
    _dataWriter.EndStep();
}

// Write one node field into the current step of the training data file.
void adios_client_t::putField(const std::string& name, dfloat *field, int num_dim)
{
    unsigned long field_num_dim = num_dim;
    auto it = _dataVars.find(name);
    if (it == _dataVars.end()) {
        auto var = _data_io.DefineVariable<dfloat>(
            name,
            {_global_field_offset * field_num_dim},
            {_offset_field_offset * field_num_dim},
            {_field_offset * field_num_dim});
        it = _dataVars.emplace(name, var).first;
    }
    _dataWriter.Put<dfloat>(it->second, field);
}

// Write a rank-0 scalar int
void adios_client_t::putScalar(const std::string& name, int value)
{
    auto it = _dataIntVars.find(name);
    if (it == _dataIntVars.end()) {
        auto var = _data_io.DefineVariable<int>(name, {1}, {0}, {1});
        it = _dataIntVars.emplace(name, var).first;
    }
    if (_rank == 0) {
        // Sync mode: `value` is a local and would be dangling by EndStep()
        // under the default deferred mode.
        _dataWriter.Put<int>(it->second, &value, adios2::Mode::Sync);
    }
}

// Write a rank-0 scalar float
void adios_client_t::putScalar(const std::string& name, dfloat value)
{
    auto it = _dataRealVars.find(name);
    if (it == _dataRealVars.end()) {
        auto var = _data_io.DefineVariable<dfloat>(name, {1}, {0}, {1});
        it = _dataRealVars.emplace(name, var).first;
    }
    if (_rank == 0) {
        _dataWriter.Put<dfloat>(it->second, &value, adios2::Mode::Sync);
    }
}

#endif // NEKRS_ENABLE_ADIOS
