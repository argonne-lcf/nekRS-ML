#if !defined(nekrs_trajGen_hpp_)
#define nekrs_trajGen_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <filesystem>
#ifdef NEKRS_ENABLE_SMARTREDIS
#include "smartRedis.hpp"
#endif
#include "adiosStreamer.hpp"

void deleteDirectoryContents(const std::filesystem::path& dir);


class trajGen_t 
{
public:
    trajGen_t(nrs_t *nrs, int dt_factor_, int skip_, dfloat time_init_);
    ~trajGen_t(); 

    // public variables
    std::string writePath;
    dfloat time_init;
    int dt_factor;
    int skip;
    bool first_step = true;
    std::string irank, nranks;
    dfloat *previous_U = 0, *U = 0;
    dfloat /* *previous_P, */ *P = 0;
    //int previous_tstep;
    
    // member functions 
    void trajGenSetup();
    void trajGenWrite(dfloat time, int tstep, const std::string& field_name);
#ifdef NEKRS_ENABLE_SMARTREDIS
    void trajGenWriteDB(smartredis_client_t* client, 
                        dfloat time, 
                        int tstep, 
                        const std::string& field_name);
#endif
    void trajGenWriteADIOS(adios_client_t* client,
                           dfloat time, 
                           int tstep, 
                           const std::string& field_name);

private:
    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // MPI stuff 
    int rank;
    int size;

    // for prints 
    bool verbose = false; 

    // for writing 
    bool write = true;
};

#endif
