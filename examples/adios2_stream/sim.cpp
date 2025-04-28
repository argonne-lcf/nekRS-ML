#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>

#include "adios2.h"
#include <mpi.h>


int check_run(MPI_Comm comm, adios2::IO bpIO)
{
    int exit_val = 1;
    int exists;
    std::string fname = "check-run.bp";
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Check if check-run file exists
    if (rank == 0) {
        if (std::filesystem::exists(fname)) {
            printf("Found check-run file!\n");
            fflush(stdout);
            exists = 1;
        } else {
            exists = 0;
        }
    }
    MPI_Bcast(&exists, 1, MPI_INT, 0, comm);

    // Read check-run file if exists
    if (exists) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        adios2::Engine reader = bpIO.Open(fname, adios2::Mode::Read);
        reader.BeginStep();
        adios2::Variable<int> var = bpIO.InquireVariable<int>("check-run");
        if (rank == 0 and var) {
            reader.Get(var, &exit_val);
        }
        reader.EndStep();
        reader.Close();
        MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);
    }

    if (exit_val == 0 && rank == 0) {
        printf("ML training says time to quit ...\n");
    }
    fflush(stdout);

    return exit_val;
}


int main(int argc, char *argv[])
{

    int rank;
    int size;
    int provide;

    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;
    if (rank == 0) {
        std::cout << "Running with " << size << " MPI ranks \n" << std::endl;
    }

    try
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        adios2::IO bpIO = adios.DeclareIO("graphStream");
        adios2::IO sstIO = adios.DeclareIO("solutionStream");
        sstIO.SetEngine("Sst");
        adios2::Params params;
        // sync setup
        //params["RendezvousReaderCount"] = "1"; // proceed only when 1 reader is present, blocking
        //params["QueueFullPolicy"] = "Block";
        //params["QueueLimit"] = "1"; // number of steps writes allows to be queued before taking action
        // async setup
        params["RendezvousReaderCount"] = "1"; // proceed even if no reader is present, non-blocking
        params["QueueFullPolicy"] = "Discard"; // (Block,Discard) action to perform when queue is full
        params["QueueLimit"] = "3"; // number of steps writer allows to be queued before taking action (0 means no limit)
        params["ReserveQueueLimit"] = "0"; // number of steps writer allows to be queued before taking action when no reader is connected (0 means no limit)
        //*/
        params["DataTransport"] = "RDMA";
        params["OpenTimeoutSecs"] = "600";
        sstIO.SetParameters(params);

        // Define graph data
        int N = 100000 + rank;
        int num_edges = 200000 + rank;
        double *pos_node = new double[N * 3](); 
        int *edge_index = new int[num_edges * 2]();

        for (int n=0; n<N; n++) {
            pos_node[n + 0*N] = static_cast<double>(n+N*0);
            pos_node[n + 1*N] = static_cast<double>(n+N*1);
            pos_node[n + 2*N] = static_cast<double>(n+N*2);
        }
        for (int n=0; n<num_edges; n++) {
            edge_index[n + 0*num_edges] = n+num_edges*0;
            edge_index[n + 1*num_edges] = n+num_edges*1;
        }

        // Get global size of data
        int global_N, global_num_edges;
        MPI_Allreduce(&N, &global_N, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&num_edges, &global_num_edges, 1, MPI_INT, MPI_SUM, comm);

        // Gather size of data
        int* gathered_N = new int[size];
        int* gathered_num_edges = new int[size];
        MPI_Allgather(&N, 1, MPI_INT, gathered_N, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&num_edges, 1, MPI_INT, gathered_num_edges, 1, MPI_INT, MPI_COMM_WORLD);

        // Get global offset
        int offset_N = 0;
        int offset_num_edges = 0;
        for (int i=0; i<rank; i++) {
            offset_N += gathered_N[i];
            offset_num_edges += gathered_num_edges[i];
        }

        // Define ADIOS2 variables and send
        unsigned long _size = size;
        unsigned long _rank = rank;
        unsigned long _N = N;
        unsigned long _global_N = global_N;
        unsigned long _offset_N = offset_N;
        unsigned long _num_edges = num_edges;
        unsigned long _global_num_edges = global_num_edges;
        unsigned long _offset_num_edges = offset_num_edges;
        auto posVar = bpIO.DefineVariable<double>("pos_node", 
                                                  {_global_N * 3}, // global dim
                                                  {_offset_N * 3}, // starting offset in global dim
                                                  {_N * 3}); // local size
        auto edgeVar = bpIO.DefineVariable<int>("edge_index", 
                                                {2 * _global_num_edges}, 
                                                {2 * _offset_num_edges}, 
                                                {2 * _num_edges});
        auto NVar = bpIO.DefineVariable<int>("N", {_size}, {_rank}, {1});
        auto numedgesVar = bpIO.DefineVariable<int>("num_edges", {_size}, {_rank}, {1});

        adios2::Engine graphWriter = bpIO.Open("graph.bp", adios2::Mode::Write);
        graphWriter.BeginStep();

        graphWriter.Put<int>(NVar, N);
        graphWriter.Put<int>(numedgesVar, num_edges);
        graphWriter.Put<double>(posVar, pos_node);
        graphWriter.Put<int>(edgeVar, edge_index);

        graphWriter.EndStep();
        graphWriter.Close();
        MPI_Barrier(comm);
        if (rank == 0) std::cout << "Done sending graph data" << std::endl;

        // Setup iteration loop
        int iters = 500;
        double *U = new double[N * 3]();
        auto UVar = sstIO.DefineVariable<double>("U", 
                                                {_global_N * 3}, 
                                                {_offset_N * 3}, 
                                                {_N * 3});
        // Open stream before the iter loop
        if (rank == 0) {
            std::cout << "[Sim] Opening stream ... " << std::endl;
        }
        adios2::Engine solWriter = sstIO.Open("solutionStream", adios2::Mode::Write);
        for (int iter=0; iter<iters; iter++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            int exit_val = check_run(comm, bpIO);
            if (exit_val == 0) {
                break;
            }

            double frac = (iter != 0) ? (1.0 / iter) : 0.0;
            for (int n=0; n<N; n++) {
                U[n + 0*N] = static_cast<double>(n+N*0+frac);
                U[n + 1*N] = static_cast<double>(n+N*1+frac);
                U[n + 2*N] = static_cast<double>(n+N*2+frac);
            }

            if (rank == 0) {
                std::cout << "[Sim] Sending data for step " << iter << std::endl;
            }
            solWriter.BeginStep();
            solWriter.Put<double>(UVar, U);
            solWriter.EndStep();
            MPI_Barrier(comm);
            if (rank == 0) {
                std::cout << "[Sim] Done writing solution data for step " << iter << std::endl;
            }
        }
        solWriter.Close();
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM from rank " << rank
                  << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }

    MPI_Finalize();

    return 0;
}

