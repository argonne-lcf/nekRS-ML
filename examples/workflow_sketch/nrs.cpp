#include "client.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <mpi.h>
#include <random>

//using namespace SmartRedis;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Hello from rank " << rank << "/" << size << std::endl;

    // Initialize SR Client
//    const char* SSDB = std::getenv("SSDB");
//    SmartRedis::Client client(SSDB, false);
    SmartRedis::Client *client_ptr;
//    Client *client_ptr;
    std::string logger_name("Client");
    client_ptr = new SmartRedis::Client(false, logger_name); // allocates on heap

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\nAll clients initialized\n" << std::endl;

    // Get input data from DB
    std::vector<double> vel_dist(3,0);
    client_ptr->unpack_tensor("vel_dist", vel_dist.data(), {3},
		  SRTensorTypeDouble, SRMemLayoutContiguous);

    if (vel_dist[0] < 0.5) {
        if (rank == 0) std::cout << "\nRead velocity distribution of uniform type in range " << vel_dist[1] << " - " << vel_dist[2] << "\n" << std::endl;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(vel_dist[1], vel_dist[2]);
    float inflow_vel = dis(gen);

    // Pretend to do some calculations for a few time steps
    for (int step = 0; step < 5; ++step) {
        sleep(5);
        std::uniform_real_distribution<> temp_dis(0.0, 1.0);
        float maxT = temp_dis(gen);
        if (rank == 0) std::cout << "\nComputed max Temp = " << maxT << "\n" << std::endl;

        // From rank 0, create a DataSet with training data and append to a list
        if (rank == 0) {
	    SmartRedis::DataSet train_dataset("train_data_" + std::to_string(maxT));
	    //DataSet train_dataset("train_data_" + std::to_string(maxT));
            //train_dataset.add_tensor("train", std::vector<float>{inflow_vel, maxT});

            double *train_data = new double[2]();
            train_data[0] = inflow_vel;
            train_data[1] = maxT;
            train_dataset.add_tensor("train",  train_data, {2}, SRTensorTypeDouble, SRMemLayoutContiguous);
            client_ptr->put_dataset(train_dataset);
            client_ptr->append_to_list("training_list", train_dataset);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Exit
    if (rank == 0) std::cout << "\nExiting ...\n" << std::endl;
    MPI_Finalize();
    return 0;
}

