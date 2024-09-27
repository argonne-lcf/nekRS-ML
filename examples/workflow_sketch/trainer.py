import os
import numpy as np
from time import sleep

from smartredis import Client

LIST_NAME = 'training_list'

## Initialize Redis clients
def init_client(SSDB, args):
    if (args.dbnodes==1):
        client = Client(address=SSDB,cluster=False)
    else:
        client = Client(address=SSDB,cluster=True)
    return client


## Define the Neural Network Structure
class NeuralNetwork(nn.Module):
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim, outputDim, numNeurons):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.ndOut),
        )

    # Define the method to do a forward pass
    def forward(self, x):
        return self.net(x)

## Training subroutine
def train(comm, model, train_sampler, train_loader, optimizer, epoch, 
          batch, ndIn, client, args, logger_data):
    rank = comm.Get_rank()
    size = comm.Get_size()

    model.train()
    running_loss = 0.0
    train_sampler.set_epoch(epoch)

    loss_fn = nn.functional.mse_loss

    for batch_idx, dbdata in enumerate(train_loader):
        # with this a small model, slow down training a little for purpses of example problem
        sleep(0.01)

        # split inputs and outputs
        if (args.device != 'cpu'):
            dbdata = dbdata.to(args.device)
        features = dbdata[:, :ndIn]
        target = dbdata[:, ndIn:]

        optimizer.zero_grad()
        output = model.forward(features)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if ((batch_idx)%10==0):
            print(f'Train Epoch: {epoch} | ' + \
                      f'[{tensor_idx+1}/{len(train_tensor_loader)}] | ' + \
                      f'[{batch_idx+1}/{len(train_loader)}] | ' + \
                      f'Loss: {loss.item():>8e}', flush=True)

    running_loss = running_loss / len(train_loader)
    loss_avg = metric_average(comm, size, running_loss)

    if rank == 0:
        print(f"Training set: Average loss: {loss_avg:>8e}", flush=True)

    return model, loss_avg


## Average across hvd ranks
def metric_average(comm, size, val):
    avg_val = comm.allreduce(val, op=MPI.SUM)
    avg_val = avg_val / size
    return avg_val


## Define the dataset
class MinibDataset(torch.utils.data.Dataset):
    def __init__(self,concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]


## Main
def main():
    # MPI import and initialization
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    rankl = int(os.getenv("PALS_LOCAL_RANKID"))
    print(f'Rank {rank}/{size}, local rank {rankl} says hello from {name}', flush=True)
    comm.Barrier()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    args = parser.parse_args()

    # Initialize Torch Distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (args.device=='cpu'): backend = 'gloo'
    elif (args.device=='cuda'): backend = 'nccl'
    dist.init_process_group(backend,
                            rank=int(rank),
                            world_size=int(size),
                            init_method='env://',
                            timeout=datetime.timedelta(seconds=120))

    # Initialize Redis clients on each rank
    address = os.getenv('SSDB')
    client = init_client(address, args)
    comm.Barrier()
    if (rank == 0):
        print("All Python clients initialized\n", flush=True)

    # NN Training Hyper-Parameters
    ndIn = 1
    ndOut = 1
    Nepochs = 100 # number of epochs
    mini_batch = 16 # batch size once tensors obtained from db and concatenated 
    learning_rate = 0.001 # learning rate
    nNeurons = 20 # number of neuronsining settings
    tol = 1.0e-7 # convergence tolerance on loss function

    # Set device to run on
    if (rank == 0):
        print(f"\nRunning on device: {args.device} \n")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            device_id = rankl if torch.cuda.device_count()>1 else 0
            torch.cuda.set_device(device_id)

    # Instantiate the NN model and optimizer
    model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons)
    if (args.device != 'cpu'):
        model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate*size)
    
    # Wrap model with DDP
    model = DDP(model) 

    # Training setup and variable initialization
    iepoch = 1 # epoch number

    # Check to see if traing data list has length >0, if not cycle
    while True:
        list_length = client.get_list_length(LIST_NAME)
        if list_length == 0:
            sleep(10)
            continue
        else:
            break

    # While loop that checks when training data is available on database
    if (rank == 0):
        print("Starting training loop ... \n")
    while True:
        datasets = client.get_datasets_from_list(LIST_NAME)
        print(f'Read list with length {len(datasets)}',flush=True)
        data = np.zeros((len(datasets),2))
        for i in range(len(datasets)):
            data[i] = datasets[i].get_tensor('train')
        print('Current training data: \n',data,'\n',flush=True)

        datasetTrain = MiniBatchDataset(torch.from_numpy(data))
        train_sampler = DistributedSampler(datasetTrain, num_replicas=size, 
                                               rank=rank, drop_last=False)
        train_tensor_loader = DataLoader(datasetTrain, batch_size=mini_batch, 
                                             sampler=train_sampler)
        
        if (rank == 0):
            print(f"\n Epoch {iepoch}\n-------------------------------", flush=True)
        
        model, global_loss = train(comm, model, train_sampler, train_tensor_loader, optimizer,
                                    iepoch, mini_batch, ndIn, client, args, logger_data)
            
        # check if tolerance on loss is satisfied
        if (global_loss <= tol):
            if (rank == 0):
                print("Convergence tolerance met. Stopping training loop. \n", flush=True)
            break
        
        # check if max number of epochs is reached
        if (iepoch >= Nepochs):
            if (rank == 0):
                print("Max number of epochs reached. Stopping training loop. \n", flush=True)
            break

        iepoch = iepoch + 1        
        
    # Save model to file before exiting
    model = model.module
    model.eval()
    if (rank == 0):
        model.double()
        model_name = "model"
        torch.save(model.state_dict(), f"{model_name}.pt", _use_new_zipfile_serialization=False)
        # save jit traced model to be used for online inference with SmartSim
        features = np.double(np.random.uniform(low=0, high=10, size=(npts,ndIn)))
        features = torch.from_numpy(features).to(args.device)
        module = torch.jit.trace(model, features)
        torch.jit.save(module, f"{model_name}_jit.pt")
        print("Saved model to disk\n", flush=True)


    # Exit
    comm.Barrier()
    dist.destroy_process_group()

    if (rank==0):
        print("Exiting ...")
    

###
if __name__ == '__main__':
    main()
