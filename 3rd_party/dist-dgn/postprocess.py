"""
Postprocessing utilities for the DGN model.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_field(comm, pos: np.ndarray, field: np.ndarray, filename: str):
    """
    Plot a 2D field (useful for the ext_cyl example)
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size > 1:
        global_pos = comm.gather(pos, root=0)
        global_field = comm.gather(field, root=0)
        if rank == 0:
            global_pos = np.concatenate(global_pos, axis=0)
            global_field = np.concatenate(global_field, axis=0)
    else:
        global_pos = pos
        global_field = field

    if rank == 0:
        # sample pos and field at min_z
        min_z = np.amin(global_pos[:,2]); max_z = np.amax(global_pos[:,2])
        #print(f"Min z: {min_z}, Max z: {max_z}", flush=True)
        indices = np.where(global_pos[:, 2] <= (min_z+1.0e-6))[0]
        #print(f"Number of indeces to plot: {len(indices)}", flush=True)
        #field_sampled = field[indices]
        #pos_sampled = pos[indices]
        field_sampled = global_field
        pos_sampled = global_pos

        # scatter plot of the field (x, y, z velocities) in subplots
        n_vars = field_sampled.shape[1]
        fig, axs = plt.subplots(n_vars, 1, figsize=(10, 10))
        titles = ['x velocity', 'y velocity', 'z velocity']
        for i, ax in enumerate(axs):
            sc = ax.scatter(pos_sampled[:,0], pos_sampled[:, 1], 
                            c=field_sampled[:,i], cmap='viridis', s=1.2)
            ax.set_title(titles[i])
            fig.colorbar(sc, ax=ax)
        plt.savefig(filename)
        plt.close()

def plot_training_loss(log_file: str):
    """
    Plot the training loss from a log file
    """
    iterations = []
    losses = []
    running_losses = []
    mse_losses = []
    mse_steps = []  # Training steps corresponding to MSE losses
    vlb_losses = []
    per_step_mse_losses = {i: [] for i in range(100)}
    current_training_step = None
    last_diffusion_steps = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract step number and loss value
            iter_match = re.search(r'\[STEP (\d+)\]', line)
            loss_match = re.search(r'loss=([\d.e+-]+)', line)
            running_loss_match = re.search(r'r_loss=([\d.e+-]+)', line)            
            mse_loss_match = re.search(r'MSE loss term: \[([\d.,\s.e+-]+)\], mean = ([\d.e+-]+)', line)
            #mse_loss_match_old = re.search(r'MSE loss term: ([\d.e+-]+)$', line)
            vlb_loss_match = re.search(r'VLB loss term: ([\d.e+-]+)', line)            
            diffusion_steps_match = re.search(r'Sampled diffusion steps: \[([\d,\s]+)\]', line)
                
            if iter_match and loss_match:
                iteration = int(iter_match.group(1))
                loss = float(loss_match.group(1))
                iterations.append(iteration)
                losses.append(loss)
            if iter_match and running_loss_match:
                running_loss = float(running_loss_match.group(1))
                running_losses.append(running_loss)

            if diffusion_steps_match:
                steps_str = diffusion_steps_match.group(1)
                steps = [int(x.strip()) for x in steps_str.split(',')]
            
            if mse_loss_match:
                # Extract the mean
                mse_loss_mean = float(mse_loss_match.group(2))
                mse_losses.append(mse_loss_mean)
                # Extract the list of loss values
                loss_list_str = mse_loss_match.group(1)
                loss_values = [float(x.strip()) for x in loss_list_str.split(',')]
                for step, loss_val in zip(steps, loss_values):
                    per_step_mse_losses[step].append(loss_val)
            if vlb_loss_match:
                vlb_loss = float(vlb_loss_match.group(1))
                vlb_losses.append(vlb_loss)
                

    # Plot loss vs iterations
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, losses, label='Loss', linewidth=1)
    if len(mse_losses) > 0:
        plt.plot(iterations, mse_losses, label='MSE Loss', linewidth=1)
    if len(vlb_losses) > 0:
        plt.plot(iterations, vlb_losses, label='VLB Loss', linewidth=1)
    if len(running_losses) > 0:
        plt.plot(iterations, running_losses, label='Running Loss', linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Steps')
    plt.grid(True, alpha=0.3)
    #plt.yscale('log')
    plt.xlim(-100,2600)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=150)

    # Plot loss vs diffusion steps
    plt.figure(figsize=(8, 6))
    steps = sorted([s for s in per_step_mse_losses.keys() if len(per_step_mse_losses[s]) > 0])
    # Filter to plot only every step_interval steps
    steps_to_plot = steps[::10]
    for step in steps_to_plot:
        plt.plot([i for i in range(len(per_step_mse_losses[step]))], per_step_mse_losses[step], label=f'{step}', linewidth=1)
    plt.xlabel('Instances in training')
    plt.ylabel('Loss')
    plt.title('Training Loss for Diffusion Steps')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('loss_plot_diffusion_steps.png', dpi=150)


if __name__ == '__main__':
    # use argparse to get the log file
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['loss'], help='Task to perform')
    parser.add_argument('--log_file', type=str, default='train.log', help='Log file to plot')
    args = parser.parse_args()

    if args.task == 'loss':
        plot_training_loss(args.log_file)
