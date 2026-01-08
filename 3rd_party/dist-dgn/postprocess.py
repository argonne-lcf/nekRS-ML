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
    steps = []
    losses = []
    running_losses = []
    with open(log_file, 'r') as f:
        for line in f:
            # Extract step number and loss value
            step_match = re.search(r'\[STEP (\d+)\]', line)
            loss_match = re.search(r'loss=([\d.e+-]+)', line)
            running_loss_match = re.search(r'r_loss=([\d.e+-]+)', line)

            if step_match and loss_match and running_loss_match:
                step = int(step_match.group(1))
                loss = float(loss_match.group(1))
                running_loss = float(running_loss_match.group(1))
                steps.append(step)
                running_losses.append(running_loss)
                losses.append(loss)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(steps, losses, label='Loss', linewidth=1)
    plt.plot(steps, running_losses, label='Running Loss', linewidth=1)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Steps')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=150)


if __name__ == '__main__':
    # use argparse to get the log file
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['loss'], help='Task to perform')
    parser.add_argument('--log_file', type=str, default='log.txt', help='Log file to plot')
    args = parser.parse_args()

    if args.task == 'loss':
        plot_training_loss(args.log_file)