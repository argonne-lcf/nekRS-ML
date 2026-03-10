"""
Postprocessing utilities for the DGN model.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_2d_field(comm, pos: np.ndarray, field: np.ndarray, filename: str):
    """
    Plot a 2D field as filled contours (useful for the ext_cyl example).
    Uses Delaunay triangulation on the unstructured mesh points.
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
        # Sample pos and field at the min-z plane
        min_z = np.amin(global_pos[:, 2])
        indices = np.where(global_pos[:, 2] <= (min_z + 1.0e-6))[0]
        field_sampled = global_field[indices]
        pos_sampled = global_pos[indices]

        x = pos_sampled[:, 0]
        y = pos_sampled[:, 1]

        # Build Delaunay triangulation and mask out large triangles (e.g. across the cylinder hole)
        triang = tri.Triangulation(x, y)
        # Compute characteristic edge length per triangle and mask outliers
        triangles = triang.triangles
        xt = x[triangles]  # (ntri, 3)
        yt = y[triangles]
        # Edge lengths squared for each triangle's three edges
        d01 = (xt[:, 0] - xt[:, 1])**2 + (yt[:, 0] - yt[:, 1])**2
        d12 = (xt[:, 1] - xt[:, 2])**2 + (yt[:, 1] - yt[:, 2])**2
        d20 = (xt[:, 2] - xt[:, 0])**2 + (yt[:, 2] - yt[:, 0])**2
        max_edge2 = np.maximum(np.maximum(d01, d12), d20)
        # Mask triangles whose longest edge exceeds 4x the median edge length
        median_edge2 = np.median(max_edge2)
        triang.set_mask(max_edge2 > 5.0 * median_edge2)

        # Contour plot of each velocity component
        n_vars = field_sampled.shape[1]
        fig, axs = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))
        if n_vars == 1:
            axs = [axs]
        titles = ['x velocity', 'y velocity', 'z velocity']
        n_levels = 64
        for i, ax in enumerate(axs):
            vmin = np.percentile(field_sampled[:, i], 1)
            vmax = np.percentile(field_sampled[:, i], 99)
            levels = np.linspace(vmin, vmax, n_levels)
            tc = ax.tricontourf(triang, field_sampled[:, i], levels=levels,
                                cmap='RdBu_r', extend='both')
            fig.colorbar(tc, ax=ax, shrink=0.8)
            ax.set_title(titles[i])
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
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
    per_step_vlb_losses = {i: [] for i in range(100)}
    per_step_weighted_mse_losses = {i: [] for i in range(100)}
    current_training_step = None
    last_diffusion_steps = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract step number and loss value
            iter_match = re.search(r'\[STEP (\d+)\]', line)
            loss_match = re.search(r'loss=([\d.e+-]+)', line)
            running_loss_match = re.search(r'r_loss=([\d.e+-]+)', line)
            #mse_loss_match_old = re.search(r'MSE loss term: ([\d.e+-]+)$', line)
            #vlb_loss_match_old = re.search(r'VLB loss term: ([\d.e+-]+)', line)        
            mse_loss_match = re.search(r'MSE loss term: \[([\d.,\s.e+-]+)\], mean = ([\d.e+-]+)', line) 
            vlb_loss_match = re.search(r'VLB loss term: \[([\d.,\s.e+-]+)\], mean = ([\d.e+-]+)', line)
            weighted_mse_match = re.search(r'Weighted MSE loss: \[([\d.,\s.e+-]+)\], mean = ([\d.e+-]+)', line)
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
                # Extract the mean
                vlb_loss_mean = float(vlb_loss_match.group(2))
                vlb_losses.append(vlb_loss_mean)
                # Extract the list of loss values
                loss_list_str = vlb_loss_match.group(1)
                loss_values = [abs(float(x.strip())) for x in loss_list_str.split(',')]
                for step, loss_val in zip(steps, loss_values):
                    per_step_vlb_losses[step].append(loss_val)
            if weighted_mse_match:
                # Extract the list of weighted loss values
                loss_list_str = weighted_mse_match.group(1)
                loss_values = [float(x.strip()) for x in loss_list_str.split(',')]
                for step, loss_val in zip(steps, loss_values):
                    per_step_weighted_mse_losses[step].append(loss_val)

    # Adjust the lengths of the loss lists to match the number of iterations
    if len(mse_losses) > len(iterations):
        mse_losses.pop(-1)
    if len(vlb_losses) > len(iterations):
        vlb_losses.pop(-1)

    # Plot loss vs iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, label='Loss', linewidth=1)
    if len(running_losses) > 0:
        plt.plot(iterations, running_losses, label='Running Loss', linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    #plt.xlim(-100,4100)
    plt.title('Training Loss vs Iterations')
    if np.max(losses) / np.min(losses) > 10:
        plt.yscale('log')
    plt.subplot(1, 2, 2)
    if len(mse_losses) > 0:
        plt.plot(iterations, mse_losses, label='MSE Loss', linewidth=1)
    if len(vlb_losses) > 0:
        plt.plot(iterations, vlb_losses, label='VLB Loss', linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Components vs Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=150)

    # Plot loss vs diffusion steps 
    # Make subplots for weighted MSE and VLB losses
    has_weighted = any(len(v) > 0 for v in per_step_weighted_mse_losses.values())
    has_vlb = any(len(v) > 0 for v in per_step_vlb_losses.values())
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    if has_weighted:
        w_steps = sorted([s for s in per_step_weighted_mse_losses.keys() if len(per_step_weighted_mse_losses[s]) > 0])
        w_steps_to_plot = w_steps[::10]
        for step in w_steps_to_plot:
            plt.plot(range(len(per_step_weighted_mse_losses[step])), per_step_weighted_mse_losses[step], label=f'{step}', linewidth=1)
        plt.ylabel('Weighted MSE Loss')
        plt.title('Weighted MSE Loss for Diffusion Steps')
    else:
        mse_steps = sorted([s for s in per_step_mse_losses.keys() if len(per_step_mse_losses[s]) > 0])
        mse_steps_to_plot = mse_steps[::10]
        for step in mse_steps_to_plot:
            plt.plot(range(len(per_step_mse_losses[step])), per_step_mse_losses[step], label=f'{step}', linewidth=1)
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss for Diffusion Steps')
    plt.xlabel('Instances in training')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.subplot(1, 2, 2)
    vlb_steps = sorted([s for s in per_step_vlb_losses.keys() if len(per_step_vlb_losses[s]) > 0])
    vlb_steps_to_plot = vlb_steps[::10]
    for step in vlb_steps_to_plot:
        plt.plot(range(len(per_step_vlb_losses[step])), per_step_vlb_losses[step], label=f'{step}', linewidth=1)
    plt.xlabel('Instances in training')
    plt.ylabel('VLB Loss')
    plt.title('VLB Loss for Diffusion Steps')
    plt.grid(True, alpha=0.3)
    #plt.legend(loc='upper right')
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
