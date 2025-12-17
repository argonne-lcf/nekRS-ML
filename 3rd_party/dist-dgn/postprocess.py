"""
Postprocessing utilities for the DGN model.
"""

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