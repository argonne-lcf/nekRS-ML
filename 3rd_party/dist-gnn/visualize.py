import numpy as np
import pyvista as pv
pv.start_xvfb()
import matplotlib.pyplot as plt
import time

root_dir = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekRS-GNN-devel/3rd_party/gnn"

# Load mesh 
#mesh = point_cloud.delaunay_3d()
#mesh = pv.read("/Users/sbarwey/Files/gmsh_files/bfs_nek/bfs.msh")
mesh = pv.read(f"{root_dir}/outputs/meshes/bfs_nek/bfs.msh")

# visualize mesh 
# p = pv.Plotter()
# p.add_mesh(mesh, show_edges=True, color='white')  # You can customize appearance
# p.show_bounds(
#     grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
#     location='outer',      # Options: 'all', 'front', 'back', 'outer'.
#     all_edges=True,        # Show all edges of the bounding box
#     #corner_factor=0.5,     # Fractional position of the labels along the axis
#     xtitle='X Axis',       # Label for the X-axis
#     ytitle='Y Axis',       # Label for the Y-axis
#     ztitle='Z Axis',       # Label for the Z-axis
#     fmt="%.2f",            # Format for the tick labels
#     font_size=12,          # Font size of the labels
#     color='black',         # Color of the labels and ticks
#     show_xlabels=True,     # Show X-axis labels
#     show_ylabels=True,     # Show Y-axis labels
#     show_zlabels=True,     # Show Z-axis labels
# )
# p.add_axes()
# p.show()

# Load 3d coordinates and velocity field  
#model_str = "POLY_3_SIZE_32_SEED_64_3_4_64_3_2_4_all_to_all_opt"
#model_str = "POLY_3_SIZE_32_SEED_64_3_4_128_3_2_4_all_to_all_opt"
#model_str = "POLY_3_SIZE_32_SEED_64_3_4_256_3_2_4_all_to_all_opt"
#model_str = "POLY_3_SIZE_32_SEED_64_3_4_64_3_2_8_all_to_all_opt"
model_str = "POLY_3_SIZE_32_SEED_64_3_4_128_3_2_8_all_to_all_opt"
data_path = f"{root_dir}/outputs/inference/" + model_str
N_snaps = 100
for i in range(N_snaps):
    print(f"step {i}")
    pos_path = data_path + f"/pos_{i}.npy"
    target_path = data_path + f"/target_{i}.npy"
    pred_path = data_path + f"/pred_{i}.npy"
    error_path = data_path + f"/error_{i}.npy"

    pos = np.load(pos_path)
    target = np.load(target_path)
    pred = np.load(pred_path)
    error = np.load(error_path)

    # 1. create a pyvista point cloud 
    point_cloud = pv.PolyData(pos)
    point_cloud['target_x'] = target[:,0]
    point_cloud['target_y'] = target[:,1]
    point_cloud['target_z'] = target[:,2]
    point_cloud['target_mag'] = np.linalg.norm(target, axis=1)

    point_cloud['pred_x'] = pred[:,0]
    point_cloud['pred_y'] = pred[:,1]
    point_cloud['pred_z'] = pred[:,2]
    point_cloud['pred_mag'] = np.linalg.norm(pred, axis=1)

    point_cloud['error_x'] = error[:,0]
    point_cloud['error_y'] = error[:,1]
    point_cloud['error_z'] = error[:,2]
    point_cloud['error_mag'] = np.linalg.norm(error, axis=1)

    print("interpolating...")
    t_interp = time.time()
    imesh = mesh.interpolate(point_cloud, n_points=6)
    #imesh = mesh.interpolate(point_cloud, radius=0.2)
    t_interp = time.time() - t_interp
    print(f"interpolation took {t_interp} sec")

    # # 3. extract contours 
    # scalars = 'vel_mag'
    # iso_values = [0.1, 0.2, 0.3]
    # contours = imesh.contour(isosurfaces=iso_values, scalars=scalars)

    # ~~~~ # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ # # 4. visualize contours 
    # ~~~~ # plotter = pv.Plotter()
    # ~~~~ # plotter.add_mesh(
    # ~~~~ #     contours,
    # ~~~~ #     scalars='vel_mag',
    # ~~~~ #     cmap='viridis',
    # ~~~~ #     opacity=0.6,
    # ~~~~ #     show_scalar_bar=True,
    # ~~~~ # )
    # ~~~~ # 
    # ~~~~ # # Optionally add the original point cloud for reference
    # ~~~~ # plotter.add_mesh(
    # ~~~~ #     imesh,
    # ~~~~ #     color='white',
    # ~~~~ #     show_edges=True,
    # ~~~~ #     edge_color="black",
    # ~~~~ #     opacity=0.2,
    # ~~~~ # )
    # ~~~~ # 
    # ~~~~ # # Display axes and show the plot
    # ~~~~ # plotter.show_bounds(
    # ~~~~ #     grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
    # ~~~~ #     location='outer',      # Options: 'all', 'front', 'back', 'outer'.
    # ~~~~ #     all_edges=True,        # Show all edges of the bounding box
    # ~~~~ #     #corner_factor=0.5,     # Fractional position of the labels along the axis
    # ~~~~ #     xtitle='X Axis',       # Label for the X-axis
    # ~~~~ #     ytitle='Y Axis',       # Label for the Y-axis
    # ~~~~ #     ztitle='Z Axis',       # Label for the Z-axis
    # ~~~~ #     fmt="%.2f",            # Format for the tick labels
    # ~~~~ #     font_size=12,          # Font size of the labels
    # ~~~~ #     color='black',         # Color of the labels and ticks
    # ~~~~ #     show_xlabels=True,     # Show X-axis labels
    # ~~~~ #     show_ylabels=True,     # Show Y-axis labels
    # ~~~~ #     show_zlabels=True,     # Show Z-axis labels
    # ~~~~ # )
    # ~~~~ # plotter.add_axes()
    # ~~~~ # plotter.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot planar surfaces 

    #feature_id_list = ["target_x", "pred_x", "error_x"]
    feature_id_list = ["target_x", "pred_x", "error_x",
                       "target_y", "pred_y", "error_y",
                       "target_z", "pred_z", "error_z"]

    feature_id_list = ["error_x", "error_y", "error_z"]

    for feature_id in feature_id_list:
        print(f"\tplotting feature: {feature_id}")
        # Example: Slice at z = 0.5
        origin = (0, 0, 1)
        normal = (0, 0, 1)

        # Extract the slice
        slice_plane = imesh.slice(
            origin=origin,
            normal=normal,
            generate_triangles=True  # Ensures the output is a triangulated surface
        )

        # Initialize the plotter
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Add the slice to the plotter
        plotter.add_mesh(
            slice_plane,
            clim = [-0.05,0.05],
            scalars=feature_id,
            cmap='seismic',  # Choose a colormap
            show_scalar_bar=True,
            show_edges=False
        )

        # Set the camera to look along the Z-axis
        slice_origin = slice_plane.center
        z_pos = origin[2]  # Z position of your plane
        distance = 35      # Distance from the plane (adjust as needed)
        plotter.camera_position = [
            (slice_origin[0], slice_origin[1], z_pos + distance),  # Camera position above the plane
            (slice_origin[0], slice_origin[1], z_pos),             # Focal point (center of the plane)
            (0, 1, 0),                                 # View-up vector
        ]

        # Display axes and show the plot
        plotter.show_bounds(
            grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
            location='outer',      # Options: 'all', 'front', 'back', 'outer'.
            all_edges=True,        # Show all edges of the bounding box
            corner_factor=0.5,     # Fractional position of the labels along the axis
            xtitle='X Axis',       # Label for the X-axis
            ytitle='Y Axis',       # Label for the Y-axis
            ztitle='Z Axis',       # Label for the Z-axis
            fmt="%.2f",            # Format for the tick labels
            font_size=12,          # Font size of the labels
            color='black',         # Color of the labels and ticks
            show_xlabels=True,     # Show X-axis labels
            show_ylabels=True,     # Show Y-axis labels
            show_zlabels=True,     # Show Z-axis labels
        )
        plotter.add_axes()
        plotter.add_text(f"Step {i} [{feature_id}]", position="upper_left", font_size=20, color='black')
        plotter.show(screenshot=f"{data_path}/{feature_id}_{i}.png")
