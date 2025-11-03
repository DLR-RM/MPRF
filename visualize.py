import open3d as o3d
import os

def main():

    # Change to your candidate_path-derived base name
    base = "img1625660046.0920656"
    save_dir = "./Results/Plots"

    # File paths
    pcd1_path = os.path.join(save_dir, f"{base}_pcd1.ply")
    pcd2_path = os.path.join(save_dir, f"{base}_pcd2.ply")
    line_path = os.path.join(save_dir, f"{base}_lines.ply")
    camera_path = os.path.join(save_dir, f"{base}camera.json")

    # Load geometries
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    line_set = o3d.io.read_line_set(line_path)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(line_set)

    vis.run()  # 🟢 Interact and close manually
    vis.destroy_window()

if __name__ == "__main__":
    main()