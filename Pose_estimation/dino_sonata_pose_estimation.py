import numpy as np
import pandas as pd
import torch
import os
import open3d as o3d
import pickle
import time
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity,  euclidean_distances
from scipy.spatial.transform import Rotation as R
from store_sonata_descriptors import apply_transform, load_model, run_inference, load_pointcloud_from_array, extract_final_features
from store_dino_feat import load_model_v2 as load_dino_model
from preprocess_img import enhance_contrast, preprocess_image
import csv
import vis

def find_pointcloud(dataset, query_path):
    df = pd.read_pickle(dataset)
    # Find the row where 'img_path' matches query_path
    match = df[df['img_path'] == query_path]
    if match.empty:
        raise ValueError(f"No entry found for query path: {query_path}")
    # Return the corresponding point_cloud (assumes unique match)
    return match.iloc[0]['point_cloud']

def extract_patchwise_features(model, image_tensor, device):
    """Extracts patch-wise normalized feature vectors (one per 14x14 patch)."""
    image_tensor = image_tensor.to(device)  # Move image to the model's device

    with torch.no_grad():
        features = model.get_intermediate_layers(image_tensor, n=3)
        features = [feat[:, 1:] for feat in features]  # remove CLS token
        features = torch.cat(features, dim=-1)  # (1, n_patches, feature_dim_total)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features[0].cpu().numpy()  # (n_patches, feature_dim)

def map_dino_to_points(pts_cam, K, dino_features, img_shape=(224, 224), patch_size=14):
    H, W = img_shape
    n_patches_y = H // patch_size
    n_patches_x = W // patch_size

    expected_patches = n_patches_y * n_patches_x
    if dino_features.shape[0] == expected_patches - 1:
        dino_features = np.vstack([dino_features, np.zeros((1, dino_features.shape[1]))])

    assert dino_features.shape[0] == expected_patches

    # Project to image plane
    u = (K[0, 0] * pts_cam[0, :] / pts_cam[2, :]) + K[0, 2]
    v = (K[1, 1] * pts_cam[1, :] / pts_cam[2, :]) + K[1, 2]

    # Scale to 224x224
    u = u * img_shape[1] / 688
    v = v * img_shape[0] / 512

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts_cam[2, :] > 0)
    if np.sum(valid) == 0:
        print("No valid projections.")
        return None, None, None, None

    u = u[valid].astype(int)
    v = v[valid].astype(int)
    pts_cam_valid = pts_cam[:, valid]
    pts_cam_invalid = pts_cam[:, ~valid]

    patch_x = u // patch_size
    patch_y = v // patch_size
    patch_x = np.clip(patch_x, 0, n_patches_x - 1)
    patch_y = np.clip(patch_y, 0, n_patches_y - 1)

    patch_indices = patch_y * n_patches_x + patch_x
    pointwise_dino = dino_features[patch_indices]  # (N_valid, D)

    return pts_cam_valid, pts_cam_invalid, pointwise_dino, valid


def get_pca_color(feat, center=True):
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat).float()  # Convert to tensor
    u, s, v = torch.pca_lowrank(feat, center=center, q=3, niter=5)
    projection = feat @ v
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div
    return color

def compute_matches(feat1, feat2, threshold=0.90):
    # Compute cosine similarity matrix (shape: [N1, N2])
    sim = cosine_similarity(feat1, feat2)
    # Convert to a cost matrix by negating similarity (because Hungarian finds min-cost)
    cost = -sim
    # Hungarian algorithm for optimal 1-to-1 assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # Get similarity scores for the selected pairs
    scores = sim[row_ind, col_ind]
    # Filter by similarity threshold
    mask = scores >= threshold
    idx1 = row_ind[mask]
    idx2 = col_ind[mask]
    filtered_scores = scores[mask]

    return idx1, idx2, filtered_scores

def match_with_lowes_ratio(query_descriptors, target_descriptors, ratio_thresh=0.8):
    sim_matrix = cosine_similarity(query_descriptors, target_descriptors)  # shape (N, M)

    matches = []
    for i, row in enumerate(sim_matrix):
        sorted_idx = row.argsort()[::-1]  # descending similarity
        best_idx = sorted_idx[0]
        second_best_idx = sorted_idx[1]

        if row[best_idx] > ratio_thresh * row[second_best_idx]:
            matches.append((i, best_idx))
    return matches

def estimate_pose_ransac(pts1, pts2):
    # Convert to Open3D PointClouds
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    corres = np.array([[i, i] for i in range(len(pts1))])
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=pcd1, target=pcd2,
        corres=o3d.utility.Vector2iVector(corres),
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

def refine_pose_icp(source_points, target_points, init_transform):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.05,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result_icp.transformation, result_icp.fitness


def extract_yaw_angle(transformation_matrix):
    # Extract rotation matrix and make it writable
    rotation_matrix = np.array(transformation_matrix[:3, :3], dtype=float, copy=True)

    # Convert to Euler angles (yaw is rotation around Z axis in 'zyx' order)
    rotation = R.from_matrix(rotation_matrix)
    yaw, pitch, roll = rotation.as_euler('zyx', degrees=True)  # use degrees=False for radians

    return yaw

def visualize_matches_open3d(query_path, candidate_path, coords1, coords2, idx1, idx2, combined_feat1, combined_feat2, results_folder, offset=np.array([[0.0, 7.0, 0.0]])):
    """
    Visualize two point clouds and their matches using Open3D.

    Args:
        coords1: (N1, 3) numpy array of first point cloud coordinates
        coords2: (N2, 3) numpy array of second point cloud coordinates
        idx1: matched indices in coords1
        idx2: matched indices in coords2
        combined_feat1: features for coords1 (for PCA-based coloring)
        combined_feat2: features for coords2
        offset: offset to apply to coords2 for side-by-side visualization
    """
    # Shift second cloud for side-by-side viewing
    coords2_shifted = coords2 + offset

    # PCA-based color (same as before)
    color1 = get_pca_color(combined_feat1, center=True).cpu().numpy()
    color2 = get_pca_color(combined_feat2, center=True).cpu().numpy()

    # Create Open3D PointClouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(coords1)
    pcd1.colors = o3d.utility.Vector3dVector(color1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(coords2_shifted)
    pcd2.colors = o3d.utility.Vector3dVector(color2)

    # Create lines for matches
    match_lines = []
    match_colors = []
    all_points = np.vstack((coords1, coords2_shifted))
    for i1, i2 in zip(idx1, idx2):
        start = i1
        end = coords1.shape[0] + i2  # account for stacking
        match_lines.append([start, end])
        match_colors.append([0, 1, 0])  # green

    # Build Open3D line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(match_lines)
    line_set.colors = o3d.utility.Vector3dVector(match_colors)

    # Visualize
    #o3d.visualization.draw_geometries([pcd1, pcd2, line_set])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()

    # Save point clouds and line set
    query_filename = os.path.splitext(os.path.basename(query_path))[0].replace(".", "")
    save_dir = results_folder+query_filename+"/Plots"
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(candidate_path))[0]

    o3d.io.write_point_cloud(f"{save_dir}/{base}_pcd1.ply", pcd1)
    o3d.io.write_point_cloud(f"{save_dir}/{base}_pcd2.ply", pcd2)
    o3d.io.write_line_set(f"{save_dir}/{base}_lines.ply", line_set)

    vis.run()  # 🟢 Opens the window and lets you interact manually
    # The script waits here until you close the window
    # After closing, this part is executed:
    vis.destroy_window()


def estimate_pose(device, query_path, candidate_path, query_point_cloud, candidate_dataset, model, dino_model, results_folder, visualize_matches, threshold=0.9):

    img_shape = (560, 560)
    K = np.array([577.331309, 0.0, 353.747907, 0.0, 577.326066, 256.683265, 0.0, 0.0, 1.0]).reshape((3,3))

    # In case I want to precompute and load sonata features
    """
    sonata_feat_path= "./Datasets/Sonata_descriptors/features00.pkl"
    print("🔹 Loading Sonata features...")

    try:
        with open(sonata_feat_path, "rb") as f:
            data_dict = pickle.load(f)
        print(f"📂 Loaded {len(data_dict)} features.")
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"⚠️ Failed to load existing features: {e}")
        data_dict = {}

    # Check if it exists in the dictionary
    if candidate_path in data_dict:
        coords1, point1 = data_dict[candidate_path]

    else:
    """

    #print(f"❌ Image path '{candidate_path}' not found in the feature dictionary.")
    print("🔹 Extracting Sonata features from candidate...")
    point1 = find_pointcloud(candidate_dataset, candidate_path)
    point1 = load_pointcloud_from_array(point1)
    coords1 = point1["coord"].copy()
    point1 = apply_transform(point1)
    point1 = run_inference(model, point1)
    point1 = extract_final_features(point1)


    print("🔹 Extracting Sonata features from query...")
    point2 = load_pointcloud_from_array(query_point_cloud)
    coords2 = point2["coord"].copy()
    point2 = apply_transform(point2)
    point2 = run_inference(model, point2)
    point2 = extract_final_features(point2)


    # === Load image and extract DINO features ===
    print("🔹 Loading Dino features from candidate...")
    #candidate_path_aux = "~" + candidate_path
    #candidate_path_aux = os.path.expanduser(candidate_path_aux)
    image_tensor1 = preprocess_image(candidate_path, image_size=img_shape)
    dino_feat_grid1 = extract_patchwise_features(dino_model, image_tensor1, device)

    print("🔹 Loading Dino features from query...")
    #query_path_aux = "~" + query_path
    #query_path_aux = os.path.expanduser(query_path_aux)
    image_tensor2 = preprocess_image(query_path, image_size=img_shape)
    dino_feat_grid2 = extract_patchwise_features(dino_model, image_tensor2, device)


    # === Map to points ===
    print("🔹 Projection Dino features into Lidar points...")
    pts_valid_dino1, pts_invalid_dino1, dino_feat1, dino_valid_mask1 = map_dino_to_points(
        coords1.T, K, dino_feat_grid1, img_shape)

    pts_valid_dino2, pts_invalid_dino2, dino_feat2, dino_valid_mask2 = map_dino_to_points(
        coords2.T, K, dino_feat_grid2, img_shape)

    if pts_valid_dino2 is None or dino_valid_mask1 is None or dino_valid_mask2 is None:
        print("Not possible to compute the pose due to not enough valid dino projections")
        return None, None

    # === Filter and align point cloud ===
    point1_filtered = coords1[dino_valid_mask1.T]
    sonata_feat1_filtered = point1.feat[point1.inverse][dino_valid_mask1.T].cpu()

    point2_filtered = coords2[dino_valid_mask2.T]
    sonata_feat2_filtered = point2.feat[point2.inverse][dino_valid_mask2.T].cpu()

    # Concatenate features along the last axis
    # === Combine features ===
    print("🔹 Combining Sonata and Dino features ...")
    feat1 = torch.cat([sonata_feat1_filtered, torch.tensor(dino_feat1).float()], dim=1)
    feat2 = torch.cat([sonata_feat2_filtered, torch.tensor(dino_feat2).float()], dim=1)

    print("🔹 Computing matches...")
    idx1, idx2, scores = compute_matches(feat1, feat2, threshold)
    #idx1, idx2, scores = compute_matches_euclidean(feat1, feat2)
    #idx1, idx2, scores = compute_matches_hybrid(feat1, feat2)

    print(f"✅ Found {len(idx1)} correspondences above threshold {threshold}")
    matched_coords1 = point1_filtered[idx1]
    matched_coords2 = point2_filtered[idx2]

    if len(matched_coords1) < 6:
        print(f"❌ Not enough correspondences ({len(matched_coords1)}) to compute pose.")
        return None, None

    print("🔹 Estimating pose with RANSAC...")
    transformation = estimate_pose_ransac(matched_coords1, matched_coords2)

    #print("🔹 Refining pose with ICP...")
    #transformation, fitness = refine_pose_icp(matched_coords1, matched_coords2, transformation)

    print("✅ Estimated transformation:\n", transformation)

    yaw_deg = extract_yaw_angle(transformation)
    print(f"Relative Yaw (deg): {yaw_deg:.2f}°")

    if visualize_matches:
        print("🔹 Visualizing matches...")
        visualize_matches_open3d(query_path, candidate_path, point1_filtered, point2_filtered, idx1, idx2, feat1, feat2, results_folder)

    return transformation, yaw_deg

def get_candidate_dataset_path(matched_img_path):
    # Trim to second last slash
    base_dir = os.path.dirname(os.path.dirname(matched_img_path))  # up two levels
    folder_name = os.path.basename(base_dir)  # get last folder name
    candidate_dataset = os.path.join("./Datasets/s3li-dataset", folder_name + ".pkl")
    return candidate_dataset


def save_top10_results(results, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Sort results by yaw angle
    sorted_results = sorted(results, key=lambda x: abs(x['yaw_angle']))

    # Save to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['query_path', 'candidate_path', 'transformation', 'yaw_angle']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(sorted_results):

            writer.writerow({
                'query_path': result['query_path'],
                'candidate_path': result['candidate_path'],
                'transformation': result['transformation'].tolist(),
                'yaw_angle': result['yaw_angle']
            })

    print(f"✅ Saved top 10 results to {output_csv_path}")

def pose_estimation_top10(device, query_path, query_point_cloud, sonata_model, dino_model,
                          results_folder, visualize_matches, similarity_threshold):
    query_filename = os.path.splitext(os.path.basename(query_path))[0].replace(".", "")
    candidates = results_folder+query_filename+"_refinement_top10.pkl"

    # Load the DataFrame
    with open(candidates, "rb") as f:
        candidates = pickle.load(f)

    results = []
    # Iterate and estimate pose
    for entry in candidates:
        # Then inside your main script logic, build a results list like this:
        candidate_path = str(entry["matched_image"])
        candidate_dataset = get_candidate_dataset_path(candidate_path)

        image_start = time.time()  # Start timing per image
        transformation_matrix, yaw_angle = estimate_pose(device, query_path, candidate_path, query_point_cloud, candidate_dataset,
                                                         sonata_model, dino_model, results_folder, visualize_matches, threshold=similarity_threshold)
        time_taken_ms = (time.time() - image_start) * 1000  # Convert to ms
        print(f"Time taken to estimate pose: {time_taken_ms:.2f} ms")

        if yaw_angle is not None:
            # After each pose estimation for a query-candidate pair:
            results.append({
                'query_path': query_path,
                'candidate_path': candidate_path,
                'transformation': transformation_matrix,  # a 4x4 numpy array
                'yaw_angle': yaw_angle
            })

    print(f"Number of results received: {len(results)}")

    # After collecting top 10 results:
    save_top10_results(results[:10], results_folder+query_filename+'pose_estimation.csv')


