from Retrieval.image_retrieval import retrieval_top_k
from Retrieval.descriptor_refinement import refine_to_top_k
from Pose_estimation.dino_sonata_pose_estimation import pose_estimation_top10
import torch
from Store_descriptors.store_dino_feat import load_model_v2, load_model as load_dino_model
from Store_descriptors.store_salad_descriptors import load_model, load_model_v2 as load_salad_model
from Store_descriptors.store_sonata_descriptors import load_model as load_sonata_model
import pandas as pd
import os
import time

def find_pointcloud(df, query_path):

    # Find the row where 'img_path' matches query_path
    match = df[df['img_path'] == query_path]
    if match.empty:
        raise ValueError(f"No entry found for query path: {query_path}")
    # Return the corresponding point_cloud (assumes unique match)
    return match.iloc[0]['point_cloud']

def get_timestamp(df, query_path):

    # Find the row where 'img_path' matches query_path
    match = df[df['img_path'] == query_path]
    if match.empty:
        raise ValueError(f"No entry found for query path: {query_path}")
    # Return the corresponding point_cloud (assumes unique match)
    return match.iloc[0]["time_stamp"]


def get_candidate_dataset_path(matched_img_path):
    """
        get_candidate_dataset_path finds the $dataset.pkl file, output of the s3li-toolkit, that matches 
        the subfolder name where the matched_img_path lives. E.g., if matched_img_path lives in 
        $path_to_dataset/moon_lake/images/001.png, then this function looks for $path_to_dataset/moon_lake/moon_lake.pkl
    """
    base_dir = os.path.dirname(os.path.dirname(matched_img_path))  # up two levels
    folder_name = os.path.basename(base_dir)  # get last folder name
    target_pkl_file = os.path.join(base_dir, folder_name + '.pkl')
    print(f"Looking for: {target_pkl_file}")
    return target_pkl_file


def pipeline(query_original_path,
             dino_model, salad_model, sonata_model, device, results_folder,
             salad_features_file, dino_features_file, pose_estimation,
             visualize_matches, dataset_cache,
             k=20, k_refine=10, similarity_threshold_3d=0.9, time_threshold=100):

    query_dataset = get_candidate_dataset_path(query_original_path)

    if query_dataset not in dataset_cache:
        print(f"Loading dataset {query_dataset}")
        dataset_cache[query_dataset] = pd.read_pickle(query_dataset)

    df = dataset_cache[query_dataset]

    query_timestamp = get_timestamp(df, query_original_path)
    query_point_cloud = find_pointcloud(df, query_original_path)

    start = time.time()
    retrieval_top_k(query_timestamp, query_original_path, salad_model, device, results_folder, salad_features_file, k=k, time_threshold = time_threshold)
    refine_to_top_k(query_original_path, dino_model, device, results_folder, dino_features_file, k_orig=k, k_refine=k_refine)
    print(f"Image Retrieval time: {(time.time() - start) * 1000:.2f} ms")

    if pose_estimation:
        pose_estimation_top10(device,query_original_path, query_point_cloud, sonata_model, dino_model,
                              results_folder, visualize_matches, similarity_threshold_3d)

