from image_retrieval import retrieval_top20
from descriptor_refinement import refine_to_top10
from dino_sonata_pose_estimation import pose_estimation_top10
import torch
from store_dino_feat import load_model_v2, load_model as load_dino_model
from store_salad_descriptors import load_model, load_model_v2 as load_salad_model
from store_sonata_descriptors import load_model as load_sonata_model
import pandas as pd
import os
import time

def find_pointcloud(dataset, query_path):
    df = pd.read_pickle(dataset)
    # Find the row where 'img_path' matches query_path
    match = df[df['img_path'] == query_path]
    if match.empty:
        raise ValueError(f"No entry found for query path: {query_path}")
    # Return the corresponding point_cloud (assumes unique match)
    return match.iloc[0]['point_cloud']

def get_timestamp(dataset, query_path):
    df = pd.read_pickle(dataset)
    # Find the row where 'img_path' matches query_path
    match = df[df['img_path'] == query_path]
    if match.empty:
        raise ValueError(f"No entry found for query path: {query_path}")
    # Return the corresponding point_cloud (assumes unique match)
    return match.iloc[0]["time_stamp"]


def get_candidate_dataset_path(matched_img_path, dataset_path):
    # Trim to second last slash
    base_dir = os.path.dirname(os.path.dirname(matched_img_path))  # up two levels
    folder_name = os.path.basename(base_dir)  # get last folder name
    candidate_dataset = os.path.join(dataset_path, folder_name + ".pkl")
    return candidate_dataset


def pipeline(query_original_path, dino_model, salad_model, sonata_model, device, results_folder,
             salad_features_file, dino_features_file, pose_estimation, visualize_matches,
             similarity_threshold, dataset_path):

    #query_img = "~"+ query_original_path # to run from server
    #query_img = os.path.expanduser(query_img)
    query_img = query_original_path

    query_dataset = get_candidate_dataset_path(query_original_path, dataset_path)
    #print(query_dataset)

    query_timestamp = get_timestamp(query_dataset, query_original_path)
    query_point_cloud = find_pointcloud(query_dataset, query_original_path)

    start = time.time()
    retrieval_top20(query_timestamp, query_img, salad_model, device, results_folder, salad_features_file)
    refine_to_top10(query_img, dino_model, device, results_folder, dino_features_file)
    print(f"Image Retrieval time: {(time.time() - start) * 1000:.2f} ms")

    if pose_estimation:
        pose_estimation_top10(device,query_img, query_point_cloud, sonata_model, dino_model,
                              results_folder, visualize_matches, similarity_threshold)


if __name__ == "__main__":
    query_original_path = "/home_local/enci_la/Etna/dataset/s3li_crater/images/img1625658929.4591641.png"  # path that can be found in the database
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dino_model = load_model_v2("dinov2_vitb14_reg", weights_path="./Weights/finetuned_dinov2_v3.pth").to(device)
    dino_model = load_dino_model("dinov2_vits14_reg") #old dino

    #salad_model = load_salad_model(ckpt_path="./Weights/last.ckpt")
    salad_model = load_model() #old salad
    sonata_model = load_sonata_model()

    results_folder = "./Results/New_results_2/"
    dino_features_file = "./Datasets/Dinov2_descriptors/features_vulcano_new.pkl"
    salad_features_file = "./Datasets/SALAD_descriptors/features_vulcano_new.pkl"

    pipeline(query_original_path, dino_model, salad_model, sonata_model, device, results_folder, salad_features_file, dino_features_file)