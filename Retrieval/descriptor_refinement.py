
import numpy as np
import pickle
import torch
import pandas as pd
import os
import cv2
from pathlib import Path
import time
from Indexing.FAISS_indexing import build_faiss_index, create_filtered_index, run_similarity_search
from Store_descriptors.store_dino_feat import extract_features, load_features
from Retrieval.preprocess_img import preprocess_image
from Retrieval.image_retrieval import store_results

def format_results(query_path, similarities, temp_idx, image_paths, timestamps, data):
    results = []
    top_k_paths = []

    print(f"Top {len(temp_idx[0])} similar images to file://{Path(query_path).resolve()}:")

    # Convert list of dicts to DataFrame for filtering
    data = pd.DataFrame(data)

    for rank, idx in enumerate(temp_idx[0]):
        matched_path = Path(image_paths[idx]).resolve()
        matched_ts = timestamps[idx]
        similarity = similarities[0][rank]  # Convert from inner product to cosine

        print(f"file://{matched_path} (Timestamp: {matched_ts}) with similarity: {similarity:.4f}")

        row = data[data["matched_image"] == matched_path].iloc[0]
        results.append({
            "query_image": row["query_image"],
            "query_timestamp": row["query_timestamp"],
            "matched_image": matched_path,
            "matched_timestamp": matched_ts,
            #"northing": row["northing"],
            #"easting": row["easting"],
            "similarity": similarity
        })

        top_k_paths.append(str(matched_path))

    return results, top_k_paths

def image_retrieval(query_path, query_features, index, data, image_paths, timestamps, top_k=10,
                    results_file="./Results/refinement.csv", pickle_file="./Results/refinement.pkl"):

    """Find similar images and store results in a structured format."""

    # Run FAISS search on filtered index
    # Calculate similarities and extract the topk
    similarities, temp_idx = run_similarity_search(index, query_features, top_k)

    # Print results for each query
    results, top_k_paths = format_results(query_path, similarities, temp_idx, image_paths, timestamps, data)

    # Store results
    store_results(results, results_file, pickle_file)

    return top_k_paths

def descriptor(image_path, model, device):
    image_tensor = preprocess_image(image_path)
    return extract_features(model, image_tensor, device).astype('float32').reshape(1, -1)

def extract_matching_features(features_dict, pickle_path):
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    if not features_dict:
        print(f"Error! features_dict is empty!")

    matched_dict = {}
    for entry in results:
        matched_image = entry["matched_image"]
        matched_image_str = str(matched_image)  # ensure consistent key format

        if matched_image_str in features_dict:
            matched_dict[matched_image_str] = features_dict[matched_image_str]
        else:
            print(f"Warning: {matched_image_str} not found in features_dict")
    return matched_dict

def refine_to_top_k(query_image_path, model, device, results_folder, feature_save_path, k_orig=20, k_refine=10):
    query_filename = os.path.splitext(os.path.basename(query_image_path))[0].replace(".", "")
    candidates = results_folder+query_filename + f"_retrieval_top{k_orig}.pkl"

    features_dict = load_features(feature_save_path)
    candidates_dict = extract_matching_features(features_dict, candidates)

    if not candidates_dict:
        return

    index, image_paths, timestamps = build_faiss_index(candidates_dict)

    image_start = time.time()
    query_features=descriptor(query_image_path, model, device)

    # Query using all images in the pickle file
    with open(candidates, 'rb') as f:
        data = pickle.load(f)

    # Start timing per image
    image_retrieval(query_image_path, query_features, index, data, image_paths, timestamps, top_k=k_refine,
                    results_file=results_folder+query_filename+f"_refinement_top{k_refine}.csv",
                    pickle_file=results_folder+query_filename+f"_refinement_top{k_refine}.pkl")
    image_end = time.time()  # End timing per image

    time_taken_ms = (image_end - image_start) * 1000  # Convert to ms
    print(f"Time taken for descriptor refinement: {time_taken_ms:.2f} ms")



