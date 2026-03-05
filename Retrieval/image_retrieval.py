import torch
import pickle
import os
from pathlib import Path
import pandas as pd
import time
from Retrieval.preprocess_img import preprocess_image
from Store_descriptors.store_salad_descriptors import load_model, load_model_v2, extract_salad_descriptor
from Indexing.FAISS_indexing import build_faiss_index, create_filtered_index, run_similarity_search


def preprocess_query_image(image_path):
    start = time.time()
    image_tensor = preprocess_image(image_path)
    print(f"Preprocessing time: {(time.time() - start)*1000:.2f} ms")
    return image_tensor

def extract_query_features(model, image_tensor, device):
    start = time.time()
    features = extract_salad_descriptor(model, image_tensor, device)
    print(f"Feature extraction time: {(time.time() - start)*1000:.2f} ms")
    return features.astype('float32').reshape(1, -1)

def load_features(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Features in {save_path} not found!")
    return {}

def store_results(results, csv_file, pickle_file):
    """Stores results in both CSV and Pickle formats."""
    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # **Save to CSV**
    if os.path.exists(csv_file):
        results_df.to_csv(csv_file, mode="a", header=False, index=False)  # Append mode
    else:
        results_df.to_csv(csv_file, index=False)

    # **Save to Pickle**
    if os.path.exists(pickle_file):
        # Load existing data
        with open(pickle_file, "rb") as f:
            existing_results = pickle.load(f)

        # Convert to DataFrame and append
        existing_df = pd.DataFrame(existing_results)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)

        # Save updated results
        with open(pickle_file, "wb") as f:
            pickle.dump(combined_df.to_dict(orient="records"), f)
    else:
        # Save new results if file does not exist
        with open(pickle_file, "wb") as f:
            pickle.dump(results_df.to_dict(orient="records"), f)

    print(f"Saved {len(results)} results to {csv_file} and {pickle_file}")


def format_results(query_path, query_ts, similarities, temp_idx, filtered_indices, image_paths, timestamps):
    results = []
    top_k_paths = []

    print(f"Top {len(temp_idx[0])} similar images to file://{Path(query_path).resolve()} (Timestamp: {query_ts}):")

    for rank, idx in enumerate(temp_idx[0]):
        original_idx = filtered_indices[idx]
        matched_path = Path(image_paths[original_idx]).resolve()
        matched_ts = timestamps[original_idx]
        similarity = 1 - similarities[0][rank] / 2  # Convert from inner product to cosine

        print(f"file://{matched_path} (Timestamp: {matched_ts}) with similarity: {similarity:.4f}")

        results.append({
            "query_image": query_path,
            "query_timestamp": query_ts,
            "matched_image": matched_path,
            "matched_timestamp": matched_ts,
            #"northing": row["northing"],
            #"easting": row["easting"],
            "similarity": similarity
        })

        top_k_paths.append(str(matched_path))

    return results, top_k_paths


def image_retrieval(device, query_image_path, query_timestamp, index, model, image_paths, timestamps,
                    time_window=100, top_k=20,
                    results_file="./Results/retrieval_results.csv", pickle_file="./Results/retrieval_results.pkl"):

    """Find similar images and store results in a structured format."""
    # Preprocess query and extract query features
    query_tensor = preprocess_query_image(query_image_path)
    query_features = extract_query_features(model, query_tensor, device)

    #Filter candidates based on timestamp
    filtered_indices = [i for i, ts in enumerate(timestamps) if abs(ts - query_timestamp) > time_window]
    if not filtered_indices:
        print("No candidates found after timestamp filtering.")
        return []

    #Create temporary index with filtered features
    temp_index, filtered_indices = create_filtered_index(index, filtered_indices)

    #Run FAISS search on filtered index
    #Calculate similarities and extract the topk
    similarities, temp_idx = run_similarity_search(temp_index, query_features, top_k)

    # Print results for each query
    results, top_k_paths = format_results(query_image_path, query_timestamp, similarities, temp_idx, filtered_indices,
                                          image_paths, timestamps)
    # Store results
    store_results(results, results_file, pickle_file)
    return top_k_paths

def retrieval_top_k(query_timestamp, query_image_path, model, device, results_folder, feature_save_path, k=20, time_threshold = 100.0):

    query_filename = os.path.splitext(os.path.basename(query_image_path))[0].replace(".", "")

    features_dict = load_features(feature_save_path)
    if not features_dict:
        print(f"⚠️ Features file in {feature_save_path} not found")
        return

    index, image_paths, timestamps = build_faiss_index(features_dict)

    start_time = time.time()  # Start timer

    image_start = time.time()  # Start timing per image
    image_retrieval(device, query_image_path, query_timestamp, index, model, image_paths, timestamps,
                    time_window = time_threshold,
                    top_k=k,
                    results_file=results_folder+query_filename+f"_retrieval_top{k}.csv",
                    pickle_file=results_folder+query_filename+f"_retrieval_top{k}.pkl")
    image_end = time.time()  # End timing per image

    time_taken_ms = (image_end - image_start) * 1000  # Convert to ms
    print(f"Time taken for image: {time_taken_ms:.2f} ms")

    end_time = time.time()  # End timer

    # Total time for image retrieval
    total_time_ms = (end_time - start_time) * 1000
    #average_time_ms = total_time_ms / len(data)

    print(f"\nTotal retrieval time: {total_time_ms:.2f} ms")

