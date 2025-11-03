import pandas as pd
import torch
import pickle
from pipeline import pipeline
import os
from store_dino_feat import load_model as load_old_dino_model, load_model_v2 as load_dino_model
from store_salad_descriptors import load_model, load_model_v2 as load_salad_model
from store_sonata_descriptors import load_model as load_sonata_model
from pathlib import Path
import yaml
import argparse

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def strip_first_two(path: str) -> str:
    """Remove the first two folders from an absolute path."""
    parts = Path(path).parts
    return "/"+str(Path(*parts[3:]))  # skip "", "home/enci_la"

def load_top10_matches(query_image_path, results_folder):
    query_filename = os.path.splitext(os.path.basename(query_image_path))[0].replace(".", "")
    #query_filename = os.path.basename(query_image_path).split('.')[0]

    results_file = f"{results_folder}{query_filename}_refinement_top10.pkl"

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return []

    # Use pickle directly, specifying encoding for cross-version compatibility
    with open(results_file, "rb") as f:
        raw_data = pickle.load(f)

    if isinstance(raw_data, list):
        df = pd.DataFrame(raw_data, columns=[
            "query_image", "query_timestamp", "matched_image",
            "matched_timestamp", "similarity"
        ])
    else:
        df = raw_data

    df["matched_image"] = df["matched_image"].apply(str)
    df["query_image"] = df["query_image"].apply(str)

    # Filter only rows where the query matches the input path exactly
    filtered_df = df[df["query_image"] == query_image_path]

    if filtered_df.empty:
        print(f"No entries found for query: {query_image_path}")
        return []

    # return top 10 matched images
    return filtered_df["matched_image"].tolist()[:10]


def main(config_path="config.yaml"):
    config = load_config(config_path)
    eval_pairs_path = config["eval_pairs_path"]
    results_folder = config["results_folder"]
    dino_features_file = config["dino_features_file"]
    salad_features_file = config["salad_features_file"]
    dino_weights = config["dino_weights"]
    dino_model_name = config["dino_model"]
    pose_estimation = config["pose_estimation"]
    run_pipeline = config["run_pipeline"]
    visualize_matches = config["visualize_matches"]
    similarity_threshold = config["similarity_threshold"]
    dataset_path = config["dataset_path"]

    raw_pairs = pd.read_pickle(eval_pairs_path)

    if isinstance(raw_pairs, list):
        pairs_df = pd.DataFrame(raw_pairs, columns=["query", "matched_img", "overlap_score"])
    else:
        pairs_df = raw_pairs

    # Build ground-truth dict
    gt_dict = {}

    for _, row in pairs_df.iterrows():
        query = row['query']
        match = str(row['matched_img'])
        overlap = row['overlap_score']
        gt_dict.setdefault(query, {})[match] = overlap

    unique_queries = [
        query for query, matches in gt_dict.items()
    ]
    unique_queries = sorted(set(unique_queries))

    total_queries = 0
    tp_at_1 = 0
    tp_at_5 = 0
    tp_at_10 = 0

    if run_pipeline:
        print("LOADING MODELS")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_model = load_dino_model(dino_model_name, weights_path=dino_weights).to(device)
        if config["salad_model_version"] == "pt":
            salad_model = load_model()
        else:
            salad_weights = config["salad_weights"]
            salad_model= load_salad_model(ckpt_path=salad_weights, dino_model= dino_model_name, dino_weights = dino_weights)
        sonata_model = load_sonata_model()

    matched_pairs = []  # ← To collect matched query-candidate pairs

    print("True Positives and Overlap Scores:")
    print("-----------------------------------")

    for query in unique_queries:
        if run_pipeline:
            pipeline(query, dino_model, salad_model, sonata_model, device, results_folder,
                     salad_features_file, dino_features_file, pose_estimation, visualize_matches,
                     similarity_threshold, dataset_path)
        top10 = load_top10_matches(query, results_folder)
        if not top10 or len(top10) < 10:
            print("Less than 10 candidates")
            continue

        total_queries += 1

        top1 = top10[0]
        top5 = top10[:5]

        # Precision@1
        if top1 in gt_dict[query]:
            tp_at_1 += 1
            print(f"[P@1] Query: {os.path.basename(query)} | Match: {os.path.basename(top1)} | Overlap: {gt_dict[query][top1]:.4f}")

        # Precision@5
        tp_5 = sum([1 for img in top5 if img in gt_dict[query]])
        tp_at_5 += tp_5
        for img in top5:
            if img in gt_dict[query]:
                print(f"[P@5] Query: {os.path.basename(query)} | Match: {os.path.basename(img)} | Overlap: {gt_dict[query][img]:.4f}")

        # Precision@10
        tp_10 = sum([1 for img in top10 if img in gt_dict[query]])
        tp_at_10 += tp_10
        for img in top10:
            if img in gt_dict[query]:
                print(f"[P@10] Query: {os.path.basename(query)} | Match: {os.path.basename(img)} | Overlap: {gt_dict[query][img]:.4f}")
                matched_pairs.append((query, img))  # Save match

    if total_queries == 0:
        print("No valid queries found.")
        return

    # Compute precision
    p_at_1 = tp_at_1 / total_queries
    p_at_5 = tp_at_5 / (5 * total_queries)
    p_at_10 = tp_at_10 / (10 * total_queries)

    print("\nSummary:")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"True Positives in Top-1:  {tp_at_1}")
    print(f"True Positives in Top-5:  {tp_at_5}")
    print(f"True Positives in Top-10: {tp_at_10}")
    print(f"Precision@1:  {p_at_1:.4f}")
    print(f"Precision@5:  {p_at_5:.4f}")
    print(f"Precision@10: {p_at_10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
