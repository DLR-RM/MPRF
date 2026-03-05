import pandas as pd
import numpy as np
import torch
import pickle

#from tests.c_integration.cmds.test_run import test_run_hello_addition

from pipeline import pipeline
import os
from Store_descriptors.store_dino_feat import load_model as load_old_dino_model, load_model_v2 as load_dino_model
from Store_descriptors.store_salad_descriptors import load_model, load_model_v2 as load_salad_model
from pathlib import Path
import yaml
import argparse
from collections import defaultdict


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def strip_first_two(path: str) -> str:
    """Remove the first two folders from an absolute path."""
    parts = Path(path).parts
    return "/"+str(Path(*parts[3:]))  # skip "", "home/enci_la"

def load_top_k_matches(query_image_path, results_folder, k=10, similarity_threshold = 0.985):

    query_filename = os.path.splitext(os.path.basename(query_image_path))[0].replace(".", "")

    results_file = f"{results_folder}{query_filename}_refinement_top{k}.pkl"


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

    filtered_df = df[df["query_image"] == query_image_path]

    # Remove potential duplicates, conserving the order (query, match)
    filtered_df = filtered_df.drop_duplicates(subset=['query_image', 'matched_image'], keep='first')

    # Filter for similarity 
    similarity_mask = filtered_df["similarity"] >= similarity_threshold     
    filtered_df = filtered_df[similarity_mask].sort_values(by = "similarity", ascending=False)

    if filtered_df.empty:
        return []   

    # return top k matched images
    return filtered_df["matched_image"].tolist()[:k]



def compute_metrics(res: dict, gt_pairs: dict):
    """
    Computes P@k. 
    @param res          results from the retrieval step, dict that associates
                        image_query_path -> [positive_sample_1, positive_sample_2, etc..]
                        must be all the possible images evaluated, even if no decision is associated.
                        samples must be SORTED in decreasing order for similarity values!
    @param gt_pairs     results from the ground truth generation step, dict that associates
                        image_query_path -> [true_sample_1, true_sample_2, etc..]
    """
    tot_tp = 0
    tot_fp = 0 
    tot_fn = 0
    tot_positive_samples = 0
    tot_true_samples = 0
    tot_successful_queries_for_recall_at_k = defaultdict(int)
    tot_precision_at_k = defaultdict(float)
    tot_queries = 0 

    for query_image, positive_samples in res.items():
        true_samples = [] 
        if query_image in gt_pairs.keys():
            #print("gt paths: ", gt_pairs[query_image][0])
            true_samples = set(gt_pairs[query_image])
        
        # Should not happen because of the way the gt_set is assembled
        # but we check anyway
        if not true_samples:
            continue

        # At best, the top-k, otherwise all 
        for k in [1, 5, 10, 20]:
            #print("Model paths: ", positive_samples)
            top_k_samples = set(positive_samples[:k])
            if top_k_samples & true_samples:
                tot_successful_queries_for_recall_at_k[k] += 1

            # ----- Precision@K -----
            correct_in_top_k = len(top_k_samples & true_samples)
            precision_k = correct_in_top_k / float(min(k, len(positive_samples))) if positive_samples else 0
            tot_precision_at_k[k] += precision_k
        tot_queries += 1

        positive_samples = set(positive_samples)
        tot_tp += len(positive_samples & true_samples)
        tot_fp += len(positive_samples - true_samples)
        tot_fn += len(true_samples - positive_samples)
        tot_positive_samples += len(positive_samples)
        tot_true_samples += len(true_samples)

    precision = tot_tp / tot_positive_samples if tot_positive_samples > 0 else 0
    recall    = tot_tp / tot_true_samples if tot_true_samples > 0 else 0
    recall_at_k = {k: tot_successful_queries_for_recall_at_k[k] / float(tot_queries) for k in tot_successful_queries_for_recall_at_k}
    precision_at_k = { k: tot_precision_at_k[k] / float(tot_queries) for k in tot_precision_at_k}

    return precision, recall, recall_at_k, precision_at_k, tot_tp, tot_fp, tot_fn, tot_positive_samples, tot_true_samples

def print_metrics(p, r, rk, pk, tp, fp, fn, positive_samples, true_samples, thresh):
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    table = Table(title=f"Evaluation Summary (Threshold: {thresh:.2f})")

    table.add_column("Category", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Precision (Global)", f"{p:.4f}")
    table.add_row("Recall (Global)", f"{r:.4f}")
    table.add_section()
    for k, val in rk.items():
        table.add_row(f"Recall@{k} (Success Rate) ", f"{val:.4f}")
    table.add_section()
    for k, val in pk.items():
        table.add_row(f"Precision@{k} ", f"{val:.4f}")
    table.add_section()
    table.add_row("True Positives", str(int(tp)), style="green")
    table.add_row("False Positives", str(int(fp)), style="red")
    table.add_row("False Negatives", str(int(fn)), style="yellow")
    table.add_row("Total GT Positives (TP+FN)", str(int(true_samples)), style="bold white")

    console.print(table)

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import matplotlib.ticker as ticker

def plot_pr_curve(precision, recall, output_dir="/tmp"):
    # Calculate Area Under the Curve (AUC)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(5, 5))
    
    # Plot the curve with a nice fill
    plt.plot(recall, precision, color='#2ecc71', lw=3, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='#2ecc71')
    
    # Baseline for a random classifier (usually horizontal line at proportion of positives)
    # Adjust this value based on your dataset balance
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

    # Formatting
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.xlabel('Recall (Completeness)')
    plt.ylabel('Precision (Exactness)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(9)
    
    filename = f"precision_recall_curve_global.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close() 

def plot_and_save_recall(recall_at_k_dict, threshold, output_dir="/tmp"):
    """
    Plots the Recall@k curve and saves it to a directory.
    """
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ks = sorted(recall_at_k_dict.keys())
    values = [recall_at_k_dict[k] for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, values, marker='o', linestyle='-', color='teal', linewidth=2)
    
    plt.title(f"Recall@k Performance\nSimilarity Threshold: {threshold}")
    plt.xlabel("k")
    plt.ylabel("Recall (Success Rate)")
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save logic
    filename = f"recall_curve_thresh_{str(threshold).replace('.', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close() 

def plot_recall_comparison(summary_results, output_dir="/tmp"):

    # Slightly increase font sizes
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    plt.figure(figsize=(5, 5))  # 3x3 inches

    for thresh, rk in summary_results.items():
        ks = sorted(rk.keys())
        values = [rk[k] for k in ks]
        plt.plot(ks, values, marker='s', label=f"Thresh {thresh:.2f}")

    plt.title("Recall@k Comparison: Threshold Sweep")
    plt.xlabel("k")
    plt.ylabel("Success Rate")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "master_comparison.png"),
        bbox_inches="tight"    # remove extra whitespace
    )
    plt.show()


def plot_precision_comparison(summary_results, output_dir="/tmp"):

    plt.figure(figsize=(5, 5))

    for thresh, pk in summary_results.items():
        ks = sorted(pk.keys())
        values = [pk[k] for k in ks]
        plt.plot(ks, values, marker='o', label=f"Thresh {thresh:.2f}")

    plt.title("Precision@K Comparison: Threshold Sweep")
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "precision_at_k_comparison.png"),
        bbox_inches = "tight"  # remove extra whitespace
    )
    plt.show()


def main(config_path="config.yaml"):
    config = load_config(config_path)

    # Ground-truth for place recognition
    eval_pairs_path = config["eval_pairs_path"]

    # Output folder to store results
    results_folder = config["results_folder"]

    # Pre-computed DinoV2/Salad features
    dino_features_file = config["dino_features_file"]
    salad_features_file = config["salad_features_file"]
    dino_weights = config["dino_weights"]
    dino_model_name = config["dino_model"]

    # Toggles
    pose_estimation = config["pose_estimation"]
    run_pipeline = config["run_pipeline"]
    visualize_matches = config["visualize_matches"]
    similarity_threshold_3d = config["similarity_threshold_3d"]
    similarity_threshold_retrieval = config["similarity_threshold_retrieval"]
    similarity_threshold_retrieval_pr_range = config["similarity_threshold_retrieval_pr_range"]
    time_threshold = config["time_threshold"]

    # Input dataset to use for evaluation
    #path_to_pickled_test_dataset = config["path_to_test_dataset"]

    # Top-k params 
    k = config['k']
    k_refine = config['k_refine']

    raw_pairs = pd.read_pickle(eval_pairs_path)

    if isinstance(raw_pairs, list):
        pairs_df = pd.DataFrame(raw_pairs, columns=["query", "matched_img", "overlap_score"])
    else:
        pairs_df = raw_pairs

    # Build ground-truth dict
    overlap_dict = defaultdict(list)
    gt_dict = defaultdict(list)

    for _, row in pairs_df.iterrows():
        query = row['query']
        match = str(row['matched_img'])
        overlap = row['overlap_score']
        overlap_dict.setdefault(query, {})[match] = overlap
        gt_dict[query].append(match)

    print("Length pairs_df", len(gt_dict))

    # TODO: careful!! This assumes we only check those for which we have overlap > threshold
    # that was set in s3li-toolkit. We should rather do it for all known queries that were 
    # used to computed descriptors!!
    unique_queries = sorted(gt_dict.keys())

    total_queries = 0
    
    if run_pipeline:
        print("LOADING MODELS")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_model = load_dino_model(dino_model_name, weights_path=dino_weights).to(device)
        sonata_model = None
        if config["salad_model_version"] == "pt":
            salad_model = load_model()
        else:
            salad_weights = config["salad_weights"]
            salad_model= load_salad_model(ckpt_path=salad_weights)
        if pose_estimation:
            from Store_descriptors.store_sonata_descriptors import load_model as load_sonata_model
            sonata_model = load_sonata_model()

    # To collect matched query-candidate pairs
    # It's a dict {query_img: [positive_sample_1, positive_sample_2, etc..]}
    matched_pairs = defaultdict(list)  
    
    iteration = 0
    tot_iterations = len(unique_queries)
    dataset_cache = {}

    for image_query_path in unique_queries:
        if run_pipeline:
            pipeline(image_query_path,
                     dino_model, salad_model, sonata_model, device, results_folder,
                     salad_features_file, dino_features_file, pose_estimation,
                     visualize_matches, dataset_cache,
                     k=k, k_refine=k_refine,
                     similarity_threshold_3d=similarity_threshold_3d,
                     time_threshold=time_threshold)
        top_k = load_top_k_matches(image_query_path, results_folder, k=k_refine, similarity_threshold=similarity_threshold_retrieval)
        
        iteration +=1
        print(f"Completeness: {float(iteration) / float(tot_iterations) * 100:.2f}%")

        if not top_k:
            print("No matches found")
            matched_pairs[image_query_path] # access the key without doing anything so an empty list is initialized
            continue 

        for img in top_k:
            matched_pairs[image_query_path].append(img) 
            #print(f"({image_query_path}) <-> ({img})")
        
        total_queries += 1


    p, r, rk, pk, tp, fp, fn, positive_samples, true_samples = compute_metrics(matched_pairs, gt_dict)
    print_metrics(p, r, rk, pk, tp, fp, fn, positive_samples, true_samples, similarity_threshold_retrieval)




    # Compute here PR curve! ======================================================================================================
    if not plot_pr_curve:
        import sys
        sys.exit() 

    p_array = []
    r_array = []
    rk_summary = {}
    pk_summary = {}
    for thresh in np.linspace(similarity_threshold_retrieval_pr_range[0],
                              similarity_threshold_retrieval_pr_range[1],
                              similarity_threshold_retrieval_pr_range[2]):
        print(f"Computing P/R value for thresh [min: {similarity_threshold_retrieval_pr_range[0]:.2f} < {thresh:.2f} < max: {similarity_threshold_retrieval_pr_range[1]:.2f}]")
        matched_pairs = defaultdict(list)  
        for image_query_path in unique_queries:
            top_k = load_top_k_matches(image_query_path, results_folder, k=k_refine, similarity_threshold=thresh)
            if not top_k: 
                matched_pairs[image_query_path] # access the key without doing anything so an empty list is initialized
                continue 

            for img in top_k:
                matched_pairs[image_query_path].append(img) 
                #print(f"({image_query_path}) <-> ({img})")
            
            total_queries += 1
        
        p, r, rk, pk, tp, fp, fn, positive_samples, true_samples = compute_metrics(matched_pairs, gt_dict)
        print("Length pairs_df", len(gt_dict))
        print_metrics(p, r, rk, pk, tp, fp, fn, positive_samples, true_samples, thresh)
        p_array.append(p)
        r_array.append(r)
        rk_summary[thresh] = rk
        pk_summary[thresh] = pk

    plot_pr_curve(p_array, r_array, output_dir="./tmp")
    plot_recall_comparison(rk_summary, output_dir="./tmp")
    plot_precision_comparison(pk_summary, output_dir="./tmp")

    # Dump pickle
    results_bundle = {
        "rk_summary": rk_summary,
        "p_array": p_array,
        "r_array": r_array
    } 
    with open('./tmp/results.pickle', 'wb') as file: 
        pickle.dump(results_bundle, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
