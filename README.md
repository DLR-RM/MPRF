# MPRF
This repository contains the official code for the paper [MPRF](URL).

### ✅ Tested Configuration
- Python 3.10.18  
- PyTorch 2.5.0+cu124  
- CUDA 12.4  

---

## ⚙️ Environment Setup

**Option 1:**
```bash
conda env create -f environment.yml -n mprf_env
```

**Option 2:**
```bash
conda create -n mprf_env python=3.10.18
conda activate mprf_env
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

---

## 📂 1. Prepare Dataset
- Download and prepare the dataset using the [s3li toolkit](https://github.com/DLR-RM/s3li-toolkit) following the instructions in that repository.  
- You can also use the preprocessed `.pkl` files provided here: [Datasets](https://drive.google.com/drive/folders/1l0-7jXz42KJxYig_2yLFdvd2wBWLoCw1?usp=drive_link)

  This folder includes:
  - `.pkl` files for each processed sequence  
  - `.pkl` files containing valid match pairs used as ground truth for evaluation  
  
  Note that the query images should be downloaded and stored in the same directories referenced inside the `.pkl` files

---

## 🧠 2. Download Weights
Weights for Dinov2 and SALAD can be found here: 
[Weights](https://drive.google.com/drive/folders/1XueYvt4Uy_dHzec81hhHKim0h-YPeAOB?usp=drive_link)

---

## 💾 3. Generate Feature Files
Run the scripts from the root folder `MPRF`:

**DINOv2 features:**
```bash
source .env && python Store_descriptors/store_dino_feat.py   --pickle_folder ./Processed_datasets/vulcano   --feature_save_path ./Dinov2_features/vulcano_dino_features.pkl   --model_type finetuned   --weights_path ./Weights/finetuned_dinov2.pth
```

**SALAD features:**
```bash
source .env && python Store_descriptors/store_salad_descriptor.py   --pickle_folder ./Processed_datasets/vulcano   --feature_save_path ./SALAD_features/vulcano_salad_features.pkl   --model_type pretrained
```

Pre-generated feature files for Etna and Vulcano datasets:  [Features](https://drive.google.com/drive/folders/1fonONDhNA7DOp-giDtmPlSIej3IIshWu?usp=drive_link)

---

## ⚙️ 4. Update Configuration

Edit the `config.yaml` file to set your dataset paths, model weights, and preferences.  
Below is an example configuration with inline explanations:

```yaml
# config.yaml

# === Paths ===
eval_pairs_path: "./Datasets/s3li-dataset/gt_dataset/s3li_etna_pairs.pkl"   # Ground truth file with valid matches

dino_features_file: "./Datasets/Dinov2_descriptors/etna_finetuned_dinov2_features.pkl"   # Features generated from store_dino_feat.py

salad_features_file: "./Datasets/SALAD_descriptors/etna_pretrained_salad_features.pkl"   # Features generated from store_salad_descriptors.py

results_folder: "./Results/New_results/"    # Output directory where results will be stored
dataset_path: "./Datasets/s3li-etna/"       # Path to the dataset


# === Models ===
dino_model: "dinov2_vitb14"
dino_weights: "./Weights/finetuned_dinov2.pth"

salad_model_version: "pt"                   # Options: "pt" (pretrained) or "rt" (retrained)
salad_weights: "./Weights/pretrained_salad.ckpt"    # Required if using retrained model

# === Flags ===
run_pipeline: true          # Run full pipeline for all queries (if false, only evaluate stored results)
pose_estimation: false      # Enable or disable pose estimation
visualize_matches: false    # Visualize feature correspondences used for pose estimation

# === Thresholds ===
similarity_threshold: 0.95  # Threshold for 3D correspondences
```

## ▶️ 5. Run the Main Script
```bash
source .env && python eval_precision.py --config config.yaml
```

This command runs the full pipeline:

### 🖼️ Image Retrieval
Processes all query images from the evaluation `.pkl` file and retrieves the most similar candidates.

Generates two files per query (identified by the query’s timestamp):
- `*_top20.pkl` → Top-20 most similar candidates (first retrieval stage)  
- `*_top10.pkl` → Top-10 candidates (final retrieval stage)

### 🤖 Pose Estimation
For each query, generates a `.csv` file containing one row per candidate.  
Each row includes the estimated transformation and yaw angle between the query and its candidate image.

### 📊 Evaluation
Computes and reports precision metrics at **Top-1**, **Top-5**, and **Top-10**.

---

💡 **To Skip Processing and Only Compute Metrics**

If results are already stored and you only want to evaluate precision without running the full pipeline again, set the following in your `config.yaml`:
```yaml
run_pipeline: false
```
