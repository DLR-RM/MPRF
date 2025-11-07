import numpy as np
import torch
import sonata
import pandas as pd
import os
import pickle

def load_pointcloud_from_array(coords):
    coords = coords.astype(np.float32)
    color = np.zeros_like(coords)  # Dummy RGB
    normal = np.zeros_like(coords)  # Dummy normals
    return {"coord": coords, "color": color, "normal": normal}

def apply_transform(point_dict):
    transform = sonata.transform.default()
    return transform(point_dict)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = sonata.model.load("sonata", repo_id="facebook/sonata").to(device)
    except:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = sonata.model.load("sonata", repo_id="facebook/sonata", custom_config=custom_config).cuda()
    return model

def run_inference(model, point):
    with torch.inference_mode():
        for key in point:
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)
    return point

def extract_final_features(point):
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point

def main():
    folder_path = "./Datasets/s3li-dataset"
    output_path = "./Datasets/Sonata_descriptors/features.pkl"
    BATCH_SIZE = 16

    print("🔹 Loading model...")
    model = load_model()

    # Load existing features
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            data_dict = pickle.load(f)
        print(f"📂 Loaded existing feature file with {len(data_dict)} entries.")
    else:
        data_dict = {}

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".pkl"):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"\n📁 Processing file: {file_name}")

        try:
            df = pd.read_pickle(file_path)

            if "point_cloud" not in df.columns or "img_path" not in df.columns:
                print(f"⚠️ Skipping {file_name}: missing required columns.")
                continue

            for start in range(0, len(df), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(df))
                batch_coords = df["point_cloud"].iloc[start:end]
                batch_paths = df["img_path"].iloc[start:end]

                # Skip entire batch if all images are already processed
                if all(img_path in data_dict for img_path in batch_paths):
                    print(f"⏩ Skipping already processed batch {start}-{end} of {file_name}")
                    continue

                for coords, img_path in zip(batch_coords, batch_paths):
                    if img_path in data_dict:
                        continue

                    point = load_pointcloud_from_array(coords)
                    print(point)
                    original_coord = point["coord"].copy()
                    point = apply_transform(point)
                    point = run_inference(model, point)
                    point = extract_final_features(point)

                    point_numpy = {
                        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in point.items()
                    }

                    data_dict[img_path] = (original_coord, point_numpy)

                    # Clean up
                    del point, point_numpy
                    torch.cuda.empty_cache()

                # Save after each batch
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                tmp_path = output_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(data_dict, f)
                os.replace(tmp_path, output_path)
                print(f"💾 Saved features after batch {start}-{end} of {file_name}")

        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")
            continue

if __name__ == "__main__":
    main()
