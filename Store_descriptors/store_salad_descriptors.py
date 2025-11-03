import numpy as np
import torch
import pickle
import os
from preprocess_img import preprocess_image
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from vpr_model import VPRModel
from vpr_model_v2 import VPRModel as VPRModelv2

def load_model(ckpt_path="./Weights/dino_salad.ckpt", dino_model= "dinov2_vitb14", dino_weights= "./Weights/finetuned_dinov2_v3.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VPRModel(
        backbone_arch=dino_model,
        backbone_config={
            'dino_weight': dino_weights,
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,

        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            #'num_channels': 2304,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"Loaded model from {ckpt_path} successfully on {device}!")
    return model


def load_model_v2(ckpt_path="./Weights/last.ckpt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VPRModelv2(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 2304,
            'num_clusters': 64,
            'cluster_dim': 128
        },
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract state_dict from the checkpoint
    state_dict = checkpoint["state_dict"]

    # OPTIONAL: remove "model." or "module." prefix if needed
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")  # or whatever prefix your model uses
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)  # strict=False is helpful if there are mismatches
    model = model.to(device).eval()

    print(f"Loaded model from {ckpt_path} successfully on {device}!")
    return model

def extract_salad_descriptor(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        desc = model(image_tensor)  # SALAD outputs global descriptor
    return desc.squeeze().cpu().numpy()  # shape: (descriptor_dim,)


def save_features(features_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

def load_features(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    return {}

def compute_and_store_features(device, pickle_folder_path, model, feature_save_path):
    features_dict = load_features(feature_save_path)

    for file_name in os.listdir(pickle_folder_path):
        if file_name.endswith(".pkl"):
            pickle_path = os.path.join(pickle_folder_path, file_name)
            print(f"\n📂 Processing: {file_name}")

            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)

                for _, row in data.iterrows():
                    timestamp = row["time_stamp"]
                    image_path = row["img_path"]
                    if image_path not in features_dict:
                        image_tensor = preprocess_image(image_path)
                        features = extract_salad_descriptor(model, image_tensor, device)
                        features_dict[image_path] = (timestamp, features.astype('float32'))
                        print(f"✅ Extracted: {image_path}")

            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    save_features(features_dict, feature_save_path)
    print(f"\n💾 Saved all features to: {feature_save_path}")
    return features_dict


def main():
    pickle_folder = "./Datasets/s3li-dataset/vulcano"  # Path to your pickle file
    feature_save_path = "./Datasets/SALAD_descriptors/features_vulcano_new.pkl"

    #model = load_model_v2() #retrained salad
    model = load_model()  # old salad
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_and_store_features(device, pickle_folder, model, feature_save_path)

if __name__ == "__main__":
    main()

