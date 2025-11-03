import torch
import numpy as np
import pickle
import os
from preprocess_img import preprocess_image

def load_model(model_name="dinov2_vitb14_reg"):
    #model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = torch.hub.load('./Models/dinov2', model_name, source='local')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def load_model_v2(model_name="dinov2_vitb14_reg", weights_path="./Weights/finetuned_dinov2_v3.pth"):
    # Load base model architecture (from local hub)
    model = torch.hub.load('./Models/dinov2', model_name, source='local')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unfreeze last 3 transformer blocks (just like during training)
    for block in model.blocks[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    # Load the fine-tuned weights
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))  # or 'cuda' if needed
    model.load_state_dict(state_dict)

    # Move the model to the device (CPU or GPU)
    model.to(device)
    model.eval()
    return model

def extract_features(model, image_tensor, device):
    """Extract features using deeper layers and patch-based embeddings instead of CLS token."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.get_intermediate_layers(image_tensor, n=3)  # Extract deeper layers
        features = [feat[:, 1:] for feat in features]  # Remove CLS token
        features = torch.cat(features, dim=-1).mean(dim=1)  # Average pooling
    return torch.nn.functional.normalize(features, dim=-1).cpu().numpy().flatten()  # Normalize

def extract_patch_features(model, image_tensor):
    """Extracts patch-wise normalized feature vectors (one per 14x14 patch)."""
    with torch.no_grad():
        features = model.get_intermediate_layers(image_tensor, n=3)
        features = [feat[:, 1:] for feat in features]  # remove CLS token
        features = torch.cat(features, dim=-1)  # (1, n_patches, feature_dim_total)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features[0].cpu().numpy()  # (n_patches, feature_dim)


def save_features(features_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

def load_features(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    return {}

def compute_and_store_features(pickle_folder_path, model, feature_save_path):
    features_dict = load_features(feature_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                        features = extract_features(model, image_tensor, device)
                        features_dict[image_path] = (timestamp, features)
                        print(f"✅ Extracted: {image_path}")

            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    save_features(features_dict, feature_save_path)
    print(f"\n💾 Saved all features to: {feature_save_path}")
    return features_dict


def main():
    pickle_folder = "./Datasets/s3li-dataset/vulcano"  # Path to your pickle file
    feature_save_path =  "./Datasets/Dinov2_descriptors/features_vulcano_pt_dino.pkl"

    #model = load_model_v2("dinov2_vitb14_reg", weights_path="./Weights/finetuned_dinov2_v3.pth")
    model = load_model()
    compute_and_store_features(pickle_folder, model, feature_save_path)

if __name__ == "__main__":
    main()

