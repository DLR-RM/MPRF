import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def enhance_contrast(image):
    """Apply histogram equalization to enhance contrast in grayscale images."""
    image = cv2.equalizeHist(image)  # Enhance contrast
    return Image.fromarray(image)

def preprocess_image(image_path, image_size=(224, 224)):
    """Convert grayscale to 3-channel, enhance contrast, and apply DINO preprocessing."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    print("Processing image: ", image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = enhance_contrast(image)  # Improve contrast
    image = np.stack([np.array(image)] * 3, axis=-1)  # Convert grayscale to RGB

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    return transform(image).unsqueeze(0)
