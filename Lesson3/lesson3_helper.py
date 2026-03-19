"""
lesson3_setup.py — Helper module for Lab 3: The Stress Test

Students import this module in their notebook. It handles downloading
the dataset, setting up PyTorch datasets, and providing visualization
helpers. Like answers.py in Lesson 1, this hides boilerplate complexity.
"""

import os
import zipfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn

# A simple wrapper to ensure calling forward returns a Tensor, not a HF Object
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

# ── Configuration ──────────────────────────────────────────────────────
# UPDATE THIS URL to point to the raw GitHub URL of your skin_lesions.zip
DATA_URL = "https://github.com/SimonIlic/ML4Epidemiology/raw/main/Lesson3/skin_lesions.zip"
DATA_DIR = "skin_lesions"
ZIP_FILE = "skin_lesions.zip"

# ImageNet normalization (required for MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CLASS_NAMES = ["benign", "malignant"]


# ── Data Loading ───────────────────────────────────────────────────────

def load_data():
    """Download the skin lesion dataset and return train/test datasets.

    Returns:
        (train_dataset, test_dataset): PyTorch ImageFolder datasets
    """
    if not os.path.exists(DATA_DIR):
        print("Downloading skin lesion dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            zf.extractall(".")
        os.remove(ZIP_FILE)
        print("Done!")

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=TRANSFORM
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=TRANSFORM
    )

    print(f"Training images: {len(train_dataset)} ({_class_counts(train_dataset)})")
    print(f"Test images:     {len(test_dataset)} ({_class_counts(test_dataset)})")
    return train_dataset, test_dataset


def _class_counts(dataset):
    counts = {}
    for _, label in dataset.samples:
        name = CLASS_NAMES[label]
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


# ── Visualization Helpers ──────────────────────────────────────────────

def _denormalize(tensor):
    """Convert a normalized tensor back to a displayable numpy image."""
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def show_images(dataset, n=8):
    """Display a grid of sample images from the dataset."""
    indices = np.random.choice(len(dataset), n, replace=False)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        axes[i].imshow(_denormalize(img_tensor))
        axes[i].set_title(CLASS_NAMES[label], fontsize=12,
                          color="green" if label == 0 else "red")
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
