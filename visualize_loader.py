import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from datapreprocess import ISSDockingDataset
from torch.utils.data import DataLoader

# Create output directory
OUTPUT_DIR = "Visualization"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def imshow_minimal(img_tensor, target, img_id, ax):
    """Simple denormalize and plot on axis with Image ID."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.numpy().transpose((1, 2, 0))
    
    x_px = target[0] * 224
    y_px = target[1] * 224
    
    ax.imshow(img_np)
    ax.scatter(x_px, y_px, c='red', marker='x', s=60, linewidth=2)
    ax.set_title(f"ID: {img_id}", fontsize=12, fontweight='bold')
    ax.axis('off')

def generate_10_batches_with_ids():
    main_csv = os.path.join("Dataset", "train.csv")
    main_img = os.path.join("Dataset", "images")
    
    dataset = ISSDockingDataset(csv_path=main_csv, img_dir=main_img, is_train=False)
    
    # We need access to ImageIDs, which are in the dataset's dataframe
    df = dataset.df
    
    for b in range(10):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        start_idx = b * 10
        
        for i in range(10):
            idx = start_idx + i
            img_tensor, target = dataset[idx]
            img_id = int(df.iloc[idx]['ImageID'])
            imshow_minimal(img_tensor, target, img_id, axes[i])
        
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f"docking_batch_{b+1}.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"Generated: {filename}")

if __name__ == "__main__":
    generate_10_batches_with_ids()
