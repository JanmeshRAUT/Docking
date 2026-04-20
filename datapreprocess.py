import os
import ast
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split

# 1. Configuration & Paths
BASE_DIR = r"e:\Docking"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
IMG_DIR = os.path.join(DATASET_DIR, "images")
RAW_CSV = os.path.join(BASE_DIR, "raw_data", "train.csv")

# Model Training CSVs
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

IMG_SIZE = 512  # Original dimension
TARGET_SIZE = 224
BATCH_SIZE = 32

def prepare_dataset_splits():
    """Load raw data, clean, and create Train/Val/Test CSVs in the Dataset folder."""
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    
    # safe parsing helper
    def parse_loc(val):
        try: return ast.literal_eval(val)
        except: return None

    df['coords'] = df['location'].apply(parse_loc)
    df = df.dropna(subset=['coords'])
    df['x'] = df['coords'].apply(lambda c: float(c[0]))
    df['y'] = df['coords'].apply(lambda c: float(c[1]))
    
    # Image existence check
    def check_img(idx):
        return os.path.exists(os.path.join(IMG_DIR, f"{int(idx)}.jpg"))
    df = df[df['ImageID'].apply(check_img)]

    # Normalized Targets
    max_dist = df['distance'].max()
    df['x_norm'] = df['x'] / IMG_SIZE
    df['y_norm'] = df['y'] / IMG_SIZE
    df['dist_norm'] = df['distance'] / max_dist

    # 70/15/15 Split
    train_df, rest_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(rest_df, test_size=0.50, random_state=42)

    # Save to Dataset folder
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    print(f"Dataset splits saved to {DATASET_DIR}")
    return train_df, val_df, test_df

class ISSDockingDataset(Dataset):
    def __init__(self, csv_path, img_dir, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_train = is_train
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{int(row['ImageID'])}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        x, y, dist = row['x_norm'], row['y_norm'], row['dist_norm']

        if self.is_train:
            # Sync augmentation for coordinates
            if random.random() > 0.5:
                image = TF.hflip(image)
                x = 1.0 - x
            
            # Apply color jitter with error handling
            try:
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
                image = color_jitter(image)
            except Exception as e:
                # Skip augmentation if it fails (e.g., corrupted image)
                pass

        image = TF.resize(image, (TARGET_SIZE, TARGET_SIZE))
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        target = torch.tensor([x, y, dist], dtype=torch.float32)
        return image, target

# Helper to get Loaders
def get_dataloaders():
    train_ds = ISSDockingDataset(TRAIN_CSV, IMG_DIR, is_train=True)
    val_ds = ISSDockingDataset(VAL_CSV, IMG_DIR, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 1. Run splitting (only needed once, or if parameters change)
    # prepare_dataset_splits()
    
    # 2. Test Loaders
    t_loader, v_loader = get_dataloaders()
    images, labels = next(iter(t_loader))
    print(f"Loaded batch of size {images.shape[0]} from {TRAIN_CSV}")
