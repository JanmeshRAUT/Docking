import os
import pandas as pd
import numpy as np
import cv2
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
CROPPED_IMG_DIR = "cropped_images"

# =========================
# 1. Crop Black Background
# =========================
def crop_black(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 10

    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return image, (0, 0)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped, (x_min, y_min)


# =========================
# 2. Crop Images + FIX LABELS
# =========================
def process_and_save(csv_file, img_dir, output_img_dir, output_csv):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    df = pd.read_csv(csv_file)
    new_rows = []

    print(f"\nProcessing {csv_file}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = f"{int(row['ImageID'])}.jpg"
        img_path = os.path.join(img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w, _ = image.shape

        # Parse location from CSV
        try:
            loc = ast.literal_eval(row["location"])
            x = float(loc[0])
            y = float(loc[1])
        except:
            continue

        # Normalize original coordinates
        x_norm_orig = x / w
        y_norm_orig = y / h

        # Crop
        cropped, (x_min, y_min) = crop_black(image)

        new_h, new_w, _ = cropped.shape

        # Adjust coordinates AFTER crop
        x_new = (x - x_min) / new_w
        y_new = (y - y_min) / new_h

        # Clip to [0,1]
        x_new = np.clip(x_new, 0, 1)
        y_new = np.clip(y_new, 0, 1)

        # Save image
        save_path = os.path.join(output_img_dir, img_id)
        cv2.imwrite(save_path, cropped)

        # Save updated row
        new_rows.append({
            "ImageID": img_id,
            "x_norm": x_new,
            "y_norm": y_new
        })

    # Save NEW CSV
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)

    print(f"✓ Saved: {output_csv}")
    print(f"✓ Images saved in: {output_img_dir}\n")


# =========================
# 3. Dataset Class
# =========================
class ISSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, use_coordconv=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.use_coordconv = use_coordconv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["ImageID"])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        if self.use_coordconv:
            image = self.add_coord_channels(image)

        target = torch.tensor(
            [row["x_norm"], row["y_norm"]],
            dtype=torch.float32
        )

        return image, target

    def add_coord_channels(self, img):
        _, h, w = img.shape

        x_coords = torch.linspace(-1, 1, w).repeat(h, 1)
        y_coords = torch.linspace(-1, 1, h).unsqueeze(1).repeat(1, w)

        x_coords = x_coords.unsqueeze(0)
        y_coords = y_coords.unsqueeze(0)

        return torch.cat([img, x_coords, y_coords], dim=0)


# =========================
# 4. Transforms
# =========================
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# 5. DataLoaders
# =========================
def get_dataloaders(
    train_csv,
    val_csv,
    img_dir,
    use_coordconv=False
):
    transform = get_transforms()

    train_dataset = ISSDataset(train_csv, img_dir, transform, use_coordconv)
    val_dataset = ISSDataset(val_csv, img_dir, transform, use_coordconv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader


# =========================
# 6. MAIN PIPELINE
# =========================
if __name__ == "__main__":
    print("="*60)
    print("FULL PREPROCESSING (WITH CORRECT LABELS)")
    print("="*60)

    process_and_save("../Dataset/train.csv", "images", CROPPED_IMG_DIR, "train_cropped.csv")
    process_and_save("../Dataset/val.csv", "images", CROPPED_IMG_DIR, "val_cropped.csv")
    process_and_save("../Dataset/test.csv", "images", CROPPED_IMG_DIR, "test_cropped.csv")

    print("✓ DONE: Images + labels are now consistent!")
