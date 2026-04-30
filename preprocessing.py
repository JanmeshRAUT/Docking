import os
import pandas as pd
import numpy as np
import cv2
import ast
import random
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
AUG_PER_IMAGE = 2      # 10k → 20k
SHIFT = 60

OUTPUT_DIR = "Preprocess"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# =========================
# ENHANCE IMAGE
# =========================
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# =========================
# RANDOM SHIFT CROP
# =========================
def crop_with_shift(image, x, y, size=224, shift=60):
    h, w, _ = image.shape

    dx = random.randint(-shift, shift)
    dy = random.randint(-shift, shift)

    x1 = int(x - size // 2 + dx)
    y1 = int(y - size // 2 + dy)

    x1 = max(0, min(x1, w - size))
    y1 = max(0, min(y1, h - size))

    x2 = x1 + size
    y2 = y1 + size

    cropped = image[y1:y2, x1:x2]

    return cropped, (x1, y1)

# =========================
# LIGHT AUGMENTATION
# =========================
def random_augment(image):
    if random.random() < 0.5:
        alpha = 0.8 + random.random() * 0.4
        beta = random.randint(-20, 20)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

# =========================
# PROCESS DATASET
# =========================
def process_dataset(csv_path, img_dir, output_csv):

    df = pd.read_csv(csv_path)

    # Normalize distance
    max_dist = df["distance"].max()
    df["distance_norm"] = df["distance"] / max_dist

    new_rows = []

    print(f"\nProcessing: {csv_path}")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        base_name = str(int(row["ImageID"]))
        img_path = os.path.join(img_dir, f"{base_name}.jpg")

        image = cv2.imread(img_path)
        if image is None:
            continue

        orig_h, orig_w, _ = image.shape

        loc = ast.literal_eval(row["location"])
        x, y = loc

        image = enhance_image(image)

        for i in range(AUG_PER_IMAGE):

            # =========================
            # Crop with shift
            # =========================
            cropped, (x_min, y_min) = crop_with_shift(image, x, y, IMG_SIZE, SHIFT)

            # =========================
            # Convert to crop space
            # =========================
            x_rel = x - x_min
            y_rel = y - y_min

            # =========================
            # SKIP invalid crops (VERY IMPORTANT)
            # =========================
            if not (0 <= x_rel < IMG_SIZE and 0 <= y_rel < IMG_SIZE):
                continue

            # =========================
            # Flip (CORRECT)
            # =========================
            if random.random() < 0.5:
                cropped = cv2.flip(cropped, 1)
                x_rel = IMG_SIZE - 1 - x_rel   # ✅ FIXED

            # =========================
            # Extra augmentation
            # =========================
            cropped = random_augment(cropped)

            # =========================
            # Normalize
            # =========================
            new_x = x_rel / IMG_SIZE
            new_y = y_rel / IMG_SIZE
            # =========================
            # Save image
            # =========================
            new_name = f"{base_name}_{i}.jpg"
            save_path = os.path.join(OUTPUT_IMG_DIR, new_name)
            cv2.imwrite(save_path, cropped)

            # =========================
            # Save row
            # =========================
            new_rows.append({
                "ImageID": new_name,
                "x_norm": new_x,
                "y_norm": new_y,
                "distance": row["distance"],
                "distance_norm": row["distance_norm"],
                "x_offset": x_min,
                "y_offset": y_min,
                "orig_w": orig_w,
                "orig_h": orig_h
            })

    # =========================
    # SAVE CSV
    # =========================
    output_path = os.path.join(OUTPUT_DIR, output_csv)
    pd.DataFrame(new_rows).to_csv(output_path, index=False)

    print(f"\n✅ Saved: {output_path}")
    print(f"📊 Total samples: {len(new_rows)}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    DATASET_PATH = "raw_data"
    IMAGE_PATH = os.path.join(DATASET_PATH, "images")

    process_dataset(
        os.path.join(DATASET_PATH, "train.csv"),
        IMAGE_PATH,
        "train_processed.csv"
    )

    print("\n🚀 DONE: 20K dataset created (Correct + Stable)")