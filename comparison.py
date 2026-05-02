import os
import pandas as pd
import numpy as np
import cv2
import ast
from tqdm import tqdm

RAW_CSV = "raw_data/train.csv"
RAW_IMG_DIR = "raw_data/images"
PROC_CSV = "Preprocess/train_processed.csv"

OUTPUT_DIR = "Validation_Compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

# Load data
raw_df = pd.read_csv(RAW_CSV)
proc_df = pd.read_csv(PROC_CSV)

raw_dict = {
    int(row["ImageID"]): row
    for _, row in raw_df.iterrows()
}

errors = []

for _, row in tqdm(proc_df.iterrows(), total=len(proc_df)):

    img_name = row["ImageID"]
    base_id = int(img_name.split("_")[0])

    if base_id not in raw_dict:
        continue

    raw_row = raw_dict[base_id]

    img_path = os.path.join(RAW_IMG_DIR, f"{base_id}.jpg")
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Original point
    loc = ast.literal_eval(raw_row["location"])
    x_raw, y_raw = int(loc[0]), int(loc[1])

    # Processed
    x_norm = row["x_norm"]
    y_norm = row["y_norm"]
    x_offset = row["x_offset"]
    y_offset = row["y_offset"]
    orig_w = row["orig_w"]
    orig_h = row["orig_h"]
    flipped = row["flipped"] if "flipped" in row else False

    # =========================
    # CORRECT RECONSTRUCTION (accounts for flips)
    # =========================
    # Denormalize from [0, 1] to [0, 224]
    x_in_crop = x_norm * IMG_SIZE
    y_in_crop = y_norm * IMG_SIZE
    
    # If flipped, reverse the x coordinate
    if flipped:
        x_in_crop = IMG_SIZE - 1 - x_in_crop
    
    # Transform back to original image space
    x_reconstructed = x_in_crop + x_offset
    y_reconstructed = y_in_crop + y_offset
    
    x_new = int(x_reconstructed)
    y_new = int(y_reconstructed)

    err_new = np.sqrt((x_raw - x_new)**2 + (y_raw - y_new)**2)
    errors.append(err_new)

    # =========================
    # VISUALIZATION
    # =========================
    vis = image.copy()

    # Ground truth (Red)
    cv2.circle(vis, (x_raw, y_raw), 6, (0, 0, 255), -1)

    # Reconstructed (Green)
    cv2.circle(vis, (x_new, y_new), 6, (0, 255, 0), -1)

    # Line
    cv2.line(vis, (x_raw, y_raw), (x_new, y_new), (0, 255, 0), 2)

    # Text
    cv2.putText(vis, f"Error: {err_new:.1f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), vis)

# =========================
# RESULTS
# =========================
print("\n===== RECONSTRUCTION ACCURACY =====")
print(f"Mean Error: {np.mean(errors):.2f}px")
print(f"Median Error: {np.median(errors):.2f}px")
print(f"Max Error: {np.max(errors):.2f}px")
print(f"Min Error: {np.min(errors):.2f}px")
print(f"Std Dev: {np.std(errors):.2f}px")
print("====================================")
print(f"📁 Output: {OUTPUT_DIR}")