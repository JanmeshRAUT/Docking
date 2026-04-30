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

errors_old = []
errors_new = []

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

    # =========================
    # OLD WRONG RECONSTRUCTION
    # =========================
    x_old = int(x_norm * IMG_SIZE + x_offset)
    y_old = int(y_norm * IMG_SIZE + y_offset)

    err_old = np.sqrt((x_raw - x_old)**2 + (y_raw - y_old)**2)
    errors_old.append(err_old)

    # =========================
    # NEW CORRECT RECONSTRUCTION
    # =========================
    x_new = int(round(x_norm * IMG_SIZE)) + x_offset
    y_new = int(round(y_norm * IMG_SIZE)) + y_offset

    err_new = np.sqrt((x_raw - x_new)**2 + (y_raw - y_new)**2)
    errors_new.append(err_new)

    # =========================
    # VISUALIZATION
    # =========================
    vis = image.copy()

    # Ground truth (Red)
    cv2.circle(vis, (x_raw, y_raw), 6, (0, 0, 255), -1)

    # Old (Blue)
    cv2.circle(vis, (x_old, y_old), 6, (255, 0, 0), -1)

    # New (Green)
    cv2.circle(vis, (x_new, y_new), 6, (0, 255, 0), -1)

    # Lines
    cv2.line(vis, (x_raw, y_raw), (x_old, y_old), (255, 0, 0), 2)
    cv2.line(vis, (x_raw, y_raw), (x_new, y_new), (0, 255, 0), 2)

    # Text
    cv2.putText(vis, f"Old Err: {err_old:.1f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(vis, f"New Err: {err_new:.1f}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), vis)

# =========================
# RESULTS
# =========================
print("\n===== COMPARISON REPORT =====")
print(f"Old Mean Error: {np.mean(errors_old):.2f}px")
print(f"Old Max Error: {np.max(errors_old):.2f}px")

print(f"New Mean Error: {np.mean(errors_new):.2f}px")
print(f"New Max Error: {np.max(errors_new):.2f}px")

print("================================")
print(f"📁 Output: {OUTPUT_DIR}")