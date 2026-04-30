import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
preprocess_dir = r"e:\Docking\Preprocess"
csv_path = os.path.join(preprocess_dir, "train_processed.csv")
images_dir = os.path.join(preprocess_dir, "images")

# Create output directories
output_dirs = {
    'train': r"e:\Docking\Train",
    'test': r"e:\Docking\Test",
    'valid': r"e:\Docking\Validation"
}

for split_dir in output_dirs.values():
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)

# Read the CSV
df = pd.read_csv(csv_path)

# First split: 70% train, 30% temp (test + valid)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)

# Second split: split temp into 50% test, 50% valid (15% and 15% of original)
test_df, valid_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Validation samples: {len(valid_df)}")

# Save CSVs
train_df.to_csv(os.path.join(output_dirs['train'], "train.csv"), index=False)
test_df.to_csv(os.path.join(output_dirs['test'], "test.csv"), index=False)
valid_df.to_csv(os.path.join(output_dirs['valid'], "validation.csv"), index=False)

print("\nCSV files created successfully!")

# Copy images to respective folders
def copy_images(df, split_name, output_dir):
    image_output_dir = os.path.join(output_dir, "images")
    for idx, row in df.iterrows():
        image_name = row['ImageID']
        source_path = os.path.join(images_dir, image_name)
        dest_path = os.path.join(image_output_dir, image_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: Image {image_name} not found!")
        
        if (idx + 1) % 1000 == 0:
            print(f"  Copied {idx + 1} images to {split_name}...")

print("\nCopying images...")
copy_images(train_df, "train", output_dirs['train'])
copy_images(test_df, "test", output_dirs['test'])
copy_images(valid_df, "validation", output_dirs['valid'])

print("\nData split completed successfully!")
print(f"\nDirectory structure created:")
for split_name, split_dir in output_dirs.items():
    print(f"  {split_dir}/")
    print(f"    - {split_name}.csv")
    print(f"    - images/")
