import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get script directory for all paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pth")

# Use preprocessed data for validation
PREPROCESS_CSV = os.path.join(SCRIPT_DIR, "Preprocess", "train_processed.csv")
PREPROCESS_IMG_DIR = os.path.join(SCRIPT_DIR, "Preprocess", "images")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Visualization")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =========================
# CUSTOM DATASET CLASS
# =========================
class DockingDataset(Dataset):
    """Load docking dataset from CSV and images folder."""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Handle image ID with or without extension
        img_id = str(row["ImageID"])
        if not img_id.endswith('.jpg'):
            img_id = f"{img_id}.jpg"
        
        # Load image
        img_path = os.path.join(self.img_dir, img_id)
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Load target values
        target = torch.tensor([
            row["x_norm"],
            row["y_norm"],
            row["distance_norm"]
        ], dtype=torch.float32)
        
        return image, target

# =========================
# TRANSFORMS
# =========================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Load Model ---
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# --- 3. Denormalization Function ---
def denormalize_coords(x_norm, y_norm, dist_norm, img_size=512, max_dist=None):
    """Convert normalized values back to original pixel/distance values."""
    x_pixel = x_norm * img_size
    y_pixel = y_norm * img_size
    distance = dist_norm * max_dist if max_dist else dist_norm
    return x_pixel, y_pixel, distance

# --- 4. Make Predictions ---
def predict_on_test_set(model):
    test_dataset = DockingDataset(csv_path=PREPROCESS_CSV, img_dir=PREPROCESS_IMG_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    predictions = []
    actuals = []
    image_ids = []
    
    print(f"Making predictions on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    image_ids = test_dataset.df['ImageID'].values
    
    return predictions, actuals, image_ids, test_dataset

# --- 4b. Make Predictions on Validation Set ---
def predict_on_val_set(model):
    val_dataset = DockingDataset(csv_path=PREPROCESS_CSV, img_dir=PREPROCESS_IMG_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    predictions = []
    actuals = []
    image_ids = []
    
    print(f"Making predictions on {len(val_dataset)} samples...")
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    image_ids = val_dataset.df['ImageID'].values
    
    return predictions, actuals, image_ids, val_dataset

# --- 5. Visualization: Predictions vs Actuals (Scatter Plots) ---
def plot_predictions_vs_actuals(predictions, actuals):
    """Create scatter plots for X, Y, Distance comparisons."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    labels = ['X (normalized)', 'Y (normalized)', 'Distance (normalized)']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.scatter(actuals[:, i], predictions[:, i], alpha=0.6, edgecolors='k', s=50)
        
        # Perfect prediction line
        min_val = min(actuals[:, i].min(), predictions[:, i].min())
        max_val = max(actuals[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_predictions_vs_actuals_scatter.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_predictions_vs_actuals_scatter.png")
    plt.close()

# --- 6. Visualization: Error Distribution ---
def plot_error_distribution(predictions, actuals):
    """Plot error distribution for each output."""
    errors = predictions - actuals
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['X Error', 'Y Error', 'Distance Error']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(errors[:, i], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(errors[:, i].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors[:, i].mean():.6f}')
        ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        
        ax.set_xlabel('Error Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_error_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_error_distribution.png")
    plt.close()

# --- 7. Visualization: Residuals Plot ---
def plot_residuals(predictions, actuals):
    """Plot residuals for each output."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['X', 'Y', 'Distance']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        residuals = actuals[:, i] - predictions[:, i]
        ax.scatter(predictions[:, i], residuals, alpha=0.6, edgecolors='k', s=50)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Residuals', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_residuals_plot.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_residuals_plot.png")
    plt.close()

# --- 8. Visualization: Sample Images with Predictions vs Actual ---
def plot_sample_predictions(model, test_dataset, num_samples=20):
    """Show images with predicted vs actual markers."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    sample_count = 0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images_gpu = images.to(DEVICE)
            predictions = model(images_gpu).cpu().numpy()
            targets = targets.numpy()
            
            for i in range(len(images)):
                if sample_count >= num_samples:
                    break
                
                # Denormalize image
                img = images[i] * std + mean
                img = torch.clamp(img, 0, 1)
                img_np = img.numpy().transpose((1, 2, 0))
                
                ax = axes[sample_count]
                ax.imshow(img_np)
                
                # Convert normalized coords to pixel coords (224x224)
                actual_x, actual_y = targets[i, 0] * 224, targets[i, 1] * 224
                pred_x, pred_y = predictions[i, 0] * 224, predictions[i, 1] * 224
                
                # Plot actual (red X)
                ax.scatter(actual_x, actual_y, c='red', marker='x', s=100, linewidth=3, label='Actual')
                # Plot predicted (green O)
                ax.scatter(pred_x, pred_y, c='lime', marker='o', s=100, linewidth=2, 
                          facecolors='none', edgecolors='lime', label='Predicted')
                
                ax.set_title(f'Sample {sample_count+1}', fontsize=12, fontweight='bold')
                ax.axis('off')
                if sample_count == 0:
                    ax.legend(loc='upper right', fontsize=10)
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
            
            if sample_count >= num_samples:
                break
    
    # Hide unused subplots
    for j in range(sample_count, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_sample_predictions_visual.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_sample_predictions_visual.png")
    plt.close()

# --- 9. Visualization: Metrics Summary ---
def plot_metrics_summary(predictions, actuals):
    """Plot MAE, RMSE, and R² metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae_scores = [mean_absolute_error(actuals[:, i], predictions[:, i]) for i in range(3)]
    rmse_scores = [np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i])) for i in range(3)]
    r2_scores = [r2_score(actuals[:, i], predictions[:, i]) for i in range(3)]
    
    labels = ['X', 'Y', 'Distance']
    x_pos = np.arange(len(labels))
    width = 0.25
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE
    ax1.bar(x_pos, mae_scores, width, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Output', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_scores):
        ax1.text(i, v + 0.001, f'{v:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    ax2.bar(x_pos, rmse_scores, width, color='salmon', edgecolor='black')
    ax2.set_xlabel('Output', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_scores):
        ax2.text(i, v + 0.001, f'{v:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    ax3.bar(x_pos, r2_scores, width, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Output', fontsize=12, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax3.set_title('R² Score (1.0 = Perfect)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(r2_scores):
        ax3.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_metrics_summary.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_metrics_summary.png")
    plt.close()
    
    # Print metrics to console
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    for i, label in enumerate(labels):
        print(f"\n{label}:")
        print(f"  MAE:  {mae_scores[i]:.6f}")
        print(f"  RMSE: {rmse_scores[i]:.6f}")
        print(f"  R²:   {r2_scores[i]:.6f}")

# --- 10. Save Predictions to CSV ---
def save_predictions_to_csv(predictions, actuals, image_ids, set_name='test'):
    """Save predictions and actuals to CSV for reference."""
    df = pd.DataFrame({
        'ImageID': image_ids,
        'Actual_X': actuals[:, 0],
        'Actual_Y': actuals[:, 1],
        'Actual_Distance': actuals[:, 2],
        'Predicted_X': predictions[:, 0],
        'Predicted_Y': predictions[:, 1],
        'Predicted_Distance': predictions[:, 2],
        'Error_X': actuals[:, 0] - predictions[:, 0],
        'Error_Y': actuals[:, 1] - predictions[:, 1],
        'Error_Distance': actuals[:, 2] - predictions[:, 2],
    })
    
    csv_path = os.path.join(OUTPUT_DIR, f'predictions_{set_name}_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed {set_name} predictions: {csv_path}")
    print(f"\nFirst 10 {set_name} predictions:")
    print(df.head(10).to_string())

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading trained model...")
    model = load_model()
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"Using device: {DEVICE}")
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_pred, test_actual, test_ids, test_dataset = predict_on_test_set(model)
    print(f"✓ Generated {len(test_pred)} test predictions")
    
    print("\n" + "="*60)
    print("EVALUATING ON VALIDATION SET")
    print("="*60)
    val_pred, val_actual, val_ids, val_dataset = predict_on_val_set(model)
    print(f"✓ Generated {len(val_pred)} validation predictions")
    
    print("\n" + "="*60)
    print("GENERATING TEST SET VISUALIZATIONS")
    print("="*60)
    plot_predictions_vs_actuals(test_pred, test_actual)
    plot_error_distribution(test_pred, test_actual)
    plot_residuals(test_pred, test_actual)
    plot_sample_predictions(model, test_dataset, num_samples=20)
    plot_metrics_summary(test_pred, test_actual)
    save_predictions_to_csv(test_pred, test_actual, test_ids, set_name='test')
    
    print("\n" + "="*60)
    print("GENERATING VALIDATION SET VISUALIZATIONS")
    print("="*60)
    plot_predictions_vs_actuals(val_pred, val_actual)
    plot_error_distribution(val_pred, val_actual)
    plot_residuals(val_pred, val_actual)
    plot_sample_predictions(model, val_dataset, num_samples=20)
    plot_metrics_summary(val_pred, val_actual)
    save_predictions_to_csv(val_pred, val_actual, val_ids, set_name='validation')
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETED!")
    print(f"✓ Check the '{OUTPUT_DIR}' folder for all visualization graphs")
    print("="*60)
