import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datapreprocess import get_dataloaders
import os

# --- 1. Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20  # Increased epochs with early stopping
LEARNING_RATE = 1e-4 # 10^-4
MODEL_SAVE_PATH = "resnet50_docking_best.pth"
CHECKPOINT_PATH = "training_checkpoint.pth"  # For resuming training
EARLY_STOP_PATIENCE = 15  # Stop if val loss doesn't improve for 15 epochs
L2_REGULARIZATION = 1e-5  # Weight decay to prevent overfitting
AUTO_RESUME = True  # Automatically resume from checkpoint if it exists

def train_model():
    # --- 2. Data Loaders ---
    train_loader, val_loader = get_dataloaders()

    # --- 3. Model Definition (ResNet50) ---
    model = models.resnet50(pretrained=True)
    
    # Modify the fully connected layer for regression (3 outputs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) # Predicts: [x, y, distance]
    model = model.to(DEVICE)

    # --- 4. Optimization ---
    criterion = nn.SmoothL1Loss() # More robust to outliers than MSE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

    # --- 5. Resume from Checkpoint ---
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    
    # Check if checkpoint exists
    checkpoint_exists = os.path.exists(CHECKPOINT_PATH)
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    
    if checkpoint_exists and AUTO_RESUME:
        print("="*60)
        print("CHECKPOINT FOUND! Resuming training...")
        print("="*60)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            patience_counter = checkpoint['patience_counter']
            
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            print(f"✓ Checkpoint loaded successfully!")
            print(f"  - Resuming from epoch: {start_epoch + 1}/{EPOCHS}")
            print(f"  - Best validation loss: {best_loss:.6f}")
            print(f"  - Patience counter: {patience_counter}/{EARLY_STOP_PATIENCE}")
            print("="*60 + "\n")
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            print("Starting fresh training...\n")
            start_epoch = 0
            best_loss = float('inf')
            patience_counter = 0
    elif model_exists and AUTO_RESUME:
        print(f"⚠ Model weights found but no checkpoint.")
        print("Starting fresh training from epoch 1...\n")
        start_epoch = 0
    else:
        print("Starting fresh training from epoch 1...\n")
        start_epoch = 0

    # --- 6. Training Loop ---
    print(f"Training on {DEVICE} for {EPOCHS} epochs (with Early Stopping)...")
    print(f"Early Stopping Patience: {EARLY_STOP_PATIENCE} epochs\n")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                v_loss = criterion(outputs, targets)
                running_val_loss += v_loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", end="")

        # --- Save Best Model & Early Stopping ---
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" ✓ Better weights saved", end="")
        else:
            patience_counter += 1
            print(f" (No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epochs)", end="")
        
        # --- Save Checkpoint after Every Epoch ---
        try:
            checkpoint_data = {
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'patience_counter': patience_counter,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss
            }
            torch.save(checkpoint_data, CHECKPOINT_PATH)
            print(" | Checkpoint saved")
        except Exception as e:
            print(f" | ⚠ Checkpoint save failed: {e}")
        
        # Early stopping check
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠ Early Stopping triggered! Validation loss did not improve for {EARLY_STOP_PATIENCE} epochs.")
            print(f"Best Model saved with Val Loss: {best_loss:.6f}")
            break

    print("\nTraining Finished.")

def show_checkpoint_info():
    """Display information about the saved checkpoint."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            print("\n" + "="*60)
            print("CHECKPOINT INFORMATION")
            print("="*60)
            print(f"Last Epoch: {checkpoint['epoch']}")
            print(f"Best Validation Loss: {checkpoint['best_loss']:.6f}")
            print(f"Patience Counter: {checkpoint['patience_counter']}/{EARLY_STOP_PATIENCE}")
            print(f"Last Training Loss: {checkpoint['train_loss']:.6f}")
            print(f"Last Validation Loss: {checkpoint['val_loss']:.6f}")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
    else:
        print("No checkpoint file found.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RESNET50 DOCKING POSITION PREDICTION - TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"L2 Regularization: {L2_REGULARIZATION}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"Auto Resume: {AUTO_RESUME}")
    print("="*60 + "\n")
    
    train_model()
    show_checkpoint_info()
