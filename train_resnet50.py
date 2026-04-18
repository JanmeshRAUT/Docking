import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datapreprocess import get_dataloaders
import os

# --- 1. Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4 # 10^-4
MODEL_SAVE_PATH = "resnet50_docking_best.pth"

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    best_loss = float('inf')

    print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
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

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # --- Save Best Model ---
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved better weights to {MODEL_SAVE_PATH}")

    print("\nTraining Finished.")

if __name__ == "__main__":
    train_model()
