import os
import time
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# =========================
# DATASET
# =========================
class DockingDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, str(row['ImageID']))
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Missing image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([
            row["x_norm"],
            row["y_norm"],
            row["distance_norm"]
        ], dtype=torch.float32)

        return image, target

# =========================
# LOSS
# =========================
def loss_fn(pred, target):
    loss_xy = nn.MSELoss()(pred[:, :2], target[:, :2])
    loss_d  = nn.MSELoss()(pred[:, 2], target[:, 2])
    return loss_xy + 0.3 * loss_d, loss_xy, loss_d

# =========================
# MAIN
# =========================
def main():

    IMG_SIZE = 224
    BATCH_SIZE = 32
    MAX_EPOCHS = 200
    PATIENCE = 15

    # Get script directory for all paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(script_dir, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(script_dir, "best_model.pth")
    RUNS_DIR = os.path.join(script_dir, "runs")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using Device: {DEVICE}\n")

    # AMP (NEW API)
    if DEVICE.type == "cuda":
        from torch.amp import autocast, GradScaler
        scaler = GradScaler("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        autocast = None
        scaler = None

    # =========================
    # DATA
    # =========================
    train_csv_path = os.path.join(script_dir, "Preprocess/train_processed.csv")
    img_dir = os.path.join(script_dir, "Preprocess/images")
    
    df = pd.read_csv(train_csv_path)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_loader = DataLoader(
        DockingDataset(train_df, img_dir, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        DockingDataset(val_df, img_dir, val_transform),
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )

    # =========================
    # MODEL
    # =========================
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 3),
        nn.Sigmoid()
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # =========================
    # RESUME
    # =========================
    start_epoch = 0
    best_val = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        print("🔄 Resuming from checkpoint...\n")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint["best_val"]

    patience_counter = 0
    global_start = time.time()

    # =========================
    # TENSORBOARD
    # =========================
    writer = SummaryWriter(RUNS_DIR)
    print(f"📊 TensorBoard logs saved to: {RUNS_DIR}\n")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(start_epoch, MAX_EPOCHS):

        epoch_start = time.time()
        model.train()
        train_loss = 0
        train_loss_xy = 0
        train_loss_d = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):

            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            if DEVICE.type == "cuda":
                with autocast("cuda"):
                    preds = model(images)
                    loss, loss_xy, loss_d = loss_fn(preds, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss, loss_xy, loss_d = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_loss_xy += loss_xy.item()
            train_loss_d += loss_d.item()

        # VALIDATION
        model.eval()
        val_loss = 0
        val_loss_xy = 0
        val_loss_d = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(images)
                loss, loss_xy, loss_d = loss_fn(preds, targets)
                val_loss += loss.item()
                val_loss_xy += loss_xy.item()
                val_loss_d += loss_d.item()

        train_loss /= len(train_loader)
        train_loss_xy /= len(train_loader)
        train_loss_d /= len(train_loader)
        val_loss /= len(val_loader)
        val_loss_xy /= len(val_loader)
        val_loss_d /= len(val_loader)

        # TENSORBOARD LOGGING
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/val_total", val_loss, epoch)
        writer.add_scalar("Loss/train_xy", train_loss_xy, epoch)
        writer.add_scalar("Loss/train_distance", train_loss_d, epoch)
        writer.add_scalar("Loss/val_xy", val_loss_xy, epoch)
        writer.add_scalar("Loss/val_distance", val_loss_d, epoch)

        # SAVE CHECKPOINT
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val
        }, CHECKPOINT_PATH)

        # BEST MODEL
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            status = "✅ Improved"
        else:
            patience_counter += 1
            status = "⚠️ No Improvement"

        # REPORT
        epoch_time = time.time() - epoch_start
        total_time = (time.time() - global_start) / 60

        print("\n" + "="*50)
        print(f"📌 Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"📉 Train Loss : {train_loss:.6f}")
        print(f"📈 Val Loss   : {val_loss:.6f}")
        print(f"🏆 Best Loss  : {best_val:.6f}")
        print(f"⏱ Time       : {epoch_time:.2f} sec")
        print(f"⏳ Total Time : {total_time:.2f} min")
        print(f"📊 Status     : {status}")
        print("="*50)

        # MORE TENSORBOARD METRICS
        writer.add_scalar("Metrics/patience_counter", patience_counter, epoch)
        writer.add_scalar("Metrics/best_val_loss", best_val, epoch)
        writer.add_scalar("Metrics/epoch_time_sec", epoch_time, epoch)
        writer.add_scalar("Metrics/total_time_min", total_time, epoch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Optimizer/learning_rate", current_lr, epoch)
        
        # Flush writer for live updates
        writer.flush()

        if patience_counter >= PATIENCE:
            print("\n🛑 Early stopping triggered")
            break

    writer.flush()
    writer.close()
    print("\n🚀 Training Complete")


# =========================
# ENTRY POINT (CRITICAL)
# =========================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()