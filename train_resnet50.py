import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

TRAIN_IMG_DIR = "Train/images"
TRAIN_CSV_PATH = "Train/train.csv"

VAL_IMG_DIR = "Validation/images"
VAL_CSV_PATH = "Validation/validation.csv"

TEST_IMG_DIR = "Test/images"
TEST_CSV_PATH = "Test/test.csv"

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

        img_path = os.path.join(self.img_dir, row["ImageID"])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

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
    return loss_xy + 0.3 * loss_d

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    import torch.multiprocessing as mp
    mp.freeze_support()

    # =========================
    # FORCE CUDA
    # =========================
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available")

    DEVICE = torch.device("cuda")

    # =========================
    # LOAD DATA
    # =========================
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # =========================
    # TRANSFORMS
    # =========================
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # =========================
    # LOADERS
    # =========================
    train_loader = DataLoader(
        DockingDataset(train_df, TRAIN_IMG_DIR, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        DockingDataset(val_df, VAL_IMG_DIR, val_transform),
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    test_loader = DataLoader(
        DockingDataset(test_df, TEST_IMG_DIR, val_transform),
        batch_size=BATCH_SIZE,
        num_workers=0
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

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_val_loss = float("inf")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(images)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # =========================
        # VALIDATION + TEST
        # =========================
        model.eval()
        val_loss = 0
        test_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(images)
                loss = loss_fn(preds, targets)
                val_loss += loss.item()

            for images, targets in test_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(images)
                loss = loss_fn(preds, targets)
                test_loss += loss.item()

        scheduler.step()

        val_loss /= len(val_loader)
        test_loss /= len(test_loader)
        train_loss /= len(train_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    print("\n🚀 Training Complete")