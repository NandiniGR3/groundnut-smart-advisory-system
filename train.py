# train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# --------------------------------------------------
# PATH FIX (important for Windows)
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
SEQ_LEN = 6
NUM_CLASSES = 5
SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# --------------------------------------------------
# DATASET
# --------------------------------------------------
dataset = GroundnutMultimodalDataset(seq_len=SEQ_LEN)

print(f"[INFO] Total samples loaded: {len(dataset)}")
print(f"[INFO] Env features: {len(dataset.env_features)}")
print(f"[INFO] Soil features: {len(dataset.soil_features)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # IMPORTANT: Windows safe
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,      # IMPORTANT: Windows safe
    pin_memory=True
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = GroundnutCustomMultimodalModel(
    num_env_features=len(dataset.env_features),
    num_soil_features=len(dataset.soil_features),
    num_classes=NUM_CLASSES
).to(DEVICE)

# --------------------------------------------------
# LOSS & OPTIMIZER
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------
# TRAIN ONE EPOCH
# --------------------------------------------------
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, env, soil, labels in loader:
        images = images.to(DEVICE)
        env = env.to(DEVICE)
        soil = soil.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, env, soil)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total if total > 0 else 0
    return running_loss / len(loader), acc

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
def validate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, env, soil, labels in loader:
            images = images.to(DEVICE)
            env = env.to(DEVICE)
            soil = soil.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, env, soil)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0
    return running_loss / len(loader), acc

# --------------------------------------------------
# MAIN TRAIN LOOP (WINDOWS SAFE)
# --------------------------------------------------
def main():
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "best_model.pth")
            )
            print("[INFO] Best model saved")

    print("[INFO] Training completed successfully")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
