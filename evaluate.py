# evaluate.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader

# --------------------------------------------------
# PATH FIX (VERY IMPORTANT)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "model", "pytorch"))

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
dataset = GroundnutMultimodalDataset()

test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size

_, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --------------------------------------------------
# LOAD MODEL (✔ CORRECT ARGUMENT NAMES)
# --------------------------------------------------
model = GroundnutCustomMultimodalModel(
    num_env_features=len(dataset.env_features),
    num_soil_features=len(dataset.soil_features),
    num_classes=5
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------------------------
# EVALUATION LOOP
# --------------------------------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, env_seq, soil, labels in test_loader:
        images = images.to(DEVICE)
        env_seq = env_seq.to(DEVICE)
        soil = soil.to(DEVICE)

        outputs = model(images, env_seq, soil)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n===== MODEL EVALUATION RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# --------------------------------------------------
# CONFUSION MATRIX (✔ FIXED)
# --------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("\n[INFO] Confusion matrix saved to outputs/confusion_matrix.png")
