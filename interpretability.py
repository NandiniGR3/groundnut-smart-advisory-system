# interpretability.py
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_model.pth"
BATCH_SIZE = 8

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
dataset = GroundnutMultimodalDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------------
# LOAD MODEL
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
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate(model, loader, mask_image=False, mask_env=False, mask_soil=False):
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, env, soil, label in loader:
            img, env, soil = img.to(DEVICE), env.to(DEVICE), soil.to(DEVICE)

            if mask_image:
                img = torch.zeros_like(img)
            if mask_env:
                env = torch.zeros_like(env)
            if mask_soil:
                soil = torch.zeros_like(soil)

            outputs = model(img, env, soil)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(label.numpy())
            y_pred.extend(preds.cpu().numpy())

    return accuracy_score(y_true, y_pred)

# --------------------------------------------------
# ABLATION STUDY
# --------------------------------------------------
full_acc = evaluate(model, loader)
no_image_acc = evaluate(model, loader, mask_image=True)
no_env_acc = evaluate(model, loader, mask_env=True)
no_soil_acc = evaluate(model, loader, mask_soil=True)

results = {
    "Configuration": [
        "Full Multimodal",
        "Without Image",
        "Without Weather",
        "Without Soil"
    ],
    "Accuracy": [
        full_acc,
        no_image_acc,
        no_env_acc,
        no_soil_acc
    ]
}

df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/interpretability_ablation.csv", index=False)

print("\n===== INTERPRETABILITY (ABLATION STUDY) =====")
print(df)
print("\n[INFO] Saved to outputs/interpretability_ablation.csv")
