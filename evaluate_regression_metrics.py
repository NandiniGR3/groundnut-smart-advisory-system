import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_model.pth"
OUTPUT_DIR = "output"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5   # ✅ 5 outputs (ordinal regression)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ordered severity levels
SEVERITY_LEVELS = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32).to(DEVICE)

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# DATASET & LOADER
# --------------------------------------------------
test_dataset = GroundnutMultimodalDataset(transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = GroundnutCustomMultimodalModel(
    num_env_features=len(test_dataset.env_features),
    num_soil_features=len(test_dataset.soil_features),
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------------------------
# NON-LINEAR ORDINAL REGRESSION EVALUATION
# --------------------------------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, weather, soil, targets in test_loader:

        images = images.to(DEVICE)
        weather = weather.to(DEVICE)
        soil = soil.to(DEVICE)

        outputs = model(images, weather, soil)  # (B, 5)

        # 🔥 Convert logits → probability distribution
        probs = F.softmax(outputs, dim=1)

        # 🔥 Expected severity (continuous, 0–4)
        severity_pred = torch.sum(probs * SEVERITY_LEVELS, dim=1)

        y_true.extend(targets.cpu().numpy())
        y_pred.extend(severity_pred.cpu().numpy())

# --------------------------------------------------
# METRICS
# --------------------------------------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
results_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

csv_path = os.path.join(OUTPUT_DIR, "regression_metrics.csv")
results_df.to_csv(csv_path, index=False)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print("\nNon-Linear Ordinal Regression Metrics (Disease Severity Index)")
print("-------------------------------------------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"\nSaved metrics to: {csv_path}")
