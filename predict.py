# predict.py
import torch
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_model.pth"
IMAGE_SIZE = 224

# LABEL MAP (CONFIRMED)
IDX_TO_LABEL = {
    0: "Healthy Leaf",
    1: "Leaf Spot",
    2: "Rust",
    3: "Stem Rot",
    4: "Nutrient Deficiency"
}

# --------------------------------------------------
# IMAGE TRANSFORM
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
# LOAD DATASET (FOR WEATHER + SOIL LOOKUP)
# --------------------------------------------------
dataset = GroundnutMultimodalDataset()

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
# PREDICT FUNCTION
# --------------------------------------------------
def predict(image_path, district):
    # -------- Load image --------
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    # -------- Get env + soil --------
    env_seq = dataset._get_env_sequence(district)
    soil_vec = dataset._get_soil(district)

    env_seq = torch.tensor(env_seq).unsqueeze(0).to(DEVICE)
    soil_vec = torch.tensor(soil_vec).unsqueeze(0).to(DEVICE)

    # -------- Forward --------
    with torch.no_grad():
        outputs = model(img, env_seq, soil_vec)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return IDX_TO_LABEL[pred.item()], conf.item()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "C:/Somalingaiah/minor_proj_MSc/groundnut_proj/data/cleaned/nutrition deficiency/21.JPG"   
    DISTRICT = "Bagalkot"                          

    disease, confidence = predict(IMAGE_PATH, DISTRICT)

    print("\n===== PREDICTION RESULT =====")
    print(f"Predicted Disease : {disease}")
    print(f"Confidence        : {confidence*100:.2f}%")
