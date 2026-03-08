import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --------------------------------------------------
# LABEL MAP (AS PROVIDED)
# --------------------------------------------------
LABEL_MAP = {
    "healthy_leaf": 0,
    "early_leaf_spot": 1,
    "early_leaf_spot_1": 1,
    "late_leaf_spot": 1,
    "late_leaf_spot_1": 1,
    "rust": 2,
    "rust_1": 2,
    "stem_rot": 3,
    "nutrition_deficiency": 4,
    "nutrition_deficiency_1": 4
}

# --------------------------------------------------
# DATASET CLASS
# --------------------------------------------------
class GroundnutMultimodalDataset(Dataset):
    def __init__(
        self,
        image_root="data/cleaned",
        env_csv="clean_data/environment_monthly_merged.csv",
        soil_csv="clean_data/soil_clean.csv",
        transform=None,
        seq_len=6
    ):
        self.image_root = image_root
        self.seq_len = seq_len

        # ==================================================
        # ENVIRONMENT DATA
        # ==================================================
        self.env_df = pd.read_csv(env_csv)

        # normalize column names (CRITICAL FIX)
        self.env_df.columns = [c.strip().lower() for c in self.env_df.columns]

        # date handling
        self.env_df["year_month"] = pd.to_datetime(
            self.env_df["year_month"], errors="coerce"
        )

        # explicit env features
        self.env_features = [
            "t2m",
            "prectotcorr",
            "rh2m",
            "ws2m",
            "t2m_max",
            "t2m_min",
            "ndvi",
            "evi"
        ]

        # keep only existing columns
        self.env_features = [
            c for c in self.env_features if c in self.env_df.columns
        ]

        # global fallback mean
        self.global_env_mean = (
            self.env_df[self.env_features]
            .mean()
            .fillna(0)
            .astype(np.float32)
            .values
        )

        # districts from ENV DATA (correct column)
        self.districts = (
            self.env_df["district"]
            .dropna()
            .unique()
            .tolist()
        )

        # ==================================================
        # SOIL DATA
        # ==================================================
        if os.path.exists(soil_csv):
            self.soil_df = pd.read_csv(soil_csv)

            # normalize columns
            self.soil_df.columns = [
                c.strip().lower() for c in self.soil_df.columns
            ]

            self.soil_features = ["ph", "organic_carbon"]
        else:
            self.soil_df = None
            self.soil_features = []

        # ==================================================
        # IMAGE TRANSFORM
        # ==================================================
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # ==================================================
        # BUILD SAMPLES
        # ==================================================
        self.samples = []
        self._build_samples()

        print(f"[INFO] Total samples loaded: {len(self.samples)}")
        print(f"[INFO] Env features: {len(self.env_features)}")
        print(f"[INFO] Soil features: {len(self.soil_features)}")

    # --------------------------------------------------
    # BUILD IMAGE SAMPLES
    # --------------------------------------------------
    def _build_samples(self):
        for disease in os.listdir(self.image_root):
            class_dir = os.path.join(self.image_root, disease)

            if not os.path.isdir(class_dir):
                continue
            if disease not in LABEL_MAP:
                continue

            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(class_dir, img_name)

                # detect district from filename
                district = None
                for d in self.districts:
                    if d.lower().replace(" ", "_") in img_name.lower():
                        district = d
                        break

                # fallback
                if district is None:
                    district = random.choice(self.districts)

                self.samples.append({
                    "image_path": img_path,
                    "label": LABEL_MAP[disease],
                    "district": district
                })

    # --------------------------------------------------
    # ENV SEQUENCE (LSTM INPUT)
    # --------------------------------------------------
    def _get_env_sequence(self, district):
        df = self.env_df[self.env_df["district"] == district]

        if len(df) < self.seq_len:
            seq = np.tile(self.global_env_mean, (self.seq_len, 1))
        else:
            seq = (
                df.sort_values("year_month")
                .iloc[-self.seq_len:][self.env_features]
                .fillna(0)
                .values
            )

        return seq.astype(np.float32)

    # --------------------------------------------------
    # SOIL FEATURES
    # --------------------------------------------------
    def _get_soil(self, district):
        if self.soil_df is None or len(self.soil_features) == 0:
            return np.zeros(len(self.soil_features), dtype=np.float32)

        row = self.soil_df[self.soil_df["district"] == district]

        if row.empty:
            return np.zeros(len(self.soil_features), dtype=np.float32)

        return row[self.soil_features].values[0].astype(np.float32)

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        env_seq = torch.tensor(
            self._get_env_sequence(sample["district"]),
            dtype=torch.float32
        )

        soil = torch.tensor(
            self._get_soil(sample["district"]),
            dtype=torch.float32
        )

        label = torch.tensor(sample["label"], dtype=torch.long)

        return image, env_seq, soil, label


# --------------------------------------------------
# SANITY TEST
# --------------------------------------------------
if __name__ == "__main__":
    ds = GroundnutMultimodalDataset()
    img, env, soil, label = ds[0]

    print("Image:", img.shape)
    print("Env:", env.shape)
    print("Soil:", soil.shape)
    print("Label:", label)

    assert img.shape == (3, 224, 224)
    assert env.shape == (6, len(ds.env_features))
    assert soil.shape == (len(ds.soil_features),)
    assert label.dtype == torch.long    