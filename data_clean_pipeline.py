import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


class GroundnutWeatherDataset(Dataset):
    def __init__(self, csv_path, target_col=None):
        """
        Args:
            csv_path (str): path to weather_clean.csv
            target_col (str or None): column name to predict (optional)
        """
        self.df = pd.read_csv(csv_path)

        # -----------------------------
        # Handle date
        # -----------------------------
        self.df["year_month"] = pd.to_datetime(self.df["year_month"], errors="coerce")
        self.df["year"] = self.df["year_month"].dt.year
        self.df["month"] = self.df["year_month"].dt.month
        self.df.drop(columns=["year_month"], inplace=True)

        # -----------------------------
        # Encode district (multiple districts supported)
        # -----------------------------
        self.le = LabelEncoder()
        self.df["district_encoded"] = self.le.fit_transform(self.df["district"])
        self.df.drop(columns=["district"], inplace=True)

        # -----------------------------
        # Handle missing NDVI / EVI
        # -----------------------------
        if "ndvi" in self.df.columns:
            self.df["ndvi"] = self.df["ndvi"].fillna(0.0)
        if "evi" in self.df.columns:
            self.df["evi"] = self.df["evi"].fillna(0.0)

        # -----------------------------
        # Feature columns
        # -----------------------------
        self.feature_cols = [
            "t2m",
            "prectotcorr",
            "rh2m",
            "ws2m",
            "t2m_max",
            "t2m_min",
            "ndvi",
            "evi",
            "year",
            "month",
            "district_encoded"
        ]

        # Ensure columns exist
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]

        X = self.df[self.feature_cols].values.astype(np.float32)

        # -----------------------------
        # Scaling
        # -----------------------------
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)

        # -----------------------------
        # Target handling
        # -----------------------------
        self.target_col = target_col
        if target_col and target_col in self.df.columns:
            y = self.df[target_col].values.astype(np.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
