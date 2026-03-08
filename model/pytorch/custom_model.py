import torch
import torch.nn as nn


# -------------------------------------------------------
# Custom CNN Encoder (FROM SCRATCH)
# -------------------------------------------------------
class CustomCNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------------------------------------------
# LSTM Encoder (Weather + NDVI + EVI)
# -------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # last hidden state


# -------------------------------------------------------
# Soil Encoder (Tabular Data)
# -------------------------------------------------------
class SoilEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=64):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


# -------------------------------------------------------
# MULTIMODAL FUSION MODEL
# -------------------------------------------------------
class GroundnutCustomMultimodalModel(nn.Module):
    def __init__(
        self,
        num_env_features,
        num_soil_features,
        num_classes=5  # default to 5 classes
    ):
        super().__init__()

        self.image_encoder = CustomCNNEncoder()
        self.env_encoder = LSTMEncoder(num_env_features)
        self.soil_encoder = SoilEncoder(num_soil_features)

        fusion_dim = 256 + 128 + 64

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    # --------------------------------------------------
    # Weight Initialization
    # --------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, image, env_seq, soil):
        img_feat = self.image_encoder(image)
        env_feat = self.env_encoder(env_seq)
        soil_feat = self.soil_encoder(soil)

        fused = torch.cat([img_feat, env_feat, soil_feat], dim=1)
        return self.classifier(fused)


# -------------------------------------------------------
# SANITY CHECK
# -------------------------------------------------------
if __name__ == "__main__":
    model = GroundnutCustomMultimodalModel(
        num_env_features=10,
        num_soil_features=5,
        num_classes=5
    )

    img = torch.randn(2, 3, 224, 224)
    env = torch.randn(2, 6, 10)
    soil = torch.randn(2, 5)

    out = model(img, env, soil)
    print("Output shape:", out.shape)  # (2, 5)
