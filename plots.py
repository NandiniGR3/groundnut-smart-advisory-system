import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
LOG_PATH = "checkpoints/training_logs.npy"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD TRAINING LOGS
# --------------------------------------------------
if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(
        f"[ERROR] Training log not found at {LOG_PATH}. "
        "Ensure train.py saves training_logs.npy"
    )

logs = np.load(LOG_PATH, allow_pickle=True).item()

train_loss = logs.get("train_loss", [])
val_loss = logs.get("val_loss", [])
train_acc = logs.get("train_acc", [])
val_acc = logs.get("val_acc", [])

epochs = range(1, len(train_loss) + 1)

# --------------------------------------------------
# PLOT LOSS CURVES
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

# --------------------------------------------------
# PLOT ACCURACY CURVES
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

# --------------------------------------------------
# DONE
# --------------------------------------------------
print("[INFO] Plots saved successfully:")
print(" - outputs/loss_curve.png")
print(" - outputs/accuracy_curve.png")
