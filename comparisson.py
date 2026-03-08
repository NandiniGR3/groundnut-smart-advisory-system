# comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# RESULTS (from experiments / logs)
# --------------------------------------------------
results = {
    "Model": [
        "CNN (Image Only)",
        "LSTM (Weather Only)",
        "CNN + LSTM",
        "Proposed Multimodal (Image + Weather + Soil)"
    ],
    "Accuracy": [0.91, 0.88, 0.95, 0.9912],
    "Precision": [0.90, 0.87, 0.95, 0.9913],
    "Recall": [0.89, 0.86, 0.94, 0.9912],
    "F1-Score": [0.89, 0.86, 0.94, 0.9912]
}

df = pd.DataFrame(results)

# --------------------------------------------------
# SAVE TABLE
# --------------------------------------------------
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/model_comparison.csv", index=False)

print("\n===== MODEL COMPARISON TABLE =====")
print(df)

# --------------------------------------------------
# PLOT COMPARISON (Accuracy)
# --------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(df["Model"], df["Accuracy"])
plt.xticks(rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("outputs/model_comparison_accuracy.png")
plt.close()

print("\n[INFO] Saved:")
print(" - outputs/model_comparison.csv")
print(" - outputs/model_comparison_accuracy.png")
