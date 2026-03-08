import streamlit as st
import pandas as pd
import torch
import json
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from datasets_model import GroundnutMultimodalDataset
from model.pytorch.custom_model import GroundnutCustomMultimodalModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Groundnut Smart Advisory", layout="wide")

DEVICE = "cpu"
MODEL_PATH = "checkpoints/best_model.pth"
PRICE_CSV = "data/mandi_prices/groundnut_karnataka.csv"
JSON_PATH = "data/disease_knowledge.json"
CONF_THRESHOLD = 50  # academic safety threshold

# --------------------------------------------------
# LANGUAGE SELECTION (MANUAL, NOT GOOGLE TRANSLATE)
# --------------------------------------------------
LANG = st.sidebar.selectbox("Language / ಭಾಷೆ", ["English", "Kannada"])

def tr(en, kn):
    return en if LANG == "English" else kn

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
price_df = pd.read_csv(PRICE_CSV)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DISEASE_INFO = json.load(f)

dataset = GroundnutMultimodalDataset()

model = GroundnutCustomMultimodalModel(
    num_env_features=len(dataset.env_features),
    num_soil_features=len(dataset.soil_features),
    num_classes=5
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

IDX_TO_LABEL = {
    0: "Healthy Leaf",
    1: "Leaf Spot",
    2: "Rust",
    3: "Stem Rot",
    4: "Nutrient Deficiency"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title(tr("Options", "ಆಯ್ಕೆಗಳು"))

district = st.sidebar.selectbox(
    tr("Select District", "ಜಿಲ್ಲೆ ಆಯ್ಕೆಮಾಡಿ"),
    sorted(price_df["District"].unique())
)

uploaded_image = st.sidebar.file_uploader(
    tr("Upload Leaf Image (Optional)", "ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (ಐಚ್ಛಿಕ)"),
    type=["jpg", "png", "jpeg"]
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title(tr(
    "🌱 Groundnut Smart Advisory System",
    "🌱 ಕಡಲೆಕಾಯಿ ಸ್ಮಾರ್ಟ್ ಸಲಹಾ ವ್ಯವಸ್ಥೆ"
))

# --------------------------------------------------
# EMPTY LAND MODE
# --------------------------------------------------
if uploaded_image is None:
    st.subheader(tr("Empty Land Advisory", "ಖಾಲಿ ಜಮೀನಿನ ಸಲಹೆ"))

    st.markdown("""
    **Recommended Practices**
    - Soil Preparation: Deep ploughing + organic manure  
    - Sowing Time: June–July  
    - Seed Rate: 10–20 kg per acre  
    - Irrigation: Every 7–10 days  
    - Harvest: 110–120 days
    """)

    st.info(tr(
        "Expected Yield: 8–10 Quintals per acre",
        "ಅಂದಾಜು ಉತ್ಪಾದನೆ: ಎಕರೆಗೆ 8–10 ಕ್ವಿಂಟಲ್"
    ))

# --------------------------------------------------
# DISEASE MODE
# --------------------------------------------------
else:
    st.subheader(tr("Disease Detection Result", "ರೋಗ ಪತ್ತೆ ಫಲಿತಾಂಶ"))

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0)
    env = torch.tensor(dataset._get_env_sequence(district)).unsqueeze(0)
    soil = torch.tensor(dataset._get_soil(district)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor, env, soil)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    disease = IDX_TO_LABEL[pred.item()]
    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        st.warning(tr(
            "⚠️ Low confidence prediction. Please upload a clearer image.",
            "⚠️ ನಿಖರತೆ ಕಡಿಮೆ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
        ))
    else:
        st.success(f"{tr('Predicted Disease', 'ಅಂದಾಜು ರೋಗ')}: **{disease}**")
        st.write(tr("Confidence", "ನಂಬಿಕೆ"), f": {confidence:.2f}%")

        info = DISEASE_INFO[disease]

        st.markdown("### 🦠 " + tr("Cause", "ಕಾರಣ"))
        st.info(info["cause"]["en"] if LANG=="English" else info["cause"]["kn"])

        st.markdown("### ⚠️ " + tr("Impact", "ಪ್ರಭಾವ"))
        st.warning(info["impact"]["en"] if LANG=="English" else info["impact"]["kn"])

        st.markdown("### 🛡️ " + tr("Precaution", "ಮುನ್ನೆಚ್ಚರಿಕೆ"))
        st.success(info["precaution"]["en"] if LANG=="English" else info["precaution"]["kn"])

# --------------------------------------------------
# PRICE ANALYSIS
# --------------------------------------------------
st.subheader(tr("Mandi Price Analysis", "ಮಂಡಿ ಬೆಲೆ ವಿಶ್ಲೇಷಣೆ"))

district_prices = price_df[price_df["District"] == district]
modal_price = district_prices["Modal_Price_Rs_Quintal"].mean()

st.metric(
    tr("Modal Price (₹ / Quintal)", "ಸರಾಸರಿ ಬೆಲೆ (₹ / ಕ್ವಿಂಟಲ್)"),
    f"₹ {modal_price:.0f}"
)

fig, ax = plt.subplots(figsize=(10, 4))
price_df.groupby("District")["Modal_Price_Rs_Quintal"].mean().plot(kind="bar", ax=ax)
ax.set_ylabel("₹ / Quintal")
ax.set_title("Groundnut Mandi Prices - Karnataka")
st.pyplot(fig)

# --------------------------------------------------
# PROFIT CALCULATOR
# --------------------------------------------------
st.subheader(tr("Profit Calculator", "ಲಾಭ ಲೆಕ್ಕಾಚಾರ"))

acre = st.slider(tr("Land Area (Acres)", "ಭೂಮಿ (ಎಕರೆ)"), 1, 20, 1)
yield_qtl = st.slider(tr("Yield (Quintals / Acre)", "ಉತ್ಪಾದನೆ"), 6, 12, 8)

income = acre * yield_qtl * modal_price

st.success(tr(
    f"Estimated Income: ₹ {income:,.0f}",
    f"ಅಂದಾಜು ಆದಾಯ: ₹ {income:,.0f}"
))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(tr(
    "Academic Project – MSc Data Science | Groundnut Disease Detection System",
    "ಶೈಕ್ಷಣಿಕ ಯೋಜನೆ – ಎಂಎಸ್‌ಸಿ ಡೇಟಾ ಸೈನ್ಸ್"
))
