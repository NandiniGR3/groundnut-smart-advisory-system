# config/settings.py
LOCAL_BASE = "data"

DRIVE_BASE_FOLDER = "Groundnut_Project"
DRIVE_FOLDERS = {
    "leaf_images": "Leaf_Images",
    "soil_data": "Soil_Data",
    "sentinel_images": "Sentinel_Images",
    "nasa_images": "NASA_Images",
    "gee_images": "GEE_Images",
    "climate_yield": "Climate_Yield",
    "labels": "Labels",
    "reports": "Reports"
}

LANGUAGES = ["english", "kannada", "hindi"]

KARNATAKA_BBOX = [74.0, 11.5, 78.5, 18.5]

GROUNDNUT_VARIETIES = {
    "red soil": ["TMV 2", "K 6"],
    "black soil": ["JL 24", "K 9"],
    "sandy soil": ["AK 12", "KDG 1"]
}

DISEASE_CLASSES = ["Leaf Spot", "Rust", "Stem Rot", "Healthy", "Other"]

GAN_OUTPUT_DIR = "data/leaf_images/synthetic"
GAN_IMAGE_SIZE = (224, 224)
