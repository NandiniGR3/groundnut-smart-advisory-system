DISEASE_INFO = {
    "Leaf Spot": {
        "causes": "High humidity, continuous cropping, infected seeds",
        "organic": "Neem oil spray, Trichoderma",
        "government": "ICAR recommends timely fungicide spraying"
    },
    "Rust": {
        "causes": "Cool temperature, excess moisture",
        "organic": "Sulphur-based organic sprays",
        "government": "KVK advisory for early detection"
    },
    "Blight": {
        "causes": "Bacterial infection, poor drainage",
        "organic": "Copper fungicide",
        "government": "Crop rotation advised"
    }
}

def get_disease_info(disease):
    return DISEASE_INFO.get(disease, {})
