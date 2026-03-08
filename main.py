'''import os

folders = [
    "backend", "frontend", 
    "model/pytorch", "model/tensorflow", "model/ml_models", "model/fusion",
    "data/sentinel_images", "data/nasa_images", "data/gee_images",
    "data/leaf_images", "data/soil_data", "data/climate_yield", "data/labels",
    "config"
]

base_path = "C:/Somalingaiah/minor_proj_MSc/groundnut_proj"

for f in folders:
    os.makedirs(os.path.join(base_path, f), exist_ok=True)

print("All folders created successfully!")'''


from backend.drive_connect import connect_drive

if __name__ == "__main__":
    drive = connect_drive()
    print("Drive connected!")

