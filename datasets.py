import os
import shutil
import kagglehub

# ------------------------------------------------------
# 1. DOWNLOAD DATASETS FROM KAGGLE USING KAGGLEHUB
# ------------------------------------------------------

def download_kaggle_datasets():
    print("\n Downloading datasets from Kaggle...\n")

    datasets = {
        "dataset1": "muhammadazeemabbas/groundnut-leaves-dataset",
        "dataset2": "warcoder/groundnut-plant-leaf-data",
        "dataset3": "avyaya/groundnut"
    }

    kaggle_raw = "data/kaggle_raw"

    # Clean old data
    if os.path.exists(kaggle_raw):
        shutil.rmtree(kaggle_raw)
    os.makedirs(kaggle_raw, exist_ok=True)

    downloaded_paths = {}

    for name, kaggle_id in datasets.items():
        print(f" Downloading {name} from Kaggle: {kaggle_id}")
        path = kagglehub.dataset_download(kaggle_id)

        dst = os.path.join(kaggle_raw, name)
        shutil.copytree(path, dst)
        downloaded_paths[name] = dst

        print(f" Saved {name} → {dst}\n")

    return downloaded_paths


# ------------------------------------------------------
# 2. MERGE ALL DATASETS INTO A SINGLE FOLDER
# ------------------------------------------------------

def merge_datasets(downloaded_paths):
    print("\n Merging datasets...\n")

    leaf_images_folder = "data/leaf_images"

    # Clean existing merged folder
    if os.path.exists(leaf_images_folder):
        shutil.rmtree(leaf_images_folder)

    os.makedirs(leaf_images_folder, exist_ok=True)

    for idx, (name, src) in enumerate(downloaded_paths.items(), start=1):
        dst = os.path.join(leaf_images_folder, f"dataset{idx}")
        shutil.copytree(src, dst)
        print(f" Copied {name} -> {dst}")

    print("\n Dataset merging finished!\n")


# ------------------------------------------------------
# 3. SHOW FINAL FOLDER STRUCTURE
# ------------------------------------------------------

def show_folder_tree(path):
    print("\n Folder Structure:\n")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 8 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}    {f}")


# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    downloaded = download_kaggle_datasets()
    merge_datasets(downloaded)
    show_folder_tree("data/leaf_images")
