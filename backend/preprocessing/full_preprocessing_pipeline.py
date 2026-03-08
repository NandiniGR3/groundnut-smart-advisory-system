import os
import cv2
import hashlib
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------
# PATHS
# ------------------------
RAW_DATA = "data/kaggle_raw"
CLEAN_DATA = "data/cleaned"

TARGET_SIZE = (224, 224)


# ------------------------
# FUNCTION 1 – Count Raw Images
# ------------------------
def count_raw_images():
    print("\n Counting RAW images before preprocessing...")

    raw_counts = {}
    total = 0

    for root, dirs, files in os.walk(RAW_DATA):
        label = os.path.basename(root)
        count = len([f for f in files if f.lower().endswith(("jpg", "png", "jpeg"))])
        if count > 0:
            raw_counts[label] = count
            total += count

    print("\n--- RAW DATA COUNT ---")
    for label, count in raw_counts.items():
        print(f"{label} : {count}")

    print(f"\nTOTAL RAW IMAGES = {total}\n")
    return raw_counts, total


# ------------------------
# FUNCTION 2 – Clean + Resize Images
# ------------------------
def clean_and_resize_images():
    print("\n Cleaning & Standardizing images...")

    if not os.path.exists(CLEAN_DATA):
        os.makedirs(CLEAN_DATA)

    removed = 0

    for root, dirs, files in os.walk(RAW_DATA):
        label = os.path.basename(root)

        if len(files) == 0:
            continue

        save_dir = os.path.join(CLEAN_DATA, label)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in files:
            if not img_name.lower().endswith(("jpg", "jpeg", "png")):
                continue

            src_path = os.path.join(root, img_name)
            dst_path = os.path.join(save_dir, img_name)

            try:
                img = Image.open(src_path)

                # Skip corrupt images
                img.verify()
                img = Image.open(src_path).convert("RGB")

                # Resize
                img = img.resize(TARGET_SIZE)

                img.save(dst_path)

            except Exception:
                removed += 1
                continue

    print(f" Cleaning Completed — Removed Corrupt Images: {removed}")
    return removed


# ------------------------
# FUNCTION 3 – Remove Duplicate Images
# ------------------------
def remove_duplicates():
    print("\n Removing duplicate images...")

    hashes = set()
    removed = 0

    for label_folder in os.listdir(CLEAN_DATA):
        folder_path = os.path.join(CLEAN_DATA, label_folder)

        for img_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, img_name)

            with open(full_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                os.remove(full_path)
                removed += 1
            else:
                hashes.add(file_hash)

    print(f" Duplicate Removal Completed — Removed: {removed}")
    return removed


# ------------------------
# FUNCTION 4 – Balance Classes (Augmentation)
# ------------------------
def balance_classes():
    print("\n Balancing class distribution using augmentation...")

    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    counts = {cls: len(os.listdir(os.path.join(CLEAN_DATA, cls)))
              for cls in os.listdir(CLEAN_DATA)}

    max_count = max(counts.values())
    augmented_total = 0

    for cls, count in counts.items():
        if count == 0:
            continue

        folder = os.path.join(CLEAN_DATA, cls)
        images = os.listdir(folder)

        needed = max_count - count

        if needed <= 0:
            continue

        print(f"  Augmenting {cls}: Need {needed} more images")

        i = 0
        while i < needed:
            img = cv2.imread(os.path.join(folder, images[i % len(images)]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, 0)

            for batch in datagen.flow(img, batch_size=1, save_to_dir=folder,
                                      save_prefix="aug_", save_format="jpg"):
                i += 1
                augmented_total += 1
                if i >= needed:
                    break

    print(f"Augmentation Completed — Added: {augmented_total}")
    return augmented_total


# ------------------------
# FUNCTION 5 – Count Cleaned Images
# ------------------------
def count_cleaned_images():
    print("\n Counting CLEANED images after preprocessing...")

    clean_counts = {}
    total = 0

    for cls in os.listdir(CLEAN_DATA):
        folder = os.path.join(CLEAN_DATA, cls)
        if os.path.isdir(folder):
            count = len(os.listdir(folder))
            clean_counts[cls] = count
            total += count

    print("\n--- CLEANED DATA COUNT ---")
    for cls, count in clean_counts.items():
        print(f"{cls} : {count}")

    print(f"\nTOTAL CLEANED IMAGES = {total}\n")
    return clean_counts, total



# ------------------------
# FUNCTION 6 – SUMMARY
# ------------------------
def summary(raw_total, cleaned_total, removed_corrupt, removed_dup, augmented):
    total_removed = removed_corrupt + removed_dup

    print("\n================= SUMMARY =================")
    print(f"RAW IMAGES TOTAL             : {raw_total}")
    print(f"CLEANED IMAGES TOTAL         : {cleaned_total}")
    print(f"REMOVED (Corrupt+Duplicates) : {total_removed}")
    print(f"AUGMENTED IMAGES             : {augmented}")
    print("===========================================\n")


# ------------------------
# MAIN PIPELINE
# ------------------------
if __name__ == "__main__":
    print("\n=========== IMAGE PREPROCESSING STARTED ===========")

    # 1) Raw before preprocessing
    raw_counts, raw_total = count_raw_images()

    # 2) Clean images
    removed_corrupt = clean_and_resize_images()

    # 3) Remove duplicates
    removed_dup = remove_duplicates()

    # 4) Balance classes
    augmented = balance_classes()

    # 5) Count after preprocessing
    clean_counts, cleaned_total = count_cleaned_images()

    # 6) Summary
    summary(raw_total, cleaned_total, removed_corrupt, removed_dup, augmented)

    print("=========== IMAGE PREPROCESSING COMPLETED ===========")
