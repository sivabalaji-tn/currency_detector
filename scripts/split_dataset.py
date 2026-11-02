
from pathlib import Path
import random
import shutil

# Project paths
ROOT = Path(r"M:\currency-detector")
COLLECTION = ROOT / "dataset" / "collection"
DATASET = ROOT / "dataset"

# Reproducible random split
random.seed(42)

SPLITS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Collect all images from positive and negative folders
images = []

for category in ["positive", "negative"]:
    category_folder = COLLECTION / category

    for image_path in category_folder.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(image_path)

# Check matching annotation files
valid_images = []

for image_path in images:
    label_path = image_path.with_suffix(".txt")

    if label_path.exists():
        valid_images.append(image_path)
    else:
        print(f"Missing label: {image_path.name}")

if not valid_images:
    raise SystemExit("No valid images with matching label files found.")

# Shuffle images
random.shuffle(valid_images)

total = len(valid_images)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.20)

split_images = {
    "train": valid_images[:train_end],
    "val": valid_images[train_end:val_end],
    "test": valid_images[val_end:]
}

# Copy images and labels
for split_name, split_list in split_images.items():

    image_output = DATASET / "images" / split_name
    label_output = DATASET / "labels" / split_name

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_path in split_list:
        label_path = image_path.with_suffix(".txt")

        # Use unique names to avoid filename collisions
        unique_name = f"{image_path.parent.name}_{image_path.stem}"

        shutil.copy2(
            image_path,
            image_output / f"{unique_name}{image_path.suffix.lower()}"
        )

        shutil.copy2(
            label_path,
            label_output / f"{unique_name}.txt"
        )

# Summary
print("\nDATASET SPLIT COMPLETED")
print("-----------------------")
print(f"Total images: {total}")

for split_name, split_list in split_images.items():
    print(f"{split_name}: {len(split_list)} images")

print("\nOriginal collection remains unchanged.")
