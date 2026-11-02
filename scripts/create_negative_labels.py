
from pathlib import Path

# Negative image collection folder
negative_folder = Path(r"M:\currency-detector\dataset\collection\negative")

image_extensions = {".jpg", ".jpeg", ".png"}

created = 0
skipped = 0

for image_path in negative_folder.rglob("*"):
    if image_path.is_file() and image_path.suffix.lower() in image_extensions:

        label_path = image_path.with_suffix(".txt")

        if label_path.exists():
            skipped += 1
            continue

        label_path.touch()
        created += 1

print(f"Empty label files created: {created}")
print(f"Existing label files skipped: {skipped}")
print("Done!")
