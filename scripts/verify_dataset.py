
from pathlib import Path

ROOT = Path(r"M:\currency-detector\dataset")
SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

errors = 0

for split in SPLITS:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    label_names = {p.stem for p in labels}
    image_names = {p.stem for p in images}

    missing = image_names - label_names
    extra = label_names - image_names

    if missing:
        print("Missing labels:", missing)
        errors += len(missing)

    if extra:
        print("Labels without images:", extra)
        errors += len(extra)

    for label_file in labels:
        for line_number, line in enumerate(
            label_file.read_text().splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                print(f"Invalid format: {label_file.name}, line {line_number}")
                errors += 1
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                if class_id != 0:
                    raise ValueError("Unexpected class ID")

                if not all(0 <= value <= 1 for value in (x, y, w, h)):
                    raise ValueError("Coordinates outside 0-1")

                if w <= 0 or h <= 0:
                    raise ValueError("Invalid box dimensions")

            except ValueError as error:
                print(f"Invalid annotation: {label_file.name}: {error}")
                errors += 1

print("\n-------------------------")

if errors == 0:
    print("SUCCESS: Dataset integrity checks passed!")
else:
    print(f"FOUND {errors} PROBLEM(S). Please review them.")
