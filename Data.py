import os
import random
import shutil

# ======================================================
# PATHS
# ======================================================

TRAIN_DIR = r"C:\Users\Nilesh\Downloads\archive (2)\SkinDisease\SkinDisease\train"

TEST_DIR = r"C:\Users\Nilesh\Downloads\archive (2)\SkinDisease\SkinDisease\test"

OUTPUT_DIR = r"C:\Users\Nilesh\Downloads\SkinDisease_Prepared"

# ======================================================
# CHANGE THESE TO MATCH YOUR EXACT FOLDER NAMES
# ======================================================

CLASSES = [
    "Unknown_Normal",
    "Benign_Tumors",
    "Vascular_Tumors",
    "Skin_Cancer"
]

# ======================================================

VALIDATION_SPLIT = 0.15
random.seed(42)

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp"
)

# ======================================================
# CREATE OUTPUT FOLDERS
# ======================================================

for split in ["train", "validation", "test"]:
    for cls in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

print("=" * 60)
print("Preparing Dataset...")
print("=" * 60)

# ======================================================
# TRAIN -> TRAIN + VALIDATION
# ======================================================

for cls in CLASSES:

    class_path = os.path.join(TRAIN_DIR, cls)

    if not os.path.exists(class_path):
        print(f"Folder not found : {class_path}")
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    val_size = int(len(images) * VALIDATION_SPLIT)

    validation_images = images[:val_size]
    train_images = images[val_size:]

    # Copy Training Images
    for img in train_images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "train", cls, img)
        )

    # Copy Validation Images
    for img in validation_images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "validation", cls, img)
        )

    print(f"{cls}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(validation_images)}")
    print()

# ======================================================
# COPY TEST SET AS IT IS
# ======================================================

for cls in CLASSES:

    class_path = os.path.join(TEST_DIR, cls)

    if not os.path.exists(class_path):
        print(f"Folder not found : {class_path}")
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for img in images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "test", cls, img)
        )

    print(f"{cls} Test Images : {len(images)}")

print("\n" + "=" * 60)
print("Dataset Prepared Successfully!")
print(f"Saved at : {OUTPUT_DIR}")
print("=" * 60)