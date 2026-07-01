from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent

# Source folders
SOURCE_FOLDERS = [
    "Earlier_Collection",
    "Puneet_Collection_1",
    "Puneet_Collection_2"
]

DEST_FOLDER = BASE_DIR / "layer_0_raw_images"
DEST_FOLDER.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".heic",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif"
}

all_images = []

for folder_name in SOURCE_FOLDERS:
    folder_path = BASE_DIR / folder_name

    if not folder_path.exists():
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            all_images.append(file_path)

all_images.sort(key=lambda x: str(x).lower())

for index, image_path in enumerate(all_images, start=1):
    extension = image_path.suffix.lower()

    new_filename = f"IMG_{index:05d}{extension}"
    destination_path = DEST_FOLDER / new_filename

    shutil.copy2(image_path, destination_path)

    print(f"Copied: {image_path} -> {destination_path}")

print(f"\nDone. Total images copied: {len(all_images)}")
print(f"Images saved in: {DEST_FOLDER}")