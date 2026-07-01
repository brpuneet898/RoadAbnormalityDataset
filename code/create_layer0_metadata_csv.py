import csv
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def create_image_metadata_csv():
    project_root = Path(__file__).resolve().parent.parent
    image_folder = project_root / "layer_0_raw_images"

    output_csv = image_folder / "layer_0_raw_image_metadata.csv"

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm",
        ".heic", ".heif"
    }

    rows = []

    for image_path in sorted(image_folder.iterdir()):
        if not image_path.is_file():
            continue

        file_extension = image_path.suffix.lower()

        if file_extension not in image_extensions:
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size

            file_size_mb = image_path.stat().st_size / (1024 * 1024)

            rows.append({
                "image_id": image_path.stem,
                "file_type": file_extension.replace(".", ""),
                "file_size": round(file_size_mb, 4),
                "resolution_width": width,
                "resolution_height": height
            })

        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "image_id",
            "file_type",
            "file_size",
            "resolution_width",
            "resolution_height"
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV created successfully: {output_csv}")
    print(f"Total images processed: {len(rows)}")


if __name__ == "__main__":
    create_image_metadata_csv()