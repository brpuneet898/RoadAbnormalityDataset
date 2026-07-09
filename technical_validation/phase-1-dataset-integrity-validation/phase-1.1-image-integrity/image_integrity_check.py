from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import pandas as pd
import pillow_heif
pillow_heif.register_heif_opener()

LAYER_0_DIR = Path("../../../layer_0_raw_images")
CSV_PATH = LAYER_0_DIR / "layer_0_raw_image_metadata.csv"

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".heic"}
IMAGE_EXTENSIONS = SUPPORTED_FORMATS
IGNORED_EXTENSIONS = {".md"}

def find_column(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def main():
    image_files = [
        p for p in LAYER_0_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    all_files = [p for p in LAYER_0_DIR.rglob("*") if p.is_file()]
    invalid_format_files = [
        p for p in all_files
        if (
            p.suffix.lower() not in SUPPORTED_FORMATS
            and p.suffix.lower() not in IGNORED_EXTENSIONS
            and p.name != CSV_PATH.name
        )
    ]

    filename_counts = Counter(p.name.lower() for p in image_files)
    duplicate_filenames = sum(1 for count in filename_counts.values() if count > 1)

    corrupted_images = []
    resolution_mismatches = []
    successfully_opened = 0

    metadata_df = pd.read_csv(CSV_PATH)

    file_col = find_column(metadata_df, ["image_id"])
    width_col = find_column(metadata_df, ["resolution_width"])
    height_col = find_column(metadata_df, ["resolution_height"])

    if file_col is None or width_col is None or height_col is None:
        raise ValueError(
            "CSV must contain filename, resolution_width, and resolution_height columns."
        )

    metadata_lookup = {
        str(row[file_col]).strip(): (
            int(row[width_col]),
            int(row[height_col])
        )
        for _, row in metadata_df.iterrows()
        if pd.notna(row[file_col])
    }

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                actual_width, actual_height = img.size

            successfully_opened += 1

            expected_resolution = metadata_lookup.get(image_path.stem)

            if expected_resolution is None:
                resolution_mismatches.append(
                    f"{image_path.name}: missing from CSV"
                )
            else:
                expected_width, expected_height = expected_resolution
                if actual_width != expected_width or actual_height != expected_height:
                    resolution_mismatches.append(
                        f"{image_path.name}: actual {actual_width}x{actual_height}, "
                        f"expected {expected_width}x{expected_height}"
                    )

        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupted_images.append(f"{image_path.name}: {e}")

    results = [
        ["Total images", len(image_files)],
        ["Successfully opened images", successfully_opened],
        ["Corrupted images", len(corrupted_images)],
        ["Duplicate filenames", duplicate_filenames],
        ["Invalid formats", len(invalid_format_files)],
        ["Resolution mismatches", len(resolution_mismatches)],
        ["Supported formats", ", ".join(sorted(SUPPORTED_FORMATS))],
    ]

    print("\nIMAGE INTEGRITY VALIDATION")
    print("=" * 40)
    print(pd.DataFrame(results, columns=["Metric", "Value"]).to_string(index=False))

    if corrupted_images:
        print("\nCORRUPTED IMAGES")
        print("-" * 40)
        for item in corrupted_images:
            print(item)

    if invalid_format_files:
        print("\nINVALID FORMAT FILES")
        print("-" * 40)
        for item in invalid_format_files:
            print(item)

    if resolution_mismatches:
        print("\nRESOLUTION MISMATCHES")
        print("-" * 40)
        for item in resolution_mismatches:
            print(item)


if __name__ == "__main__":
    main()