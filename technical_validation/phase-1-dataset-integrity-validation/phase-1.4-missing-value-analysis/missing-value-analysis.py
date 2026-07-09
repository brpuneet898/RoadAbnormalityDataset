from pathlib import Path
import pandas as pd

ROOT_DIR = Path("../../..")

LAYER_DIRS = {
    "Layer 0": ROOT_DIR / "layer_0_raw_images",
    "Layer 1": ROOT_DIR / "layer_1_annotations",
    "Layer 2": ROOT_DIR / "layer_2_metadata",
    "Layer 3": ROOT_DIR / "layer_3_geospatial",
}

CHECK_FILE_TYPES = {".csv"}
IGNORED_EXTENSIONS = {".md"}

def count_missing_values_in_folder(folder_path):
    total_missing = 0
    files_checked = 0

    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            continue

        if file_path.suffix.lower() not in CHECK_FILE_TYPES:
            continue

        try:
            df = pd.read_csv(file_path)

            missing_count = df.isna().sum().sum()
            empty_string_count = df.astype(str).apply(
                lambda col: col.str.strip().eq("")
            ).sum().sum()

            total_missing += int(missing_count + empty_string_count)
            files_checked += 1

        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")

    return total_missing, files_checked


def print_table(rows):
    col1_width = max(len(str(row[0])) for row in rows + [["Layer", "Missing values"]])
    col2_width = max(len(str(row[1])) for row in rows + [["Layer", "Missing values"]])

    print("\nMISSING VALUE ANALYSIS")
    print("=" * (col1_width + col2_width + 7))
    print(f"{'Layer':<{col1_width}} | {'Missing values':>{col2_width}}")
    print("-" * (col1_width + col2_width + 7))

    for layer, missing_values in rows:
        print(f"{layer:<{col1_width}} | {missing_values:>{col2_width}}")

def main():
    rows = []

    for layer_name, layer_path in LAYER_DIRS.items():
        missing_values, files_checked = count_missing_values_in_folder(layer_path)
        rows.append([layer_name, missing_values])

    print_table(rows)

if __name__ == "__main__":
    main()