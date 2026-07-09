from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

ROOT_DIR = Path("../../..")

LAYER_0_DIR = ROOT_DIR / "layer_0_raw_images"
LAYER_1_DIR = ROOT_DIR / "layer_1_annotations"
LAYER_2_DIR = ROOT_DIR / "layer_2_metadata"
LAYER_3_DIR = ROOT_DIR / "layer_3_geospatial"

FIGURES_DIR = ROOT_DIR / "figures"
OUTPUT_FIGURE = FIGURES_DIR / "phase_1.3_venn_diagram.png"

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".heic"}
IGNORED_EXTENSIONS = {".md"}

def collect_ids_from_images(folder):
    return {
        p.stem.strip()
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    }


def collect_ids_from_csvs(folder):
    ids = set()

    for csv_path in folder.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)

            possible_id_cols = [
                "image_id",
                "Image_ID",
                "image_name",
                "filename",
                "file_name",
            ]

            id_col = None
            for col in possible_id_cols:
                if col in df.columns:
                    id_col = col
                    break

            if id_col is None:
                continue

            for value in df[id_col].dropna():
                image_id = str(value).strip()
                image_id = Path(image_id).stem
                ids.add(image_id)

        except Exception as e:
            print(f"Warning: Could not read {csv_path.name}: {e}")

    return ids


def print_table(rows):
    col1_width = max(len(str(row[0])) for row in rows + [["Metric", "Value"]])
    col2_width = max(len(str(row[1])) for row in rows + [["Metric", "Value"]])

    print("\nMETADATA CONSISTENCY TABLE")
    print("=" * (col1_width + col2_width + 7))
    print(f"{'Metric':<{col1_width}} | {'Value':>{col2_width}}")
    print("-" * (col1_width + col2_width + 7))

    for metric, value in rows:
        print(f"{metric:<{col1_width}} | {str(value):>{col2_width}}")


def main():
    layer0_ids = collect_ids_from_images(LAYER_0_DIR)
    layer1_ids = collect_ids_from_csvs(LAYER_1_DIR)
    layer2_ids = collect_ids_from_csvs(LAYER_2_DIR)
    layer3_ids = collect_ids_from_csvs(LAYER_3_DIR)

    l0_l1 = layer0_ids & layer1_ids
    l0_l2 = layer0_ids & layer2_ids
    l0_l3 = layer0_ids & layer3_ids

    all_common = layer0_ids & layer1_ids & layer2_ids & layer3_ids

    rows = [
        ["Layer 0 image IDs", len(layer0_ids)],
        ["Layer 1 annotation IDs", len(layer1_ids)],
        ["Layer 2 metadata IDs", len(layer2_ids)],
        ["Layer 3 geospatial IDs", len(layer3_ids)],
        ["Layer0 ∩ Layer1", len(l0_l1)],
        ["Layer0 ∩ Layer2", len(l0_l2)],
        ["Layer0 ∩ Layer3", len(l0_l3)],
        ["Common in all layers", len(all_common)],
        ["Missing in Layer 1", len(layer0_ids - layer1_ids)],
        ["Missing in Layer 2", len(layer0_ids - layer2_ids)],
        ["Missing in Layer 3", len(layer0_ids - layer3_ids)],
        ["Extra IDs in Layer 1", len(layer1_ids - layer0_ids)],
        ["Extra IDs in Layer 2", len(layer2_ids - layer0_ids)],
        ["Extra IDs in Layer 3", len(layer3_ids - layer0_ids)],
    ]

    print_table(rows)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    venn3(
        subsets=(layer1_ids, layer2_ids, layer3_ids),
        set_labels=("Layer 1", "Layer 2", "Layer 3"),
    )

    plt.title("Metadata Consistency Across Layers 1, 2, and 3")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    plt.close()

    print(f"\nVenn diagram saved to:")
    print(OUTPUT_FIGURE.resolve())

    if len(all_common) == len(layer0_ids):
        print("\nResult: PASS - All Layer 0 image IDs are consistent across Layers 1, 2, and 3.")
    else:
        print("\nResult: CHECK REQUIRED - Some Layer 0 image IDs are missing in one or more layers.")


if __name__ == "__main__":
    main()