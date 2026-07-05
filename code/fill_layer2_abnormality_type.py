import pandas as pd
from pathlib import Path


def main():
    code_dir = Path(__file__).resolve().parent
    dataset_root = code_dir.parent

    layer1_csv = dataset_root / "layer_1_annotations" / "layer_1_annotation_metadata.csv"
    layer2_csv = dataset_root / "layer_2_metadata" / "layer_2_semantic_metadata.csv"

    output_csv = dataset_root / "layer_2_metadata" / "layer_2_semantic_metadata_updated.csv"

    layer1_df = pd.read_csv(layer1_csv)
    layer2_df = pd.read_csv(layer2_csv)

    abnormality_map = (
        layer1_df
        .dropna(subset=["abnormality_type"])
        .groupby("image_id")["abnormality_type"]
        .apply(lambda values: "; ".join(dict.fromkeys(values.astype(str).str.strip())))
        .to_dict()
    )

    layer2_df["abnormality_type"] = layer2_df["image_id"].map(abnormality_map)
    layer2_df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()