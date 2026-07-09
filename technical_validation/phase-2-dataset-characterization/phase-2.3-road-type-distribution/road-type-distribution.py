from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

SEMANTIC_METADATA = BASE_DIR / "layer_2_metadata" / "layer_2_semantic_metadata.csv"

df = pd.read_csv(SEMANTIC_METADATA)

road_type_counts = (
    df["road_type"]
    .value_counts()
    .reindex(["urban", "highway", "residential", "rural"], fill_value=0)
    .rename_axis("Road Type")
    .reset_index(name="Count")
)

road_type_counts["Percentage"] = (
    road_type_counts["Count"] / road_type_counts["Count"].sum() * 100
).round(2)

print("\nROAD-TYPE DISTRIBUTION")
print("=" * 50)
print(road_type_counts.to_string(index=False))