from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

ANNOTATION_METADATA = BASE_DIR / "layer_2_metadata" / "layer_2_semantic_metadata.csv"

df = pd.read_csv(ANNOTATION_METADATA)

severity_counts = (
    df["severity"]
    .value_counts()
    .reindex(["minor", "moderate", "severe"], fill_value=0)
    .rename_axis("Severity")
    .reset_index(name="Count")
)

severity_counts["Percentage"] = (
    severity_counts["Count"] / severity_counts["Count"].sum() * 100
).round(2)

print("\nSEVERITY DISTRIBUTION")
print("=" * 50)
print(severity_counts.to_string(index=False))