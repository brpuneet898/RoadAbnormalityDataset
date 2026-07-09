from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]

ANNOTATION_METADATA = BASE_DIR / "layer_1_annotations" / "layer_1_annotation_metadata.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ANNOTATION_METADATA)

objects_per_image = (
    df.groupby("image_id")
    .size()
    .reset_index(name="object_count")
)

objects_per_image["Object Group"] = objects_per_image["object_count"].apply(
    lambda x: "4+" if x >= 4 else str(x)
)

group_order = ["1", "2", "3", "4+"]

summary_table = (
    objects_per_image["Object Group"]
    .value_counts()
    .reindex(group_order, fill_value=0)
    .rename_axis("Objects per Image")
    .reset_index(name="Number of Images")
)

summary_table["Percentage"] = (
    summary_table["Number of Images"] / summary_table["Number of Images"].sum() * 100
).round(2)

print("\nNUMBER OF OBJECTS PER IMAGE")
print("=" * 60)
print(summary_table.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(summary_table["Objects per Image"], summary_table["Number of Images"])
plt.title("Number of Objects per Image")
plt.xlabel("Objects per Image")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase-2.7-objects-per-image-histogram.png", dpi=300)
plt.close()

print("\nPlot saved to:")
print(FIGURES_DIR / "phase-2.7-objects-per-image-histogram.png")