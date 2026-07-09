from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]

ANNOTATION_METADATA = BASE_DIR / "layer_1_annotations" / "layer_1_annotation_metadata.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ANNOTATION_METADATA)

df["aspect_ratio"] = df["width_scaled"] / df["height_scaled"]
df["object_area"] = df["width_scaled"] * df["height_scaled"]
df["relative_object_size"] = df["object_area"] * 100

metrics = [
    "width_scaled",
    "height_scaled",
    "aspect_ratio",
    "object_area",
    "relative_object_size"
]

stats_table = df[metrics].agg(["mean", "median", "min", "max"]).T
stats_table = stats_table.round(4)
stats_table = stats_table.rename(columns={
    "mean": "Average",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum"
})

print("\nBOUNDING BOX STATISTICS")
print("=" * 70)
print(stats_table.to_string())

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.hist(df[metric].dropna(), bins=30)
    plt.title(f"Histogram of {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"phase-2.6-{metric}-histogram.png", dpi=300)
    plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot([df[m].dropna() for m in metrics], label=[m.replace("_", "\n") for m in metrics])
plt.title("Bounding Box Statistics Boxplots")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase-2.6-bounding-box-statistics-boxplots.png", dpi=300)
plt.close()

print("\nPlots saved to figures folder.")