from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]

GEOSPATIAL_METADATA = BASE_DIR / "layer_3_geospatial" / "layer_3_geospatial_metadata.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(GEOSPATIAL_METADATA)

state_counts = (
    df["collection_state"]
    .value_counts()
    .rename_axis("State")
    .reset_index(name="Count")
)

state_counts["Percentage"] = (
    state_counts["Count"] / state_counts["Count"].sum() * 100
).round(2)

print("\nSTATE-WISE GEOGRAPHIC DISTRIBUTION")
print("=" * 60)
print(state_counts.to_string(index=False))

district_counts = (
    df["district"]
    .value_counts()
    .rename_axis("District")
    .reset_index(name="Count")
)

district_counts["Percentage"] = (
    district_counts["Count"] / district_counts["Count"].sum() * 100
).round(2)

print("\nDISTRICT-WISE GEOGRAPHIC DISTRIBUTION")
print("=" * 60)
print(district_counts.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(state_counts["State"], state_counts["Count"])
plt.title("State-wise Geographic Distribution")
plt.xlabel("State")
plt.ylabel("Number of Images")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(district_counts["District"], district_counts["Count"])
plt.title("District-wise Geographic Distribution")
plt.xlabel("District")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.close()

heatmap_data = pd.crosstab(df["collection_state"], df["district"])

plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, aspect="auto")
plt.colorbar(label="Number of Images")

plt.title("Geographic Heat Map: State vs District")
plt.xlabel("District")
plt.ylabel("State")

plt.xticks(
    ticks=range(len(heatmap_data.columns)),
    labels=heatmap_data.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=range(len(heatmap_data.index)),
    labels=heatmap_data.index
)

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        plt.text(j, i, heatmap_data.iloc[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase-2.5-geographic-heat-map.png", dpi=300)
plt.close()

print("\nPlots saved to:")
print(FIGURES_DIR / "phase-2.5-geographic-heat-map.png")