from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]

ANNOTATION_METADATA = BASE_DIR / "layer_1_annotations" / "layer_1_annotation_metadata.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ANNOTATION_METADATA)

class_counts = (
    df["abnormality_type"]
    .value_counts()
    .rename_axis("Abnormality Type")
    .reset_index(name="Number of Images/Annotations")
)

class_counts["Percentage"] = (
    class_counts["Number of Images/Annotations"]
    / class_counts["Number of Images/Annotations"].sum()
    * 100
).round(2)

print("\nCLASS DISTRIBUTION")
print("=" * 60)
print(class_counts.to_string(index=False))

most_common = class_counts.iloc[0]
least_common = class_counts.iloc[-1]
imbalance_ratio = round(
    most_common["Number of Images/Annotations"]
    / least_common["Number of Images/Annotations"],
    2
)

print("\nCLASS IMBALANCE DISCUSSION")
print("=" * 60)
print(
    f"The most frequent class is '{most_common['Abnormality Type']}' "
    f"with {most_common['Number of Images/Annotations']} instances."
)
print(
    f"The least frequent class is '{least_common['Abnormality Type']}' "
    f"with {least_common['Number of Images/Annotations']} instances."
)
print(f"The imbalance ratio is approximately {imbalance_ratio}:1.")

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    class_counts["Number of Images/Annotations"],
    labels=class_counts["Abnormality Type"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Class Distribution by Abnormality Type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase-2.1-class-distribution-pie-chart.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(
    class_counts["Abnormality Type"],
    class_counts["Number of Images/Annotations"]
)
plt.title("Number of Images per Abnormality Type")
plt.xlabel("Abnormality Type")
plt.ylabel("Number of Images/Annotations")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "phase-2.1-class-distribution-histogram.png", dpi=300)
plt.close()

print("\nPlots saved to:")
print(FIGURES_DIR / "phase-2.1-class-distribution-pie-chart.png")
print(FIGURES_DIR / "phase-2.1-class-distribution-histogram.png")