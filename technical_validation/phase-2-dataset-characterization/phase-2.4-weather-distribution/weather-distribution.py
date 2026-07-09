from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

SEMANTIC_METADATA = BASE_DIR / "layer_2_metadata" / "layer_2_semantic_metadata.csv"

df = pd.read_csv(SEMANTIC_METADATA)

weather_counts = (
    df["weather"]
    .value_counts(dropna=False)
    .rename_axis("Weather")
    .reset_index(name="Count")
)

weather_counts["Percentage"] = (
    weather_counts["Count"] / weather_counts["Count"].sum() * 100
).round(2)

print("\nWEATHER DISTRIBUTION")
print("=" * 50)
print(weather_counts.to_string(index=False))


lighting_counts = (
    df["lighting"]
    .value_counts(dropna=False)
    .rename_axis("Lighting Condition")
    .reset_index(name="Count")
)

lighting_counts["Percentage"] = (
    lighting_counts["Count"] / lighting_counts["Count"].sum() * 100
).round(2)

print("\nLIGHTING DISTRIBUTION")
print("=" * 50)
print(lighting_counts.to_string(index=False))