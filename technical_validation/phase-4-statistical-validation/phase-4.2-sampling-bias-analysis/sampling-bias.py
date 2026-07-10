import pandas as pd

GEOSPATIAL_METADATA = "../../../layer_3_geospatial/layer_3_geospatial_metadata.csv"

df = pd.read_csv(GEOSPATIAL_METADATA)

state_counts = (
    df["collection_state"]
    .value_counts(dropna=False)
    .rename_axis("State")
    .reset_index(name="Count")
)

total = len(df)
state_counts["Percentage"] = (state_counts["Count"] / total * 100).round(2)

print("\n" + "=" * 70)
print("SAMPLING BIAS ANALYSIS")
print("=" * 70)

print("\nState Distribution\n")
print(state_counts.to_string(index=False))

tn_row = state_counts[state_counts["State"].str.lower() == "tamil nadu".lower()]

if not tn_row.empty:
    tn_percentage = float(tn_row["Percentage"].iloc[0])

    print("\n" + "-" * 70)
    print(f"Tamil Nadu samples: {tn_percentage:.2f}% of the dataset")

    if tn_percentage > 50:
        print("\nResult:")
        print("Tamil Nadu dominates the dataset.")
    else:
        print("\nResult:")
        print("No significant dominance by Tamil Nadu was observed.")
else:
    print("\nTamil Nadu is not present in the metadata.")