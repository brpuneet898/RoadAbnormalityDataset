import pandas as pd
import numpy as np

SEMANTIC_METADATA = "../../../layer_2_metadata/layer_2_semantic_metadata.csv"

df = pd.read_csv(SEMANTIC_METADATA)

def confidence_interval_table(data, column, confidence=0.95):
    z = 1.96  
    n = len(data)

    counts = data[column].value_counts(dropna=False).sort_index()

    rows = []

    for category, count in counts.items():
        p = count / n
        se = np.sqrt((p * (1 - p)) / n)

        lower = max(0, p - z * se)
        upper = min(1, p + z * se)

        rows.append({
            column: category,
            "Count": count,
            "Proportion": f"{p:.4f}",
            "95% CI Lower": f"{lower:.4f}",
            "95% CI Upper": f"{upper:.4f}"
        })

    return pd.DataFrame(rows)

print("\n" + "=" * 80)
print("95% CONFIDENCE INTERVALS - CLASS PROPORTIONS")
print("=" * 80)
print(confidence_interval_table(df, "abnormality_type").to_string(index=False))


print("\n" + "=" * 80)
print("95% CONFIDENCE INTERVALS - SEVERITY PROPORTIONS")
print("=" * 80)
print(confidence_interval_table(df, "severity").to_string(index=False))


print("\n" + "=" * 80)
print("95% CONFIDENCE INTERVALS - ROAD TYPE PROPORTIONS")
print("=" * 80)
print(confidence_interval_table(df, "road_type").to_string(index=False))