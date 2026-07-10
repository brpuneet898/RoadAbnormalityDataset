from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

LAYER_2_PATH = Path(
    "../../../layer_2_metadata/layer_2_semantic_metadata.csv"
)

LAYER_3_PATH = Path(
    "../../../layer_3_geospatial/layer_3_geospatial_metadata.csv"
)

STATE_COLUMN = "collection_state"

DISTRIBUTION_COLUMNS = {
    "Class": "abnormality_type",
    "Severity": "severity",
    "Road Type": "road_type",
}

SIGNIFICANCE_LEVEL = 0.05

def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_columns}"
        )


def cramers_v(contingency_table: pd.DataFrame, chi_square: float) -> float:
    observed = contingency_table.to_numpy()
    sample_size = observed.sum()

    rows, columns = observed.shape

    if sample_size <= 1 or rows <= 1 or columns <= 1:
        return np.nan

    phi_squared = chi_square / sample_size

    corrected_phi_squared = max(
        0,
        phi_squared
        - ((columns - 1) * (rows - 1)) / (sample_size - 1),
    )

    corrected_rows = (
        rows - ((rows - 1) ** 2) / (sample_size - 1)
    )

    corrected_columns = (
        columns - ((columns - 1) ** 2) / (sample_size - 1)
    )

    denominator = min(
        corrected_columns - 1,
        corrected_rows - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(np.sqrt(corrected_phi_squared / denominator))


def interpret_cramers_v(value: float) -> str:
    if pd.isna(value):
        return "Not available"
    if value < 0.10:
        return "Negligible"
    if value < 0.30:
        return "Weak"
    if value < 0.50:
        return "Moderate"
    return "Strong"


def format_p_value(value: float) -> str:
    if value < 0.0001:
        return "< 0.0001"

    return f"{value:.4f}"


def print_distribution_table(
    dataframe: pd.DataFrame,
    category_column: str,
    title: str,
) -> None:

    counts = pd.crosstab(
        dataframe[STATE_COLUMN],
        dataframe[category_column],
    )

    percentages = pd.crosstab(
        dataframe[STATE_COLUMN],
        dataframe[category_column],
        normalize="index",
    ) * 100

    print("\n" + "=" * 100)
    print(f"{title.upper()} — STATE-WISE COUNTS")
    print("=" * 100)
    print(counts.to_string())

    print("\n" + "-" * 100)
    print(f"{title.upper()} — STATE-WISE PERCENTAGES")
    print("-" * 100)

    formatted_percentages = percentages.map(
        lambda value: f"{value:.2f}%"
    )

    print(formatted_percentages.to_string())


def run_chi_square_test(
    dataframe: pd.DataFrame,
    category_column: str,
    analysis_name: str,
) -> dict:
    analysis_data = dataframe[
        [STATE_COLUMN, category_column]
    ].dropna()

    contingency_table = pd.crosstab(
        analysis_data[STATE_COLUMN],
        analysis_data[category_column],
    )

    if contingency_table.shape[0] < 2:
        return {
            "Distribution": analysis_name,
            "Chi-square": np.nan,
            "Degrees of freedom": np.nan,
            "P-value": np.nan,
            "Significant": "Not testable",
            "Cramér's V": np.nan,
            "Effect": "Not available",
            "Low expected cells": np.nan,
            "Sample size": len(analysis_data),
        }

    if contingency_table.shape[1] < 2:
        return {
            "Distribution": analysis_name,
            "Chi-square": np.nan,
            "Degrees of freedom": np.nan,
            "P-value": np.nan,
            "Significant": "Not testable",
            "Cramér's V": np.nan,
            "Effect": "Not available",
            "Low expected cells": np.nan,
            "Sample size": len(analysis_data),
        }

    chi_square, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    effect_size = cramers_v(
        contingency_table,
        chi_square,
    )

    low_expected_count = int((expected < 5).sum())
    total_expected_cells = int(expected.size)

    return {
        "Distribution": analysis_name,
        "Chi-square": chi_square,
        "Degrees of freedom": degrees_of_freedom,
        "P-value": p_value,
        "Significant": (
            "Yes" if p_value < SIGNIFICANCE_LEVEL else "No"
        ),
        "Cramér's V": effect_size,
        "Effect": interpret_cramers_v(effect_size),
        "Low expected cells": (
            f"{low_expected_count}/{total_expected_cells}"
        ),
        "Sample size": len(analysis_data),
    }

if not LAYER_2_PATH.exists():
    raise FileNotFoundError(
        f"Layer 2 metadata not found: {LAYER_2_PATH.resolve()}"
    )

if not LAYER_3_PATH.exists():
    raise FileNotFoundError(
        f"Layer 3 metadata not found: {LAYER_3_PATH.resolve()}"
    )

layer_2 = pd.read_csv(LAYER_2_PATH)
layer_3 = pd.read_csv(LAYER_3_PATH)

validate_columns(
    layer_2,
    ["image_id", *DISTRIBUTION_COLUMNS.values()],
    "Layer 2 metadata",
)

validate_columns(
    layer_3,
    ["image_id", STATE_COLUMN],
    "Layer 3 metadata",
)

geospatial_columns = layer_3[
    ["image_id", STATE_COLUMN]
].drop_duplicates(subset="image_id")

merged = layer_2.merge(
    geospatial_columns,
    on="image_id",
    how="inner",
    validate="one_to_one",
)


# Clean text values
columns_to_clean = [
    STATE_COLUMN,
    *DISTRIBUTION_COLUMNS.values(),
]

for column in columns_to_clean:
    merged[column] = (
        merged[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

print("\n" + "=" * 100)
print("PHASE 4.3 — STATE-WISE DISTRIBUTION COMPARISON")
print("=" * 100)

print(f"\nLayer 2 records:       {len(layer_2):,}")
print(f"Layer 3 records:       {len(layer_3):,}")
print(f"Successfully matched:  {len(merged):,}")
print(f"States represented:    {merged[STATE_COLUMN].nunique(dropna=True)}")

state_summary = (
    merged[STATE_COLUMN]
    .value_counts(dropna=False)
    .rename_axis("State")
    .reset_index(name="Count")
)

state_summary["Percentage"] = (
    state_summary["Count"] / len(merged) * 100
).map(lambda value: f"{value:.2f}%")

print("\nState sample sizes:\n")
print(state_summary.to_string(index=False))

test_results = []

for distribution_name, column_name in DISTRIBUTION_COLUMNS.items():
    print_distribution_table(
        dataframe=merged.dropna(
            subset=[STATE_COLUMN, column_name]
        ),
        category_column=column_name,
        title=distribution_name,
    )

    result = run_chi_square_test(
        dataframe=merged,
        category_column=column_name,
        analysis_name=distribution_name,
    )

    test_results.append(result)

results = pd.DataFrame(test_results)

display_results = results.copy()

display_results["Chi-square"] = display_results["Chi-square"].map(
    lambda value: (
        "N/A" if pd.isna(value) else f"{value:.4f}"
    )
)

display_results["Degrees of freedom"] = display_results[
    "Degrees of freedom"
].map(
    lambda value: (
        "N/A" if pd.isna(value) else str(int(value))
    )
)

display_results["P-value"] = display_results["P-value"].map(
    lambda value: (
        "N/A" if pd.isna(value) else format_p_value(value)
    )
)

display_results["Cramér's V"] = display_results["Cramér's V"].map(
    lambda value: (
        "N/A" if pd.isna(value) else f"{value:.4f}"
    )
)

print("\n" + "=" * 100)
print("FINAL CHI-SQUARE TEST RESULTS")
print("=" * 100)
print(display_results.to_string(index=False))

print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)

for result in test_results:
    distribution = result["Distribution"]

    if pd.isna(result["P-value"]):
        print(
            f"\n{distribution}: The comparison could not be performed "
            "because at least two states and two categories are required."
        )
        continue

    p_value = result["P-value"]
    effect = result["Effect"]
    effect_size = result["Cramér's V"]

    if p_value < SIGNIFICANCE_LEVEL:
        print(
            f"\n{distribution}: State-wise distributions are "
            f"statistically different "
            f"(p {format_p_value(p_value)}, "
            f"Cramér's V = {effect_size:.4f}, "
            f"{effect.lower()} association)."
        )
    else:
        print(
            f"\n{distribution}: No statistically significant "
            f"state-wise distribution difference was detected "
            f"(p = {p_value:.4f}, "
            f"Cramér's V = {effect_size:.4f}, "
            f"{effect.lower()} association)."
        )

    low_cells = result["Low expected cells"]

    if isinstance(low_cells, str):
        low_count, total_cells = map(int, low_cells.split("/"))

        if low_count > 0:
            percentage = low_count / total_cells * 100

            print(
                f"  Warning: {low_count} of {total_cells} expected "
                f"cells ({percentage:.1f}%) have an expected count "
                "below 5. Interpret the Chi-square result cautiously."
            )

print(
    "\nDecision rule: p < 0.05 indicates that the distribution "
    "differs significantly between states."
)