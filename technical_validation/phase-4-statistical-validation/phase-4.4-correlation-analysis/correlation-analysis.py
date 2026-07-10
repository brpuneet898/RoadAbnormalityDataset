"""
Phase 4.4 - Correlation Analysis

Analyses:
1. Weather -> Severity
2. Road type -> Abnormality type
3. Traffic density -> Abnormality frequency

Methods:
- Chi-square test of independence for categorical variables
- Cramér's V for categorical association strength
- Spearman rank correlation for ordinal traffic density versus frequency
- Kruskal-Wallis test for differences in frequency across traffic groups

Output:
- Terminal tables only
- No CSV files are created
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    kruskal,
    spearmanr,
)


# =====================================================================
# Configuration
# =====================================================================

# Script location:
# project_root/
# ├── layer_1_annotations/
# ├── layer_2_metadata/
# ├── layer_3_geospatial/
# └── technical_validation/
#     └── phase-4-statistical-validation/
#         └── phase-4.4-correlation-analysis/
#             └── correlation_analysis.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LAYER_1_PATH = (
    PROJECT_ROOT
    / "layer_1_annotations"
    / "layer_1_annotation_metadata.csv"
)

LAYER_2_PATH = (
    PROJECT_ROOT
    / "layer_2_metadata"
    / "layer_2_semantic_metadata.csv"
)

LAYER_3_PATH = (
    PROJECT_ROOT
    / "layer_3_geospatial"
    / "layer_3_geospatial_metadata.csv"
)

SIGNIFICANCE_LEVEL = 0.05

# Ordinal scores used for Spearman correlation.
TRAFFIC_DENSITY_ORDER = {
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "very high": 4,
    "very_high": 4,
}


# =====================================================================
# General utilities
# =====================================================================

def print_heading(title: str, character: str = "=") -> None:
    """Print a formatted terminal heading."""

    print("\n" + character * 100)
    print(title)
    print(character * 100)


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Verify that a dataframe contains all required columns."""

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def clean_text_column(series: pd.Series) -> pd.Series:
    """Strip whitespace and convert blank strings to missing values."""

    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def format_p_value(value: float) -> str:
    """Format a p-value for terminal output."""

    if pd.isna(value):
        return "N/A"

    if value < 0.0001:
        return "<0.0001"

    return f"{value:.4f}"


def interpret_p_value(p_value: float) -> str:
    """Interpret statistical significance at alpha = 0.05."""

    if pd.isna(p_value):
        return "Not testable"

    if p_value < SIGNIFICANCE_LEVEL:
        return "Significant"

    return "Not significant"


# =====================================================================
# Categorical association utilities
# =====================================================================

def calculate_cramers_v(
    contingency_table: pd.DataFrame,
    chi_square: float,
) -> float:
    """
    Calculate bias-corrected Cramér's V.

    Cramér's V ranges from 0 to 1:
    - 0 means no observed association
    - 1 means perfect association
    """

    observed = contingency_table.to_numpy()
    sample_size = observed.sum()
    rows, columns = observed.shape

    if sample_size <= 1 or rows <= 1 or columns <= 1:
        return np.nan

    phi_squared = chi_square / sample_size

    correction = (
        (columns - 1) * (rows - 1)
    ) / (sample_size - 1)

    corrected_phi_squared = max(
        0,
        phi_squared - correction,
    )

    corrected_rows = (
        rows
        - ((rows - 1) ** 2) / (sample_size - 1)
    )

    corrected_columns = (
        columns
        - ((columns - 1) ** 2) / (sample_size - 1)
    )

    denominator = min(
        corrected_rows - 1,
        corrected_columns - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(corrected_phi_squared / denominator)
    )


def interpret_cramers_v(value: float) -> str:
    """Give a simple interpretation of Cramér's V."""

    if pd.isna(value):
        return "Not available"
    if value < 0.10:
        return "Negligible"
    if value < 0.30:
        return "Weak"
    if value < 0.50:
        return "Moderate"

    return "Strong"


def categorical_association(
    dataframe: pd.DataFrame,
    predictor: str,
    outcome: str,
    analysis_name: str,
) -> dict:
    """
    Run a Chi-square test and calculate Cramér's V.

    Also prints count and row-percentage contingency tables.
    """

    analysis_data = dataframe[
        [predictor, outcome]
    ].dropna()

    contingency_table = pd.crosstab(
        analysis_data[predictor],
        analysis_data[outcome],
    )

    print_heading(
        f"{analysis_name.upper()} — OBSERVED COUNTS",
        "-",
    )
    print(contingency_table.to_string())

    row_percentages = (
        contingency_table
        .div(contingency_table.sum(axis=1), axis=0)
        .mul(100)
    )

    print_heading(
        f"{analysis_name.upper()} — ROW PERCENTAGES",
        "-",
    )

    formatted_percentages = row_percentages.map(
        lambda value: f"{value:.2f}%"
    )

    print(formatted_percentages.to_string())

    if (
        contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        return {
            "Analysis": analysis_name,
            "Method": "Chi-square",
            "Statistic": np.nan,
            "Degrees of freedom": np.nan,
            "P-value": np.nan,
            "Effect size": np.nan,
            "Effect measure": "Cramér's V",
            "Effect strength": "Not available",
            "Significance": "Not testable",
            "Sample size": len(analysis_data),
            "Expected cells < 5": "N/A",
        }

    chi_square, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    cramers_v = calculate_cramers_v(
        contingency_table,
        chi_square,
    )

    low_expected_cells = int((expected < 5).sum())
    total_expected_cells = int(expected.size)

    return {
        "Analysis": analysis_name,
        "Method": "Chi-square",
        "Statistic": chi_square,
        "Degrees of freedom": degrees_of_freedom,
        "P-value": p_value,
        "Effect size": cramers_v,
        "Effect measure": "Cramér's V",
        "Effect strength": interpret_cramers_v(cramers_v),
        "Significance": interpret_p_value(p_value),
        "Sample size": len(analysis_data),
        "Expected cells < 5": (
            f"{low_expected_cells}/{total_expected_cells}"
        ),
    }


# =====================================================================
# Traffic density versus abnormality frequency
# =====================================================================

def traffic_frequency_analysis(
    dataframe: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Analyse traffic density against abnormality frequency.

    Spearman correlation tests whether frequency tends to increase or
    decrease as traffic density increases.

    Kruskal-Wallis tests whether frequency distributions differ among
    traffic-density categories.
    """

    analysis_data = dataframe[
        [
            "traffic_density",
            "abnormality_frequency",
        ]
    ].dropna().copy()

    analysis_data["traffic_density_normalized"] = (
        analysis_data["traffic_density"]
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    normalized_order = {
        key.replace("_", " "): value
        for key, value in TRAFFIC_DENSITY_ORDER.items()
    }

    analysis_data["traffic_density_score"] = (
        analysis_data["traffic_density_normalized"]
        .map(normalized_order)
    )

    unknown_categories = sorted(
        analysis_data.loc[
            analysis_data["traffic_density_score"].isna(),
            "traffic_density",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_categories:
        print(
            "\nWarning: The following traffic-density categories do "
            "not have an ordinal score and are excluded from the "
            f"Spearman test: {unknown_categories}"
        )

    # ---------------------------------------------------------------
    # Descriptive table
    # ---------------------------------------------------------------

    summary = (
        analysis_data
        .groupby(
            "traffic_density",
            observed=True,
        )["abnormality_frequency"]
        .agg(
            Images="count",
            Total_abnormalities="sum",
            Mean="mean",
            Median="median",
            Standard_deviation="std",
            Minimum="min",
            Maximum="max",
        )
        .reset_index()
        .rename(
            columns={
                "traffic_density": "Traffic density",
                "Total_abnormalities": "Total abnormalities",
                "Standard_deviation": "Std. deviation",
            }
        )
    )

    score_lookup = (
        analysis_data[
            ["traffic_density", "traffic_density_score"]
        ]
        .drop_duplicates()
        .set_index("traffic_density")[
            "traffic_density_score"
        ]
    )

    summary["Traffic score"] = (
        summary["Traffic density"].map(score_lookup)
    )

    summary = summary.sort_values(
        by=["Traffic score", "Traffic density"],
        na_position="last",
    )

    for column in [
        "Mean",
        "Median",
        "Std. deviation",
    ]:
        summary[column] = summary[column].map(
            lambda value: (
                "N/A"
                if pd.isna(value)
                else f"{value:.3f}"
            )
        )

    print_heading(
        "TRAFFIC DENSITY → ABNORMALITY FREQUENCY — SUMMARY",
        "-",
    )
    print(summary.to_string(index=False))

    # ---------------------------------------------------------------
    # Spearman correlation
    # ---------------------------------------------------------------

    spearman_data = analysis_data.dropna(
        subset=["traffic_density_score"]
    )

    if (
        len(spearman_data) >= 3
        and spearman_data["traffic_density_score"].nunique() >= 2
        and spearman_data["abnormality_frequency"].nunique() >= 2
    ):
        spearman_rho, spearman_p = spearmanr(
            spearman_data["traffic_density_score"],
            spearman_data["abnormality_frequency"],
        )
    else:
        spearman_rho = np.nan
        spearman_p = np.nan

    if pd.isna(spearman_rho):
        correlation_strength = "Not available"
        correlation_direction = "Not available"
    else:
        absolute_rho = abs(spearman_rho)

        if absolute_rho < 0.10:
            correlation_strength = "Negligible"
        elif absolute_rho < 0.30:
            correlation_strength = "Weak"
        elif absolute_rho < 0.50:
            correlation_strength = "Moderate"
        else:
            correlation_strength = "Strong"

        if spearman_rho > 0:
            correlation_direction = "Positive"
        elif spearman_rho < 0:
            correlation_direction = "Negative"
        else:
            correlation_direction = "None"

    spearman_result = {
        "Analysis": "Traffic density → abnormality frequency",
        "Method": "Spearman correlation",
        "Statistic": spearman_rho,
        "Degrees of freedom": np.nan,
        "P-value": spearman_p,
        "Effect size": spearman_rho,
        "Effect measure": "Spearman rho",
        "Effect strength": (
            f"{correlation_strength}, "
            f"{correlation_direction.lower()}"
            if correlation_strength != "Not available"
            else "Not available"
        ),
        "Significance": interpret_p_value(spearman_p),
        "Sample size": len(spearman_data),
        "Expected cells < 5": "N/A",
    }

    # ---------------------------------------------------------------
    # Kruskal-Wallis test
    # ---------------------------------------------------------------

    frequency_groups = [
        group["abnormality_frequency"].to_numpy()
        for _, group in analysis_data.groupby(
            "traffic_density",
            observed=True,
        )
        if len(group) > 0
    ]

    if (
        len(frequency_groups) >= 2
        and analysis_data[
            "abnormality_frequency"
        ].nunique() >= 2
    ):
        kruskal_statistic, kruskal_p = kruskal(
            *frequency_groups
        )
    else:
        kruskal_statistic = np.nan
        kruskal_p = np.nan

    kruskal_result = {
        "Analysis": "Traffic density → abnormality frequency",
        "Method": "Kruskal-Wallis",
        "Statistic": kruskal_statistic,
        "Degrees of freedom": (
            len(frequency_groups) - 1
            if len(frequency_groups) >= 2
            else np.nan
        ),
        "P-value": kruskal_p,
        "Effect size": np.nan,
        "Effect measure": "N/A",
        "Effect strength": "N/A",
        "Significance": interpret_p_value(kruskal_p),
        "Sample size": len(analysis_data),
        "Expected cells < 5": "N/A",
    }

    return spearman_result, kruskal_result


# =====================================================================
# Load metadata
# =====================================================================

for path in [
    LAYER_1_PATH,
    LAYER_2_PATH,
    LAYER_3_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

layer_1 = pd.read_csv(LAYER_1_PATH)
layer_2 = pd.read_csv(LAYER_2_PATH)
layer_3 = pd.read_csv(LAYER_3_PATH)


# =====================================================================
# Validate columns
# =====================================================================

validate_columns(
    layer_1,
    [
        "image_id",
        "abnormality_id",
    ],
    "Layer 1 metadata",
)

validate_columns(
    layer_2,
    [
        "image_id",
        "weather",
        "severity",
        "road_type",
        "abnormality_type",
    ],
    "Layer 2 metadata",
)

validate_columns(
    layer_3,
    [
        "image_id",
        "traffic_density",
    ],
    "Layer 3 metadata",
)


# =====================================================================
# Prepare metadata
# =====================================================================

# One Layer 2 record is expected per image.
if layer_2["image_id"].duplicated().any():
    raise ValueError(
        "Layer 2 contains duplicate image_id values. "
        "A one-record-per-image structure is required."
    )

# One Layer 3 record is expected per image.
if layer_3["image_id"].duplicated().any():
    raise ValueError(
        "Layer 3 contains duplicate image_id values. "
        "A one-record-per-image structure is required."
    )

# Count Layer 1 annotations for each image.
abnormality_counts = (
    layer_1.groupby("image_id")
    .size()
    .rename("abnormality_frequency")
    .reset_index()
)

# Include images with no Layer 1 annotation by starting from Layer 2.
merged = (
    layer_2.merge(
        layer_3[
            [
                "image_id",
                "traffic_density",
            ]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        abnormality_counts,
        on="image_id",
        how="left",
        validate="one_to_one",
    )
)

merged["abnormality_frequency"] = (
    merged["abnormality_frequency"]
    .fillna(0)
    .astype(int)
)

for column in [
    "weather",
    "severity",
    "road_type",
    "abnormality_type",
    "traffic_density",
]:
    merged[column] = clean_text_column(merged[column])


# =====================================================================
# Dataset summary
# =====================================================================

print_heading("PHASE 4.4 — CORRELATION ANALYSIS")

print(f"\nLayer 1 annotations:       {len(layer_1):,}")
print(f"Layer 2 image records:     {len(layer_2):,}")
print(f"Layer 3 image records:     {len(layer_3):,}")
print(f"Merged image records:      {len(merged):,}")
print(
    "Total abnormalities:      "
    f"{merged['abnormality_frequency'].sum():,}"
)
print(
    "Mean abnormalities/image: "
    f"{merged['abnormality_frequency'].mean():.3f}"
)


# =====================================================================
# Run analyses
# =====================================================================

results = []

# 1. Weather -> Severity
results.append(
    categorical_association(
        dataframe=merged,
        predictor="weather",
        outcome="severity",
        analysis_name="Weather → severity",
    )
)

# 2. Road type -> Abnormality type
results.append(
    categorical_association(
        dataframe=merged,
        predictor="road_type",
        outcome="abnormality_type",
        analysis_name="Road type → abnormality type",
    )
)

# 3. Traffic density -> Abnormality frequency
spearman_result, kruskal_result = (
    traffic_frequency_analysis(merged)
)

results.extend([
    spearman_result,
    kruskal_result,
])


# =====================================================================
# Final results table
# =====================================================================

results_table = pd.DataFrame(results)

display_table = results_table.copy()

display_table["Statistic"] = display_table["Statistic"].map(
    lambda value: (
        "N/A"
        if pd.isna(value)
        else f"{value:.4f}"
    )
)

display_table["Degrees of freedom"] = display_table[
    "Degrees of freedom"
].map(
    lambda value: (
        "N/A"
        if pd.isna(value)
        else str(int(value))
    )
)

display_table["P-value"] = display_table["P-value"].map(
    format_p_value
)

display_table["Effect size"] = display_table[
    "Effect size"
].map(
    lambda value: (
        "N/A"
        if pd.isna(value)
        else f"{value:.4f}"
    )
)

print_heading("FINAL CORRELATION AND ASSOCIATION RESULTS")
print(display_table.to_string(index=False))


# =====================================================================
# Automated interpretation
# =====================================================================

print_heading("INTERPRETATION")

for result in results:
    analysis = result["Analysis"]
    method = result["Method"]
    p_value = result["P-value"]

    print(f"\n{analysis} — {method}")

    if pd.isna(p_value):
        print(
            "The analysis could not be completed because there was "
            "insufficient variation or too little valid data."
        )
        continue

    if p_value < SIGNIFICANCE_LEVEL:
        print(
            f"A statistically significant relationship was detected "
            f"(p {format_p_value(p_value)})."
        )
    else:
        print(
            "No statistically significant relationship was detected "
            f"(p = {p_value:.4f})."
        )

    if method == "Chi-square":
        print(
            f"Association strength: {result['Effect strength']} "
            f"(Cramér's V = {result['Effect size']:.4f})."
        )

        expected_cells = result["Expected cells < 5"]

        if expected_cells != "N/A":
            low_count, total_count = map(
                int,
                expected_cells.split("/"),
            )

            if low_count > 0:
                low_percentage = (
                    low_count / total_count * 100
                )

                print(
                    f"Caution: {low_count} of {total_count} "
                    f"expected cells ({low_percentage:.1f}%) "
                    "have expected counts below 5."
                )

    elif method == "Spearman correlation":
        print(
            f"Correlation: {result['Effect strength']} "
            f"(rho = {result['Statistic']:.4f})."
        )

        if not pd.isna(result["Statistic"]):
            if result["Statistic"] > 0:
                print(
                    "Higher traffic density tends to be associated "
                    "with a higher abnormality count per image."
                )
            elif result["Statistic"] < 0:
                print(
                    "Higher traffic density tends to be associated "
                    "with a lower abnormality count per image."
                )

    elif method == "Kruskal-Wallis":
        if p_value < SIGNIFICANCE_LEVEL:
            print(
                "Abnormality-frequency distributions differ among "
                "at least two traffic-density groups."
            )
        else:
            print(
                "Abnormality-frequency distributions do not show a "
                "significant difference across traffic-density groups."
            )


print(
    "\nNote: These tests identify statistical associations, not "
    "causal relationships. Weather, road type, and traffic density "
    "should not be interpreted as proven causes of abnormalities."
)