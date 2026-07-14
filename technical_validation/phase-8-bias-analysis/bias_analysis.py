from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_FILES = {
    "layer0": "layer_0_raw_image_metadata.csv",
    "layer1": "layer_1_annotation_metadata.csv",
    "layer2": "layer_2_semantic_metadata.csv",
    "layer3": "layer_3_geospatial_metadata.csv",
}

STANDARD_FOLDERS = {
    "layer0": "layer_0_raw_images",
    "layer1": "layer_1_annotations",
    "layer2": "layer_2_metadata",
    "layer3": "layer_3_geospatial",
}

SEPARATOR = "=" * 88
SUBSEPARATOR = "-" * 88


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Phase VIII dataset-bias analysis tables."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Repository root. When omitted, the script searches upward from "
            "its own location and the current working directory."
        ),
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print missing-value tables for the analyzed columns.",
    )
    return parser.parse_args()


def candidate_roots(explicit_root: Path | None) -> list[Path]:
    roots: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)

    if explicit_root is not None:
        add(explicit_root)

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    for start in (script_dir, cwd):
        add(start)
        add(start.parent)
        for parent in start.parents:
            add(parent)

    return roots


def find_metadata_file(
    key: str, filename: str, roots: Iterable[Path]
) -> Path:
    checked: list[Path] = []

    for root in roots:
        direct_candidates = [
            root / filename,
            root / STANDARD_FOLDERS[key] / filename,
        ]
        for candidate in direct_candidates:
            checked.append(candidate)
            if candidate.is_file():
                return candidate

    ignored_parts = {
        ".git", ".venv", "venv", "env", "__pycache__", "node_modules"
    }
    for root in roots[:4]:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in root.rglob(filename):
                if any(part in ignored_parts for part in candidate.parts):
                    continue
                if candidate.is_file():
                    return candidate.resolve()
        except (OSError, PermissionError):
            continue

    checked_text = "\n".join(f"  - {p}" for p in checked[:20])
    raise FileNotFoundError(
        f"Could not locate {filename}.\nChecked locations included:\n"
        f"{checked_text}\n"
        "Use --project-root to provide the repository root explicitly."
    )


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s): {', '.join(missing)}"
        )

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype("string").str.strip()

    return df


def normalize_category(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.isna() | result.eq(""), "Missing")


def format_number(value: float | int) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.4f}"


def print_heading(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def distribution_table(
    series: pd.Series,
    category_name: str,
    unit_name: str,
) -> pd.DataFrame:
    clean = normalize_category(series)
    counts = clean.value_counts(dropna=False)
    total = int(counts.sum())

    table = counts.rename_axis(category_name).reset_index(name="count")
    table["percentage"] = (
        table["count"].div(total).mul(100).round(2) if total else 0.0
    )
    table["unit"] = unit_name
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", category_name, "count", "percentage", "unit"]]


def imbalance_summary(
    series: pd.Series,
    dimension: str,
    unit_name: str,
) -> pd.DataFrame:
    clean = normalize_category(series)
    counts = clean.value_counts(dropna=False).astype(float)
    total = float(counts.sum())
    number_categories = int(len(counts))

    if number_categories == 0 or total == 0:
        values = {
            "dimension": dimension,
            "unit": unit_name,
            "total": 0,
            "categories": 0,
            "largest_count": np.nan,
            "smallest_count": np.nan,
            "max_min_ratio": np.nan,
            "coefficient_of_variation": np.nan,
            "normalized_entropy": np.nan,
            "largest_share_pct": np.nan,
        }
        return pd.DataFrame([values])

    maximum = float(counts.max())
    minimum = float(counts.min())
    mean = float(counts.mean())
    std = float(counts.std(ddof=0))
    probabilities = counts / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized_entropy = (
        entropy / math.log(number_categories) if number_categories > 1 else 1.0
    )

    values = {
        "dimension": dimension,
        "unit": unit_name,
        "total": int(total),
        "categories": number_categories,
        "largest_count": int(maximum),
        "smallest_count": int(minimum),
        "max_min_ratio": round(maximum / minimum, 4) if minimum > 0 else np.inf,
        "coefficient_of_variation": round(std / mean, 4) if mean > 0 else np.nan,
        "normalized_entropy": round(normalized_entropy, 4),
        "largest_share_pct": round(maximum / total * 100, 2),
    }
    return pd.DataFrame([values])


def print_table(table: pd.DataFrame, index: bool = False) -> None:
    if table.empty:
        print("(No rows available)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 180,
        "display.max_colwidth", 50,
    ):
        print(table.to_string(index=index))


def print_distribution_section(
    title: str,
    series: pd.Series,
    category_name: str,
    unit_name: str,
) -> None:
    print_heading(title)
    table = distribution_table(series, category_name, unit_name)
    print_table(table)

    print(f"\n{SUBSEPARATOR}\nImbalance indicators\n{SUBSEPARATOR}")
    summary = imbalance_summary(series, title, unit_name)
    print_table(summary)


def unique_image_table(
    df: pd.DataFrame,
    category_column: str,
    category_name: str,
) -> pd.DataFrame:
    working = df[["image_id", category_column]].copy()
    working[category_column] = normalize_category(working[category_column])
    working = working.drop_duplicates(["image_id", category_column])
    return distribution_table(
        working[category_column], category_name, "unique images"
    )


def print_class_imbalance(layer1: pd.DataFrame, layer2: pd.DataFrame) -> None:
    print_heading("1. CLASS IMBALANCE")

    print("A. Annotation-instance distribution")
    annotation_table = distribution_table(
        layer1["abnormality_type"],
        "abnormality_type",
        "annotation instances",
    )
    print_table(annotation_table)

    print(f"\n{SUBSEPARATOR}\nB. Unique-image distribution from Layer 1\n{SUBSEPARATOR}")
    image_table_l1 = unique_image_table(
        layer1, "abnormality_type", "abnormality_type"
    )
    print_table(image_table_l1)

    print(f"\n{SUBSEPARATOR}\nC. Image-level semantic distribution from Layer 2\n{SUBSEPARATOR}")
    semantic_table = distribution_table(
        layer2["abnormality_type"],
        "abnormality_type",
        "metadata rows",
    )
    print_table(semantic_table)

    print(f"\n{SUBSEPARATOR}\nD. Class imbalance indicators\n{SUBSEPARATOR}")
    summaries = pd.concat(
        [
            imbalance_summary(
                layer1["abnormality_type"],
                "Class — annotation instances",
                "annotation instances",
            ),
            imbalance_summary(
                layer1.drop_duplicates(
                    ["image_id", "abnormality_type"]
                )["abnormality_type"],
                "Class — unique images",
                "unique images",
            ),
            imbalance_summary(
                layer2["abnormality_type"],
                "Class — Layer 2 rows",
                "metadata rows",
            ),
        ],
        ignore_index=True,
    )
    print_table(summaries)


def print_regional_imbalance(layer3: pd.DataFrame) -> None:
    print_heading("2. REGIONAL IMBALANCE")

    print("A. State distribution")
    state_table = distribution_table(
        layer3["collection_state"], "state", "images"
    )
    print_table(state_table)

    print(f"\n{SUBSEPARATOR}\nB. District distribution\n{SUBSEPARATOR}")
    district_table = distribution_table(
        layer3["district"], "district", "images"
    )
    print_table(district_table)

    print(
        f"\n{SUBSEPARATOR}\n"
        "C. State × district distribution\n"
        f"{SUBSEPARATOR}"
    )
    cross = layer3[["collection_state", "district", "image_id"]].copy()
    cross["collection_state"] = normalize_category(cross["collection_state"])
    cross["district"] = normalize_category(cross["district"])
    cross_table = (
        cross.groupby(["collection_state", "district"], dropna=False)["image_id"]
        .nunique()
        .reset_index(name="image_count")
        .sort_values(
            ["image_count", "collection_state", "district"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    total_images = max(layer3["image_id"].nunique(), 1)
    cross_table["dataset_percentage"] = (
        cross_table["image_count"].div(total_images).mul(100).round(2)
    )
    print_table(cross_table)

    print(f"\n{SUBSEPARATOR}\nD. Regional imbalance indicators\n{SUBSEPARATOR}")
    summaries = pd.concat(
        [
            imbalance_summary(
                layer3["collection_state"], "Region — state", "images"
            ),
            imbalance_summary(
                layer3["district"], "Region — district", "images"
            ),
        ],
        ignore_index=True,
    )
    print_table(summaries)


def object_size_analysis(layer1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    boxes = layer1[
        ["image_id", "abnormality_type", "width_scaled", "height_scaled"]
    ].copy()
    boxes["width_scaled"] = pd.to_numeric(
        boxes["width_scaled"], errors="coerce"
    )
    boxes["height_scaled"] = pd.to_numeric(
        boxes["height_scaled"], errors="coerce"
    )

    boxes["relative_area"] = boxes["width_scaled"] * boxes["height_scaled"]
    valid = (
        boxes["width_scaled"].between(0, 1, inclusive="both")
        & boxes["height_scaled"].between(0, 1, inclusive="both")
        & boxes["relative_area"].notna()
    )
    boxes = boxes.loc[valid].copy()

    bins = [-np.inf, 0.01, 0.05, 0.15, np.inf]
    labels = [
        "Tiny (<1%)",
        "Small (1%–<5%)",
        "Medium (5%–<15%)",
        "Large (>=15%)",
    ]
    boxes["size_category"] = pd.cut(
        boxes["relative_area"],
        bins=bins,
        labels=labels,
        right=False,
        ordered=True,
    )

    size_table = distribution_table(
        boxes["size_category"], "object_size", "annotation instances"
    )

    descriptive = pd.DataFrame(
        {
            "metric": [
                "valid_boxes",
                "invalid_or_missing_boxes",
                "mean_relative_area",
                "median_relative_area",
                "std_relative_area",
                "minimum_relative_area",
                "25th_percentile",
                "75th_percentile",
                "90th_percentile",
                "95th_percentile",
                "maximum_relative_area",
            ],
            "value": [
                len(boxes),
                len(layer1) - len(boxes),
                boxes["relative_area"].mean(),
                boxes["relative_area"].median(),
                boxes["relative_area"].std(ddof=0),
                boxes["relative_area"].min(),
                boxes["relative_area"].quantile(0.25),
                boxes["relative_area"].quantile(0.75),
                boxes["relative_area"].quantile(0.90),
                boxes["relative_area"].quantile(0.95),
                boxes["relative_area"].max(),
            ],
        }
    )
    descriptive["value"] = descriptive["value"].map(format_number)

    return boxes, size_table, descriptive


def print_object_size_imbalance(layer1: pd.DataFrame) -> None:
    print_heading("6. OBJECT-SIZE IMBALANCE")
    boxes, size_table, descriptive = object_size_analysis(layer1)

    print(
        "Size bins use normalized bounding-box area "
        "(width_scaled × height_scaled)."
    )
    print("Tiny <1%; Small 1–<5%; Medium 5–<15%; Large >=15%.\n")

    print("A. Object-size distribution")
    print_table(size_table)

    print(f"\n{SUBSEPARATOR}\nB. Relative-area descriptive statistics\n{SUBSEPARATOR}")
    print_table(descriptive)

    print(f"\n{SUBSEPARATOR}\nC. Object size by abnormality class\n{SUBSEPARATOR}")
    if boxes.empty:
        print("(No valid bounding boxes available)")
    else:
        class_size = pd.crosstab(
            normalize_category(boxes["abnormality_type"]),
            boxes["size_category"],
            dropna=False,
        )
        class_size["Total"] = class_size.sum(axis=1)
        class_size = class_size.sort_values("Total", ascending=False)
        print_table(class_size.reset_index())

        print(
            f"\n{SUBSEPARATOR}\n"
            "D. Within-class object-size percentages\n"
            f"{SUBSEPARATOR}"
        )
        pct = pd.crosstab(
            normalize_category(boxes["abnormality_type"]),
            boxes["size_category"],
            normalize="index",
            dropna=False,
        ).mul(100).round(2)
        pct["Total_pct"] = pct.sum(axis=1).round(2)
        print_table(pct.reset_index())

    print(f"\n{SUBSEPARATOR}\nE. Object-size imbalance indicators\n{SUBSEPARATOR}")
    summary = imbalance_summary(
        boxes["size_category"],
        "Object size",
        "annotation instances",
    )
    print_table(summary)


def print_missing_values(
    layer0: pd.DataFrame,
    layer1: pd.DataFrame,
    layer2: pd.DataFrame,
    layer3: pd.DataFrame,
) -> None:
    print_heading("SUPPLEMENTARY: MISSING VALUES")

    datasets = {
        "Layer 0": layer0,
        "Layer 1": layer1,
        "Layer 2": layer2,
        "Layer 3": layer3,
    }
    rows = []
    for name, df in datasets.items():
        for column in df.columns:
            missing = int(df[column].isna().sum())
            blank = int(
                df[column].astype("string").str.strip().eq("").fillna(False).sum()
            )
            rows.append(
                {
                    "dataset": name,
                    "column": column,
                    "rows": len(df),
                    "missing_or_blank": missing + blank,
                    "percentage": round((missing + blank) / max(len(df), 1) * 100, 2),
                }
            )
    table = pd.DataFrame(rows)
    print_table(table)


def main() -> int:
    args = parse_args()
    roots = candidate_roots(args.project_root)

    try:
        paths = {
            key: find_metadata_file(key, filename, roots)
            for key, filename in EXPECTED_FILES.items()
        }

        layer0 = load_csv(
            paths["layer0"],
            {
                "image_id",
                "collection_date",
                "collection_state",
            },
        )
        layer1 = load_csv(
            paths["layer1"],
            {
                "image_id",
                "abnormality_type",
                "width_scaled",
                "height_scaled",
            },
        )
        layer2 = load_csv(
            paths["layer2"],
            {
                "image_id",
                "abnormality_type",
                "weather",
                "lighting",
            },
        )
        layer3 = load_csv(
            paths["layer3"],
            {
                "image_id",
                "collection_state",
                "district",
                "collection_season",
            },
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(SEPARATOR)
    print("PHASE VIII — DATASET BIAS ANALYSIS")
    print(SEPARATOR)
    print("\nMetadata files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print("\nDataset sizes:")
    print(f"  Unique raw images       : {layer0['image_id'].nunique():,}")
    print(f"  Annotation instances    : {len(layer1):,}")
    print(f"  Layer 2 metadata rows   : {len(layer2):,}")
    print(f"  Layer 3 geospatial rows : {len(layer3):,}")

    print_class_imbalance(layer1, layer2)
    print_regional_imbalance(layer3)

    print_distribution_section(
        "3. SEASONAL IMBALANCE",
        layer3["collection_season"],
        "season",
        "images",
    )
    print_distribution_section(
        "4. LIGHTING IMBALANCE",
        layer2["lighting"],
        "lighting",
        "images",
    )
    print_distribution_section(
        "5. WEATHER IMBALANCE",
        layer2["weather"],
        "weather",
        "images",
    )
    print_object_size_imbalance(layer1)

    if args.show_missing:
        print_missing_values(layer0, layer1, layer2, layer3)

    print(f"\n{SEPARATOR}")
    print("BIAS ANALYSIS COMPLETE")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
