from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEPARATOR = "=" * 88
SUBSEPARATOR = "-" * 88

EXPECTED_FILES = {
    "layer1": ("layer_1_annotations", "layer_1_annotation_metadata.csv"),
    "layer2": ("layer_2_metadata", "layer_2_semantic_metadata.csv"),
    "layer3": ("layer_3_geospatial", "layer_3_geospatial_metadata.csv"),
}

ROBUSTNESS_CONDITIONS = [
    ("Night", "lighting", "night"),
    ("Rain / wet road", "weather", "wet_road"),
    ("Shadow", "lighting", "shadow"),
    ("Urban", "road_type", "urban"),
    ("Highway", "road_type", "highway"),
    ("Rural", "road_type", "rural"),
    ("Low light", "lighting", "low_light"),
]

OBJECT_SIZE_BINS = [
    ("Tiny objects (<1%)", "tiny", 0.00, 0.01),
    ("Small objects (1%-<5%)", "small", 0.01, 0.05),
    ("Medium objects (5%-<15%)", "medium", 0.05, 0.15),
    ("Large objects (>=15%)", "large", 0.15, np.inf),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase VI robustness evaluation using metadata-defined slices. "
            "No model retraining is performed."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help=(
            "Optional prediction CSV with image_id and true/pred columns. "
            "If supplied, per-slice accuracy is computed."
        ),
    )
    parser.add_argument(
        "--true-column",
        default=None,
        help="Ground-truth column in prediction CSV. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--pred-column",
        default=None,
        help="Prediction column in prediction CSV. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <this script folder>/outputs.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=6,
        help="Number of top abnormality classes to show per robustness slice.",
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
        for parent in start.parents:
            add(parent)

    return roots


def find_metadata_file(key: str, roots: Iterable[Path]) -> Path:
    folder, filename = EXPECTED_FILES[key]
    checked: list[Path] = []
    for root in roots:
        candidate = root / folder / filename
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked[:20])
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n{checked_text}"
    )


def load_metadata(project_root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roots = candidate_roots(project_root)
    layer1_path = find_metadata_file("layer1", roots)
    layer2_path = find_metadata_file("layer2", roots)
    layer3_path = find_metadata_file("layer3", roots)

    layer1 = pd.read_csv(layer1_path)
    layer2 = pd.read_csv(layer2_path)
    layer3 = pd.read_csv(layer3_path)
    return layer1, layer2, layer3


def build_image_table(layer1: pd.DataFrame, layer2: pd.DataFrame, layer3: pd.DataFrame) -> pd.DataFrame:
    image_df = layer2.merge(layer3, on="image_id", how="left")

    boxes = layer1.copy()
    boxes["object_area"] = boxes["width_scaled"] * boxes["height_scaled"]
    box_summary = (
        boxes.groupby("image_id", as_index=False)
        .agg(
            object_count=("abnormality_id", "count"),
            mean_object_area=("object_area", "mean"),
            max_object_area=("object_area", "max"),
        )
    )
    return image_df.merge(box_summary, on="image_id", how="left")


def condition_mask(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().eq(value.lower())


def object_size_mask(df: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    area = df["max_object_area"].fillna(-1)
    if np.isinf(upper):
        return area >= lower
    return (area >= lower) & (area < upper)


def summarize_slice(df: pd.DataFrame, name: str, mask: pd.Series, total_images: int) -> dict[str, object]:
    subset = df.loc[mask].copy()
    count = int(len(subset))
    percentage = (count / total_images * 100) if total_images else 0.0

    dominant_class = "N/A"
    dominant_class_pct = np.nan
    if count and "abnormality_type" in subset.columns:
        class_counts = subset["abnormality_type"].fillna("Missing").value_counts()
        dominant_class = str(class_counts.index[0])
        dominant_class_pct = float(class_counts.iloc[0] / count * 100)

    dominant_severity = "N/A"
    if count and "severity" in subset.columns:
        severity_counts = subset["severity"].fillna("Missing").value_counts()
        dominant_severity = str(severity_counts.index[0])

    return {
        "Slice": name,
        "Images": count,
        "Dataset percentage": round(percentage, 2),
        "Dominant abnormality type": dominant_class,
        "Dominant class percentage": round(dominant_class_pct, 2)
        if not np.isnan(dominant_class_pct)
        else "N/A",
        "Dominant severity": dominant_severity,
    }


def top_class_distribution(
    df: pd.DataFrame, name: str, mask: pd.Series, top_n: int
) -> pd.DataFrame:
    subset = df.loc[mask]
    if subset.empty:
        return pd.DataFrame(
            columns=["Slice", "Rank", "Abnormality Type", "Count", "Percentage"]
        )

    counts = subset["abnormality_type"].fillna("Missing").value_counts().head(top_n)
    rows = []
    for rank, (label, count) in enumerate(counts.items(), start=1):
        rows.append(
            {
                "Slice": name,
                "Rank": rank,
                "Abnormality Type": label,
                "Count": int(count),
                "Percentage": round(float(count / len(subset) * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def state_summary(df: pd.DataFrame, total_images: int) -> pd.DataFrame:
    counts = df["collection_state"].fillna("Missing").value_counts()
    rows = []
    for state, count in counts.items():
        rows.append(
            {
                "State": state,
                "Images": int(count),
                "Dataset percentage": round(float(count / total_images * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def detect_prediction_columns(
    predictions: pd.DataFrame, true_column: str | None, pred_column: str | None
) -> tuple[str, str]:
    true_candidates = [
        true_column,
        "y_true",
        "true",
        "true_label",
        "ground_truth",
        "label",
        "severity_true",
        "abnormality_type_true",
    ]
    pred_candidates = [
        pred_column,
        "y_pred",
        "pred",
        "prediction",
        "pred_label",
        "severity_pred",
        "abnormality_type_pred",
    ]

    true_name = next((col for col in true_candidates if col and col in predictions.columns), None)
    pred_name = next((col for col in pred_candidates if col and col in predictions.columns), None)

    if true_name is None or pred_name is None:
        raise ValueError(
            "Could not detect prediction columns. Provide --true-column and --pred-column."
        )
    return true_name, pred_name


def prediction_accuracy_by_slice(
    image_df: pd.DataFrame,
    slice_masks: dict[str, pd.Series],
    predictions_csv: Path,
    true_column: str | None,
    pred_column: str | None,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_csv)
    if "image_id" not in predictions.columns:
        raise ValueError("Prediction CSV must contain an image_id column.")

    true_name, pred_name = detect_prediction_columns(predictions, true_column, pred_column)
    merged = image_df[["image_id"]].merge(predictions, on="image_id", how="inner")
    merged["_correct"] = merged[true_name].astype(str).eq(merged[pred_name].astype(str))

    rows = []
    for name, mask in slice_masks.items():
        ids = set(image_df.loc[mask, "image_id"])
        subset = merged[merged["image_id"].isin(ids)]
        evaluated = int(len(subset))
        accuracy = float(subset["_correct"].mean()) if evaluated else np.nan
        rows.append(
            {
                "Slice": name,
                "Evaluated images": evaluated,
                "Accuracy": round(accuracy, 4) if not np.isnan(accuracy) else "N/A",
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * len(title))
    if df.empty:
        print("No rows available.")
    else:
        print(df.to_string(index=False))


def markdown_table(title: str, df: pd.DataFrame) -> str:
    lines = [f"### {title}", ""]
    if df.empty:
        lines.append("No rows available.")
    else:
        lines.append(df.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    layer1, layer2, layer3 = load_metadata(args.project_root)
    image_df = build_image_table(layer1, layer2, layer3)
    total_images = int(image_df["image_id"].nunique())

    slice_masks: dict[str, pd.Series] = {}
    for name, column, value in ROBUSTNESS_CONDITIONS:
        slice_masks[name] = condition_mask(image_df, column, value)

    for display_name, _, lower, upper in OBJECT_SIZE_BINS:
        slice_masks[display_name] = object_size_mask(image_df, lower, upper)

    if "collection_state" in image_df.columns:
        for state in sorted(image_df["collection_state"].dropna().unique()):
            slice_masks[f"State: {state}"] = condition_mask(
                image_df, "collection_state", str(state)
            )

    summary_df = pd.DataFrame(
        [summarize_slice(image_df, name, mask, total_images) for name, mask in slice_masks.items()]
    )

    class_tables = [
        top_class_distribution(image_df, name, mask, args.top_classes)
        for name, mask in slice_masks.items()
    ]
    class_distribution_df = pd.concat(class_tables, ignore_index=True)

    state_df = state_summary(image_df, total_images)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "phase-6-robustness-slice-summary.csv", index=False)
    class_distribution_df.to_csv(
        output_dir / "phase-6-robustness-class-distribution-by-slice.csv",
        index=False,
    )
    state_df.to_csv(output_dir / "phase-6-robustness-state-summary.csv", index=False)

    accuracy_df = None
    if args.predictions_csv is not None:
        accuracy_df = prediction_accuracy_by_slice(
            image_df,
            slice_masks,
            args.predictions_csv.expanduser().resolve(),
            args.true_column,
            args.pred_column,
        )
        accuracy_df.to_csv(
            output_dir / "phase-6-robustness-prediction-accuracy-by-slice.csv",
            index=False,
        )

    md_parts = [
        "# Phase VI - Robustness Evaluation",
        "",
        "This report evaluates dataset robustness slices without retraining models.",
        "",
        markdown_table("Robustness Slice Summary", summary_df),
        markdown_table("State Coverage", state_df),
        markdown_table("Top Class Distribution by Slice", class_distribution_df),
    ]
    if accuracy_df is not None:
        md_parts.append(markdown_table("Prediction Accuracy by Slice", accuracy_df))

    md_parts.append(
        "> **Note:** Robustness slices include night, wet-road/rain proxy, shadow, "
        "urban, highway, rural, low-light, object-size groups, and state-wise subsets."
    )

    report_path = output_dir / "phase-6-robustness-evaluation-report.md"
    report_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(SEPARATOR)
    print("PHASE VI - ROBUSTNESS EVALUATION")
    print(SEPARATOR)
    print(f"Total images evaluated: {total_images}")
    print(f"Output directory: {output_dir}")

    print_table("Robustness Slice Summary", summary_df)
    print_table("State Coverage", state_df)
    if accuracy_df is not None:
        print_table("Prediction Accuracy by Slice", accuracy_df)

    print("\nGenerated files:")
    print(f"  - {output_dir / 'phase-6-robustness-slice-summary.csv'}")
    print(f"  - {output_dir / 'phase-6-robustness-class-distribution-by-slice.csv'}")
    print(f"  - {output_dir / 'phase-6-robustness-state-summary.csv'}")
    if accuracy_df is not None:
        print(f"  - {output_dir / 'phase-6-robustness-prediction-accuracy-by-slice.csv'}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
