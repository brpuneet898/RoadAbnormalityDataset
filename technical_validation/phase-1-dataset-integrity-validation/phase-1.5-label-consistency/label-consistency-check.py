# label_consistency_check.py

from pathlib import Path
import pandas as pd


# ---------------- CONFIG ----------------
ROOT_DIR = Path("../../..")

LAYER_DIRS = {
    "Layer 1": ROOT_DIR / "layer_1_annotations",
    "Layer 2": ROOT_DIR / "layer_2_metadata",
    "Layer 3": ROOT_DIR / "layer_3_geospatial",
}

# Column names expected somewhere in your CSV files
WEATHER_COL = "weather"
LIGHTING_COL = "lighting"
ROAD_TYPE_COL = "road_type"
TRAFFIC_DENSITY_COL = "traffic_density"

# Logical consistency rules
INVALID_WEATHER_LIGHTING = {
    ("night", "daylight"),
    ("night", "bright daylight"),
    ("night", "sunlight"),
    ("day", "night"),
    ("daytime", "night"),
    ("sunny", "night"),
}

INVALID_ROAD_TRAFFIC = {
    ("highway", "very low"),
    ("highway", "low"),
    ("expressway", "very low"),
    ("expressway", "low"),
    ("residential", "very high"),
    ("village road", "very high"),
    ("rural road", "very high"),
    ("narrow road", "very high"),
}
# ----------------------------------------


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def find_column(df, possible_names):
    normalized_cols = {col.strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in normalized_cols:
            return normalized_cols[name.lower()]

    return None


def get_image_id(row, df):
    possible_id_cols = ["image_id", "Image_ID", "filename", "file_name", "image_name"]

    for col in possible_id_cols:
        actual_col = find_column(df, [col])
        if actual_col is not None:
            return Path(str(row[actual_col]).strip()).stem

    return "UNKNOWN"


def check_file(csv_path, layer_name):
    issues = []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return [{
            "layer": layer_name,
            "file": csv_path.name,
            "image_id": "N/A",
            "rule": "CSV read error",
            "details": str(e),
        }]

    weather_col = find_column(df, [WEATHER_COL, "time_of_day", "weather_condition"])
    lighting_col = find_column(df, [LIGHTING_COL, "light_condition", "illumination"])
    road_type_col = find_column(df, [ROAD_TYPE_COL, "road_category", "road_class"])
    traffic_col = find_column(df, [TRAFFIC_DENSITY_COL, "traffic", "traffic_level"])

    for _, row in df.iterrows():
        image_id = get_image_id(row, df)

        if weather_col and lighting_col:
            weather = normalize(row[weather_col])
            lighting = normalize(row[lighting_col])

            if (weather, lighting) in INVALID_WEATHER_LIGHTING:
                issues.append({
                    "layer": layer_name,
                    "file": csv_path.name,
                    "image_id": image_id,
                    "rule": "Weather vs lighting",
                    "details": f"weather='{weather}', lighting='{lighting}'",
                })

        if road_type_col and traffic_col:
            road_type = normalize(row[road_type_col])
            traffic = normalize(row[traffic_col])

            if (road_type, traffic) in INVALID_ROAD_TRAFFIC:
                issues.append({
                    "layer": layer_name,
                    "file": csv_path.name,
                    "image_id": image_id,
                    "rule": "Road type vs traffic density",
                    "details": f"road_type='{road_type}', traffic_density='{traffic}'",
                })

    return issues


def print_summary_table(rows):
    print("\nLABEL CONSISTENCY SUMMARY")
    print("=" * 70)
    print(f"{'Layer':<12} | {'Files checked':>13} | {'Issues found':>12}")
    print("-" * 70)

    for row in rows:
        print(
            f"{row['layer']:<12} | "
            f"{row['files_checked']:>13} | "
            f"{row['issues_found']:>12}"
        )


def print_issue_table(issues):
    if not issues:
        print("\nResult: PASS - No label consistency issues found.")
        return

    print("\nLABEL CONSISTENCY ISSUES")
    print("=" * 120)
    print(f"{'Layer':<10} | {'File':<30} | {'Image ID':<15} | {'Rule':<28} | Details")
    print("-" * 120)

    for issue in issues:
        print(
            f"{issue['layer']:<10} | "
            f"{issue['file']:<30} | "
            f"{issue['image_id']:<15} | "
            f"{issue['rule']:<28} | "
            f"{issue['details']}"
        )


def main():
    all_issues = []
    summary_rows = []

    for layer_name, layer_path in LAYER_DIRS.items():
        csv_files = list(layer_path.rglob("*.csv"))
        layer_issues = []

        for csv_path in csv_files:
            layer_issues.extend(check_file(csv_path, layer_name))

        all_issues.extend(layer_issues)

        summary_rows.append({
            "layer": layer_name,
            "files_checked": len(csv_files),
            "issues_found": len(layer_issues),
        })

    print_summary_table(summary_rows)
    print_issue_table(all_issues)


if __name__ == "__main__":
    main()