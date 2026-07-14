from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_NAMES = ("train", "validation", "test")
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
ANNOTATION_EXTENSIONS = {
    ".txt", ".json", ".xml", ".yaml", ".yml", ".csv", ".mask", ".npy"
}

CSV_RELATIVE_PATHS = {
    "layer_0": Path("layer_0_raw_images/layer_0_raw_image_metadata.csv"),
    "layer_1": Path("layer_1_annotations/layer_1_annotation_metadata.csv"),
    "layer_2": Path("layer_2_metadata/layer_2_semantic_metadata.csv"),
    "layer_3": Path("layer_3_geospatial/layer_3_geospatial_metadata.csv"),
}

ZIP_NAMES = {
    "random": "phase7_random_image_split_70_15_15.zip",
    "region": "phase7_region_wise_split.zip",
    "state": "phase7_state_wise_split.zip",
    "loso": "phase7_leave_one_state_out.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four Phase VII dataset-split ZIP archives."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. By default it is inferred from the script location.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Phase VII ZIP files.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Package available files even when some image files are missing.",
    )
    return parser.parse_args()


def infer_project_root(script_path: Path) -> Path:
    """Find the nearest ancestor containing the expected layer folders."""
    candidates = [script_path.parent, *script_path.parents]
    for candidate in candidates:
        if (candidate / "layer_0_raw_images").is_dir() and (
            candidate / "layer_1_annotations"
        ).is_dir():
            return candidate
    if len(script_path.parents) >= 3:
        return script_path.parents[2]
    return script_path.parent


def locate_csv(project_root: Path, key: str, expected: Path) -> Path:
    direct = project_root / expected
    if direct.exists():
        return direct

    filename = expected.name
    matches = [
        p for p in project_root.rglob(filename)
        if "phase-7-dataset-split-strategy" not in p.parts
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not locate {key} metadata CSV: expected {direct}"
        )
    raise RuntimeError(
        f"Multiple candidates found for {filename}:\n"
        + "\n".join(f"  - {p}" for p in matches)
    )


def load_metadata(project_root: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    frames: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}
    for key, relative in CSV_RELATIVE_PATHS.items():
        path = locate_csv(project_root, key, relative)
        frame = pd.read_csv(path, dtype={"image_id": "string"})
        if "image_id" not in frame.columns:
            raise ValueError(f"{path} does not contain an image_id column.")
        frame["image_id"] = frame["image_id"].astype(str).str.strip()
        frames[key] = frame
        paths[key] = path
    return frames, paths


def validate_master_metadata(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    master = frames["layer_3"].copy()
    required = {"image_id", "collection_state", "district"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(
            "Layer 3 metadata is missing required columns: " + ", ".join(sorted(missing))
        )

    if master["image_id"].duplicated().any():
        duplicate_ids = master.loc[master["image_id"].duplicated(), "image_id"].head(10)
        raise ValueError(
            "Layer 3 must have one row per image. Duplicate IDs include: "
            + ", ".join(duplicate_ids.astype(str))
        )

    master["collection_state"] = master["collection_state"].fillna("UNKNOWN_STATE").astype(str)
    master["district"] = master["district"].fillna("UNKNOWN_DISTRICT").astype(str)
    master["region"] = master["collection_state"] + " :: " + master["district"]

    master_ids = set(master["image_id"])
    layer0_ids = set(frames["layer_0"]["image_id"])
    if master_ids != layer0_ids:
        only_l3 = sorted(master_ids - layer0_ids)[:10]
        only_l0 = sorted(layer0_ids - master_ids)[:10]
        raise ValueError(
            "Layer 0 and Layer 3 image_id sets do not match. "
            f"Only in Layer 3: {only_l3}; only in Layer 0: {only_l0}"
        )
    return master


def index_files(folder: Path, extensions: Set[str]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    if not folder.exists():
        return index
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index[path.stem].append(path)
    return index


def find_layer_folder(project_root: Path, name: str) -> Path:
    direct = project_root / name
    if direct.is_dir():
        return direct
    matches = [p for p in project_root.rglob(name) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    return direct


def choose_single_image(image_id: str, index: Mapping[str, List[Path]]) -> Path | None:
    candidates = index.get(image_id, [])
    if not candidates:
        return None
    priority = {".jpg": 0, ".jpeg": 1, ".png": 2, ".webp": 3, ".tif": 4, ".tiff": 5}
    return sorted(candidates, key=lambda p: (priority.get(p.suffix.lower(), 99), str(p)))[0]


def get_annotation_files(image_id: str, index: Mapping[str, List[Path]]) -> List[Path]:
    return sorted(index.get(image_id, []), key=str)


def exact_image_split(image_ids: Sequence[str], seed: int) -> Dict[str, List[str]]:
    ids = list(dict.fromkeys(image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    return {
        "train": ids[:n_train],
        "validation": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def grouped_split(
    master: pd.DataFrame,
    group_column: str,
    seed: int,
    attempts: int = 10000,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Assign whole groups while approximating 70/15/15 by image count."""
    grouped = {
        str(group): rows["image_id"].tolist()
        for group, rows in master.groupby(group_column, sort=True)
    }
    group_names = sorted(grouped)
    if len(group_names) < 3:
        raise ValueError(
            f"{group_column} split needs at least 3 unique groups; found {len(group_names)}."
        )

    total = len(master)
    targets = {
        "train": total * TRAIN_RATIO,
        "validation": total * VAL_RATIO,
        "test": total * TEST_RATIO,
    }
    rng = random.Random(seed)
    best_assignment = None
    best_score = float("inf")

    for _ in range(attempts):
        order = group_names[:]
        rng.shuffle(order)
        assignment = {name: [] for name in SPLIT_NAMES}
        counts = {name: 0 for name in SPLIT_NAMES}

        for split_name, group_name in zip(SPLIT_NAMES, order[:3]):
            assignment[split_name].append(group_name)
            counts[split_name] += len(grouped[group_name])

        for group_name in order[3:]:
            size = len(grouped[group_name])
            choices = []
            for split_name in SPLIT_NAMES:
                trial = counts.copy()
                trial[split_name] += size
                score = sum(
                    ((trial[s] - targets[s]) / max(targets[s], 1.0)) ** 2
                    for s in SPLIT_NAMES
                )
                choices.append((score, rng.random(), split_name))
            chosen = min(choices)[2]
            assignment[chosen].append(group_name)
            counts[chosen] += size

        score = sum(
            ((counts[s] - targets[s]) / max(targets[s], 1.0)) ** 2
            for s in SPLIT_NAMES
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment

    assert best_assignment is not None
    split_ids = {
        split_name: [
            image_id
            for group_name in best_assignment[split_name]
            for image_id in grouped[group_name]
        ]
        for split_name in SPLIT_NAMES
    }
    return split_ids, best_assignment


def loso_splits(master: pd.DataFrame, seed: int) -> Dict[str, Dict[str, List[str]]]:
    folds: Dict[str, Dict[str, List[str]]] = {}
    states = sorted(master["collection_state"].unique())
    for fold_number, held_out_state in enumerate(states, start=1):
        test_ids = master.loc[
            master["collection_state"] == held_out_state, "image_id"
        ].tolist()
        remaining = master[master["collection_state"] != held_out_state].copy()

        remaining_ids = remaining["image_id"].tolist()
        rng = random.Random(seed + fold_number)
        rng.shuffle(remaining_ids)
        train_fraction_of_remaining = TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)
        n_train = round(len(remaining_ids) * train_fraction_of_remaining)

        fold_key = sanitize_name(str(held_out_state))
        folds[fold_key] = {
            "train": remaining_ids[:n_train],
            "validation": remaining_ids[n_train:],
            "test": test_ids,
        }
    return folds


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unnamed"


def verify_split(split_ids: Mapping[str, Sequence[str]], all_ids: Set[str]) -> None:
    sets = {name: set(ids) for name, ids in split_ids.items()}
    for name, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image_id values inside {name} split.")
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(
                    f"Image leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )
    union = set().union(*sets.values())
    if union != all_ids:
        raise ValueError(
            f"Split coverage error: missing={len(all_ids - union)}, extra={len(union - all_ids)}"
        )


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def write_split_contents(
    archive: zipfile.ZipFile,
    prefix: str,
    split_ids: Mapping[str, Sequence[str]],
    frames: Mapping[str, pd.DataFrame],
    csv_paths: Mapping[str, Path],
    image_index: Mapping[str, List[Path]],
    annotation_index: Mapping[str, List[Path]],
    project_root: Path,
    allow_missing_images: bool,
) -> Dict[str, dict]:
    report: Dict[str, dict] = {}

    for split_name in SPLIT_NAMES:
        ids = list(split_ids[split_name])
        id_set = set(ids)
        base = f"{prefix}/{split_name}" if prefix else split_name
        missing_images: List[str] = []
        image_count = 0
        annotation_file_count = 0

        for image_id in ids:
            image_path = choose_single_image(image_id, image_index)
            if image_path is None:
                missing_images.append(image_id)
            else:
                archive.write(image_path, f"{base}/images/{image_path.name}")
                image_count += 1

            for annotation_path in get_annotation_files(image_id, annotation_index):
                archive.write(
                    annotation_path,
                    f"{base}/annotations/{annotation_path.name}",
                )
                annotation_file_count += 1

        if missing_images and not allow_missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image files are missing in {base}. "
                f"Examples: {missing_images[:10]}. Re-run with "
                "--allow-missing-images only if this is intentional."
            )

        for key, frame in frames.items():
            filtered = frame[frame["image_id"].isin(id_set)].copy()
            filename = csv_paths[key].name
            archive.writestr(
                f"{base}/metadata/{filename}", dataframe_csv_bytes(filtered)
            )

        manifest = pd.DataFrame({"image_id": ids, "split": split_name})
        archive.writestr(
            f"{base}/split_manifest.csv", dataframe_csv_bytes(manifest)
        )

        report[split_name] = {
            "image_ids": len(ids),
            "image_files_packaged": image_count,
            "annotation_files_packaged": annotation_file_count,
            "missing_image_files": len(missing_images),
        }
    return report


def create_standard_zip(
    output_path: Path,
    strategy_name: str,
    split_ids: Mapping[str, Sequence[str]],
    group_assignment: Mapping[str, Sequence[str]] | None,
    frames: Mapping[str, pd.DataFrame],
    csv_paths: Mapping[str, Path],
    image_index: Mapping[str, List[Path]],
    annotation_index: Mapping[str, List[Path]],
    project_root: Path,
    allow_missing_images: bool,
) -> Dict[str, dict]:
    all_ids = set(frames["layer_3"]["image_id"])
    verify_split(split_ids, all_ids)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        report = write_split_contents(
            archive=archive,
            prefix="",
            split_ids=split_ids,
            frames=frames,
            csv_paths=csv_paths,
            image_index=image_index,
            annotation_index=annotation_index,
            project_root=project_root,
            allow_missing_images=allow_missing_images,
        )
        summary = {
            "strategy": strategy_name,
            "target_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "actual_counts": {name: len(split_ids[name]) for name in SPLIT_NAMES},
            "group_assignment": group_assignment,
            "file_report": report,
            "leakage_check": "passed",
        }
        archive.writestr("split_summary.json", json.dumps(summary, indent=2))
    return report


def create_loso_zip(
    output_path: Path,
    folds: Mapping[str, Mapping[str, Sequence[str]]],
    frames: Mapping[str, pd.DataFrame],
    csv_paths: Mapping[str, Path],
    image_index: Mapping[str, List[Path]],
    annotation_index: Mapping[str, List[Path]],
    project_root: Path,
    allow_missing_images: bool,
) -> Dict[str, dict]:
    all_ids = set(frames["layer_3"]["image_id"])
    full_report: Dict[str, dict] = {}

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        fold_summaries = {}
        for fold_name, split_ids in folds.items():
            verify_split(split_ids, all_ids)
            prefix = f"fold_{fold_name}"
            report = write_split_contents(
                archive=archive,
                prefix=prefix,
                split_ids=split_ids,
                frames=frames,
                csv_paths=csv_paths,
                image_index=image_index,
                annotation_index=annotation_index,
                project_root=project_root,
                allow_missing_images=allow_missing_images,
            )
            full_report[fold_name] = report
            fold_summaries[fold_name] = {
                "counts": {name: len(split_ids[name]) for name in SPLIT_NAMES},
                "file_report": report,
            }

        summary = {
            "strategy": "leave-one-state-out",
            "description": "Each fold uses one complete state as test data.",
            "folds": fold_summaries,
            "leakage_check": "passed independently for every fold",
        }
        archive.writestr("split_summary.json", json.dumps(summary, indent=2))
    return full_report


def print_counts(title: str, split_ids: Mapping[str, Sequence[str]], group_assignment=None) -> None:
    total = sum(len(split_ids[name]) for name in SPLIT_NAMES)
    print(f"\n{title}")
    print("-" * len(title))
    for name in SPLIT_NAMES:
        count = len(split_ids[name])
        pct = (100.0 * count / total) if total else 0.0
        group_text = ""
        if group_assignment is not None:
            groups = list(group_assignment[name])
            group_text = f" | groups={len(groups)}: {', '.join(groups)}"
        print(f"{name.capitalize():<11}: {count:>5} images ({pct:6.2f}%){group_text}")


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = args.project_root.resolve() if args.project_root else infer_project_root(script_path)
    output_dir = script_path.parent

    print("=" * 78)
    print("PHASE VII — CREATE AND PACKAGE DATASET SPLITS")
    print("=" * 78)
    print(f"Project root : {project_root}")
    print(f"Output folder: {output_dir}")
    print(f"Random seed  : {args.seed}")

    frames, csv_paths = load_metadata(project_root)
    master = validate_master_metadata(frames)
    all_ids = master["image_id"].tolist()

    image_folder = find_layer_folder(project_root, "layer_0_raw_images")
    annotation_folder = find_layer_folder(project_root, "layer_1_annotations")
    image_index = index_files(image_folder, IMAGE_EXTENSIONS)
    annotation_index = index_files(annotation_folder, ANNOTATION_EXTENSIONS)

    metadata_csv_resolved = csv_paths["layer_1"].resolve()
    for stem in list(annotation_index):
        annotation_index[stem] = [
            p for p in annotation_index[stem] if p.resolve() != metadata_csv_resolved
        ]
        if not annotation_index[stem]:
            del annotation_index[stem]

    found_images = sum(1 for image_id in all_ids if choose_single_image(image_id, image_index))
    found_annotation_sets = sum(1 for image_id in all_ids if get_annotation_files(image_id, annotation_index))
    print(f"Images indexed : {found_images}/{len(all_ids)}")
    print(f"Image IDs with standalone annotation files: {found_annotation_sets}/{len(all_ids)}")
    print("Layer 1 annotation metadata CSV will be included for every split.")

    random_split = exact_image_split(all_ids, args.seed)
    region_split, region_groups = grouped_split(master, "region", args.seed + 100)
    state_split, state_groups = grouped_split(master, "collection_state", args.seed + 200)
    folds = loso_splits(master, args.seed + 300)

    print_counts("Random image-wise split", random_split)
    print_counts("Region-wise split", region_split, region_groups)
    print_counts("State-wise split", state_split, state_groups)

    print("\nLeave-one-state-out folds")
    print("--------------------------")
    for fold_name, split_ids in folds.items():
        counts = ", ".join(f"{s}={len(split_ids[s])}" for s in SPLIT_NAMES)
        print(f"{fold_name:<25} {counts}")

    outputs = {key: output_dir / filename for key, filename in ZIP_NAMES.items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "The following output files already exist:\n"
            + "\n".join(f"  - {p}" for p in existing)
            + "\nUse --overwrite to replace them."
        )
    for path in existing:
        path.unlink()

    print("\nCreating ZIP archives...")
    create_standard_zip(
        outputs["random"], "random-image-wise-70-15-15", random_split, None,
        frames, csv_paths, image_index, annotation_index, project_root,
        args.allow_missing_images,
    )
    print(f"[1/4] Created: {outputs['random'].name}")

    create_standard_zip(
        outputs["region"], "region-wise", region_split, region_groups,
        frames, csv_paths, image_index, annotation_index, project_root,
        args.allow_missing_images,
    )
    print(f"[2/4] Created: {outputs['region'].name}")

    create_standard_zip(
        outputs["state"], "state-wise", state_split, state_groups,
        frames, csv_paths, image_index, annotation_index, project_root,
        args.allow_missing_images,
    )
    print(f"[3/4] Created: {outputs['state'].name}")

    create_loso_zip(
        outputs["loso"], folds, frames, csv_paths, image_index,
        annotation_index, project_root, args.allow_missing_images,
    )
    print(f"[4/4] Created: {outputs['loso'].name}")

    print("\nAll leakage checks passed.")
    print("All four ZIP files were saved in:")
    print(f"  {output_dir}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
