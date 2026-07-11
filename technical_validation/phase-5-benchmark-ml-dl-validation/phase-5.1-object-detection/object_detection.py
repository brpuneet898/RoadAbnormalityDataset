from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args(model_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate {model_name} for Phase 5.1 object detection."
    )
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Repository root. Auto-detected from this script when omitted.")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Raw image directory. Default: <root>/layer_0_raw_images")
    parser.add_argument("--annotations-csv", type=Path, default=None,
                        help="Layer-1 annotation metadata CSV.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="Examples: 0, 0,1, cpu, mps. Ultralytics auto-selects when omitted.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--pretrained", default=None,
                        help="Override the default pretrained checkpoint.")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep the generated YOLO dataset under phase-5.1-object-detection/work.")
    return parser.parse_args()


def find_project_root(script_file: str, explicit_root: Path | None) -> Path:
    if explicit_root:
        return explicit_root.expanduser().resolve()
    here = Path(script_file).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "layer_0_raw_images").exists() and (candidate / "layer_1_annotations").exists():
            return candidate
    return here.parents[1] if len(here.parents) > 1 else here


def locate_csv(root: Path, explicit: Path | None, filename: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit.expanduser())
    candidates.extend([
        root / filename,
        root / "technical-validation" / filename,
        root / "metadata" / filename,
    ])
    candidates.extend(root.glob(f"**/{filename}"))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Pass its path with --annotations-csv."
    )


def image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory does not exist: {images_dir}")
    index: dict[str, Path] = {}
    collisions: list[str] = []
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            key = path.stem.casefold()
            if key in index:
                collisions.append(path.stem)
            else:
                index[key] = path
    if collisions:
        raise ValueError(
            "Duplicate image stems found; image_id must uniquely identify a file. "
            f"Examples: {collisions[:5]}"
        )
    if not index:
        raise RuntimeError(f"No supported images found under {images_dir}")
    return index


def validate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_id", "abnormality_type", "x_centre", "y_centre",
        "width_scaled", "height_scaled"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    clean = df.copy()
    clean["image_id"] = clean["image_id"].astype(str).str.strip()
    clean["abnormality_type"] = clean["abnormality_type"].astype(str).str.strip()
    coords = ["x_centre", "y_centre", "width_scaled", "height_scaled"]
    clean[coords] = clean[coords].apply(pd.to_numeric, errors="coerce")
    if clean[coords].isna().any().any():
        bad = clean[clean[coords].isna().any(axis=1)].head()
        raise ValueError(f"Non-numeric or missing bounding-box values found:\n{bad}")
    outside = ((clean[coords] < 0) | (clean[coords] > 1)).any(axis=1)
    nonpositive = (clean[["width_scaled", "height_scaled"]] <= 0).any(axis=1)
    if outside.any() or nonpositive.any():
        bad = clean[outside | nonpositive].head()
        raise ValueError(f"YOLO coordinates must be normalized to [0,1] with positive sizes:\n{bad}")
    return clean


def multilabel_split(
    image_ids: list[str], labels_by_image: dict[str, set[str]],
    train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[str]]:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train-ratio and val-ratio must be > 0 and sum to less than 1.")
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    label_freq: dict[str, int] = {}
    for image_id in shuffled:
        for label in labels_by_image.get(image_id, set()):
            label_freq[label] = label_freq.get(label, 0) + 1
    shuffled.sort(key=lambda x: sum(1 / label_freq[l] for l in labels_by_image.get(x, set())), reverse=True)

    n = len(shuffled)
    targets = {
        "train": round(n * train_ratio),
        "val": round(n * val_ratio),
    }
    targets["test"] = n - targets["train"] - targets["val"]
    splits = {k: [] for k in targets}
    class_counts = {k: {} for k in targets}

    for image_id in shuffled:
        labels = labels_by_image.get(image_id, set())
        choices = [k for k in targets if len(splits[k]) < targets[k]]
        def cost(split: str) -> tuple[float, float]:
            fill = len(splits[split]) / max(targets[split], 1)
            imbalance = sum(class_counts[split].get(label, 0) / max(label_freq[label], 1) for label in labels)
            return (imbalance, fill)
        chosen = min(choices, key=cost)
        splits[chosen].append(image_id)
        for label in labels:
            class_counts[chosen][label] = class_counts[chosen].get(label, 0) + 1
    return splits


def materialize_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in RASTER_EXTENSIONS:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)
        return
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError as exc:
        raise ImportError(
            "HEIC/HEIF images were found. Install Pillow and pillow-heif: "
            "pip install Pillow pillow-heif"
        ) from exc
    with Image.open(source) as image:
        image.convert("RGB").save(destination.with_suffix(".jpg"), quality=95)


def write_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    # JSON syntax is valid YAML and avoids a separate PyYAML dependency.
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_dataset(root: Path, images_dir: Path, annotations_csv: Path,
                    work_dir: Path, train_ratio: float, val_ratio: float,
                    seed: int) -> tuple[Path, list[str], dict[str, int]]:
    annotations = validate_annotations(pd.read_csv(annotations_csv))
    images = image_index(images_dir)
    annotated_ids = set(annotations["image_id"])
    missing_images = sorted(i for i in annotated_ids if i.casefold() not in images)
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} annotated image IDs have no matching image file. "
            f"Examples: {missing_images[:10]}"
        )

    class_names = sorted(annotations["abnormality_type"].unique().tolist(), key=str.casefold)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    labels_by_image = annotations.groupby("image_id")["abnormality_type"].apply(set).to_dict()
    all_image_ids = sorted(annotated_ids)
    splits = multilabel_split(all_image_ids, labels_by_image, train_ratio, val_ratio, seed)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    for split, ids in splits.items():
        (work_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (work_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            source = images[image_id.casefold()]
            out_ext = ".jpg" if source.suffix.lower() in {".heic", ".heif"} else source.suffix.lower()
            destination = work_dir / "images" / split / f"{image_id}{out_ext}"
            materialize_image(source, destination)
            rows = annotations[annotations["image_id"] == image_id]
            label_lines = [
                f"{class_to_id[row.abnormality_type]} {row.x_centre:.8f} {row.y_centre:.8f} "
                f"{row.width_scaled:.8f} {row.height_scaled:.8f}"
                for row in rows.itertuples(index=False)
            ]
            (work_dir / "labels" / split / f"{image_id}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )
    yaml_path = work_dir / "data.yaml"
    write_yaml(yaml_path, work_dir, class_names)
    counts = {split: len(ids) for split, ids in splits.items()}
    return yaml_path, class_names, counts


def scalar(value) -> float:
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def print_table(headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    rows_s = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows_s:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    rule = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(rule)
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(rule)
    for row in rows_s:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(rule)


def delete_csv_outputs(folder: Path) -> None:
    for csv_path in folder.rglob("*.csv"):
        csv_path.unlink(missing_ok=True)


def train_and_evaluate(script_file: str, model_label: str, default_checkpoint: str) -> None:
    args = parse_args(model_label)
    root = find_project_root(script_file, args.project_root)
    phase_dir = Path(script_file).resolve().parent
    images_dir = (args.images_dir or (root / "layer_0_raw_images")).expanduser().resolve()
    annotations_csv = locate_csv(root, args.annotations_csv, "layer_1_annotation_metadata.csv")
    figures_dir = phase_dir / "figures" / model_label
    models_dir = phase_dir / "models"
    work_dir = phase_dir / "work" / model_label
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {root}")
    print(f"Images       : {images_dir}")
    print(f"Annotations  : {annotations_csv}")
    yaml_path, class_names, split_counts = prepare_dataset(
        root, images_dir, annotations_csv, work_dir,
        args.train_ratio, args.val_ratio, args.seed
    )
    print_table(["Split", "Images"], [[k, v] for k, v in split_counts.items()])
    print_table(["Class ID", "Class name"], [[i, n] for i, n in enumerate(class_names)])

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install Ultralytics first: pip install ultralytics") from exc

    checkpoint = args.pretrained or default_checkpoint
    model = YOLO(checkpoint)
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=str(phase_dir / "figures"),
        name=model_label,
        exist_ok=True,
        plots=True,
        cache=args.cache,
        verbose=True,

        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,

        degrees=10.0,
        translate=0.10,
        scale=0.30,
        shear=2.0,
        perspective=0.0002,

        fliplr=0.50,
        flipud=0.00,

        mosaic=0.80,
        mixup=0.05,
        cutmix=0.00,
        close_mosaic=10,
        
        erasing=0.10,
        bgr=0.00,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    run_dir = figures_dir
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    if not best_source.exists():
        raise FileNotFoundError(f"Training completed but best.pt was not found at {best_source}")
    best_target = models_dir / f"{model_label}_best.pt"
    last_target = models_dir / f"{model_label}_last.pt"
    shutil.copy2(best_source, best_target)
    if last_source.exists():
        shutil.copy2(last_source, last_target)

    best_model = YOLO(str(best_target))
    val_kwargs = dict(
        data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, plots=True, project=str(phase_dir / "figures"),
        name=model_label, exist_ok=True, verbose=False,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = best_model.val(**val_kwargs)

    precision = scalar(metrics.box.mp)
    recall = scalar(metrics.box.mr)
    map50 = scalar(metrics.box.map50)
    map5095 = scalar(metrics.box.map)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    inference_ms = float(metrics.speed.get("inference", math.nan))
    fps = 1000.0 / inference_ms if inference_ms > 0 else math.nan

    print("\nFINAL TEST METRICS")
    print_table(
        ["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "FPS", "Inference ms/image"],
        [[model_label, f"{map50:.4f}", f"{map5095:.4f}", f"{precision:.4f}",
          f"{recall:.4f}", f"{f1:.4f}", f"{fps:.2f}", f"{inference_ms:.3f}"]]
    )

    maps = list(metrics.box.maps)
    print("\nPER-CLASS AP (mAP@0.5:0.95)")
    print_table(
        ["Class ID", "Class", "AP"],
        [[i, class_names[i], f"{scalar(maps[i]):.4f}"] for i in range(min(len(class_names), len(maps)))]
    )

    delete_csv_outputs(figures_dir)
    if (figures_dir / "weights").exists():
        shutil.rmtree(figures_dir / "weights")
    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nFigures saved to: {figures_dir}")
    print(f"Best model saved to: {best_target}")
    if last_target.exists():
        print(f"Last model saved to: {last_target}")

