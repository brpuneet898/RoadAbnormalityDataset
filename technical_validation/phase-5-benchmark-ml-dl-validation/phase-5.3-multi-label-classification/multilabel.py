#!/usr/bin/env python3
"""
Shared Phase 5.3 multi-label image-classification pipeline.

Default labels are built from layer_1_annotation_metadata.csv by grouping all
unique `abnormality_type` values belonging to the same image_id. This is the
appropriate source when one image can contain several abnormalities.

Outputs:
    figures/<model_name>/*.png
    models/<model_name>_best.pt
    models/<model_name>_last.pt

Results are printed as terminal tables. No result CSV files are generated.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"
}


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_project_root(script_dir: Path) -> Path:
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "layer_0_raw_images").exists():
            return candidate
    return script_dir.parents[2] if len(script_dir.parents) >= 3 else script_dir


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            index[path.name.lower()] = path
            index[path.stem.lower()] = path
    if not index:
        raise FileNotFoundError(f"No supported images found under {image_dir}")
    return index


def resolve_image_path(value: str, index: Dict[str, Path]) -> Path | None:
    key = str(value).strip().lower()
    return index.get(key) or index.get(Path(key).stem)


def load_multilabel_dataframe(
    annotation_csv: Path,
    image_dir: Path,
    label_column: str,
    separator: str | None,
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(annotation_csv)
    required = {"image_id", label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {annotation_csv}: {sorted(missing)}")

    df = df[["image_id", label_column]].dropna().copy()
    df[label_column] = df[label_column].astype(str).str.strip()
    df = df[df[label_column] != ""]

    if separator:
        df[label_column] = df[label_column].str.split(separator)
        df = df.explode(label_column)
        df[label_column] = df[label_column].astype(str).str.strip()
        df = df[df[label_column] != ""]

    grouped = (
        df.groupby("image_id")[label_column]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="labels")
    )

    image_index = build_image_index(image_dir)
    grouped["image_path"] = grouped["image_id"].map(
        lambda value: resolve_image_path(value, image_index)
    )
    missing_images = int(grouped["image_path"].isna().sum())
    if missing_images:
        print(f"Warning: skipping {missing_images} image IDs with no matching raw image.")
    grouped = grouped.dropna(subset=["image_path"]).reset_index(drop=True)

    class_names = sorted({label for labels in grouped["labels"] for label in labels})
    if len(class_names) < 2:
        raise RuntimeError("Multi-label classification requires at least two label classes.")

    return grouped, class_names


def iterative_like_split(
    df: pd.DataFrame,
    class_names: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uses multilabel-stratified splitting when iterative-stratification is
    installed, otherwise falls back to deterministic random splitting.
    """
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be > 0 and sum to < 1.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.zeros((len(df), len(class_names)), dtype=np.int64)
    for row_idx, labels in enumerate(df["labels"]):
        for label in labels:
            y[row_idx, class_to_idx[label]] = 1

    indices = np.arange(len(df))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        first = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction + test_fraction,
            random_state=seed,
        )
        train_idx, temp_idx = next(first.split(indices, y))

        relative_test = test_fraction / (val_fraction + test_fraction)
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_test,
            random_state=seed,
        )
        val_rel, test_rel = next(second.split(temp_idx, y[temp_idx]))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
    except ImportError:
        print(
            "Warning: iterative-stratification is unavailable; "
            "using deterministic random splitting."
        )
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_fraction + test_fraction,
            random_state=seed,
        )
        relative_test = test_fraction / (val_fraction + test_fraction)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=seed,
        )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform,
    ) -> None:
        self.df = dataframe
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = Path(row["image_path"])
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in row["labels"]:
            target[self.class_to_idx[label]] = 1.0
        return image, target


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def calculate_pos_weight(
    dataframe: pd.DataFrame,
    class_names: Sequence[str],
) -> torch.Tensor:
    positives = np.zeros(len(class_names), dtype=np.float64)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for labels in dataframe["labels"]:
        for label in labels:
            positives[class_to_idx[label]] += 1
    negatives = len(dataframe) - positives
    return torch.tensor(
        negatives / np.maximum(positives, 1.0),
        dtype=torch.float32,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_exact = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_exact += (predictions == targets).all(dim=1).sum().item()
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return total_loss / max(total_samples, 1), total_exact / max(total_samples, 1)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    targets_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(images))
        targets_all.append(targets.numpy())
        probabilities_all.append(probabilities.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return np.concatenate(targets_all), np.concatenate(probabilities_all), elapsed


def tune_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.10, 0.91, 0.02)
    rows = []
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(targets, predictions, average="micro", zero_division=0)
        rows.append({"threshold": threshold, "micro_f1": score})
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows)


def save_threshold_curve(curve: pd.DataFrame, best_threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["threshold"], curve["micro_f1"])
    ax.axvline(best_threshold, linestyle="--", label=f"Best = {best_threshold:.2f}")
    ax.set(
        title="Validation threshold tuning",
        xlabel="Decision threshold",
        ylabel="Micro F1",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_exact"], label="Train")
    axes[1].plot(epochs, history["val_exact"], label="Validation")
    axes[1].set(title="Exact-match accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 8))
    auc_by_class: Dict[str, float] = {}

    for idx, name in enumerate(class_names):
        if len(np.unique(targets[:, idx])) < 2:
            auc_by_class[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(targets[:, idx], probabilities[:, idx])
        class_auc = auc(fpr, tpr)
        auc_by_class[name] = class_auc
        ax.plot(fpr, tpr, label=f"{name} ({class_auc:.3f})")

    if len(np.unique(targets.ravel())) >= 2:
        fpr, tpr, _ = roc_curve(targets.ravel(), probabilities.ravel())
        micro_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linestyle="--", linewidth=2.5, label=f"Micro ({micro_auc:.3f})")
        auc_by_class["micro-average"] = micro_auc

    ax.plot([0, 1], [0, 1], linestyle=":")
    ax.set(
        title="Multi-label ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.3)
    if len(class_names) <= 15:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc_by_class


def save_pr_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 8))
    ap_by_class: Dict[str, float] = {}

    for idx, name in enumerate(class_names):
        if targets[:, idx].sum() == 0:
            ap_by_class[name] = float("nan")
            continue
        precision, recall, _ = precision_recall_curve(
            targets[:, idx], probabilities[:, idx]
        )
        ap = average_precision_score(targets[:, idx], probabilities[:, idx])
        ap_by_class[name] = ap
        ax.plot(recall, precision, label=f"{name} ({ap:.3f})")

    precision, recall, _ = precision_recall_curve(
        targets.ravel(), probabilities.ravel()
    )
    micro_ap = average_precision_score(
        targets, probabilities, average="micro"
    )
    ax.plot(
        recall,
        precision,
        linestyle="--",
        linewidth=2.5,
        label=f"Micro ({micro_ap:.3f})",
    )
    ap_by_class["micro-average"] = micro_ap

    ax.set(
        title="Multi-label precision-recall curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.3)
    if len(class_names) <= 15:
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ap_by_class


def save_multilabel_confusions(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    matrices = multilabel_confusion_matrix(targets, predictions)
    columns = min(4, len(class_names))
    rows = math.ceil(len(class_names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, matrix) in enumerate(zip(class_names, matrices)):
        ax = axes[idx]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")

    for idx in range(len(class_names), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-label confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    average: str,
) -> float:
    valid = np.array(
        [len(np.unique(targets[:, idx])) == 2 for idx in range(targets.shape[1])]
    )
    if not valid.any():
        return float("nan")
    return roc_auc_score(
        targets[:, valid],
        probabilities[:, valid],
        average=average,
    )


def evaluate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        targets, predictions, average="samples", zero_division=0
    )

    metrics = {
        "Exact-match accuracy": accuracy_score(targets, predictions),
        "Hamming loss": hamming_loss(targets, predictions),
        "Precision micro": precision_micro,
        "Recall micro": recall_micro,
        "F1 micro": f1_micro,
        "Precision macro": precision_macro,
        "Recall macro": recall_macro,
        "F1 macro": f1_macro,
        "Precision samples": precision_samples,
        "Recall samples": recall_samples,
        "F1 samples": f1_samples,
        "ROC-AUC micro": safe_auc(targets, probabilities, "micro"),
        "ROC-AUC macro": safe_auc(targets, probabilities, "macro"),
        "Average precision micro": average_precision_score(
            targets, probabilities, average="micro"
        ),
        "Mean average precision": average_precision_score(
            targets, probabilities, average="macro"
        ),
        "Decision threshold": threshold,
    }
    return metrics, predictions


def print_results(
    model_name: str,
    metrics: Dict[str, float],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    inference_seconds: float,
) -> None:
    metrics = dict(metrics)
    metrics["Inference images/second"] = len(targets) / max(inference_seconds, 1e-12)
    metrics["Inference ms/image"] = (
        1000.0 * inference_seconds / max(len(targets), 1)
    )

    rows = []
    for key, value in metrics.items():
        rows.append([key, "N/A" if math.isnan(value) else f"{value:.6f}"])
    print(f"\nFinal test results — {model_name}")
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    per_class_ap = average_precision_score(
        targets,
        probabilities,
        average=None,
    )
    per_class_rows = []
    for idx, name in enumerate(class_names):
        positives = int(targets[:, idx].sum())
        per_class_rows.append(
            [
                name,
                f"{precision[idx]:.4f}",
                f"{recall[idx]:.4f}",
                f"{f1[idx]:.4f}",
                positives,
                f"{per_class_ap[idx]:.4f}",
            ]
        )
    print("\nPer-label results")
    print(
        tabulate(
            per_class_rows,
            headers=["Label", "Precision", "Recall", "F1", "Positive support", "AP"],
            tablefmt="rounded_grid",
        )
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    class_names: Sequence[str],
    image_size: int,
    epoch: int,
    best_val_loss: float,
    threshold: float,
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "class_names": list(class_names),
            "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
            "image_size": image_size,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
            "task": "multi_label_classification",
        },
        path,
    )


def build_parser(default_model: str) -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_root = discover_project_root(script_dir)

    parser = argparse.ArgumentParser(
        description=f"Train {default_model} for multi-label abnormality classification."
    )
    parser.add_argument(
        "--model-name",
        default=default_model,
        choices=["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"],
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--label-column", default="abnormality_type")
    parser.add_argument(
        "--label-separator",
        default=None,
        help="Optional separator when one CSV cell itself contains multiple labels.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use this threshold instead of tuning it on validation predictions.",
    )
    return parser


def main(default_model: str) -> None:
    args = build_parser(default_model).parse_args()
    register_heif()
    seed_everything(args.seed)

    project_root = args.project_root.resolve()
    image_dir = (args.image_dir or project_root / "layer_0_raw_images").resolve()
    annotation_csv = (
        args.annotation_csv
        or project_root / "layer_1_annotation_metadata.csv"
    ).resolve()

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures" / args.model_name
    models_dir = script_dir / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Images: {image_dir}")
    print(f"Annotations: {annotation_csv}")

    dataframe, class_names = load_multilabel_dataframe(
        annotation_csv,
        image_dir,
        args.label_column,
        args.label_separator,
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_df, val_df, test_df = iterative_like_split(
        dataframe,
        class_names,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    split_rows = [
        ["Train", len(train_df)],
        ["Validation", len(val_df)],
        ["Test", len(test_df)],
    ]
    print("\nDataset split")
    print(tabulate(split_rows, headers=["Split", "Images"], tablefmt="rounded_grid"))

    distribution_rows = []
    for label in class_names:
        distribution_rows.append(
            [
                label,
                sum(label in labels for labels in dataframe["labels"]),
            ]
        )
    print("\nLabel distribution")
    print(
        tabulate(
            distribution_rows,
            headers=["Label", "Images containing label"],
            tablefmt="rounded_grid",
        )
    )

    train_transform, eval_transform = make_transforms(args.image_size)
    train_dataset = MultiLabelImageDataset(train_df, class_to_idx, train_transform)
    val_dataset = MultiLabelImageDataset(val_df, class_to_idx, eval_transform)
    test_dataset = MultiLabelImageDataset(test_df, class_to_idx, eval_transform)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = create_model(
        args.model_name,
        len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = calculate_pos_weight(train_df, class_names).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_path = models_dir / f"{args.model_name}_best.pt"
    last_path = models_dir / f"{args.model_name}_last.pt"
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_exact": [],
        "val_exact": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_exact = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_exact = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_exact"].append(train_exact)
        history["val_exact"].append(val_exact)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | train exact {train_exact:.4f} | "
            f"val loss {val_loss:.4f} | val exact {val_exact:.4f}"
        )

        save_checkpoint(
            last_path,
            model,
            args.model_name,
            class_names,
            args.image_size,
            epoch,
            best_val_loss,
            0.5,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                args.model_name,
                class_names,
                args.image_size,
                epoch,
                best_val_loss,
                0.5,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_training_curves(history, figures_dir / "training_curves.png")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_targets, val_probabilities, _ = predict(model, val_loader, device)
    if args.fixed_threshold is None:
        threshold, threshold_curve = tune_threshold(
            val_targets, val_probabilities
        )
        save_threshold_curve(
            threshold_curve,
            threshold,
            figures_dir / "threshold_tuning.png",
        )
    else:
        threshold = args.fixed_threshold

    test_targets, test_probabilities, inference_seconds = predict(
        model, test_loader, device
    )
    metrics, test_predictions = evaluate_metrics(
        test_targets,
        test_probabilities,
        threshold,
    )

    save_roc_curves(
        test_targets,
        test_probabilities,
        class_names,
        figures_dir / "roc_curves.png",
    )
    save_pr_curves(
        test_targets,
        test_probabilities,
        class_names,
        figures_dir / "precision_recall_curves.png",
    )
    save_multilabel_confusions(
        test_targets,
        test_predictions,
        class_names,
        figures_dir / "per_label_confusion_matrices.png",
    )

    save_checkpoint(
        best_path,
        model,
        args.model_name,
        class_names,
        args.image_size,
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        threshold,
    )

    print_results(
        args.model_name,
        metrics,
        test_targets,
        test_predictions,
        test_probabilities,
        class_names,
        inference_seconds,
    )
    print(f"\nBest model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "Run train_resnet50_multilabel.py, train_efficientnet_multilabel.py, "
        "train_convnext_multilabel.py, or train_vit_multilabel.py"
    )
