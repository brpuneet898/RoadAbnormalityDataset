from __future__ import annotations

import argparse
import json
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
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
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
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp",
    ".bmp", ".tif", ".tiff",
}

ABNORMALITY_CLASSES = [
    "Surface depression",
    "Manhole",
    "Pothole",
    "Road patch failure",
    "Miscellaneous abnormality",
    "Crack",
]
ABNORMALITY_ALIASES = {name.casefold(): name for name in ABNORMALITY_CLASSES}


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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def discover_project_root(script_dir: Path) -> Path:
    candidates = [script_dir, *script_dir.parents]
    for candidate in candidates:
        if (candidate / "layer_0_raw_images").exists():
            return candidate
    return script_dir.parents[2] if len(script_dir.parents) >= 3 else script_dir


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            index[path.stem.lower()] = path
            index[path.name.lower()] = path
    if not index:
        raise FileNotFoundError(f"No supported images found under: {image_dir}")
    return index


def resolve_image_path(image_id: str, index: Dict[str, Path]) -> Path | None:
    key = str(image_id).strip().lower()
    if key in index:
        return index[key]
    return index.get(Path(key).stem)


def parse_abnormality_labels(value: str) -> List[str]:
    """Parse one CSV cell into unique canonical abnormality labels."""
    labels: List[str] = []
    for raw_label in str(value).split(";"):
        cleaned = " ".join(raw_label.strip().split())
        if not cleaned:
            continue
        canonical = ABNORMALITY_ALIASES.get(cleaned.casefold())
        if canonical is None:
            print(f"Warning: ignoring unknown abnormality label: {cleaned!r}")
            continue
        if canonical not in labels:
            labels.append(canonical)
    return labels


def load_dataframe(metadata_csv: Path, image_dir: Path, label_column: str) -> pd.DataFrame:
    if label_column != "abnormality_type":
        raise ValueError(
            "This revised file performs six-class multi-label classification and "
            "therefore requires --label-column abnormality_type."
        )

    df = pd.read_csv(metadata_csv)
    required = {"image_id", label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metadata_csv}: {sorted(missing)}")

    df = df[["image_id", label_column]].dropna().copy()
    df[label_column] = df[label_column].astype(str).str.strip()
    df = df[df[label_column] != ""]

    index = build_image_index(image_dir)
    df["image_path"] = df["image_id"].map(lambda value: resolve_image_path(value, index))
    missing_paths = int(df["image_path"].isna().sum())
    if missing_paths:
        print(f"Warning: skipping {missing_paths} metadata rows with no matching image.")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    df["parsed_labels"] = df[label_column].map(parse_abnormality_labels)
    df = df[df["parsed_labels"].map(len) > 0].copy()
    if df.empty:
        raise RuntimeError("No usable image/label pairs were found.")

    merged_rows = []
    for image_key, group in df.groupby(df["image_path"].astype(str), sort=False):
        combined = []
        for labels in group["parsed_labels"]:
            for label in labels:
                if label not in combined:
                    combined.append(label)
        first = group.iloc[0]
        merged_rows.append(
            {
                "image_id": first["image_id"],
                "image_path": first["image_path"],
                label_column: "; ".join(combined),
                "parsed_labels": combined,
            }
        )

    df = pd.DataFrame(merged_rows)
    for class_name in ABNORMALITY_CLASSES:
        df[class_name] = df["parsed_labels"].map(
            lambda labels, name=class_name: int(name in labels)
        )

    if len(df) < 3:
        raise RuntimeError("At least three usable images are required.")
    return df.reset_index(drop=True)


def _split_score(
    y: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    target_right_fraction: float,
) -> float:
    """Lower is better: compare label totals and prevalence in both subsets."""
    eps = 1e-8
    total_counts = y.sum(axis=0)
    expected_right = total_counts * target_right_fraction
    right_counts = y[right_indices].sum(axis=0)
    count_error = np.mean(np.abs(right_counts - expected_right) / np.maximum(total_counts, 1))

    overall_prev = y.mean(axis=0)
    left_prev = y[left_indices].mean(axis=0)
    right_prev = y[right_indices].mean(axis=0)
    prevalence_error = np.mean(np.abs(left_prev - overall_prev) + np.abs(right_prev - overall_prev))

    left_counts = y[left_indices].sum(axis=0)
    missing_penalty = 0.0
    for total, left, right in zip(total_counts, left_counts, right_counts):
        if total >= 2 and (left == 0 or right == 0):
            missing_penalty += 10.0

    size_error = abs(len(right_indices) / len(y) - target_right_fraction)
    return float(count_error + prevalence_error + size_error + missing_penalty + eps)


def multilabel_train_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    seed: int,
    candidates: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dependency-free approximate multi-label stratified split.

    It tests multiple deterministic shuffled splits and keeps the one that best
    preserves each label's prevalence. One original image remains in one split.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    best_pair = None
    best_score = float("inf")
    local_positions = np.arange(len(indices))

    for attempt in range(candidates):
        left_pos, right_pos = train_test_split(
            local_positions,
            test_size=test_fraction,
            random_state=seed + attempt,
            shuffle=True,
        )
        score = _split_score(y, left_pos, right_pos, test_fraction)
        if score < best_score:
            best_score = score
            best_pair = (indices[left_pos], indices[right_pos])

    assert best_pair is not None
    return best_pair


def split_dataframe(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("--val-fraction and --test-fraction must be > 0 and sum to < 1.")

    all_indices = np.arange(len(df))
    y = df[ABNORMALITY_CLASSES].to_numpy(dtype=np.int64)
    temp_fraction = val_fraction + test_fraction

    train_indices, temp_indices = multilabel_train_test_split(
        all_indices, y, temp_fraction, seed
    )
    relative_test = test_fraction / temp_fraction
    temp_y = y[temp_indices]
    val_indices, test_indices = multilabel_train_test_split(
        temp_indices, temp_y, relative_test, seed + 10_000
    )

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[val_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )


class SemanticImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_names: Sequence[str],
        transform,
    ) -> None:
        self.df = dataframe
        self.class_names = list(class_names)
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
            raise RuntimeError(f"Failed to read image '{path}': {exc}") from exc

        target = row[self.class_names].to_numpy(dtype=np.float32)
        return image, torch.from_numpy(target)


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
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


def compute_positive_weights(
    train_df: pd.DataFrame, class_names: Sequence[str]
) -> torch.Tensor:
    targets = train_df[list(class_names)].to_numpy(dtype=np.float64)
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    exact_matches = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = (torch.sigmoid(logits) >= threshold).to(labels.dtype)
        running_loss += loss.item() * labels.size(0)
        exact_matches += predictions.eq(labels).all(dim=1).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), exact_matches / max(total, 1)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        labels_all.append(labels.numpy().astype(np.int64))
        probabilities_all.append(probabilities.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return np.concatenate(labels_all), np.concatenate(probabilities_all), elapsed


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Exact-match accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_multilabel_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    rows = math.ceil(len(class_names) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for idx, (name, matrix) in enumerate(zip(class_names, matrices)):
        ax = axes[idx]
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["Negative", "Positive"])
        ax.set_yticks([0, 1], labels=["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for r in range(2):
            for c in range(2):
                value = int(matrix[r, c])
                ax.text(c, r, str(value), ha="center", va="center",
                        color="white" if value > threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(class_names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 8))
    auc_per_class: Dict[str, float] = {}

    for idx, name in enumerate(class_names):
        if len(np.unique(y_true[:, idx])) < 2:
            auc_per_class[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_true[:, idx], probabilities[:, idx])
        class_auc = auc(fpr, tpr)
        auc_per_class[name] = class_auc
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={class_auc:.3f})")

    if len(np.unique(y_true.ravel())) >= 2:
        micro_fpr, micro_tpr, _ = roc_curve(y_true.ravel(), probabilities.ravel())
        micro_auc = auc(micro_fpr, micro_tpr)
        auc_per_class["micro-average"] = micro_auc
        ax.plot(micro_fpr, micro_tpr, linestyle="--", linewidth=2.5,
                label=f"Micro-average (AUC={micro_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set(title="Multi-label ROC curves", xlabel="False positive rate",
           ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc_per_class


def print_dataset_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
) -> None:
    rows = []
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        rows.append([split_name, len(split_df), int(split_df[list(class_names)].sum().sum())])
    print("\nDataset split")
    print(tabulate(rows, headers=["Split", "Images", "Positive labels"], tablefmt="rounded_grid"))

    distribution_rows = []
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for name in class_names:
        distribution_rows.append([
            name,
            int(full_df[name].sum()),
            int(train_df[name].sum()),
            int(val_df[name].sum()),
            int(test_df[name].sum()),
        ])
    print("\nClass distribution (an image may contribute to multiple classes)")
    print(tabulate(
        distribution_rows,
        headers=["Class", "Total images", "Train", "Validation", "Test"],
        tablefmt="rounded_grid",
    ))


def print_results(
    model_name: str,
    metrics: Dict[str, float],
    report: Dict[str, Dict[str, float]],
    auc_per_class: Dict[str, float],
    class_names: Sequence[str],
) -> None:
    print(f"\nFinal test results — {model_name}")
    print(tabulate(
        [[name, f"{value:.6f}"] for name, value in metrics.items()],
        headers=["Metric", "Value"], tablefmt="rounded_grid"
    ))

    class_rows = []
    for class_name in class_names:
        values = report[class_name]
        class_auc = auc_per_class.get(class_name, float("nan"))
        class_rows.append([
            class_name,
            f"{values['precision']:.4f}",
            f"{values['recall']:.4f}",
            f"{values['f1-score']:.4f}",
            int(values["support"]),
            f"{class_auc:.4f}" if not math.isnan(class_auc) else "N/A",
        ])
    print("\nPer-class results")
    print(tabulate(
        class_rows,
        headers=["Class", "Precision", "Recall", "F1", "Support", "AUC"],
        tablefmt="rounded_grid",
    ))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    class_names: Sequence[str],
    label_column: str,
    image_size: int,
    epoch: int,
    best_val_loss: float,
    threshold: float,
) -> None:
    torch.save(
        {
            "task_type": "multilabel",
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "class_names": list(class_names),
            "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
            "label_column": label_column,
            "image_size": image_size,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
        },
        path,
    )


def build_parser(default_model: str) -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_root = discover_project_root(script_dir)
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate {default_model} for six-class multi-label abnormality classification."
    )
    parser.add_argument("--model-name", default=default_model, choices=[
        "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"
    ])
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--label-column", default="abnormality_type")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser


def main(default_model: str) -> None:
    args = build_parser(default_model).parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    register_heif()
    seed_everything(args.seed)

    project_root = args.project_root.resolve()
    image_dir = (args.image_dir or project_root / "layer_0_raw_images").resolve()
    metadata_csv = (args.metadata_csv or project_root / "layer_2_semantic_metadata.csv").resolve()

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

    print(f"Using device: {device}")
    print(f"Images: {image_dir}")
    print(f"Semantic metadata: {metadata_csv}")
    print("Task: six-class multi-label abnormality classification")

    df = load_dataframe(metadata_csv, image_dir, args.label_column)
    class_names = ABNORMALITY_CLASSES
    train_df, val_df, test_df = split_dataframe(
        df, args.val_fraction, args.test_fraction, args.seed
    )
    print_dataset_summary(train_df, val_df, test_df, class_names)

    train_df.to_csv(figures_dir / "train_split.csv", index=False)
    val_df.to_csv(figures_dir / "validation_split.csv", index=False)
    test_df.to_csv(figures_dir / "test_split.csv", index=False)

    train_transform, eval_transform = make_transforms(args.image_size)
    train_dataset = SemanticImageDataset(train_df, class_names, train_transform)
    val_dataset = SemanticImageDataset(val_df, class_names, eval_transform)
    test_dataset = SemanticImageDataset(test_df, class_names, eval_transform)

    pin_memory = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = create_model(
        args.model_name, len(class_names), pretrained=not args.no_pretrained
    ).to(device)

    pos_weight = None
    if not args.no_class_weights:
        pos_weight = compute_positive_weights(train_df, class_names).to(device)
        print("Positive-class weights:")
        print(json.dumps(
            {name: round(float(weight), 4) for name, weight in zip(class_names, pos_weight.cpu())},
            indent=2,
        ))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = models_dir / f"{args.model_name}_best.pt"
    last_path = models_dir / f"{args.model_name}_last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, args.threshold, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, args.threshold
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | train exact acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | val exact acc {val_accuracy:.4f}"
        )

        save_checkpoint(
            last_path, model, args.model_name, class_names, args.label_column,
            args.image_size, epoch, best_val_loss, args.threshold
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path, model, args.model_name, class_names, args.label_column,
                args.image_size, epoch, best_val_loss, args.threshold
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_training_curves(history, figures_dir / "training_curves.png")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, probabilities, inference_seconds = predict(model, test_loader, device)
    y_pred = (probabilities >= args.threshold).astype(np.int64)

    exact_accuracy = accuracy_score(y_true, y_pred)
    hamming_accuracy = 1.0 - hamming_loss(y_true, y_pred)
    metrics = {
        "Exact-match accuracy": exact_accuracy,
        "Hamming accuracy": hamming_accuracy,
        "Precision macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "F1 micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    valid_auc_columns = [i for i in range(len(class_names)) if len(np.unique(y_true[:, i])) >= 2]
    if valid_auc_columns:
        metrics["AUC macro"] = roc_auc_score(
            y_true[:, valid_auc_columns], probabilities[:, valid_auc_columns], average="macro"
        )

    images_per_second = len(y_true) / max(inference_seconds, 1e-12)
    metrics["Inference images/second"] = images_per_second
    metrics["Inference ms/image"] = 1000.0 * inference_seconds / max(len(y_true), 1)

    auc_per_class = save_roc_curves(
        y_true, probabilities, class_names, figures_dir / "roc_curves.png"
    )
    save_multilabel_confusion_matrices(
        y_true, y_pred, class_names, figures_dir / "confusion_matrices.png"
    )

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print_results(args.model_name, metrics, report, auc_per_class, class_names)

    predictions_df = test_df[["image_id", "image_path"]].copy()
    for idx, name in enumerate(class_names):
        predictions_df[f"true_{name}"] = y_true[:, idx]
        predictions_df[f"prob_{name}"] = probabilities[:, idx]
        predictions_df[f"pred_{name}"] = y_pred[:, idx]
    predictions_df.to_csv(figures_dir / "test_predictions.csv", index=False)

    print(f"\nBest model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Figures and split CSVs: {figures_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "Run one of the model-specific scripts: train_resnet50.py, "
        "train_efficientnet.py, train_convnext.py, or train_vit.py"
    )
