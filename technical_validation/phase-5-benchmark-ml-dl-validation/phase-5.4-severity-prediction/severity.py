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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"
}
SEVERITY_ORDER = ["minor", "moderate", "severe"]


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
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return index


def resolve_image_path(image_id: str, index: Dict[str, Path]) -> Path | None:
    key = str(image_id).strip().lower()
    return index.get(key) or index.get(Path(key).stem)


def load_dataframe(metadata_csv: Path, image_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"image_id", "severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metadata_csv}: {sorted(missing)}")

    df = df[["image_id", "severity"]].dropna().copy()
    df["severity"] = df["severity"].astype(str).str.strip().str.lower()

    unknown = sorted(set(df["severity"]) - set(SEVERITY_ORDER))
    if unknown:
        print(f"Warning: dropping unsupported severity values: {unknown}")
        df = df[df["severity"].isin(SEVERITY_ORDER)]

    index = build_image_index(image_dir)
    df["image_path"] = df["image_id"].map(
        lambda value: resolve_image_path(value, index)
    )
    missing_images = int(df["image_path"].isna().sum())
    if missing_images:
        print(f"Warning: skipping {missing_images} rows with no matching image.")

    df = df.dropna(subset=["image_path"]).drop_duplicates("image_id").reset_index(drop=True)

    present = set(df["severity"])
    absent = [label for label in SEVERITY_ORDER if label not in present]
    if absent:
        raise RuntimeError(f"Required severity classes are absent: {absent}")

    counts = df["severity"].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise RuntimeError(
            "Each severity class needs at least 3 images for stratified splitting. "
            f"Insufficient classes: {too_small.to_dict()}"
        )
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be > 0 and sum to < 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        stratify=df["severity"],
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["severity"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class SeverityDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform) -> None:
        self.df = dataframe
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(SEVERITY_ORDER)}

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

        target = self.class_to_idx[row["severity"]]
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


def create_model(model_name: str, pretrained: bool) -> nn.Module:
    num_classes = len(SEVERITY_ORDER)
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


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["severity"].value_counts()
    weights = [
        len(train_df) / (len(SEVERITY_ORDER) * counts[label])
        for label in SEVERITY_ORDER
    ]
    return torch.tensor(weights, dtype=torch.float32)


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
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


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
        probabilities = torch.softmax(model(images), dim=1)
        labels_all.append(labels.numpy())
        probabilities_all.append(probabilities.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return np.concatenate(labels_all), np.concatenate(probabilities_all), elapsed


def save_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    normalized: bool,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(SEVERITY_ORDER)),
        normalize="true" if normalized else None,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
        xlabel="Predicted severity",
        ylabel="True severity",
        xticks=np.arange(len(SEVERITY_ORDER)),
        yticks=np.arange(len(SEVERITY_ORDER)),
        xticklabels=[label.title() for label in SEVERITY_ORDER],
        yticklabels=[label.title() for label in SEVERITY_ORDER],
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalized else str(int(value))
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> Dict[str, float]:
    binary = label_binarize(y_true, classes=np.arange(len(SEVERITY_ORDER)))
    auc_by_class: Dict[str, float] = {}

    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, label in enumerate(SEVERITY_ORDER):
        fpr, tpr, _ = roc_curve(binary[:, idx], probabilities[:, idx])
        class_auc = auc(fpr, tpr)
        auc_by_class[label] = class_auc
        ax.plot(fpr, tpr, label=f"{label.title()} (AUC={class_auc:.3f})")

    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    ax.plot(
        micro_fpr,
        micro_tpr,
        linestyle="--",
        linewidth=2.5,
        label=f"Micro-average (AUC={micro_auc:.3f})",
    )
    auc_by_class["micro-average"] = micro_auc

    ax.plot([0, 1], [0, 1], linestyle=":")
    ax.set(
        title="Severity ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc_by_class


def save_probability_distribution(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, label in enumerate(SEVERITY_ORDER):
        axes[idx].hist(
            probabilities[y_true == idx, idx],
            bins=20,
            alpha=0.8,
        )
        axes[idx].set(
            title=f"{label.title()} true-class confidence",
            xlabel="Predicted probability",
            ylabel="Images",
            xlim=(0, 1),
        )
        axes[idx].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_results(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    inference_seconds: float,
    auc_by_class: Dict[str, float],
) -> None:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision macro": precision_macro,
        "Recall macro": recall_macro,
        "F1 macro": f1_macro,
        "Precision weighted": precision_weighted,
        "Recall weighted": recall_weighted,
        "F1 weighted": f1_weighted,
        "ROC-AUC macro OvR": roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="macro"
        ),
        "ROC-AUC weighted OvR": roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="weighted"
        ),
        "Quadratic weighted kappa": cohen_kappa_score(
            y_true, y_pred, weights="quadratic"
        ),
        "Ordinal MAE": mean_absolute_error(y_true, y_pred),
        "Inference images/second": len(y_true) / max(inference_seconds, 1e-12),
        "Inference ms/image": 1000.0 * inference_seconds / max(len(y_true), 1),
    }

    print(f"\nFinal test results — {model_name}")
    print(
        tabulate(
            [[key, f"{value:.6f}"] for key, value in metrics.items()],
            headers=["Metric", "Value"],
            tablefmt="rounded_grid",
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(SEVERITY_ORDER)),
        target_names=[label.title() for label in SEVERITY_ORDER],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in SEVERITY_ORDER:
        values = report[label.title()]
        rows.append(
            [
                label.title(),
                f"{values['precision']:.4f}",
                f"{values['recall']:.4f}",
                f"{values['f1-score']:.4f}",
                int(values["support"]),
                f"{auc_by_class[label]:.4f}",
            ]
        )

    print("\nPer-severity results")
    print(
        tabulate(
            rows,
            headers=["Severity", "Precision", "Recall", "F1", "Support", "AUC"],
            tablefmt="rounded_grid",
        )
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    image_size: int,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "task": "severity_prediction",
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "class_names": SEVERITY_ORDER,
            "class_to_idx": {
                label: idx for idx, label in enumerate(SEVERITY_ORDER)
            },
            "severity_order": SEVERITY_ORDER,
            "image_size": image_size,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def build_parser(default_model: str) -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_root = discover_project_root(script_dir)

    parser = argparse.ArgumentParser(
        description=f"Train {default_model} to predict minor/moderate/severe."
    )
    parser.add_argument(
        "--model-name",
        default=default_model,
        choices=["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"],
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
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
    parser.add_argument("--no-class-weights", action="store_true")
    return parser


def main(default_model: str) -> None:
    args = build_parser(default_model).parse_args()
    register_heif()
    seed_everything(args.seed)

    project_root = args.project_root.resolve()
    image_dir = (args.image_dir or project_root / "layer_0_raw_images").resolve()
    metadata_csv = (
        args.metadata_csv or project_root / "layer_2_semantic_metadata.csv"
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
    print(f"Metadata: {metadata_csv}")

    dataframe = load_dataframe(metadata_csv, image_dir)
    train_df, val_df, test_df = split_dataframe(
        dataframe,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    split_rows = []
    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        split_rows.append(
            [
                name,
                len(split_df),
                int((split_df["severity"] == "minor").sum()),
                int((split_df["severity"] == "moderate").sum()),
                int((split_df["severity"] == "severe").sum()),
            ]
        )
    print("\nDataset split")
    print(
        tabulate(
            split_rows,
            headers=["Split", "Images", "Minor", "Moderate", "Severe"],
            tablefmt="rounded_grid",
        )
    )

    train_transform, eval_transform = make_transforms(args.image_size)
    train_dataset = SeverityDataset(train_df, train_transform)
    val_dataset = SeverityDataset(val_df, eval_transform)
    test_dataset = SeverityDataset(test_df, eval_transform)

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
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_accuracy:.4f}"
        )

        save_checkpoint(
            last_path,
            model,
            args.model_name,
            args.image_size,
            epoch,
            best_val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                args.model_name,
                args.image_size,
                epoch,
                best_val_loss,
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
    y_pred = probabilities.argmax(axis=1)

    save_confusion_matrix(
        y_true,
        y_pred,
        figures_dir / "confusion_matrix.png",
        normalized=False,
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        figures_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )
    auc_by_class = save_roc_curves(
        y_true,
        probabilities,
        figures_dir / "roc_curves.png",
    )
    save_probability_distribution(
        y_true,
        probabilities,
        figures_dir / "confidence_distributions.png",
    )

    print_results(
        args.model_name,
        y_true,
        y_pred,
        probabilities,
        inference_seconds,
        auc_by_class,
    )

    print(f"\nBest model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "Run train_resnet50_severity.py, train_efficientnet_severity.py, "
        "train_convnext_severity.py, or train_vit_severity.py"
    )
