# Road Abnormality Dataset

A layered computer-vision dataset for detecting, classifying, and analyzing road-surface abnormalities in Indian road scenes. The dataset combines raw road images, YOLO-format object-detection annotations, image-level semantic metadata, geospatial/context metadata, validation scripts, benchmark training code, trained model artifacts, and generated figures.

The repository is organized so that every dataset layer can be joined by the same stable key: `image_id` in the form `IMG_#####`.

## Dataset Summary

| Item | Value |
| --- | ---: |
| Images | 2,074 |
| Annotation files | 2,074 |
| Images with bounding-box labels | 1,984 |
| Negative / no-abnormality images | 90 |
| Object instances | 3,391 |
| Abnormality classes | 6 |
| Image formats | JPG, PNG, HEIC |
| Approximate image volume | 2.84 GiB |
| Geographic coverage | Delhi, Maharashtra, Tamil Nadu |
| District coverage | Chengalpattu, Chennai, North West, Sangli, Tiruvallur |
| Primary tasks | Object detection, image classification, multi-label classification, severity prediction, robustness analysis, bias analysis |

## Repository Structure

```text
RoadAbnormalityDataset/
+-- code/
|   +-- create_layer0_metadata_csv.py
|   +-- fill_layer2_abnormality_type.py
|   +-- layer_1_annotation_metadata_extractor.ipynb
|   +-- merge_and_rename_images.py
+-- figures/
|   +-- phase-5.1-object-detection/
|   +-- phase-5.2-image-classification/
|   +-- phase-5.4-severity-prediction/
|   +-- phase-*.png
+-- layer_0_raw_images/
|   +-- IMG_#####.jpg|png|heic
|   +-- info.md
|   +-- layer_0_raw_image_metadata.csv
+-- layer_1_annotations/
|   +-- IMG_#####.txt
+-- layer_2_metadata/
|   +-- info.md
|   +-- layer_2_semantic_metadata.csv
+-- layer_3_geospatial/
|   +-- info.md
|   +-- layer_3_geospatial_metadata.csv
+-- models/
|   +-- phase-5.1-object-detection/
|   +-- phase-5.2-image-classification/
|   +-- phase-5.4-severity-prediction/
+-- technical_validation/
|   +-- phase-1-dataset-integrity-validation/
|   +-- phase-2-dataset-characterization/
|   +-- phase-4-statistical-validation/
|   +-- phase-5-benchmark-ml-dl-validation/
|   +-- phase-6-robustness-evaluation/
|   +-- phase-7-dataset-split-strategy/
|   +-- phase-8-bias-analysis/
|   +-- comparison.md
|   +-- technical_validation_results_info.md
+-- LICENSE
+-- README.md
+-- requirements.txt
```

## Dataset Layers

### Layer 0: Raw Images

Path: `layer_0_raw_images/`

Layer 0 contains the source road-scene images and one metadata CSV row per image.

Image files are named as `IMG_#####` with one of the supported extensions: `.jpg`, `.jpeg`, `.png`, or `.heic`.

`layer_0_raw_image_metadata.csv` columns:

| Column | Description |
| --- | --- |
| `image_id` | Unique image identifier and cross-layer join key. |
| `file_type` | Image format without the leading dot. |
| `file_size` | File size in MiB, rounded to four decimals. |
| `resolution_width` | Image width in pixels. |
| `resolution_height` | Image height in pixels. |
| `collection_date` | Acquisition date in `DD-MM-YYYY` format. |
| `collection_state` | State or territory where the image was collected. |

### Layer 1: Bounding-Box Annotations

Path: `layer_1_annotations/`

Layer 1 contains one YOLO-format text file per image. Each annotation line follows:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image width and height.

Example:

```text
4 0.513675 0.697760532150776 0.08515625 0.23143015521064303
```

Class mapping:

| Class ID | Class Name | Instances |
| ---: | --- | ---: |
| 0 | Crack | 33 |
| 1 | Manhole | 523 |
| 2 | Miscellaneous abnormality | 24 |
| 3 | Pothole | 639 |
| 4 | Road patch failure | 335 |
| 5 | Surface depression | 1,837 |

Annotation notes:

- Empty label files represent images with no visible road abnormality.
- There are 90 negative images.
- Keep image and annotation filenames unchanged so `IMG_00001.jpg` and `IMG_00001.txt` continue to align by `image_id`.

### Layer 2: Semantic Metadata

Path: `layer_2_metadata/layer_2_semantic_metadata.csv`

Layer 2 provides image-level semantic labels describing abnormality and scene conditions.

| Column | Description / Allowed Values |
| --- | --- |
| `image_id` | Unique cross-layer join key. |
| `abnormality_type` | One or more abnormality classes separated by `; `. Blank for negative images. |
| `severity` | `minor`, `moderate`, `severe` |
| `road_type` | `highway`, `residential`, `rural`, `urban` |
| `weather` | `clear_dry`, `cloudy`, `wet_road` |
| `lighting` | `daylight`, `low_light`, `night`, `shadow` |
| `occlusion` | `none`, `partial`, `heavy`, `object_occluded`, `vehicle_occluded` |
| `shape` | `circular`, `clustered`, `irregular`, `longitudinal`, `transverse` |

Severity distribution:

| Severity | Images |
| --- | ---: |
| minor | 611 |
| moderate | 896 |
| severe | 567 |

### Layer 3: Geospatial and Road-Context Metadata

Path: `layer_3_geospatial/layer_3_geospatial_metadata.csv`

Layer 3 provides location and contextual road metadata.

| Column | Description / Allowed Values |
| --- | --- |
| `image_id` | Unique cross-layer join key. |
| `collection_state` | `Delhi`, `Maharashtra`, `Tamil Nadu` |
| `district` | `Chengalpattu`, `Chennai`, `North West`, `Sangli`, `Tiruvallur` |
| `latitude_grid` | Latitude coordinate associated with the image. |
| `longitude_grid` | Longitude coordinate associated with the image. |
| `pavement_type` | `asphalt`, `concrete`, `gravel`, `mixed`, `unpaved` |
| `traffic_density` | `low`, `medium`, `high` |
| `collection_season` | `summer`, `monsoon`, `post_monsoon`, `winter` |
| `nearby_feature` | `bridge`, `bus_stop`, `commercial_area`, `construction_site`, `drainage_channel`, `hospital`, `market`, `none`, `parking_area`, `residential_building`, `school`, `traffic_signal` |
| `slope_type` | `flat`, `mild_slope` |
| `road_width_category` | `narrow`, `medium`, `wide`, `very_wide` |
| `intersection_proximity` | `at_intersection`, `near_intersection`, `mid_block` |

State distribution:

| State | Images |
| --- | ---: |
| Delhi | 779 |
| Maharashtra | 227 |
| Tamil Nadu | 1,068 |

## File Naming and Joining

All layers use the same `image_id` convention:

```text
IMG_00001
IMG_00002
...
IMG_02074
```

Join examples:

- Raw image: `layer_0_raw_images/IMG_00001.jpg`
- Annotation: `layer_1_annotations/IMG_00001.txt`
- Semantic metadata row: `image_id == IMG_00001`
- Geospatial metadata row: `image_id == IMG_00001`

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt` and include:

- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `Pillow`, `pillow-heif`, `imagehash`
- `matplotlib`, `matplotlib-venn`, `tabulate`
- `torch`, `torchvision`, `ultralytics`
- `iterative-stratification`

## Quick Start

Load and join metadata:

```python
from pathlib import Path
import pandas as pd

root = Path("RoadAbnormalityDataset")

layer0 = pd.read_csv(root / "layer_0_raw_images" / "layer_0_raw_image_metadata.csv")
layer2 = pd.read_csv(root / "layer_2_metadata" / "layer_2_semantic_metadata.csv")
layer3 = pd.read_csv(root / "layer_3_geospatial" / "layer_3_geospatial_metadata.csv")

metadata = layer0.merge(layer2, on="image_id").merge(layer3, on="image_id")
print(metadata.head())
```

Read YOLO annotations for one image:

```python
from pathlib import Path

label_path = Path("layer_1_annotations/IMG_00001.txt")

with label_path.open("r", encoding="utf-8") as file:
    boxes = [line.strip().split() for line in file if line.strip()]

print(boxes)
```

Open an image, including HEIC support:

```python
from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

image_path = Path("layer_0_raw_images/IMG_00001.jpg")
image = Image.open(image_path)
print(image.size)
```

## Supported Research Tasks

This dataset supports several computer-vision and data-analysis tasks:

- Object detection using Layer 0 images and Layer 1 YOLO bounding boxes.
- Image-level abnormality classification using Layer 2 labels.
- Multi-label classification for images containing more than one abnormality type.
- Severity prediction using `severity` labels.
- Metadata-aware learning using road type, weather, lighting, occlusion, shape, pavement, traffic density, season, and other context variables.
- Geographic and spatial analysis using Layer 3 coordinates and district/state fields.
- Robustness, bias, and sampling analysis using the validation scripts.

## Benchmark Code and Models

Benchmark scripts are stored under:

```text
technical_validation/phase-5-benchmark-ml-dl-validation/
```

Included benchmark groups:

| Task | Path |
| --- | --- |
| Object detection | `phase-5.1-object-detection/` |
| Image classification | `phase-5.2-image-classification/` |
| Multi-label classification | `phase-5.3-multi-label-classification/` |
| Severity prediction | `phase-5.4-severity-prediction/` |

The repository includes trained model artifacts in `models/`, including YOLO object-detection weights and classification/severity model archives for ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, and ViT-B/16 variants.

Generated plots and training diagnostics are stored in `figures/`.

## Technical Validation

Validation scripts and results are stored under `technical_validation/`.

The validation package covers:

- Phase 1: Dataset integrity validation.
- Phase 2: Dataset characterization.
- Phase 4: Statistical validation.
- Phase 5: Benchmark machine-learning and deep-learning validation.
- Phase 6: Robustness evaluation.
- Phase 7: Dataset split strategy.
- Phase 8: Bias analysis.

Key reported validation results:

| Check | Result |
| --- | ---: |
| Total images opened successfully | 2,074 / 2,074 |
| Corrupted images | 0 |
| Duplicate filenames | 0 |
| Invalid formats | 0 |
| Layer 0 metadata missing values | 0 |
| Layer 2 metadata records | 2,074 |
| Layer 3 metadata records | 2,074 |
| Layer 1 label consistency issues | 0 |

Duplicate analysis identified 3 exact duplicates and 6 near duplicates. These were manually reviewed and treated as repeated captures of the same road abnormality from slightly different viewpoints. The validation notes state that duplicate and near-duplicate samples were assigned to the same split to avoid train-validation-test leakage.

See `technical_validation/technical_validation_results_info.md` for the complete validation report.

## Dataset Splits

Dataset split tooling is provided in:

```text
technical_validation/phase-7-dataset-split-strategy/
```

Important scripts:

- `split_statistics.py`
- `package_dataset_splits.py`

Use these scripts when creating train, validation, and test partitions so that duplicate-aware and metadata-aware split behavior remains reproducible.

## Reproducing Analyses

Run individual scripts from the repository root. Examples:

```bash
python technical_validation/phase-1-dataset-integrity-validation/phase-1.1-image-integrity/image_integrity_check.py
python technical_validation/phase-2-dataset-characterization/phase-2.1-class-distribution/class-distribution.py
python technical_validation/phase-4-statistical-validation/phase-4.1-confidence-intervals/confidence-interval.py
python technical_validation/phase-6-robustness-evaluation/robustness_evaluation.py
python technical_validation/phase-8-bias-analysis/bias_analysis.py
```

Some benchmark scripts may require a GPU-enabled PyTorch installation for practical training time.

## Annotation Format Details

For each non-empty label file:

```text
class_id x_center y_center width height
```

Where:

- `class_id` is an integer from 0 to 5.
- `x_center` and `y_center` are normalized center coordinates.
- `width` and `height` are normalized bounding-box dimensions.
- All coordinate values are relative to image dimensions and typically fall in `[0, 1]`.

To convert one YOLO box to pixel coordinates:

```python
def yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return x1, y1, x2, y2
```

## Comparison with Existing Road-Damage Datasets

The repository includes a detailed comparison with RDD2019, RDD2020, and RDD2022 in `technical_validation/comparison.md`.

Compared with many road-damage datasets that primarily provide bounding boxes, this dataset emphasizes annotation richness:

- Bounding boxes plus semantic metadata.
- Severity labels.
- Multi-label image support.
- Weather, lighting, road type, traffic density, pavement, season, and location context.
- Validation, robustness, bias-analysis, and benchmark scripts.

## Best Practices

- Do not rename image or annotation files.
- Use `image_id` as the only cross-layer join key.
- Treat Layer 2 and Layer 3 attributes as image-level metadata, not object-level labels.
- Treat empty annotation files as valid negative samples.
- Use duplicate-aware splitting to avoid leakage.
- Preserve HEIC support when loading all raw images.
- Report class imbalance when publishing model results, because `Surface depression` is the dominant class.

## Limitations

- The dataset is geographically focused on selected Indian regions rather than global road conditions.
- Class distribution is imbalanced, with `Surface depression` much more frequent than `Crack` or `Miscellaneous abnormality`.
- Some metadata attributes are image-level and should not be interpreted as per-object labels.
- The dataset size is smaller than large-scale RDD releases, but it provides richer semantic and geospatial metadata.

## License

This repository is licensed under the MIT License. See `LICENSE` for the full license text.

## Citation

If you use this dataset in academic or applied research, cite the dataset/repository and include the version or commit hash used for your experiments.

## Contact

For questions, corrections, or contributions, open an issue or contact the dataset maintainer.
