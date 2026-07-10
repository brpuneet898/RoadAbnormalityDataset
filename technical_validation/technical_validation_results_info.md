# Technical Validation Results

## Phase 1 - Dataset Integrity Validation

### 1.1 - Image Integrity

| Metric | Value |
| --- | ---: |
| Total images | 2074 |
| Successfully opened images | 2074 |
| Corrupted images | 0 |
| Duplicate filenames | 0 |
| Invalid formats | 0 |
| Resolution mismatches | 0 |
| Supported formats | .heic, .jpeg, .jpg, .png |

### 1.2 - Duplicate Image Detection

| Metric | Value |
| --- | ---: |
| Total image files scanned | 2074 |
| Successfully processed images | 2074 |
| Unreadable images | 0 |
| Invalid format files ignored | 0 |
| Exact duplicates detected | 3 |
| Near duplicates detected | 6 |
| pHash threshold used | 2 |
| Cosine similarity threshold | 0.995 |

### 1.3 - Metadata Consistency

| Metric | Value |
| --- | ---: |
| Layer 0 image IDs | 2074 |
| Layer 1 annotation IDs | 1984 |
| Layer 2 metadata IDs | 2074 |
| Layer 3 geospatial IDs | 2074 |
| Layer0 ∩ Layer1 | 1984 |
| Layer0 ∩ Layer2 | 2074 |
| Layer0 ∩ Layer3 | 2074 |
| Common in all layers | 1984 |
| Missing in Layer 1 | 90 |
| Missing in Layer 2 | 0 |
| Missing in Layer 3 | 0 |
| Extra IDs in Layer 1 | 0 |
| Extra IDs in Layer 2 | 0 |
| Extra IDs in Layer 3 | 0 |

> **Note:** The Venn diagram image has been saved in the `/figures` folder.

### 1.4 - Missing Value Analysis

| Layer | Missing values |
| --- | ---: |
| Layer 0 | 0 |
| Layer 1 | 0 |
| Layer 2 | 90 |
| Layer 3 | 0 |

> **Note:** In Layer 2, fields were left empty only for images for which annotations could not be provided.

### 1.5 - Label Consistency

| Layer | Files checked | Issues found |
| --- | ---: | ---: |
| Layer 1 | 1 | 0 |
| Layer 2 | 1 | 0 |
| Layer 3 | 1 | 0 |

**Result: PASS - No label consistency issues found.**

## Phase 2 - Dataset Characterization

### 2.1 - Class Distribution

| Abnormality Type | Number of Images/Annotations | Percentage |
| --- | ---: | ---: |
| Surface depression | 1837 | 54.17 |
| Pothole | 639 | 18.84 |
| Manhole | 523 | 15.42 |
| Road patch failure | 335 | 9.88 |
| Crack | 33 | 0.97 |
| Miscellaneous abnormality | 24 | 0.71 |

#### Class Imbalance Discussion

The most frequent class is `Surface depression` with 1837 instances.

The least frequent class is `Miscellaneous abnormality` with 24 instances.

> **Note:** The class distribution pie chart and histogram plots have been saved in the `/figures` folder.

### 2.2 - Severity Distribution

| Severity | Count | Percentage |
| --- | ---: | ---: |
| minor | 611 | 29.46 |
| moderate | 896 | 43.20 |
| severe | 567 | 27.34 |

### 2.3 - Road Type Distribution

| Road Type | Count | Percentage |
| --- | ---: | ---: |
| urban | 1473 | 71.02 |
| highway | 7 | 0.34 |
| residential | 284 | 13.69 |
| rural | 310 | 14.95 |

### 2.4 - Weather Distribution

| Weather | Count | Percentage |
| --- | ---: | ---: |
| clear_dry | 1835 | 88.48 |
| wet_road | 185 | 8.92 |
| cloudy | 54 | 2.60 |

| Lighting Condition | Count | Percentage |
| --- | ---: | ---: |
| daylight | 1704 | 82.16 |
| night | 286 | 13.79 |
| shadow | 67 | 3.23 |
| low_light | 17 | 0.82 |

### 2.5 - Geographic Distribution

#### State-wise Geographic Distribution

| State | Count | Percentage |
| --- | ---: | ---: |
| Tamil Nadu | 1068 | 51.49 |
| Delhi | 779 | 37.56 |
| Maharashtra | 227 | 10.95 |

| District | Count | Percentage |
| --- | ---: | ---: |
| North West | 779 | 37.56 |
| Chennai | 434 | 20.93 |
| Chengalpattu | 319 | 15.38 |
| Tiruvallur | 315 | 15.19 |
| Sangli | 227 | 10.95 |

> **Note:** The geographic heatmap has been saved in the `/figures` folder.

### 2.6 - Bounding Box Distribution

| Metric | Average | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| width_scaled | 0.2863 | 0.2429 | 0.0265 | 1.0000 |
| height_scaled | 0.1930 | 0.1507 | 0.0192 | 1.0000 |
| aspect_ratio | 1.8080 | 1.5364 | 0.1604 | 13.4409 |
| object_area | 0.0715 | 0.0378 | 0.0006 | 0.9746 |
| relative_object_size | 7.1510 | 3.7783 | 0.0603 | 97.4554 |

> **Note:** The bounding box histogram and boxplots have been saved in the `/figures` folder.

### 2.7 - Number of Objects per Image

| Objects per Image | Number of Images | Percentage |
| --- | ---: | ---: |
| 1 | 1108 | 55.85 |
| 2 | 530 | 26.71 |
| 3 | 227 | 11.44 |
| 4+ | 119 | 6.00 |

> **Note:** The objects-per-image histogram has been saved in the `/figures` folder.

## Phase 3 - Annotation Quality Validation

### 3.1 - Annotation Protocol

The annotation protocol defines how road abnormality objects were identified, labelled, and recorded across the dataset. The purpose of this protocol is to ensure that annotations are consistent, reproducible, and suitable for downstream computer vision and dataset analysis tasks.

| Protocol Item | Documentation |
| --- | --- |
| Who annotated? | The dataset images were annotated by the dataset preparation team (2 members - Saransh Saini and Prathamesh Pise) using a consistent road-abnormality labelling protocol. Annotators reviewed each image and marked visible road abnormalities such as potholes, cracks, manholes, road patch failures, surface depressions, and miscellaneous abnormalities. |
| Training given? | Annotators were briefed on the dataset class definitions, severity categories, and bounding-box drawing rules before annotation. Example images were used to explain how each abnormality type should be identified and how ambiguous cases should be handled. |
| Software used? | An image annotation tool from HuggingFace was used to draw bounding boxes around road abnormalities and export annotation metadata for further validation. The exported annotations were stored and validated through the Layer 1 annotation metadata files. |
| Instructions? | Annotators were instructed to mark only clearly visible road abnormalities, assign the most appropriate abnormality class, and avoid labelling unrelated road objects or background regions. Each bounding box was expected to tightly cover the visible abnormality while preserving enough context to represent the full damaged area. |
| Boundary rules? | Bounding boxes were drawn around the complete visible extent of each abnormality. If an abnormality was partially occluded or cut off by the image boundary, only the visible portion was annotated. Overlapping abnormalities were annotated separately when they represented distinct objects. Very unclear or non-visible abnormalities were not annotated. |

> **Note:** This protocol was followed to maintain consistency in class labels, bounding-box placement, and annotation quality across the dataset.

### 3.2 - Manual Review

| Review Item | Value |
| --- | ---: |
| Total dataset images | 2074 |
| Sampling percentage | 10% |
| Images manually reviewed | 207 |
| Reviewed by | Puneet |
| Correct annotations | 207 |
| Incorrect annotations | 0 |
| Missing annotations | 0 |
| False positives | 0 |

> **Note:** A random 10% sample of the dataset, corresponding to 207 images, was manually reviewed by Puneet. All reviewed annotations were found to be correct, with no incorrect annotations, missing annotations, or false positives observed.

### 3.3 - Inter-Annotator Agreement

Inter-annotator agreement was evaluated using the annotations produced by the two annotators, Saransh Saini and Prathamesh Pise, on a common reviewed subset of images.

| Agreement Measure | Purpose | Result |
| --- | --- | --- |
| IoU agreement | Measures bounding-box overlap agreement between annotators. | 0.91 |
| Class agreement | Measures whether both annotators assigned the same abnormality class. | 96.80% |
| Cohen's Kappa | Measures pairwise class-label agreement between the two annotators while accounting for chance agreement. | 0.94 |
| Percentage agreement | Measures the overall percentage of matching annotation decisions between annotators. | 96.80% |

> **Note:** Since the dataset has two annotators, Cohen's Kappa was used as the primary kappa-based agreement statistic. The results indicate very strong agreement between annotators for both bounding-box placement and class labelling. Fleiss Kappa is not applicable, as it requires 3 or more annotators.

### 3.4 - Annotation Accuracy

Annotation accuracy was assessed by computing the mean Intersection over Union (IoU) between bounding boxes produced by Annotator A and Annotator B on the common reviewed subset.

| Accuracy Metric | Annotator A | Annotator B | Value |
| --- | --- | --- | ---: |
| Mean IoU | Saransh Saini | Prathamesh Pise | 0.91 |
