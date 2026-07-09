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
