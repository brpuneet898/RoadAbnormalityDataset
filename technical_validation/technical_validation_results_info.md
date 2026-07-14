# Technical Validation Results

## Phase 1 - Dataset Integrity Validation

### 1.1 - Image Integrity

| Metric                     |                    Value |
| -------------------------- | -----------------------: |
| Total images               |                     2074 |
| Successfully opened images |                     2074 |
| Corrupted images           |                        0 |
| Duplicate filenames        |                        0 |
| Invalid formats            |                        0 |
| Resolution mismatches      |                        0 |
| Supported formats          | .heic, .jpeg, .jpg, .png |

### 1.2 - Duplicate Image Detection

| Metric                        | Value |
| ----------------------------- | ----: |
| Total image files scanned     |  2074 |
| Successfully processed images |  2074 |
| Unreadable images             |     0 |
| Invalid format files ignored  |     0 |
| Exact duplicates detected     |     3 |
| Near duplicates detected      |     6 |
| pHash threshold used          |     2 |
| Cosine similarity threshold   | 0.995 |

### 1.3 - Metadata Consistency

| Metric                 | Value |
| ---------------------- | ----: |
| Layer 0 image IDs      |  2074 |
| Layer 1 annotation IDs |  1984 |
| Layer 2 metadata IDs   |  2074 |
| Layer 3 geospatial IDs |  2074 |
| Layer0 ∩ Layer1        |  1984 |
| Layer0 ∩ Layer2        |  2074 |
| Layer0 ∩ Layer3        |  2074 |
| Common in all layers   |  1984 |
| Missing in Layer 1     |    90 |
| Missing in Layer 2     |     0 |
| Missing in Layer 3     |     0 |
| Extra IDs in Layer 1   |     0 |
| Extra IDs in Layer 2   |     0 |
| Extra IDs in Layer 3   |     0 |

> **Note:** The Venn diagram image has been saved in the `/figures` folder.

### 1.4 - Missing Value Analysis

| Layer   | Missing values |
| ------- | -------------: |
| Layer 0 |              0 |
| Layer 1 |              0 |
| Layer 2 |             90 |
| Layer 3 |              0 |

> **Note:** In Layer 2, fields were left empty only for images for which annotations could not be provided.

### 1.5 - Label Consistency

| Layer   | Files checked | Issues found |
| ------- | ------------: | -----------: |
| Layer 1 |             1 |            0 |
| Layer 2 |             1 |            0 |
| Layer 3 |             1 |            0 |

**Result: PASS - No label consistency issues found.**

## Phase 2 - Dataset Characterization

### 2.1 - Class Distribution

| Abnormality Type          | Number of Images/Annotations | Percentage |
| ------------------------- | ---------------------------: | ---------: |
| Surface depression        |                         1837 |      54.17 |
| Pothole                   |                          639 |      18.84 |
| Manhole                   |                          523 |      15.42 |
| Road patch failure        |                          335 |       9.88 |
| Crack                     |                           33 |       0.97 |
| Miscellaneous abnormality |                           24 |       0.71 |

#### Class Imbalance Discussion

The most frequent class is `Surface depression` with 1837 instances.

The least frequent class is `Miscellaneous abnormality` with 24 instances.

> **Note:** The class distribution pie chart and histogram plots have been saved in the `/figures` folder.

### 2.2 - Severity Distribution

| Severity | Count | Percentage |
| -------- | ----: | ---------: |
| minor    |   611 |      29.46 |
| moderate |   896 |      43.20 |
| severe   |   567 |      27.34 |

### 2.3 - Road Type Distribution

| Road Type   | Count | Percentage |
| ----------- | ----: | ---------: |
| urban       |  1473 |      71.02 |
| highway     |     7 |       0.34 |
| residential |   284 |      13.69 |
| rural       |   310 |      14.95 |

### 2.4 - Weather Distribution

| Weather   | Count | Percentage |
| --------- | ----: | ---------: |
| clear_dry |  1835 |      88.48 |
| wet_road  |   185 |       8.92 |
| cloudy    |    54 |       2.60 |

| Lighting Condition | Count | Percentage |
| ------------------ | ----: | ---------: |
| daylight           |  1704 |      82.16 |
| night              |   286 |      13.79 |
| shadow             |    67 |       3.23 |
| low_light          |    17 |       0.82 |

### 2.5 - Geographic Distribution

#### State-wise Geographic Distribution

| State       | Count | Percentage |
| ----------- | ----: | ---------: |
| Tamil Nadu  |  1068 |      51.49 |
| Delhi       |   779 |      37.56 |
| Maharashtra |   227 |      10.95 |

| District     | Count | Percentage |
| ------------ | ----: | ---------: |
| North West   |   779 |      37.56 |
| Chennai      |   434 |      20.93 |
| Chengalpattu |   319 |      15.38 |
| Tiruvallur   |   315 |      15.19 |
| Sangli       |   227 |      10.95 |

> **Note:** The geographic heatmap has been saved in the `/figures` folder.

### 2.6 - Bounding Box Distribution

| Metric               | Average | Median | Minimum | Maximum |
| -------------------- | ------: | -----: | ------: | ------: |
| width_scaled         |  0.2863 | 0.2429 |  0.0265 |  1.0000 |
| height_scaled        |  0.1930 | 0.1507 |  0.0192 |  1.0000 |
| aspect_ratio         |  1.8080 | 1.5364 |  0.1604 | 13.4409 |
| object_area          |  0.0715 | 0.0378 |  0.0006 |  0.9746 |
| relative_object_size |  7.1510 | 3.7783 |  0.0603 | 97.4554 |

> **Note:** The bounding box histogram and boxplots have been saved in the `/figures` folder.

### 2.7 - Number of Objects per Image

| Objects per Image | Number of Images | Percentage |
| ----------------- | ---------------: | ---------: |
| 1                 |             1108 |      55.85 |
| 2                 |              530 |      26.71 |
| 3                 |              227 |      11.44 |
| 4+                |              119 |       6.00 |

> **Note:** The objects-per-image histogram has been saved in the `/figures` folder.

## Phase 3 - Annotation Quality Validation

### 3.1 - Annotation Protocol

The annotation protocol defines how road abnormality objects were identified, labelled, and recorded across the dataset. The purpose of this protocol is to ensure that annotations are consistent, reproducible, and suitable for downstream computer vision and dataset analysis tasks.

| Protocol Item   | Documentation                                                                                                                                                                                                                                                                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Who annotated?  | The dataset images were annotated by the dataset preparation team (2 members - Saransh Saini and Prathamesh Pise) using a consistent road-abnormality labelling protocol. Annotators reviewed each image and marked visible road abnormalities such as potholes, cracks, manholes, road patch failures, surface depressions, and miscellaneous abnormalities.  |
| Training given? | Annotators were briefed on the dataset class definitions, severity categories, and bounding-box drawing rules before annotation. Example images were used to explain how each abnormality type should be identified and how ambiguous cases should be handled.                                                                                                 |
| Software used?  | An image annotation tool from HuggingFace was used to draw bounding boxes around road abnormalities and export annotation metadata for further validation. The exported annotations were stored and validated through the Layer 1 annotation metadata files.                                                                                                   |
| Instructions?   | Annotators were instructed to mark only clearly visible road abnormalities, assign the most appropriate abnormality class, and avoid labelling unrelated road objects or background regions. Each bounding box was expected to tightly cover the visible abnormality while preserving enough context to represent the full damaged area.                       |
| Boundary rules? | Bounding boxes were drawn around the complete visible extent of each abnormality. If an abnormality was partially occluded or cut off by the image boundary, only the visible portion was annotated. Overlapping abnormalities were annotated separately when they represented distinct objects. Very unclear or non-visible abnormalities were not annotated. |

> **Note:** This protocol was followed to maintain consistency in class labels, bounding-box placement, and annotation quality across the dataset.

### 3.2 - Manual Review

| Review Item              |  Value |
| ------------------------ | -----: |
| Total dataset images     |   2074 |
| Sampling percentage      |    10% |
| Images manually reviewed |    207 |
| Reviewed by              | Puneet |
| Correct annotations      |    207 |
| Incorrect annotations    |      0 |
| Missing annotations      |      0 |
| False positives          |      0 |

> **Note:** A random 10% sample of the dataset, corresponding to 207 images, was manually reviewed by Puneet. All reviewed annotations were found to be correct, with no incorrect annotations, missing annotations, or false positives observed.

### 3.3 - Inter-Annotator Agreement

Inter-annotator agreement was evaluated using the annotations produced by the two annotators, Saransh Saini and Prathamesh Pise, on a common reviewed subset of images.

| Agreement Measure    | Purpose                                                                                                   | Result |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------ |
| IoU agreement        | Measures bounding-box overlap agreement between annotators.                                               | 0.91   |
| Class agreement      | Measures whether both annotators assigned the same abnormality class.                                     | 96.80% |
| Cohen's Kappa        | Measures pairwise class-label agreement between the two annotators while accounting for chance agreement. | 0.94   |
| Percentage agreement | Measures the overall percentage of matching annotation decisions between annotators.                      | 96.80% |

> **Note:** Since the dataset has two annotators, Cohen's Kappa was used as the primary kappa-based agreement statistic. The results indicate very strong agreement between annotators for both bounding-box placement and class labelling. Fleiss Kappa is not applicable, as it requires 3 or more annotators.

### 3.4 - Annotation Accuracy

Annotation accuracy was assessed by computing the mean Intersection over Union (IoU) between bounding boxes produced by Annotator A and Annotator B on the common reviewed subset.

| Accuracy Metric | Annotator A   | Annotator B     | Value |
| --------------- | ------------- | --------------- | ----: |
| Mean IoU        | Saransh Saini | Prathamesh Pise |  0.91 |

## Phase 4 - Statistical Validation

### 4.1 - Confidence Intervals

| Abnormality Type                                | Count | Proportion | 95% CI Lower | 95% CI Upper |
| ----------------------------------------------- | ----: | ---------: | -----------: | -----------: |
| Crack                                           |    12 |     0.0058 |       0.0025 |       0.0091 |
| Crack; Surface depression                       |     4 |     0.0019 |       0.0000 |       0.0038 |
| Crack; Surface depression; Manhole              |     1 |     0.0005 |       0.0000 |       0.0014 |
| Manhole                                         |   233 |     0.1123 |       0.0988 |       0.1259 |
| Manhole; Crack                                  |     2 |     0.0010 |       0.0000 |       0.0023 |
| Manhole; Crack; Surface depression              |     1 |     0.0005 |       0.0000 |       0.0014 |
| Manhole; Miscellaneous abnormality              |     3 |     0.0014 |       0.0000 |       0.0031 |
| Manhole; Pothole                                |    16 |     0.0077 |       0.0039 |       0.0115 |
| Manhole; Pothole; Surface depression            |     7 |     0.0034 |       0.0009 |       0.0059 |
| Manhole; Road patch failure                     |    11 |     0.0053 |       0.0022 |       0.0084 |
| Manhole; Road patch failure; Crack              |     1 |     0.0005 |       0.0000 |       0.0014 |
| Manhole; Road patch failure; Surface depression |     1 |     0.0005 |       0.0000 |       0.0014 |
| Manhole; Surface depression                     |   113 |     0.0545 |       0.0447 |       0.0643 |
| Manhole; Surface depression; Pothole            |     2 |     0.0010 |       0.0000 |       0.0023 |
| Manhole; Surface depression; Road patch failure |     2 |     0.0010 |       0.0000 |       0.0023 |
| Miscellaneous abnormality                       |    14 |     0.0068 |       0.0032 |       0.0103 |
| Miscellaneous abnormality; Manhole              |     1 |     0.0005 |       0.0000 |       0.0014 |
| Miscellaneous abnormality; Surface depression   |     2 |     0.0010 |       0.0000 |       0.0023 |
| Pothole                                         |   211 |     0.1017 |       0.0887 |       0.1147 |
| Pothole; Manhole                                |    21 |     0.0101 |       0.0058 |       0.0144 |
| Pothole; Manhole; Surface depression            |     7 |     0.0034 |       0.0009 |       0.0059 |
| Pothole; Road patch failure                     |    11 |     0.0053 |       0.0022 |       0.0084 |
| Pothole; Road patch failure; Surface depression |     1 |     0.0005 |       0.0000 |       0.0014 |
| Pothole; Surface depression                     |   106 |     0.0511 |       0.0416 |       0.0606 |
| Pothole; Surface depression; Manhole            |     5 |     0.0024 |       0.0003 |       0.0045 |
| Pothole; Surface depression; Road patch failure |     4 |     0.0019 |       0.0000 |       0.0038 |
| Road patch failure                              |   146 |     0.0704 |       0.0594 |       0.0814 |
| Road patch failure; Crack                       |     4 |     0.0019 |       0.0000 |       0.0038 |
| Road patch failure; Manhole                     |     2 |     0.0010 |       0.0000 |       0.0023 |
| Road patch failure; Manhole; Surface depression |     1 |     0.0005 |       0.0000 |       0.0014 |
| Road patch failure; Miscellaneous abnormality   |     1 |     0.0005 |       0.0000 |       0.0014 |
| Road patch failure; Pothole                     |     1 |     0.0005 |       0.0000 |       0.0014 |
| Road patch failure; Surface depression          |    18 |     0.0087 |       0.0047 |       0.0127 |
| Road patch failure; Surface depression; Pothole |     1 |     0.0005 |       0.0000 |       0.0014 |
| Surface depression                              |   881 |     0.4248 |       0.4035 |       0.4461 |
| Surface depression; Crack                       |     4 |     0.0019 |       0.0000 |       0.0038 |
| Surface depression; Crack; Road patch failure   |     1 |     0.0005 |       0.0000 |       0.0014 |
| Surface depression; Manhole                     |    36 |     0.0174 |       0.0117 |       0.0230 |
| Surface depression; Manhole; Pothole            |     4 |     0.0019 |       0.0000 |       0.0038 |
| Surface depression; Manhole; Road patch failure |     2 |     0.0010 |       0.0000 |       0.0023 |
| Surface depression; Miscellaneous abnormality   |     3 |     0.0014 |       0.0000 |       0.0031 |
| Surface depression; Pothole                     |    48 |     0.0231 |       0.0167 |       0.0296 |
| Surface depression; Pothole; Manhole            |     3 |     0.0014 |       0.0000 |       0.0031 |
| Surface depression; Pothole; Road patch failure |     2 |     0.0010 |       0.0000 |       0.0023 |
| Surface depression; Road patch failure          |    33 |     0.0159 |       0.0105 |       0.0213 |
| Surface depression; Road patch failure; Manhole |     1 |     0.0005 |       0.0000 |       0.0014 |
| NaN                                             |    90 |     0.0434 |       0.0346 |       0.0522 |

| Severity | Count | Proportion | 95% CI Lower | 95% CI Upper |
| -------- | ----: | ---------: | -----------: | -----------: |
| minor    |   611 |     0.2946 |       0.2750 |       0.3142 |
| moderate |   896 |     0.4320 |       0.4107 |       0.4533 |
| severe   |   567 |     0.2734 |       0.2542 |       0.2926 |

| Road Type   | Count | Proportion | 95% CI Lower | 95% CI Upper |
| ----------- | ----: | ---------: | -----------: | -----------: |
| highway     |     7 |     0.0034 |       0.0009 |       0.0059 |
| residential |   284 |     0.1369 |       0.1221 |       0.1517 |
| rural       |   310 |     0.1495 |       0.1341 |       0.1648 |
| urban       |  1473 |     0.7102 |       0.6907 |       0.7297 |

### 4.2 - Sampling Bias Analysis

| State       | Count | Percentage |
| ----------- | ----: | ---------: |
| Tamil Nadu  |  1068 |      51.49 |
| Delhi       |   779 |      37.56 |
| Maharashtra |   227 |      10.95 |

Tamil Nadu samples: 51.49% of the dataset.

- The dataset is geographically concentrated, with Tamil Nadu alone contributing 51.49% of all samples.
- Model performance may be stronger for road surfaces, lighting conditions, traffic environments, and infrastructure patterns commonly seen in Tamil Nadu.
- Delhi and Maharashtra are represented, but Maharashtra has a comparatively smaller share at 10.95%, which may reduce generalization for that region.
- Road abnormality appearance can vary by climate, construction material, maintenance practices, and urban planning patterns, so this state-level imbalance may introduce regional bias.
- Additional samples from underrepresented states and districts would improve geographic diversity and make the dataset more representative across India.

### 4.3 - Distribution Comparison

| Distribution | Chi-square | Degrees of freedom | P-value | Significant | Cramér's V | Effect | Low expected cells | Sample size |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: |
| Class | 338.9292 | 90 | < 0.0001 | Yes | 0.2506 | Weak | 102/138 | 1984 |
| Severity | 85.5020 | 4 | < 0.0001 | Yes | 0.1402 | Weak | 0/9 | 2074 |
| Road Type | 360.6529 | 6 | < 0.0001 | Yes | 0.2925 | Weak | 3/12 | 2074 |

- All three distributions show statistically significant differences, with p-values below 0.0001.
- The class distribution is significantly imbalanced, which is expected because surface depression and pothole-related categories occur much more frequently than rare classes such as crack and miscellaneous abnormality.
- The severity distribution is also statistically significant, but the effect size is weak, indicating that the imbalance exists but is not extremely strong.
- The road type distribution shows a significant difference, mainly due to the high number of urban samples compared with highway, residential, and rural samples.
- Cramér's V values are in the weak-effect range for all three tests, meaning the differences are statistically detectable but the practical association strength is limited.
- The class distribution has many low expected cells, so its chi-square result should be interpreted carefully because several rare multi-label class combinations have very small counts.
- For future improvement, collecting more samples from rare abnormality classes, highway roads, and underrepresented combinations would make the dataset more balanced and statistically robust.

### 4.4 - Correlation Analysis

| Analysis | Method | Statistic | Degrees of freedom | P-value | Effect size | Effect measure | Effect strength | Significance | Sample size | Expected cells < 5 |
| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | ---: | --- |
| Weather → severity | Chi-square | 102.3951 | 4 | <0.0001 | 0.1541 | Cramér's V | Weak | Significant | 2074 | 0/9 |
| Road type → abnormality type | Chi-square | 246.7968 | 135 | <0.0001 | 0.1371 | Cramér's V | Weak | Significant | 1984 | 152/184 |
| Traffic density → abnormality frequency | Spearman correlation | -0.0003 | N/A | 0.9878 | -0.0003 | Spearman rho | Negligible, negative | Not significant | 2074 | N/A |
| Traffic density → abnormality frequency | Kruskal-Wallis | 0.0663 | 2 | 0.9674 | N/A | N/A | N/A | Not significant | 2074 | N/A |

- Weather and road type show statistically significant but weak associations with severity and abnormality type.
- Traffic density does not show a significant relationship with abnormality frequency.
- These results indicate association only, not causation.

## Phase 5 - Benchmarking Validation

### 5.1 - Object Detection

| Model | Precision | Recall | F1-Score | mAP50 | mAP50-95 | Early Stopped At Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 0.527 | 0.465 | 0.4940625 | 0.481 | 0.222 | 46 |
| YOLOv8s | 0.609 | 0.398 | 0.48139424 | 0.449 | 0.214 | 46 |
| YOLO12s | 0.6613 | 0.449 | 0.534853103 | 0.492 | 0.224 | 44 |
| YOLO12n | 0.675 | 0.398 | 0.500745573 | 0.489 | 0.231 | 45 |
| YOLO11n | 0.601 | 0.413 | 0.489571992 | 0.465 | 0.194 | 46 |
| YOLO26n | 0.491 | 0.442 | 0.46521329 | 0.441 | 0.175 | 49 |
| YOLO26s | 0.598 | 0.355 | 0.445519412 | 0.412 | 0.173 | 45 |
| YOLO10s | 0.59 | 0.381 | 0.463007209 | 0.436 | 0.179 | 42 |

> **Note:** The relevant object-detection figures and trained model files have been added to their respective `/figures` and `/models` folders.

### 5.2/5.3 Image and Multi Label Classification

6 class, Multi-label - surface depression, manhole, pothole, road patch failure, crack, miscellaneous abnormality

| Metric | convnext_tiny | efficientnet_b0 | resnet50 | vit_b_16 |
| --- | ---: | ---: | ---: | ---: |
| Exact-match accuracy | 0.248322 | 0.174497 | 0.184564 | 0.060403 |
| Hamming accuracy | 0.803691 | 0.786913 | 0.763982 | 0.736018 |
| Precision macro | 0.355861 | 0.344583 | 0.395501 | 0.258161 |
| Recall macro | 0.526398 | 0.573307 | 0.558319 | 0.449843 |
| F1 macro | 0.406522 | 0.413229 | 0.433456 | 0.316375 |
| Precision micro | 0.532860 | 0.505710 | 0.470486 | 0.438053 |
| Recall micro | 0.773196 | 0.798969 | 0.698454 | 0.765464 |
| F1 micro | 0.630915 | 0.619381 | 0.562241 | 0.557223 |
| Training seconds | 2193.997008 | 1359.233063 | 1596.615273 | 4641.274044 |
| Inference seconds | 36.234247 | 28.897818 | 32.223555 | 35.631523 |
| Inference images/second | 8.224264 | 10.312197 | 9.247893 | 8.363381 |
| Best epoch | 6.0 | 1.0 | 3.0 | 20.0 |
| Best validation loss | 0.772340 | 0.811768 | 0.822145 | 0.893948 |
| AUC macro | 0.710334 | 0.701259 | 0.729413 | 0.576951 |

> **Note:** The relevant image-classification figures and trained model files have been uploaded in their respective folders.

### 5.4 - Severity Prediction

3 types of Severity - minor, moderate, severe

| Model | Accuracy | Balanced Accuracy | Precision Macro | Recall Macro | F1 Macro | Precision Weighted | Recall Weighted | F1 Weighted | ROC AUC Macro OVR | ROC AUC Weighted OVR | Quadratic Weighted Kappa | Ordinal MAE | Inference Images/s | Inference ms/Image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | 0.605769 | 0.653219 | 0.627373 | 0.653219 | 0.605735 | 0.637183 | 0.605769 | 0.584585 | 0.818581 | 0.802422 | 0.616294 | 0.435897 | 8.838274 | 113.144265 |
| efficientnet_b0 | 0.631410 | 0.672893 | 0.655330 | 0.672893 | 0.629643 | 0.675698 | 0.631410 | 0.614208 | 0.831919 | 0.820642 | 0.591940 | 0.429487 | 9.061245 | 110.360107 |
| convnext_tiny | 0.589744 | 0.625533 | 0.606818 | 0.625533 | 0.593272 | 0.614021 | 0.589744 | 0.578625 | 0.812605 | 0.798423 | 0.560556 | 0.464744 | 8.255720 | 121.128133 |
| vit_b_16 | 0.519231 | 0.533567 | 0.525689 | 0.533567 | 0.520553 | 0.530375 | 0.519231 | 0.516543 | 0.730362 | 0.717411 | 0.400030 | 0.564103 | 9.510258 | 105.149621 |

| Model | Severity | Precision | Recall | F1 Score | Support | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| resnet50 | minor | 0.680412 | 0.717391 | 0.698413 | 92 | 0.875840 |
| resnet50 | moderate | 0.681159 | 0.348148 | 0.460784 | 135 | 0.709730 |
| resnet50 | severe | 0.520548 | 0.894118 | 0.658009 | 85 | 0.870174 |
| efficientnet_b0 | minor | 0.593750 | 0.826087 | 0.690909 | 92 | 0.872036 |
| efficientnet_b0 | moderate | 0.791045 | 0.392593 | 0.524752 | 135 | 0.755932 |
| efficientnet_b0 | severe | 0.581197 | 0.800000 | 0.673267 | 85 | 0.867790 |
| convnext_tiny | minor | 0.670455 | 0.641304 | 0.655556 | 92 | 0.851038 |
| convnext_tiny | moderate | 0.642857 | 0.400000 | 0.493151 | 135 | 0.718728 |
| convnext_tiny | severe | 0.507143 | 0.835294 | 0.631111 | 85 | 0.868049 |
| vit_b_16 | minor | 0.564103 | 0.478261 | 0.517647 | 92 | 0.763192 |
| vit_b_16 | moderate | 0.549550 | 0.451852 | 0.495935 | 135 | 0.644947 |
| vit_b_16 | severe | 0.463415 | 0.670588 | 0.548077 | 85 | 0.782949 |

> **Note:** The best-performing model checkpoints have been saved in the `/models` folder, and the related evaluation figures have been saved in the `/figures` folder.

## Phase 6 - Robustness Evaluation

This section evaluates dataset robustness across difficult acquisition conditions, road contexts, object-size groups, and state-wise subsets. The evaluation is metadata-based and does not require model retraining.

### 6.1 - Robustness Slice Summary

| Slice | Images | Dataset Percentage | Dominant Abnormality Type | Dominant Class Percentage | Dominant Severity |
| --- | ---: | ---: | --- | ---: | --- |
| Night | 286 | 13.79 | Manhole | 35.66 | moderate |
| Rain / wet road | 185 | 8.92 | Pothole | 28.11 | severe |
| Shadow | 67 | 3.23 | Surface depression | 61.19 | moderate |
| Urban | 1473 | 71.02 | Surface depression | 38.63 | moderate |
| Highway | 7 | 0.34 | Surface depression | 57.14 | moderate |
| Rural | 310 | 14.95 | Surface depression | 62.90 | moderate |
| Low light | 17 | 0.82 | Missing | 35.29 | moderate |
| Tiny objects (<1%) | 148 | 7.14 | Surface depression | 27.03 | minor |
| Small objects (1%-<5%) | 677 | 32.64 | Surface depression | 35.75 | minor |
| Medium objects (5%-<15%) | 791 | 38.14 | Surface depression | 48.93 | moderate |
| Large objects (>=15%) | 368 | 17.74 | Surface depression | 57.61 | severe |
| State: Delhi | 779 | 37.56 | Surface depression | 47.11 | minor |
| State: Maharashtra | 227 | 10.95 | Surface depression | 28.63 | moderate |
| State: Tamil Nadu | 1068 | 51.49 | Surface depression | 42.04 | moderate |

The dataset includes several robustness-relevant subsets such as night, wet-road, shadow, low-light, road-type, state-wise, and object-size slices. However, some difficult conditions such as highway and low-light are very small, so robustness claims for those slices should be interpreted cautiously.

### 6.2 - State Coverage

| State | Images | Dataset Percentage |
| --- | ---: | ---: |
| Tamil Nadu | 1068 | 51.49 |
| Delhi | 779 | 37.56 |
| Maharashtra | 227 | 10.95 |

Tamil Nadu and Delhi provide most of the geographic coverage, while Maharashtra has a smaller representation. This supports state-wise robustness evaluation, but wider geographic sampling would improve generalization.

### 6.3 - Top Class Distribution by Robustness Slice

| Slice | Rank | Abnormality Type | Count | Percentage |
| --- | ---: | --- | ---: | ---: |
| Night | 1 | Manhole | 102 | 35.66 |
| Night | 2 | Surface depression | 90 | 31.47 |
| Night | 3 | Manhole; Surface depression | 27 | 9.44 |
| Night | 4 | Road patch failure | 25 | 8.74 |
| Night | 5 | Missing | 10 | 3.50 |
| Night | 6 | Surface depression; Manhole | 7 | 2.45 |
| Rain / wet road | 1 | Pothole | 52 | 28.11 |
| Rain / wet road | 2 | Surface depression | 45 | 24.32 |
| Rain / wet road | 3 | Pothole; Surface depression | 22 | 11.89 |
| Rain / wet road | 4 | Manhole | 20 | 10.81 |
| Rain / wet road | 5 | Road patch failure | 11 | 5.95 |
| Rain / wet road | 6 | Manhole; Surface depression | 7 | 3.78 |
| Shadow | 1 | Surface depression | 41 | 61.19 |
| Shadow | 2 | Road patch failure | 10 | 14.93 |
| Shadow | 3 | Pothole | 8 | 11.94 |
| Shadow | 4 | Manhole | 2 | 2.99 |
| Shadow | 5 | Road patch failure; Surface depression | 2 | 2.99 |
| Shadow | 6 | Missing | 1 | 1.49 |
| Urban | 1 | Surface depression | 569 | 38.63 |
| Urban | 2 | Manhole | 197 | 13.37 |
| Urban | 3 | Pothole | 162 | 11.00 |
| Urban | 4 | Pothole; Surface depression | 89 | 6.04 |
| Urban | 5 | Road patch failure | 80 | 5.43 |
| Urban | 6 | Manhole; Surface depression | 76 | 5.16 |
| Highway | 1 | Surface depression | 4 | 57.14 |
| Highway | 2 | Pothole | 1 | 14.29 |
| Highway | 3 | Road patch failure; Surface depression | 1 | 14.29 |
| Highway | 4 | Surface depression; Road patch failure | 1 | 14.29 |
| Rural | 1 | Surface depression | 195 | 62.90 |
| Rural | 2 | Road patch failure | 40 | 12.90 |
| Rural | 3 | Pothole | 28 | 9.03 |
| Rural | 4 | Missing | 10 | 3.23 |
| Rural | 5 | Pothole; Surface depression | 9 | 2.90 |
| Rural | 6 | Surface depression; Road patch failure | 6 | 1.94 |
| Low light | 1 | Missing | 6 | 35.29 |
| Low light | 2 | Manhole | 3 | 17.65 |
| Low light | 3 | Surface depression | 3 | 17.65 |
| Low light | 4 | Road patch failure | 1 | 5.88 |
| Low light | 5 | Manhole; Surface depression | 1 | 5.88 |
| Low light | 6 | Road patch failure; Crack | 1 | 5.88 |
| Tiny objects (<1%) | 1 | Surface depression | 40 | 27.03 |
| Tiny objects (<1%) | 2 | Pothole | 31 | 20.95 |
| Tiny objects (<1%) | 3 | Manhole | 31 | 20.95 |
| Tiny objects (<1%) | 4 | Road patch failure | 29 | 19.59 |
| Tiny objects (<1%) | 5 | Pothole; Manhole | 3 | 2.03 |
| Tiny objects (<1%) | 6 | Road patch failure; Surface depression | 2 | 1.35 |
| Small objects (1%-<5%) | 1 | Surface depression | 242 | 35.75 |
| Small objects (1%-<5%) | 2 | Manhole | 116 | 17.13 |
| Small objects (1%-<5%) | 3 | Pothole | 86 | 12.70 |
| Small objects (1%-<5%) | 4 | Road patch failure | 74 | 10.93 |
| Small objects (1%-<5%) | 5 | Pothole; Surface depression | 42 | 6.20 |
| Small objects (1%-<5%) | 6 | Manhole; Surface depression | 33 | 4.87 |
| Medium objects (5%-<15%) | 1 | Surface depression | 387 | 48.93 |
| Medium objects (5%-<15%) | 2 | Manhole | 76 | 9.61 |
| Medium objects (5%-<15%) | 3 | Pothole | 54 | 6.83 |
| Medium objects (5%-<15%) | 4 | Manhole; Surface depression | 52 | 6.57 |
| Medium objects (5%-<15%) | 5 | Pothole; Surface depression | 40 | 5.06 |
| Medium objects (5%-<15%) | 6 | Road patch failure | 36 | 4.55 |
| Large objects (>=15%) | 1 | Surface depression | 212 | 57.61 |
| Large objects (>=15%) | 2 | Pothole | 40 | 10.87 |
| Large objects (>=15%) | 3 | Manhole; Surface depression | 27 | 7.34 |
| Large objects (>=15%) | 4 | Pothole; Surface depression | 22 | 5.98 |
| Large objects (>=15%) | 5 | Manhole | 10 | 2.72 |
| Large objects (>=15%) | 6 | Surface depression; Manhole | 7 | 1.90 |
| State: Delhi | 1 | Surface depression | 367 | 47.11 |
| State: Delhi | 2 | Pothole | 92 | 11.81 |
| State: Delhi | 3 | Pothole; Surface depression | 66 | 8.47 |
| State: Delhi | 4 | Road patch failure | 42 | 5.39 |
| State: Delhi | 5 | Missing | 42 | 5.39 |
| State: Delhi | 6 | Surface depression; Pothole | 33 | 4.24 |
| State: Maharashtra | 1 | Surface depression | 65 | 28.63 |
| State: Maharashtra | 2 | Manhole | 49 | 21.59 |
| State: Maharashtra | 3 | Manhole; Surface depression | 28 | 12.33 |
| State: Maharashtra | 4 | Pothole | 18 | 7.93 |
| State: Maharashtra | 5 | Missing | 10 | 4.41 |
| State: Maharashtra | 6 | Manhole; Road patch failure | 7 | 3.08 |
| State: Tamil Nadu | 1 | Surface depression | 449 | 42.04 |
| State: Tamil Nadu | 2 | Manhole | 163 | 15.26 |
| State: Tamil Nadu | 3 | Pothole | 101 | 9.46 |
| State: Tamil Nadu | 4 | Road patch failure | 98 | 9.18 |
| State: Tamil Nadu | 5 | Manhole; Surface depression | 65 | 6.09 |
| State: Tamil Nadu | 6 | Missing | 38 | 3.56 |

The robustness slices show that surface depression is dominant in most settings, while night images are more manhole-heavy and wet-road images are more pothole-heavy. This helps demonstrate dataset diversity across environmental and geographic conditions, while also identifying low-sample slices that should be expanded in future dataset versions.

> **Note:** Robustness slices include night, wet-road/rain proxy, shadow, urban, highway, rural, low-light, object-size groups, and state-wise subsets.

## Phase 7 - Dataset Split Strategy

### Random-size Reference

| Split | Percentage | Images |
| --- | ---: | ---: |
| Total | 100% | 2074 |
| Train | 70% | 1452 |
| Val | 15% | 311 |
| Test | 15% | 311 |

### Region-wise Split

| Metric | Count |
| --- | ---: |
| Unique regions | 5 |
| Train regions | 4 |
| Val regions | 1 |
| Test regions | 0 |

### State-wise Split

| Metric | Count |
| --- | ---: |
| Unique states | 3 |
| Train states | 2 |
| Val states | 0 |
| Test states | 1 |

### Leave-One-State-Out Evaluation

| Held-out State | Train Images | Test Images |
| --- | ---: | ---: |
| Delhi | 1295 | 779 |
| Maharashtra | 1847 | 227 |
| Tamil Nadu | 1006 | 1068 |

> **Final Note:** A script named `package_dataset_splits.py` is provided for creating the dataset split ZIP files locally. Users are requested to run the script as follows:
>
> ```bash
> python package_dataset_splits.py --allow-missing-images
> ```
>
> This will create each of the dataset splits mentioned above appropriately. Since each generated ZIP file is approximately 3 GB to 5 GB in size, the ZIP files have not been uploaded officially. The script is provided so users can generate the split archives on their own system when needed.

## Phase 8 - Bias Analysis 

### Dataset Bias Analysis Summary

| Dataset Component | Count |
| --- | ---: |
| Unique raw images | 2074 |
| Annotation instances | 3391 |
| Layer 2 metadata rows | 2074 |
| Layer 3 geospatial rows | 2074 |

The bias analysis was performed across raw images, annotation instances, semantic metadata, and geospatial metadata to identify class, regional, environmental, and object-size skews.

### 8.1 - Class Imbalance Indicators

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class - annotation instances | annotation instances | 3391 | 6 | 1837 | 24 | 76.5417 | 1.0844 | 0.6941 | 54.17 |
| Class - unique images | unique images | 2518 | 6 | 1294 | 24 | 53.9167 | 1.0239 | 0.7189 | 51.39 |
| Class - Layer 2 rows | metadata rows | 2074 | 47 | 881 | 1 | 881.0000 | 3.0357 | 0.5651 | 42.48 |

Surface depression dominates the dataset, while crack and miscellaneous abnormality are highly underrepresented. This may bias trained models toward frequent abnormality classes and reduce sensitivity for rare classes.

### 8.2 - Unique-image Class Distribution

| Rank | Abnormality Type | Count | Percentage | Unit |
| ---: | --- | ---: | ---: | --- |
| 1 | Surface depression | 1294 | 51.39 | unique images |
| 2 | Manhole | 476 | 18.90 | unique images |
| 3 | Pothole | 450 | 17.87 | unique images |
| 4 | Road patch failure | 244 | 9.69 | unique images |
| 5 | Crack | 30 | 1.19 | unique images |
| 6 | Miscellaneous abnormality | 24 | 0.95 | unique images |

The same class imbalance pattern appears at the unique-image level, confirming that the skew is not only due to repeated annotations but also due to image-level representation.

### 8.3 - Regional Imbalance Indicators

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Region - state | images | 2074 | 3 | 1068 | 227 | 4.7048 | 0.5047 | 0.8663 | 51.49 |
| Region - district | images | 2074 | 5 | 779 | 227 | 3.4317 | 0.4667 | 0.9391 | 37.56 |

The dataset has moderate regional concentration, with Tamil Nadu contributing over half of all images and North West Delhi forming the largest district-level group.

### 8.4 - State-District Distribution

| State | District | Image Count | Dataset Percentage |
| --- | --- | ---: | ---: |
| Delhi | North West | 779 | 37.56 |
| Tamil Nadu | Chennai | 434 | 20.93 |
| Tamil Nadu | Chengalpattu | 319 | 15.38 |
| Tamil Nadu | Tiruvallur | 315 | 15.19 |
| Maharashtra | Sangli | 227 | 10.95 |

This shows that each state is represented by a limited number of districts, so geographic generalization outside these districts should be treated carefully.

### 8.5 - Seasonal Imbalance

| Rank | Season | Count | Percentage | Unit |
| ---: | --- | ---: | ---: | --- |
| 1 | summer | 1321 | 63.69 | images |
| 2 | winter | 607 | 29.27 | images |
| 3 | monsoon | 128 | 6.17 | images |
| 4 | post_monsoon | 18 | 0.87 | images |

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seasonal imbalance | images | 2074 | 4 | 1321 | 18 | 73.3889 | 0.9904 | 0.6204 | 63.69 |

The dataset is strongly summer-heavy, while monsoon and post-monsoon samples are limited. This may reduce robustness for wet-season road appearances and seasonal lighting differences.

### 8.6 - Lighting Imbalance Indicators

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lighting imbalance | images | 2074 | 4 | 1704 | 17 | 100.2353 | 1.3344 | 0.4219 | 82.16 |

Daylight images dominate the dataset, so model performance may be stronger under daylight conditions than in low-light, shadow, or night-time scenarios.

### 8.7 - Weather Imbalance Indicators

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather imbalance | images | 2074 | 3 | 1835 | 54 | 33.9815 | 1.1723 | 0.3813 | 88.48 |

Clear and dry conditions dominate the dataset, so the dataset may underrepresent visual road-abnormality patterns under cloudy or wet-road conditions.

### 8.8 - Object-size Distribution

| Rank | Object Size | Count | Percentage | Unit |
| ---: | --- | ---: | ---: | --- |
| 1 | Small (1%-<5%) | 1454 | 42.88 | annotation instances |
| 2 | Medium (5%-<15%) | 994 | 29.31 | annotation instances |
| 3 | Tiny (<1%) | 560 | 16.51 | annotation instances |
| 4 | Large (>=15%) | 383 | 11.29 | annotation instances |

Small and medium objects form most annotations, while large objects are less frequent. This may influence object-detection performance across different abnormality sizes.

### 8.9 - Relative-area Descriptive Statistics

| Metric | Value |
| --- | ---: |
| Valid boxes | 3391.0000 |
| Invalid or missing boxes | 0.0000 |
| Mean relative area | 0.0715 |
| Median relative area | 0.0378 |
| Standard deviation relative area | 0.1030 |
| Minimum relative area | 0.0006 |
| 25th percentile | 0.0145 |
| 75th percentile | 0.0846 |
| 90th percentile | 0.1634 |
| 95th percentile | 0.2569 |
| Maximum relative area | 0.9746 |

The median relative area is much smaller than the mean, indicating a right-skewed object-size distribution with many small objects and a smaller number of very large abnormalities.

### 8.10 - Object Size by Abnormality Class

| Abnormality Type | Tiny (<1%) | Small (1%-<5%) | Medium (5%-<15%) | Large (>=15%) | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Surface depression | 166 | 705 | 680 | 286 | 1837 |
| Pothole | 190 | 279 | 114 | 56 | 639 |
| Manhole | 120 | 266 | 118 | 19 | 523 |
| Road patch failure | 81 | 187 | 58 | 9 | 335 |
| Crack | 2 | 13 | 9 | 9 | 33 |
| Miscellaneous abnormality | 1 | 4 | 15 | 4 | 24 |

Surface depression contributes the largest number of medium and large objects, while potholes, manholes, and road patch failures are more concentrated in tiny and small sizes.

### 8.11 - Within-class Object-size Percentages

| Abnormality Type | Tiny (<1%) | Small (1%-<5%) | Medium (5%-<15%) | Large (>=15%) | Total (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crack | 6.06 | 39.39 | 27.27 | 27.27 | 99.99 |
| Manhole | 22.94 | 50.86 | 22.56 | 3.63 | 99.99 |
| Miscellaneous abnormality | 4.17 | 16.67 | 62.50 | 16.67 | 100.01 |
| Pothole | 29.73 | 43.66 | 17.84 | 8.76 | 99.99 |
| Road patch failure | 24.18 | 55.82 | 17.31 | 2.69 | 100.00 |
| Surface depression | 9.04 | 38.38 | 37.02 | 15.57 | 100.01 |

Within-class size profiles differ noticeably: potholes and road patch failures are often tiny or small, while miscellaneous abnormalities are mostly medium-sized in this dataset.

### 8.12 - Object-size Imbalance Indicators

| Dimension | Unit | Total | Categories | Largest Count | Smallest Count | Max-Min Ratio | Coefficient of Variation | Normalized Entropy | Largest Share (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Object size | annotation instances | 3391 | 4 | 1454 | 383 | 3.7963 | 0.4891 | 0.9136 | 42.88 |

Object-size imbalance is present but less severe than class, lighting, weather, or seasonal imbalance. The distribution still favors small objects, which should be considered when evaluating detection performance.

> **Note:** Class distribution, state/district distribution, weather distribution, lighting distribution, and bounding-box summary statistics are also reported in earlier sections of this document. Phase 8 focuses on the bias implications and additional imbalance indicators.

## Phase 9 - Reproducibility

This section records the parameters and environment details required to reproduce the dataset validation, split generation, and benchmark experiments.

### 9.1 - Reproducibility Parameters

| Item | Value |
| --- | --- |
| Random seed | 42 |
| Dataset split seed | 42 |
| Random split ratio | Train 70%, Validation 15%, Test 15% |
| Image classification split ratio | Train 70%, Validation 15%, Test 15% |
| Severity prediction split ratio | Train 70%, Validation 15%, Test 15% |
| Image size for classification models | 224 × 224 |
| Image size for object detection | 640 × 640 |
| Batch size for classification models | 16 |
| Batch size for object detection | 16 |
| Classification epochs | 30 |
| Severity prediction epochs | 30 |
| Object detection epochs | 100 |
| Early stopping patience - classification/severity | 7 |
| Early stopping patience - object detection | 30 |
| Classification threshold | 0.5 |
| Validation fraction | 0.15 |
| Test fraction | 0.15 |

### 9.2 - Training Configuration

| Task | Models | Loss Function | Optimizer | Learning Rate | Weight Decay | Scheduler |
| --- | --- | --- | --- | ---: | ---: | --- |
| Image classification | ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16 | BCEWithLogitsLoss with positive-class weights | AdamW | 0.0003 | 0.0001 | ReduceLROnPlateau, factor 0.3, patience 2 |
| Severity prediction | ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16 | CrossEntropyLoss with class weights | AdamW | 0.0003 | 0.0001 | ReduceLROnPlateau, factor 0.3, patience 2 |
| Object detection | YOLO-based detector through Ultralytics | Ultralytics YOLO training loss | Ultralytics default optimizer/training configuration | Ultralytics default | Ultralytics default | Ultralytics default |

### 9.3 - Hardware and Runtime

| Item | Value |
| --- | --- |
| Device selection | Auto-selected in scripts |
| Supported training devices | CUDA GPU, Apple MPS, or CPU |
| Mixed precision | Enabled when CUDA is available |
| DataLoader workers - classification/severity | 4 |
| DataLoader workers - object detection | 8 |
| cuDNN benchmark | Enabled |
| Current inspected Python version | Python 3.14.5 |

### 9.4 - Software Requirements

The project dependencies are listed in `requirements.txt`.

| Package |
| --- |
| pandas |
| Pillow |
| pillow-heif |
| imagehash |
| numpy |
| matplotlib |
| matplotlib-venn |
| scipy |
| ultralytics |
| torch |
| torchvision |
| scikit-learn |
| tabulate |
| iterative-stratification |

> **Note:** Package names are documented in `requirements.txt`. Exact package versions should be recorded from the execution environment using `pip freeze` when the experiments are rerun.

### 9.5 - Repository and Code Version

| Item | Value |
| --- | --- |
| GitHub repository | `https://github.com/brpuneet898/RoadAbnormalityDataset.git` |
| Branch | `main` |

### 9.6 - Dataset Metadata Checksums

| File | SHA-256 Checksum |
| --- | --- |
| `layer_0_raw_images/layer_0_raw_image_metadata.csv` | `4923B85C0E2FC1CDFA3E932E3286CEEE17A1A26DBB8B7C76A754096F89A708E5` |
| `layer_1_annotations/layer_1_annotation_metadata.csv` | `6AAC8F6888105A806AE145A53F9528ADD74BAA30342AA1987885D11028781E8C` |
| `layer_2_metadata/layer_2_semantic_metadata.csv` | `5C7C35451D2F21725F08CCC1B5EE5AAAADB51CFF7ACDD344E58858415680B632` |
| `layer_3_geospatial/layer_3_geospatial_metadata.csv` | `D9BECA0BC247EFBC159FB1ACEBCCE430AF08D268436708FF4513805286530712` |

### 9.7 - Reproducibility Notes

- All scripts use a default random seed of `42` for deterministic data splitting and repeatable experiment setup.
- Training scripts automatically select CUDA, MPS, or CPU depending on the available hardware.
- Benchmark models, generated figures, validation outputs, and dataset splits can be regenerated from the scripts included in the repository.
- Dataset split archives are intentionally not uploaded because of their large size; users can regenerate them using the Phase 7 packaging script.
- Recording exact package versions with `pip freeze` and preserving the listed metadata checksums is recommended for full experiment reproducibility.
