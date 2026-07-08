# Layer 2 — Semantic Metadata

- **Purpose:** Image-level descriptions of abnormality and scene conditions.
- **Contents:** `layer_2_semantic_metadata.csv`.
- **Rows:** 2,074—one for every Layer 0 image.
- **Key:** `image_id` uniquely links each row to Layers 0, 1, and 3.
- **Columns and allowed values:**
  - `abnormality_type` — one or more Layer 1 classes; multiple classes are separated by `; `.
  - `severity` — `minor`, `moderate`, or `severe`.
  - `road_type` — `highway`, `residential`, `rural`, or `urban`.
  - `weather` — `clear_dry`, `cloudy`, or `wet_road`.
  - `lighting` — `daylight`, `low_light`, `night`, or `shadow`.
  - `occlusion` — `none`, `partial`, `heavy`, `object_occluded`, or `vehicle_occluded`.
  - `shape` — `circular`, `clustered`, `irregular`, `longitudinal`, or `transverse`.
- **Abnormality classes:** Crack, Manhole, Miscellaneous abnormality, Pothole, Road patch failure, and Surface depression.
- **Missing values:** 90 `abnormality_type` cells are blank because their matching Layer 1 label files contain no bounding boxes; all other fields are complete.
- **Granularity:** Scene attributes are image-level, not per bounding box.
- **Usage:** Use for classification, stratified sampling, filtering, dataset analysis, or enriching Layer 1 detection tasks.
- **Note:** An image containing multiple abnormality classes still occupies one CSV row.
