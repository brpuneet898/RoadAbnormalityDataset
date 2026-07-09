from pathlib import Path
from itertools import combinations
from PIL import Image, UnidentifiedImageError
import imagehash
import numpy as np

LAYER_0_DIR = Path("../../../layer_0_raw_images")

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".heic"}
IGNORED_EXTENSIONS = {".md", ".csv"}

PHASH_THRESHOLD = 2
EMBEDDING_COSINE_THRESHOLD = 0.995

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)

    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denom)


def simple_image_embedding(image_path, size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.flatten()


def main():
    all_files = [p for p in LAYER_0_DIR.rglob("*") if p.is_file()]

    image_files = [
        p for p in all_files
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]

    invalid_format_files = [
        p for p in all_files
        if (
            p.suffix.lower() not in SUPPORTED_FORMATS
            and p.suffix.lower() not in IGNORED_EXTENSIONS
        )
    ]

    hashes = {}
    embeddings = {}
    unreadable_images = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                hashes[image_path] = imagehash.phash(img)

            embeddings[image_path] = simple_image_embedding(image_path)

        except (UnidentifiedImageError, OSError, ValueError) as e:
            unreadable_images.append((image_path.name, str(e)))

    exact_duplicates = []
    near_duplicates = []

    valid_images = list(hashes.keys())

    for img1, img2 in combinations(valid_images, 2):
        phash_distance = hashes[img1] - hashes[img2]
        cosine_score = cosine_similarity(embeddings[img1], embeddings[img2])

        if phash_distance == 0:
            exact_duplicates.append(
                (img1.name, img2.name, phash_distance, cosine_score)
            )

        elif (
            phash_distance <= PHASH_THRESHOLD
            and cosine_score >= EMBEDDING_COSINE_THRESHOLD
        ):
            near_duplicates.append(
                (img1.name, img2.name, phash_distance, cosine_score)
            )

    print("\nDUPLICATE IMAGE DETECTION")
    print("=" * 60)

    print(f"Total image files scanned      : {len(image_files)}")
    print(f"Successfully processed images  : {len(valid_images)}")
    print(f"Unreadable images              : {len(unreadable_images)}")
    print(f"Invalid format files ignored   : {len(invalid_format_files)}")
    print(f"Exact duplicates detected      : {len(exact_duplicates)}")
    print(f"Near duplicates detected       : {len(near_duplicates)}")
    print(f"pHash threshold used           : {PHASH_THRESHOLD}")
    print(f"Cosine similarity threshold    : {EMBEDDING_COSINE_THRESHOLD}")

if __name__ == "__main__":
    main()