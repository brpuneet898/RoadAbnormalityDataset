from pathlib import Path
import pandas as pd

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ROOT = Path(__file__).resolve().parents[2]

candidate_files = [
    ROOT / "layer_3_geospatial" / "layer_3_geospatial_metadata.csv",
    ROOT / "layer_0_raw_images" / "layer_0_raw_image_metadata.csv",
]

metadata_file = None
for f in candidate_files:
    if f.exists():
        metadata_file = f
        break

if metadata_file is None:
    raise FileNotFoundError(
        "Could not locate metadata CSV.\n"
        "Expected one of:\n"
        + "\n".join(str(x) for x in candidate_files)
    )

df = pd.read_csv(metadata_file)

required = {"image_id", "collection_state"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if "district" in df.columns:
    df["region"] = (
        df["collection_state"].astype(str)
        + " :: "
        + df["district"].astype(str)
    )
else:
    print("\nWARNING: No district column found.")
    print("Region-wise statistics cannot be estimated.\n")
    df["region"] = None

n = len(df)

def print_ratio(title, total):
    train = round(total * TRAIN_RATIO)
    val = round(total * VAL_RATIO)
    test = total - train - val

    print(f"\n{title}")
    print("-"*len(title))
    print(f"Total images : {total}")
    print(f"Train (70%)  : {train}")
    print(f"Val   (15%)  : {val}")
    print(f"Test  (15%)  : {test}")

print("="*70)
print("PHASE VII DATASET SPLIT SIZE ESTIMATION")
print("="*70)

print_ratio("Random-size reference", n)

if df["region"].notna().any():
    regions = sorted(df["region"].unique())

    train_regions = round(len(regions)*TRAIN_RATIO)
    val_regions = round(len(regions)*VAL_RATIO)
    test_regions = len(regions)-train_regions-val_regions

    print("\nRegion-wise Split")
    print("-----------------")
    print(f"Unique regions : {len(regions)}")
    print(f"Train regions  : {train_regions}")
    print(f"Val regions    : {val_regions}")
    print(f"Test regions   : {test_regions}")

states = sorted(df["collection_state"].unique())

train_states = round(len(states)*TRAIN_RATIO)
val_states = round(len(states)*VAL_RATIO)
test_states = len(states)-train_states-val_states

print("\nState-wise Split")
print("----------------")
print(f"Unique states : {len(states)}")
print(f"Train states  : {train_states}")
print(f"Val states    : {val_states}")
print(f"Test states   : {test_states}")

print("\nLeave-One-State-Out Evaluation")
print("------------------------------")

for state in states:
    test = (df["collection_state"] == state).sum()
    train = n - test
    print(f"{state:<20} Train={train:<8} Test={test}")

print("\nDone.")
