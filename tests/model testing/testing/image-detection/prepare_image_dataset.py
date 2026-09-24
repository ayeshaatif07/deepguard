"""Downloads a stratified sample from the gated HuggingFace dataset prithivMLmods/AI-vs-
Deepfake-vs-Real (image-classification, 3 classes: Artificial, Deepfake.
"""

import argparse
import io
import os
import sys

import pandas as pd

REPO_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real"
SHARDS = ["0000.parquet", "0001.parquet", "0002.parquet", "0003.parquet"]
CLASS_NAMES = ["Artificial", "Deepfake", "Real"]  # index = label int, per the dataset's ClassLabel feature
SEED = 42

# This file lives in Tests/testing/image-detection/, so Tests/Datasets/
# is two directories up (image-detection/ -> testing/ -> Tests/).
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "Datasets", "image-detection")


def main():
    parser = argparse.ArgumentParser(description="Prepare a stratified sample of the AI-vs-Deepfake-vs-Real image dataset.")
    parser.add_argument("--token", required=True, help="HuggingFace access token (dataset is gated).")
    parser.add_argument("--per-class", type=int, default=150, help="Number of images to sample per class (default 150).")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    print(f"Downloading {len(SHARDS)} parquet shards from {REPO_ID} (cached after first run)...")
    shard_paths = []
    for shard in SHARDS:
        p = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset", token=args.token)
        shard_paths.append(p)
        print(f"  {shard} ready")

    # Load every row's (bytes, path, label) the parquet files together are ~2GB but this is
    # metadata + compressed image bytes already local on disk at this point, not a network
    print("\nReading all rows to sample from...")
    frames = []
    for p in shard_paths:
        tbl = pq.ParquetFile(p).read(columns=["image", "label"])
        frames.append(tbl.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["class_name"] = df["label"].map(lambda i: CLASS_NAMES[i])
    # A stable, run-invariant id for each row (its position in the full 9,999-row
    # concatenation, which is deterministic same 4 parquet shards read in the same order every
    df["global_id"] = df.index
    print(f"Loaded {len(df)} rows: {df['class_name'].value_counts().to_dict()}")

    # Stratified sample, same seeded-reproducibility convention as the video dataset's
    # load_dataset_csv().
    per_class_samples = []
    for class_name in CLASS_NAMES:
        class_df = df[df["class_name"] == class_name]
        n = min(args.per_class, len(class_df))
        per_class_samples.append(class_df.sample(n=n, random_state=SEED))
    sampled = (
        pd.concat(per_class_samples, ignore_index=True)
        .sample(frac=1, random_state=SEED)  # shuffle so classes aren't in contiguous blocks
        .reset_index(drop=True)
    )
    print(f"\nSampled {len(sampled)} images ({args.per_class} per class, seed={SEED})")

    images_dir = os.path.join(OUTPUT_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    for progress_i, (_, row) in enumerate(sampled.iterrows()):
        gid = row["global_id"]
        class_name = row["class_name"]
        class_dir = os.path.join(images_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_struct = row["image"]
        img_bytes = img_struct["bytes"]
        orig_name = img_struct["path"] or f"{class_name.lower()}_{gid}.jpg"
        # Sanitize + disambiguate in case of duplicate original filenames across shards (e.g.
        # multiple "a1 (1).jpg") using the row's stable global_id rather than a per-run
        safe_name = f"{gid:04d}_{os.path.basename(orig_name)}"

        # Decode + re-save via PIL rather than writing raw bytes directly: confirms every file
        # is a valid, openable image before it's recorded in the CSV, catching any corrupt
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        out_path = os.path.join(class_dir, safe_name)
        im.save(out_path, format="JPEG", quality=95)

        rel_path = os.path.join(class_name, safe_name)
        records.append({"file_path": rel_path, "label": class_name})

        if (progress_i + 1) % 50 == 0:
            print(f"  extracted {progress_i + 1}/{len(sampled)}")

    out_df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_ROOT, f"image_detection_{len(out_df)}_sample.csv")
    out_df.to_csv(csv_path, index=False)

    print(f"\nExtraction complete.")
    print(f"Images saved under: {images_dir}/")
    print(f"CSV saved to: {csv_path}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
