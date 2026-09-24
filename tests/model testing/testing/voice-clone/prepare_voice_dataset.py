"""Downloads a stratified sample from the garystafford/deepfake-audio-detection dataset and
writes the benchmark CSV.
"""

import argparse
import os
import sys

import pandas as pd

TESTING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TESTS_ROOT = os.path.abspath(os.path.join(TESTING_ROOT, '..'))
DATASET_DIR = os.path.join(TESTS_ROOT, 'Datasets', 'voice-clone')

REPO_ID = "garystafford/deepfake-audio-detection"


def main():
    parser = argparse.ArgumentParser(description="Prepare a stratified sample of the deepfake-audio-detection dataset.")
    parser.add_argument("--per-class", type=int, default=100, help="Number of samples per class (real/fake) to extract.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Download individual .flac files directly via huggingface_hub instead of the `datasets`
    # library's audio decoding path that path requires `torchcodec`, which failed to load here
    from huggingface_hub import HfApi, hf_hub_download
    import random

    api = HfApi()
    info = api.dataset_info(REPO_ID)
    all_files = [s.rfilename for s in info.siblings]

    rng = random.Random(args.seed)
    audio_dir = os.path.join(DATASET_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    rows = []
    for cls in ["real", "fake"]:
        cls_files = sorted(f for f in all_files if f.startswith(f"{cls}/"))
        chosen = rng.sample(cls_files, min(args.per_class, len(cls_files)))
        os.makedirs(os.path.join(audio_dir, cls), exist_ok=True)
        for count, rel_path in enumerate(chosen):
            local_path = hf_hub_download(REPO_ID, rel_path, repo_type="dataset")
            filename = os.path.basename(rel_path)
            dest = os.path.join(audio_dir, cls, filename)
            import shutil
            shutil.copyfile(local_path, dest)
            rows.append({"file_path": os.path.join(cls, filename), "label": cls})
            if (count + 1) % 25 == 0:
                print(f"  [{cls}] downloaded {count + 1}/{len(chosen)}")

    sampled = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    rows = sampled.to_dict("records")

    out_df = pd.DataFrame(rows)
    n_total = len(out_df)
    csv_path = os.path.join(DATASET_DIR, f"voice_clone_{n_total}_sample.csv")
    out_df.to_csv(csv_path, index=False)

    print(f"\nWrote {n_total} samples to {audio_dir}/")
    print(f"CSV: {csv_path}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
