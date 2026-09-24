"""Builds a small LibriSpeech subset in the CSV and audio-folder format the four ASR benchmark
scripts expect.
"""

import argparse
import io
import os
import random

import pandas as pd
import soundfile as sf

# Fixed, known sizes of each split.
SPLIT_SIZES = {"clean": 2620, "other": 2939}

DATASETS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets'))


def main():
    parser = argparse.ArgumentParser(description="Build a small LibriSpeech subset (test-clean or test-other) for ASR benchmarking.")
    parser.add_argument("--config", default="other", choices=["clean", "other"], help="Which LibriSpeech split to sample from - 'other' (noisier, the one currently used) is the default.")
    parser.add_argument("--n", type=int, default=500, help="Number of clips to sample (fixed random.sample, seed 42) from the full split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_size = SPLIT_SIZES[args.config]
    out_root = os.path.join(DATASETS_ROOT, f"librispeech-test-{args.config}")
    audio_dir = os.path.join(out_root, "audio")
    csv_path = os.path.join(out_root, "librispeech_transcripts.csv")
    os.makedirs(audio_dir, exist_ok=True)

    random.seed(args.seed)
    selected_indices = set(random.sample(range(split_size), min(args.n, split_size)))
    print(f"Sampling {len(selected_indices)} of {split_size} test-{args.config} clips (seed={args.seed})")

    from datasets import Audio, load_dataset
    print(f"Streaming librispeech_asr (test-{args.config} split) - only the sampled clips get downloaded...")
    ds = load_dataset("openslr/librispeech_asr", args.config, split="test", streaming=True, trust_remote_code=True)
    # Decode raw bytes ourselves via soundfile instead of the datasets library's default
    # torchcodec-based decoder torchcodec here is built against an older FFmpeg ABI
    ds = ds.cast_column("audio", Audio(decode=False))

    records = []
    for i, ex in enumerate(ds):
        if i not in selected_indices:
            continue
        audio_bytes = ex["audio"]["bytes"]
        array, samplerate = sf.read(io.BytesIO(audio_bytes))
        filename = f"{ex['id']}.wav"
        out_path = os.path.join(audio_dir, filename)
        sf.write(out_path, array, samplerate)
        records.append({"audio_filename": filename, "transcript": ex["text"].strip()})
        if len(records) % 50 == 0:
            print(f"  [{len(records)}/{len(selected_indices)}] {filename}")
        if len(records) >= len(selected_indices):
            break

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} clips to: {audio_dir}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
