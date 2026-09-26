# Signal 1 dataset — FaceForensics++ (video deepfake detection)

All video data here comes from **FaceForensics++**, obtained through the
dataset's official research-access process run by the Technical
University of Munich, and used under the FaceForensics Terms of Use
(non-commercial research and educational use).

> Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J. and
> Nießner, M. (2019) 'FaceForensics++: Learning to Detect Manipulated
> Facial Images', *ICCV*.

## Layout

```
video-detection/
├── videos/REAL/                              ← 75 genuine clips
├── videos/FAKE/                              ← 75 manipulated clips
├── official_ff_download/                     ← official download, c23 (HQ)
└── video_detection_150_official_sample.csv   ← manifest (75 real / 75 fake)
```

`videos/REAL/` and `videos/FAKE/` are browsable copies of the exact 150
clips used in every benchmark, taken from `official_ff_download/` (c23).
Each file is named `<source>_<clip>.mp4`, so its origin is readable
without opening the manifest:

| Class | Source | Clips |
|---|---|---|
| REAL | `youtube` (original, unmanipulated) | 75 |
| FAKE | `Deepfakes` | 18 |
| FAKE | `Face2Face` | 16 |
| FAKE | `NeuralTextures` | 15 |
| FAKE | `FaceShifter` | 9 |
| FAKE | `FaceSwap` | 9 |
| FAKE | `DeepFakeDetection` | 8 |

The fake half is stratified across all six manipulation methods FF++
provides, so no single forgery technique dominates the benchmark.

## Compression level

All clips are the **c23 (HQ)** encoding — the H.264 quality setting
FaceForensics++ publishes as its standard benchmark level, and the one
results are reported against in the literature (Section 2.3.1). Every
model comparison in Section 5.2.1 uses it.
