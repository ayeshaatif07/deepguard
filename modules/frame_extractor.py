import cv2
import numpy as np

def extract_frames(video_path, num_frames=10):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Video file {video_path} has 0 frames")

    # Calculate indices for evenly spaced frames
    if total_frames < num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    target_indices = set(int(i) for i in indices)

    # Decode straight through once, keeping only the sampled frames; seeking
    # per frame forces a keyframe rewind and is far slower.
    frames_by_index = {}
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx in target_indices:
            ret2, frame = cap.retrieve()
            if ret2:
                frames_by_index[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
        if len(frames_by_index) == len(target_indices):
            break

    cap.release()

    # Preserve the original evenly-spaced, ascending order
    frames = [frames_by_index[i] for i in sorted(frames_by_index) if i in frames_by_index]
    return frames

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        frames = extract_frames(sys.argv[1])
        print(f"Extracted {len(frames)} frames from {sys.argv[1]}")
