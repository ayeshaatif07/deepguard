"""Benchmarks Naman712/Deep-fake-detection, a ResNeXt50 + LSTM sequence classifier."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel, extract_frames, WEIGHTS_DIR


MODEL_KEY = "naman"


class NamanModel(BaseModel):
    name = "Naman712/Deep-fake-detection"

    def __init__(self, device):
        self.device = device
        self.model = None
        self.im_size = 112
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self._face_cascade = None
        self.faces_found = 0
        self.faces_missed = 0

    def load(self):
        import torch
        import cv2
        naman_dir = os.path.join(WEIGHTS_DIR, 'naman')
        sys.path.insert(0, naman_dir)
        from modeling import DeepFakeDetector  # type: ignore

        self.torch = torch
        self.model = DeepFakeDetector(num_classes=2)
        weights_path = os.path.join(naman_dir, 'model_87_acc_20_frames_final_data.pt')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def _crop_face(self, frame_rgb):
        """Detect the largest face and crop to it with padding (mirrors the original repo's
        face_recognition.face_locations() + padding crop).
        """
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            self.faces_missed += 1
            return frame_rgb

        # Largest detected face (by area) - matches "faces[0]" behavior in
        # the original repo closely enough for a single-subject clip.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        padding = int(0.25 * max(w, h))
        top = max(0, y - padding)
        left = max(0, x - padding)
        bottom = min(frame_rgb.shape[0], y + h + padding)
        right = min(frame_rgb.shape[1], x + w + padding)
        self.faces_found += 1
        return frame_rgb[top:bottom, left:right]

    def _preprocess_frame(self, frame_rgb):
        from PIL import Image
        face = self._crop_face(frame_rgb)
        img = Image.fromarray(face).resize((self.im_size, self.im_size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)
        return self.torch.tensor(arr, dtype=self.torch.float32)

    def score_video(self, video_path, num_frames=20):
        torch = self.torch
        frames = extract_frames(video_path, num_frames=num_frames)
        if not frames:
            raise ValueError("No frames extracted")
        if len(frames) > num_frames:
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in idx]
        elif len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1])

        processed = [self._preprocess_frame(f) for f in frames]
        batch = torch.stack(processed).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # modules/deepfake_scorer.py (production) comments "Index 0 = FAKE, 1 = REAL" using
        # probs[0] as the fake score, matching that comment.
        return float(probs[0])


def main():
    parser = common.build_arg_parser(
        "Evaluate Naman712/Deep-fake-detection (ResNeXt50+LSTM) on a labelled video dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, NamanModel, args)


if __name__ == "__main__":
    main()
