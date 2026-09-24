"""Benchmarks the EfficientNet-B7 noisy-student frame classifier from selimsef's DFDC solution."""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel, extract_frames, WEIGHTS_DIR


MODEL_KEY = "efficientnet"
CHECKPOINT_URL = (
    "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/"
    "0.0.1/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19"
)
CHECKPOINT_FILENAME = "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19.pth"
INPUT_SIZE = 380  # native size used for B7 in the official inference script


class DeepFakeClassifierModel(BaseModel):
    name = "EfficientNet-B7-NS (selimsef, DFDC Challenge solution)"

    def __init__(self, device):
        self.device = device
        self.model = None
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self._face_cascade = None
        self.faces_found = 0
        self.faces_missed = 0

    def load(self):
        import torch
        import torch.nn as nn
        import cv2
        import timm

        self.torch = torch

        class DeepFakeClassifier(nn.Module):
            """Vendored from selimsef/dfdc_deepfake_challenge training/zoo/classifiers.py
            matches the released checkpoints' state_dict keys exactly.
            """

            def __init__(self):
                super().__init__()
                # Default num_classes (1000) to match the original code's encoder exactly it
                # keeps timm's default ImageNet classifier head as an unused module (only
                self.encoder = timm.create_model("tf_efficientnet_b7_ns", pretrained=False)
                self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(0.0)
                self.fc = nn.Linear(2560, 1)

            def forward(self, x):
                x = self.encoder.forward_features(x)
                x = self.avg_pool(x).flatten(1)
                x = self.dropout(x)
                x = self.fc(x)
                return x

        self.model = DeepFakeClassifier()

        weights_dir = os.path.join(WEIGHTS_DIR, 'efficientnet')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, CHECKPOINT_FILENAME)
        if not os.path.exists(weights_path):
            torch.hub.download_url_to_file(CHECKPOINT_URL, weights_path)

        # This is a 2020-era checkpoint containing numpy scalar types that PyTorch 2.6+'s
        # default weights_only=True rejects.
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip a possible "module." prefix left over from DataParallel training
        state_dict = {re.sub("^module.", "", k): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Warm-up pass: on MPS the first real forward pass pays a one-time cost for Metal
        # shader compilation/kernel caching (measured ~7s vs ~2s steady-state on this model).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=self.device)
            self.model(dummy)
        if self.device == "mps":
            torch.mps.synchronize()

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def _crop_face(self, frame_rgb):
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            self.faces_missed += 1
            return frame_rgb

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # 30% margin per extract_crops.py in the official repo
        padding = int(0.30 * max(w, h))
        top = max(0, y - padding)
        left = max(0, x - padding)
        bottom = min(frame_rgb.shape[0], y + h + padding)
        right = min(frame_rgb.shape[1], x + w + padding)
        self.faces_found += 1
        return frame_rgb[top:bottom, left:right]

    def _preprocess_frame(self, frame_rgb):
        from PIL import Image
        face = self._crop_face(frame_rgb)
        img = Image.fromarray(face).resize((INPUT_SIZE, INPUT_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)
        return self.torch.tensor(arr, dtype=self.torch.float32)

    def score_video(self, video_path, num_frames=20):
        torch = self.torch
        frames = extract_frames(video_path, num_frames=num_frames)
        if not frames:
            raise ValueError("No frames extracted")

        batch = torch.stack([self._preprocess_frame(f) for f in frames]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch).squeeze(-1)  # [num_frames]
            probs = torch.sigmoid(logits).cpu().numpy()

        # Single output neuron, standard DFDC convention: higher = FAKE
        return float(np.mean(probs))


def main():
    parser = common.build_arg_parser(
        "Evaluate the EfficientNet-B7-NS DFDC-winning classifier (selimsef) on a labelled video dataset."
    )
    # 10 frames (not common.py's default of 20) verified on a 30-video sample to bring MPS
    # inference under the <3s/video target (1.99s avg, 90% of videos under 3s) with no
    parser.set_defaults(frames=10)
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, DeepFakeClassifierModel, args)


if __name__ == "__main__":
    main()
