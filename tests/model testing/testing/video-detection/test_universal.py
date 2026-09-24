"""Benchmarks UniversalFakeDetect (Ojha et al., CVPR 2023), a frozen CLIP ViT-L/14 with a
linear probe.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel, extract_frames, WEIGHTS_DIR


MODEL_KEY = "universal"


class UniversalFakeDetectModel(BaseModel):
    name = "UniversalFakeDetect (CLIP ViT-L/14 + linear probe)"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        import torch
        import torch.nn as nn
        universal_dir = os.path.join(WEIGHTS_DIR, 'universal')
        sys.path.insert(0, universal_dir)
        import clip  # the pip-installed openai `clip` package (already a dependency)

        self.torch = torch
        clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        clip_model.eval()
        self.clip_model = clip_model
        self.fc = nn.Linear(768, 1).to(self.device)
        state_dict = torch.load(
            os.path.join(universal_dir, 'fc_weights.pth'), map_location=self.device
        )
        self.fc.load_state_dict(state_dict)
        self.fc.eval()

    def score_video(self, video_path, num_frames=20):
        torch = self.torch
        from PIL import Image
        frames = extract_frames(video_path, num_frames=num_frames)
        if not frames:
            raise ValueError("No frames extracted")

        batch = torch.stack(
            [self.preprocess(Image.fromarray(f)) for f in frames]
        ).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(batch).float()
            # Ojha et al.'s published UniversalFakeDetect training code L2-normalizes CLIP
            # features before the linear probe.
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self.fc(features).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        return float(np.mean(probs))


def main():
    parser = common.build_arg_parser(
        "Evaluate UniversalFakeDetect (CLIP ViT-L/14 + linear probe) on a labelled video dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, UniversalFakeDetectModel, args)


if __name__ == "__main__":
    main()
