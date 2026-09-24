"""Benchmarks prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0, a SigLIP image classification
candidate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "v2"
HF_REPO_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"


class SiglipV2Model(BaseModel):
    name = "AI-vs-Deepfake-vs-Real-v2.0 (SigLIP2)"

    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import AutoImageProcessor, SiglipForImageClassification

        self.torch = torch
        print(f"Loading {HF_REPO_ID}...")
        self.processor = AutoImageProcessor.from_pretrained(HF_REPO_ID)
        self.model = SiglipForImageClassification.from_pretrained(HF_REPO_ID)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        print(f"Model id2label: {self.id2label}")

    def score_image(self, image_path):
        from PIL import Image
        torch = self.torch

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # This model's own id2label already uses "Artificial"/"Deepfake"/"Real" directly
        # matches common.CLASS_NAMES with no remapping needed (confirmed via config.json
        return {self.id2label[idx]: float(prob) for idx, prob in enumerate(probs)}


def main():
    parser = common.build_arg_parser(
        f"Evaluate {HF_REPO_ID} (SigLIP2, 3-class: Artificial/Deepfake/Real) on a labelled image dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, SiglipV2Model, args)


if __name__ == "__main__":
    main()
