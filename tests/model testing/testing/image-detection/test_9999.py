"""Benchmarks prithivMLmods/AI-vs-Deepfake-vs-Real-9999, a SigLIP image classification
candidate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel, CLASS_NAMES


MODEL_KEY = "9999"
HF_REPO_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-9999"

# This model's own id2label uses "Real one" instead of "Real" map it to our canonical
# CLASS_NAMES so the shared metrics code (and comparisons against the other two models) all
MODEL_LABEL_TO_CANONICAL = {
    "Artificial": "Artificial",
    "Deepfake": "Deepfake",
    "Real one": "Real",
}


class Siglip9999Model(BaseModel):
    name = "AI-vs-Deepfake-vs-Real-9999 (SigLIP2)"

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

        # Confirm the model's label order matches what we expect before relying on it, rather
        # than assuming id2label is [0,1,2] in a fixed order.
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

        result = {}
        for idx, prob in enumerate(probs):
            raw_label = self.id2label[idx]
            canonical = MODEL_LABEL_TO_CANONICAL.get(raw_label, raw_label)
            result[canonical] = float(prob)
        return result


def main():
    parser = common.build_arg_parser(
        f"Evaluate {HF_REPO_ID} (SigLIP2, 3-class: Artificial/Deepfake/Real) on a labelled image dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, Siglip9999Model, args)


if __name__ == "__main__":
    main()
