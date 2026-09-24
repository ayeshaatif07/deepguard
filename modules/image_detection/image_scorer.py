import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


class ImageScorer:
    """Signal 1b - classifies a still image as Artificial, Deepfake or Real.
    Uses prithivMLmods/AI-vs-Deepfake-vs-Real (ViT).
    """

    MODEL_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real"

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        print(f"Loading {self.MODEL_ID} on device {self.device}...")
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = ViTForImageClassification.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        # id2label from the model config: {0: 'Artificial', 1: 'Deepfake', 2: 'Real'}
        self.id2label = self.model.config.id2label

    def score_image(self, image: Image.Image):
        """
        Runs inference on a single PIL image.
        Returns (confidence, predicted_label) for one of Artificial / Deepfake / Real.
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(probs.argmax())
        predicted_label = self.id2label[predicted_idx]
        confidence = float(probs[predicted_idx])

        return confidence, predicted_label

    def get_all_probabilities(self, image: Image.Image):
        """Returns {label: probability} across all 3 classes."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scorer = ImageScorer()
        img = Image.open(sys.argv[1])
        confidence, label = scorer.score_image(img)
        print(f"Verdict: {label} ({confidence:.1%} confidence)")
        print(f"Full distribution: {scorer.get_all_probabilities(img)}")
