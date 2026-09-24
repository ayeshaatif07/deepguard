"""Benchmarks BothBosu/roberta-scam-classifier-v1, a scam-text classification candidate."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel

MODEL_KEY = "bothbosu"
LABEL_MAP = {"SCAM": "manipulative", "NON-SCAM": "non-manipulative"}


class BothBosuModel(BaseModel):
    name = "BothBosu/roberta-scam-classifier-v1"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        model_id = "BothBosu/roberta-scam-classifier-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def score_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]

        out = {"manipulative": 0.0, "non-manipulative": 0.0}
        for i, p in enumerate(probs):
            native_label = self.id2label[i]
            canonical = LABEL_MAP.get(native_label, native_label)
            out[canonical] = out.get(canonical, 0.0) + float(p)
        return out


def main():
    parser = common.build_arg_parser(
        "Evaluate BothBosu/roberta-scam-classifier-v1 on the hand-constructed voice-manipulation dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, BothBosuModel, args)


if __name__ == "__main__":
    main()
