"""Benchmarks SamLowe/roberta-base-go_emotions using a derived score over its coercive-emotion
classes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel

MODEL_KEY = "goemotions"

# Emotions treated as "manipulation-associated" for this derived score the coercive/pressuring
# cluster of GoEmotions' 28 labels, not just the original 7-class fear+anger this project's
MANIPULATION_EMOTIONS = {"fear", "nervousness", "anger", "annoyance", "disapproval", "disgust"}


class GoEmotionsModel(BaseModel):
    name = "SamLowe/roberta-base-go_emotions (derived manipulation score)"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        model_id = "SamLowe/roberta-base-go_emotions"
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
            # multi_label_classification -> independent sigmoid per class,
            # NOT softmax (labels aren't mutually exclusive).
            probs = self.torch.sigmoid(logits).cpu().numpy()[0]

        emotion_probs = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        manipulation_signal = sum(emotion_probs.get(e, 0.0) for e in MANIPULATION_EMOTIONS)
        # Sum of up to 6 independent sigmoid probabilities can exceed 1.0;
        # clip before treating it as a probability.
        manipulative_score = min(1.0, manipulation_signal)
        return {"manipulative": manipulative_score, "non-manipulative": 1.0 - manipulative_score}


def main():
    parser = common.build_arg_parser(
        "Evaluate SamLowe/roberta-base-go_emotions (derived manipulation score) on the hand-constructed voice-manipulation dataset."
    )
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, GoEmotionsModel, args)


if __name__ == "__main__":
    main()
