"""Benchmarks cross-encoder/nli-deberta-v3-base, an entailment-based alternative to the
similarity-based candidates.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel

ENTAILMENT_INDEX = 1  # confirmed via this model's config.json id2label


class NLIDebertaModel(BaseModel):
    name = "cross-encoder/nli-deberta-v3-base"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.name, device=self.device)

    def score_pair(self, transcript, caption):
        import torch
        logits = self.model.predict([(str(transcript), str(caption))])
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        return float(probs[0][ENTAILMENT_INDEX])


def main():
    parser = common.build_arg_parser("Evaluate cross-encoder/nli-deberta-v3-base (NLI entailment) on the caption-coherence dataset.")
    args = parser.parse_args()
    common.run_evaluation("nli_deberta", NLIDebertaModel, args)


if __name__ == "__main__":
    main()
