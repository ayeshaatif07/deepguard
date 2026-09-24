"""Benchmarks cross-encoder/stsb-roberta-base (~125M params), a cross-encoder similarity
candidate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "stsb_crossencoder"


class STSBCrossEncoderModel(BaseModel):
    name = "cross-encoder/stsb-roberta-base"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.name, device=self.device)

    def score_pair(self, transcript, caption):
        return float(self.model.predict([(str(transcript), str(caption))])[0])


def main():
    parser = common.build_arg_parser("Evaluate cross-encoder/stsb-roberta-base (cross-encoder, STS-tuned) on the caption-coherence dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, STSBCrossEncoderModel, args)


if __name__ == "__main__":
    main()
