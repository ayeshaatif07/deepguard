"""Benchmarks sentence-transformers/all-mpnet-base-v2 (~109M params), a larger bi-encoder
candidate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import BaseModel


MODEL_KEY = "mpnet"


class MPNetModel(BaseModel):
    name = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name, device=self.device)

    def score_pair(self, transcript, caption):
        from sentence_transformers import util
        embeddings = self.model.encode([str(transcript), str(caption)], convert_to_tensor=True)
        cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return (cos_sim + 1) / 2


def main():
    parser = common.build_arg_parser("Evaluate sentence-transformers/all-mpnet-base-v2 (bi-encoder cosine similarity) on the caption-coherence dataset.")
    args = parser.parse_args()
    common.run_evaluation(MODEL_KEY, MPNetModel, args)


if __name__ == "__main__":
    main()
