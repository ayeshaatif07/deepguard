"""Signal 4 - scores whether a post's caption matches its transcript.
all-MiniLM-L6-v2 embeds both separately; cosine similarity rescaled to 0-1.
"""

from sentence_transformers import SentenceTransformer, util

COHERENCE_THRESHOLD = 0.62


class CaptionCoherenceScorer:
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device=None):
        if device is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading {self.MODEL_ID} (caption coherence) on device {device}...")
        self.model = SentenceTransformer(self.MODEL_ID, device=device)

    def score(self, transcript: str, caption: str):
        """Returns (coherence_score 0.0-1.0, verdict), or (None, None) if either input is empty."""
        transcript = (transcript or "").strip()
        caption = (caption or "").strip()
        if not transcript or not caption:
            return None, None

        embeddings = self.model.encode([transcript, caption], convert_to_tensor=True)
        cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        score = (cos_sim + 1) / 2
        verdict = "Coherent" if score >= COHERENCE_THRESHOLD else "Incoherent"
        return score, verdict
