import math
import torch
from transformers import pipeline


# All known hate-positive label strings across supported models (case-insensitive match)
HATE_LABELS = {
    "hate", "hate_speech", "hateful", "offensive", "toxic", "toxicity",
    "racism", "sexism", "1", "label_1",
}


class HateClassifier:
    """
    Hate speech classifier using cardiffnlp/twitter-roberta-base-hate-latest.
    Outputs a standardized schema:
      { "label": "hate|non_hate", "score": float, "uncertainty": float }

    - score:       probability that content is hate speech (0.0–1.0)
    - uncertainty: Shannon entropy of the prediction distribution
                   → high entropy = model is unsure → routed to human review
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate-latest"

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
            top_k=None,           # Always return all class scores
            truncation=True,
            max_length=512,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        raw = self.classifier(text)
        results = raw[0] if isinstance(raw[0], list) else raw

        hate_score, non_hate_score = self._parse_scores(results)

        # Calibrate: renormalize to sum = 1
        total = hate_score + non_hate_score
        if total > 0:
            hate_score /= total
            non_hate_score /= total

        uncertainty = self._entropy(hate_score, non_hate_score)
        label = "hate" if hate_score > 0.5 else "non_hate"

        return {
            "label": label,
            "score": round(float(hate_score), 4),
            "uncertainty": round(float(uncertainty), 4),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_scores(self, results: list) -> tuple[float, float]:
        hate_score = 0.0
        non_hate_score = 0.0

        for r in results:
            lbl = r["label"].lower()
            score = r["score"]
            if lbl in HATE_LABELS:
                hate_score = max(hate_score, score)
            else:
                non_hate_score = max(non_hate_score, score)

        # Fallback: single-result model
        if hate_score == 0.0 and non_hate_score == 0.0:
            top = results[0]
            if top["label"].lower() in HATE_LABELS:
                hate_score = top["score"]
                non_hate_score = 1.0 - hate_score
            else:
                non_hate_score = top["score"]
                hate_score = 1.0 - non_hate_score

        return hate_score, non_hate_score

    @staticmethod
    def _entropy(*probs) -> float:
        return -sum(p * math.log(p) if p > 0 else 0.0 for p in probs)
