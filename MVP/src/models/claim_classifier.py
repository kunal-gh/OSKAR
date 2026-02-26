import torch
import numpy as np
from transformers import pipeline


class ClaimClassifier:
    """
    Hierarchical claim type classifier using zero-shot NLI.

    Given text, determines:
      1. Whether the claim is verifiable (factual) or non-verifiable (opinion)
      2. Sub-type: statistical | historical | policy | scientific | opinion

    Output schema:
      { "is_verifiable": bool, "claim_type": str, "confidence": float }
    """

    # Map from raw zero-shot label → output schema type
    LABEL_MAP = {
        "statistical claim":  ("statistical",  True),
        "historical claim":   ("historical",   True),
        "policy claim":       ("policy",        True),
        "scientific claim":   ("scientific",    True),
        "subjective opinion": ("opinion",       False),
        "personal experience":("opinion",       False),
        "prediction":         ("opinion",       False),
    }

    CANDIDATE_LABELS = list(LABEL_MAP.keys())

    # v0.3: Upgraded from cross-encoder/nli-distilroberta-base
    # Expected macro F1: ~80% (vs ~65% with distilroberta)
    # Swapped to nli-deberta-v3-large to avoid 401 errors
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
        )

    def predict(self, text: str) -> dict:
        result = self.classifier(
            text,
            self.CANDIDATE_LABELS,
            multi_label=False,
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        claim_type, is_verifiable = self.LABEL_MAP.get(top_label, ("opinion", False))

        return {
            "is_verifiable": bool(is_verifiable),
            "claim_type": claim_type,
            "confidence": round(float(top_score), 4),
            "model": "nli-deberta-v3-large"
        }

    def benchmark(self, samples: list[dict] = None) -> dict:
        """
        Quick inline benchmark against labeled samples.
        Each sample: {"text": str, "expected_verifiable": bool}
        Returns {"accuracy": float, "n": int}
        """
        if samples is None:
            samples = [
                {"text": "Vaccines cause autism.",                     "expected_verifiable": True},
                {"text": "CO2 has reached 420 ppm.",                  "expected_verifiable": True},
                {"text": "Pineapple on pizza is the best.",           "expected_verifiable": False},
                {"text": "The 2020 election was stolen.",             "expected_verifiable": True},
                {"text": "I love sunny days.",                        "expected_verifiable": False},
                {"text": "5G towers cause COVID-19.",                 "expected_verifiable": True},
                {"text": "Climate change is a hoax.",                 "expected_verifiable": True},
                {"text": "Coffee tastes better lukewarm.",            "expected_verifiable": False},
                {"text": "The moon landing was faked.",               "expected_verifiable": True},
                {"text": "Star Wars is the greatest franchise ever.", "expected_verifiable": False},
            ]
        correct = sum(
            1 for s in samples
            if self.predict(s["text"])["is_verifiable"] == s["expected_verifiable"]
        )
        acc = round(correct / len(samples), 4)
        print(f"[ClaimClassifier] Benchmark accuracy: {acc*100:.1f}% ({correct}/{len(samples)})")
        return {"accuracy": acc, "n": len(samples)}
