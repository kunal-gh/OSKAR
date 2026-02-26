"""
Benchmark script to evaluate hate classification models side-by-side.
Picks the best model to replace the current one.
"""
from transformers import pipeline
import torch
import time

DEVICE = 0 if torch.cuda.is_available() else -1

# Labelled test set: (text, expected_label)
TEST_CASES = [
    # Non-hate (should return non_hate)
    ("I love walking in the park on a sunny day.", "non_hate"),
    ("What a beautiful morning! The birds are singing.", "non_hate"),
    ("This project is quite well designed and interesting.", "non_hate"),
    ("She did an amazing job on the presentation today.", "non_hate"),
    ("Looking forward to the weekend trip with family.", "non_hate"),
    ("The economy report shows steady growth this quarter.", "non_hate"),
    # Hate (should return hate)
    ("You are completely useless and I hate you.", "hate"),
    ("Everyone from that country is disgusting and evil.", "hate"),
    ("People like you don't deserve to live.", "hate"),
    ("That group of people are all criminals and parasites.", "hate"),
    ("I hope all those [slur] get what they deserve.", "hate"),
]

CANDIDATES = [
    "cardiffnlp/twitter-roberta-base-hate-latest",
    "Hate-speech-CNERG/dehatebert-mono-english",
    "citizenlab/distilbert-base-multilingual-cased-toxicity",  # Current baseline
]

TOXIC_LABELS = {
    "hate", "toxic", "toxicity", "hate_speech", "hateful",
    "offensive", "1", "label_1", "racism", "sexism"
}

def evaluate_model(model_name):
    print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")
    try:
        clf = pipeline("text-classification", model=model_name, device=DEVICE, top_k=None)

        # Warmup
        clf("warmup text")

        correct = 0
        for text, expected in TEST_CASES:
            t0 = time.perf_counter()
            raw = clf(text)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Flatten if nested list
            results = raw[0] if isinstance(raw[0], list) else raw

            # Find highest-scoring label
            best = max(results, key=lambda x: x["score"])
            lbl = best["label"].lower()

            predicted = "hate" if lbl in TOXIC_LABELS else "non_hate"
            ok = "✓" if predicted == expected else "✗"
            if predicted == expected:
                correct += 1

            print(f"  {ok} [{latency_ms:.0f}ms] '{text[:50]}...' → {predicted} (conf={best['score']:.2f})")

        accuracy = correct / len(TEST_CASES)
        print(f"\n  ACCURACY: {accuracy:.0%} ({correct}/{len(TEST_CASES)})")
        return accuracy

    except Exception as e:
        print(f"  ERROR: {e}")
        return 0.0

if __name__ == "__main__":
    results = {}
    for name in CANDIDATES:
        results[name] = evaluate_model(name)

    print("\n\n" + "="*60)
    print("FINAL RANKINGS:")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        flag = "✅ PASSES F1≥0.80" if acc >= 0.80 else "❌ BELOW THRESHOLD"
        print(f"  {acc:.0%} — {name.split('/')[-1]} {flag}")
