"""
benchmark_claim_classifier.py — OSKAR v0.3
-------------------------------------------
Benchmarks the ClaimClassifier against a hand-labeled set of 20 sentences
covering the four verifiable claim types plus opinion/subjective.

Run:
    venv\\Scripts\\python benchmark_claim_classifier.py
"""

from claim_classifier import ClaimClassifier
from sklearn.metrics import classification_report, f1_score
import json

TEST_SET = [
    # ===== Verifiable (is_verifiable=True) =====
    {"text": "Vaccines cause autism, according to suppressed CDC data.",         "expected": True, "type": "scientific"},
    {"text": "CO2 atmospheric levels reached 421 ppm in 2023.",                  "expected": True, "type": "statistical"},
    {"text": "The 2020 US election was stolen through widespread fraud.",         "expected": True, "type": "historical"},
    {"text": "5G towers are responsible for spreading COVID-19.",                 "expected": True, "type": "scientific"},
    {"text": "Climate change is a hoax invented by China.",                       "expected": True, "type": "historical"},
    {"text": "The moon landing in 1969 was staged in a Hollywood studio.",        "expected": True, "type": "historical"},
    {"text": "Drinking bleach cures COVID-19 according to alternative doctors.",  "expected": True, "type": "scientific"},
    {"text": "The earth is flat, contrary to what governments claim.",            "expected": True, "type": "scientific"},
    {"text": "Human activity is the primary driver of climate change.",           "expected": True, "type": "scientific"},
    {"text": "Voter fraud affected the outcome of the 2020 presidential race.",   "expected": True, "type": "historical"},

    # ===== Non-verifiable (is_verifiable=False) =====
    {"text": "Pineapple on pizza is a crime against Italian cuisine.",            "expected": False, "type": "opinion"},
    {"text": "I genuinely feel happier when the weather is cold.",                "expected": False, "type": "opinion"},
    {"text": "Star Wars is unquestionably the greatest film franchise ever.",     "expected": False, "type": "opinion"},
    {"text": "Coffee with oat milk is significantly better than plain black.",    "expected": False, "type": "opinion"},
    {"text": "I think tomorrow will be a great day for a walk.",                  "expected": False, "type": "opinion"},
    {"text": "Winter is by far the most beautiful season of the year.",           "expected": False, "type": "opinion"},
    {"text": "I believe this company will become very successful in five years.", "expected": False, "type": "opinion"},
    {"text": "Gaming is a higher form of art than cinema in my view.",            "expected": False, "type": "opinion"},
    {"text": "In my opinion, the original trilogy is much better.",               "expected": False, "type": "opinion"},
    {"text": "I feel that the new design looks much cleaner and modern.",         "expected": False, "type": "opinion"},
]

def run_benchmark():
    print("Loading ClaimClassifier (deberta-v3-large-zeroshot-v2)...")
    clf = ClaimClassifier()

    y_true, y_pred = [], []
    details = []

    for sample in TEST_SET:
        result = clf.predict(sample["text"])
        predicted = result["is_verifiable"]
        correct = predicted == sample["expected"]

        y_true.append(int(sample["expected"]))
        y_pred.append(int(predicted))

        details.append({
            "text":      sample["text"][:60] + "...",
            "expected":  sample["expected"],
            "predicted": predicted,
            "correct":   "✅" if correct else "❌",
            "confidence": result["confidence"]
        })

    # Print per-sample results
    print("\n" + "="*80)
    print(f"{'#':<4} {'Expected':<12} {'Got':<12} {'Conf':<6} Text")
    print("="*80)
    for i, d in enumerate(details):
        print(f"{i+1:<4} {str(d['expected']):<12} {str(d['predicted']):<12} {d['confidence']:<6} {d['correct']} {d['text']}")

    # Compute metrics
    correct_n = sum(1 for d in details if d["correct"] == "✅")
    accuracy = correct_n / len(TEST_SET)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        target_names=["Opinion", "Verifiable"],
        digits=3
    )

    print("\n" + "="*80)
    print(f"Accuracy:        {accuracy*100:.1f}% ({correct_n}/{len(TEST_SET)})")
    print(f"Macro F1:        {macro_f1:.4f}  ({'✅ PASS' if macro_f1 >= 0.80 else '❌ FAIL'} — target: ≥ 0.80)")
    print("="*80)
    print("\nClassification Report:")
    print(report)

    return {"accuracy": accuracy, "macro_f1": macro_f1, "n": len(TEST_SET)}

if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result, indent=2))
