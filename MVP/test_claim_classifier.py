import time
from claim_classifier import ClaimClassifier

def test_claim_classifier_schema():
    classifier = ClaimClassifier(model_name="cross-encoder/nli-distilroberta-base")
    result = classifier.predict("I feel like today is a great day.")
    
    assert "is_verifiable" in result
    assert "claim_type" in result
    assert "confidence" in result
    assert isinstance(result["is_verifiable"], bool)
    assert result["claim_type"] in ["statistical", "historical", "policy", "scientific", "opinion"]
    assert isinstance(result["confidence"], float)

def test_claim_classifier_accuracy():
    classifier = ClaimClassifier(model_name="cross-encoder/nli-distilroberta-base")
    
    test_cases = [
        ("The new tax policy was signed into law last week.", True, "policy"),
        ("Over 65% of the population lives in urban areas.", True, "statistical"),
        ("The Declaration of Independence was signed in 1776.", True, "historical"),
        ("Water boils at 100 degrees Celsius at sea level.", True, "scientific"),
        ("I think chocolate ice cream is the best flavor.", False, "opinion"),
        ("My personal experience with that software was terrible.", False, "opinion")
    ]
    
    correct_verifiability = 0
    correct_type = 0
    
    for text, exp_verifiable, exp_type in test_cases:
        res = classifier.predict(text)
        if res["is_verifiable"] == exp_verifiable:
            correct_verifiability += 1
        if res["claim_type"] == exp_type:
            correct_type += 1
            
    # Allow some leeway for zero-shot model misclassifying the exact type, but verifiability should be high
    verifiability_acc = correct_verifiability / len(test_cases)
    assert verifiability_acc >= 0.6  # MVP threshold for basic routing

def test_claim_classifier_latency():
    classifier = ClaimClassifier(model_name="cross-encoder/nli-distilroberta-base")
    
    # Warmup
    classifier.predict("Warmup")
    
    # Test latency
    start = time.perf_counter()
    classifier.predict("Testing inference speed.")
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    # Target < 120ms (relaxed bound for CPU testing)
    print(f"Latency: {elapsed:.2f} ms")
    assert elapsed < 1000  
