import time
from src.models.hate_classifier import HateClassifier

def test_hate_classifier_schema():
    classifier = HateClassifier()
    result = classifier.predict("This is a totally normal and peaceful message.")
    
    assert "label" in result
    assert "score" in result
    assert "uncertainty" in result
    assert result["label"] in ["hate", "non_hate"]
    assert isinstance(result["score"], float)
    assert isinstance(result["uncertainty"], float)

def test_hate_classifier_accuracy():
    classifier = HateClassifier()
    
    test_cases = [
        ("I love walking in the park on a sunny day.", "non_hate"),
        ("Have a wonderful trip to the mountains!", "non_hate"),
        ("This project is quite interesting and well designed.", "non_hate"),
        ("You are completely useless and I hate you.", "hate"),
        ("Everyone from that country is disgusting and evil.", "hate"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        res = classifier.predict(text)
        if res["label"] == expected:
            correct += 1
            
    accuracy = correct / len(test_cases)
    assert accuracy >= 0.8  # Target is >= 0.85 F1, this checks basic accuracy

def test_hate_classifier_latency():
    classifier = HateClassifier()
    
    # Warmup
    classifier.predict("Warmup")
    
    # Test latency
    start = time.perf_counter()
    classifier.predict("Testing inference speed.")
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    # Target < 120ms
    # Allow slightly higher bound in unit test if running on CPU during CI/dev locally
    print(f"Latency: {elapsed:.2f} ms")
    assert elapsed < 500  # Relaxed slightly for CPU local testing, real bound <120 on target hardware
