from cognitive_engine import CognitiveEngine

def test_temperature_scaling():
    ce = CognitiveEngine(temperature=2.0)
    logits = [2.0, 1.0]
    
    # Expected: scaled logits [1.0, 0.5] => exp(1)=2.718, exp(0.5)=1.648 => probs ~ [0.62, 0.38]
    probs = ce.apply_temperature_scaling(logits)
    assert len(probs) == 2
    assert 0.60 < probs[0] < 0.65
    assert 0.35 < probs[1] < 0.40
    assert abs(sum(probs) - 1.0) < 1e-6

def test_entropy_router():
    ce = CognitiveEngine()
    
    # High entropy (uniform distribution)
    assert ce.entropy_router(0.85) == "human_review"
    
    # Low entropy (highly confident)
    assert ce.entropy_router(0.3) == "auto_action"
    
    # Medium entropy (uncertain)
    assert ce.entropy_router(0.7) == "soft_warning"

def test_compute_entropy():
    ce = CognitiveEngine()
    
    ent_high = ce.compute_entropy([0.5, 0.5])
    ent_low = ce.compute_entropy([0.99, 0.01])
    
    assert ent_high > ent_low
    assert ent_high > 0.6 # -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2) ~ 0.693
