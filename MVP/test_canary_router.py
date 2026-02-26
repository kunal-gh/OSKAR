import time
import pytest
from canary_router import CanaryRouter

class MockModel:
    def __init__(self, name, error_rate=0.0):
        self.name = name
        self.error_rate = error_rate

    def predict(self, text: str):
        import random
        if random.random() < self.error_rate:
            raise ValueError(f"Simulated {self.name} Failure")
        return {"label": self.name, "score": 0.99}

def test_canary_traffic_split_distribution():
    """Verify that traffic routes approximately matching the split percentage."""
    stable = MockModel("stable")
    canary = MockModel("canary")
    
    # 25% Canary
    router = CanaryRouter(stable, canary, initial_split=0.25, evaluation_window_seconds=999)
    
    canary_hits = 0
    total_requests = 1000
    
    for _ in range(total_requests):
        res = router.predict("test")
        if res["label"] == "canary":
            canary_hits += 1
            
    # Allow for some statistical variance
    split_ratio = canary_hits / total_requests
    assert 0.20 < split_ratio < 0.30

def test_canary_automated_rollback():
    """Verify that a high error rate in the Canary model triggers an automatic rollback."""
    stable = MockModel("stable")
    # Canary fails 100% of the time
    canary = MockModel("canary", error_rate=1.0)
    
    # Very short evaluation window for the test
    router = CanaryRouter(stable, canary, initial_split=1.0, max_error_rate=0.5, evaluation_window_seconds=1)
    
    # Force 10 errors
    for _ in range(10):
        res = router.predict("test")
        # Should always fallback to stable when canary throws Exception
        assert res["label"] == "stable"
        assert res.get("_canary_fallback") is True
        
    # Wait for the monitor thread to execute the health check
    time.sleep(1.5)
    
    # Tripped! Traffic split should now be 0.0
    assert router.traffic_split == 0.0
    
    # Future requests should natively route to stable without even trying canary
    res2 = router.predict("test")
    assert res2["label"] == "stable"
    assert res2.get("_canary_fallback") is None # natively routed, not a fallback
