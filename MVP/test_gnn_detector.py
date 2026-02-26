import pytest
from gnn_detector import GNNDetector

def test_gnn_detector_initialization():
    detector = GNNDetector()
    assert hasattr(detector, "enabled")

def test_gnn_detector_no_context():
    detector = GNNDetector()
    prob = detector.predict(None)
    assert prob == 0.1 # Default low probability
    
def test_gnn_detector_empty_context():
    detector = GNNDetector()
    prob = detector.predict({})
    assert prob == 0.1

def test_gnn_detector_with_context():
    detector = GNNDetector()
    if not detector.enabled:
        pytest.skip("torch_geometric not installed, skipping active PyG test")
        
    # Simulate a highly-connected suspicious cluster (bot swarm)
    social_context = {
        "nodes": [
            {"id": "target_user", "features": [0.1, 0.9, 0.8]},
            {"id": "bot_1", "features": [0.1, 0.9, 0.9]},
            {"id": "bot_2", "features": [0.1, 0.8, 0.8]}
        ],
        "edges": [
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
            [1, 2],
            [2, 1]
        ]
    }
    
    prob = detector.predict(social_context)
    
    # We expect a valid probability and since it's a tight cluster with high feature values,
    # the deterministic weights should yield a probability > 0.1
    assert isinstance(prob, float)
    assert 0.1 <= prob <= 1.0
    assert prob > 0.1 # Should be elevated due to the connected bots
