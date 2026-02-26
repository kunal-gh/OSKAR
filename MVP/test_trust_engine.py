import os
from trust_engine import TrustEngine
from redis_cache import redis_cache

def test_trust_engine_lifecycle():
    # Clear local fallback dictionary for clean test environment
    redis_cache._fallback_cache.clear()
    
    # Use Redis-backed TrustEngine (will fallback to dict if Redis offline)
    te = TrustEngine()
    
    user_id = "test_user_hash_123"
    
    # 1. Default trust should be 0.5
    initial_trust = te.get_user_trust(user_id)
    assert initial_trust == 0.5
    
    # 2. Risk modifier should be 1.0 
    assert te.get_risk_modifier(user_id) == 1.0
    
    # 3. Add correct claims
    te.update_trust(user_id, claim_was_correct=True)
    te.update_trust(user_id, claim_was_correct=True)
    
    # alpha = 2 + 2 = 4
    # beta = 2 + 2 - 2 = 2
    # new_trust = 4 / (4 + 2) = 0.666
    trusted_score = te.get_user_trust(user_id)
    assert 0.66 < trusted_score < 0.67
    
    # Risk modifier for trusted user should be < 1.0
    mod = te.get_risk_modifier(user_id)
    assert mod < 1.0
    assert abs(mod - (1.5 - trusted_score)) < 1e-6
    
    # 4. Add incorrect claims
    te.update_trust(user_id, claim_was_correct=False)
    te.update_trust(user_id, claim_was_correct=False)
    te.update_trust(user_id, claim_was_correct=False)
    
    # Now user is untrusted, score should drop
    untrusted_score = te.get_user_trust(user_id)
    assert untrusted_score < trusted_score
