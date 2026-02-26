import datetime
from src.infra.redis_cache import redis_cache

class TrustEngine:
    """
    v0.7 Redis-backed Trust Engine.
    Scalable across multiple container nodes.
    """
    def __init__(self):
        pass # Now relies on the globally imported redis_cache Singleton

    def _get_user_data(self, user_id_hash: str) -> dict:
        key = f"trust:user:{user_id_hash}"
        data = redis_cache.get(key)
        if not data:
            return {
                "total_claims": 0,
                "correct_claims": 0,
                "trust_score": 0.5,
                "last_updated": datetime.datetime.utcnow().isoformat()
            }
        return data

    def _save_user_data(self, user_id_hash: str, data: dict):
        key = f"trust:user:{user_id_hash}"
        data["last_updated"] = datetime.datetime.utcnow().isoformat()
        # Persist trust scores for 30 days
        redis_cache.set(key, data, ttl_seconds=60*60*24*30)

    def get_user_trust(self, user_id_hash: str) -> float:
        """
        Get the current trust score for a user.
        Prior = beta(2, 2) which gives 0.5 mean.
        """
        data = self._get_user_data(user_id_hash)
        return float(data.get("trust_score", 0.5))

    def update_trust(self, user_id_hash: str, claim_was_correct: bool):
        """
        Implement Bayesian Update
        alpha = 2 + correct_claims
        beta = 2 + total_claims - correct_claims
        trust_score = alpha / (alpha + beta)
        """
        data = self._get_user_data(user_id_hash)
        
        data["total_claims"] += 1
        if claim_was_correct:
            data["correct_claims"] += 1
            
        alpha = 2 + data["correct_claims"]
        beta = 2 + data["total_claims"] - data["correct_claims"]
        data["trust_score"] = float(alpha) / (alpha + beta)
        
        self._save_user_data(user_id_hash, data)

    def get_risk_modifier(self, user_id_hash: str) -> float:
        """
        Calculate risk modifier based on trust score.
        risk_adjusted = base_risk * (1.5 - trust_score)
        A trusted user (score > 0.5) gets a modifier < 1.0 (lowers risk)
        An untrusted user (score < 0.5) gets a modifier > 1.0 (raises risk)
        """
        trust_score = self.get_user_trust(user_id_hash)
        return 1.5 - trust_score
