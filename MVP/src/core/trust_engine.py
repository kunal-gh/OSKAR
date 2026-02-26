import datetime
import logging
from src.infra.redis_cache import redis_cache
from src.infra.postgres_db import SessionLocal, UserTrust, init_db

# Ensure DB tables exist on first import
try:
    init_db()
except Exception as e:
    logging.warning(f"[TrustEngine] Could not initialize Postgres: {e}")

class TrustEngine:
    """
    v1.0 Microservice Architecture:
    - Postgres (Relational): Persistent source of truth for trust priors.
    - Redis (Cache): Fast access for high-frequency risk fusion lookups.
    """

    def __init__(self):
        self.db = SessionLocal()

    def _get_user_data(self, user_id_hash: str) -> dict:
        # 1. Try Redis Cache first
        cache_key = f"trust:user:{user_id_hash}"
        data = redis_cache.get(cache_key)
        if data:
            return data

        # 2. Check Postgres
        user = self.db.query(UserTrust).filter(UserTrust.user_id_hash == user_id_hash).first()
        if user:
            data = {
                "total_claims": user.total_claims,
                "correct_claims": user.correct_claims,
                "trust_score": user.trust_score,
                "last_updated": user.last_updated.isoformat(),
            }
        else:
            # 3. New User Default
            data = {
                "total_claims": 0,
                "correct_claims": 0,
                "trust_score": 0.5,
                "last_updated": datetime.datetime.utcnow().isoformat(),
            }
            # Optional: Seed Postgres for new user
            new_user = UserTrust(user_id_hash=user_id_hash)
            self.db.add(new_user)
            self.db.commit()

        # Update Cache
        redis_cache.set(cache_key, data, ttl_seconds=3600)
        return data

    def _save_user_data(self, user_id_hash: str, data: dict):
        # 1. Update Postgres
        user = self.db.query(UserTrust).filter(UserTrust.user_id_hash == user_id_hash).first()
        if user:
            user.total_claims = data["total_claims"]
            user.correct_claims = data["correct_claims"]
            user.trust_score = data["trust_score"]
            user.last_updated = datetime.datetime.utcnow()
            self.db.commit()

        # 2. Update Redis
        cache_key = f"trust:user:{user_id_hash}"
        redis_cache.set(cache_key, data, ttl_seconds=3600)

    def get_user_trust(self, user_id_hash: str) -> float:
        data = self._get_user_data(user_id_hash)
        return float(data.get("trust_score", 0.5))

    def update_trust(self, user_id_hash: str, claim_was_correct: bool):
        data = self._get_user_data(user_id_hash)
        data["total_claims"] += 1
        if claim_was_correct:
            data["correct_claims"] += 1

        alpha = 2 + data["correct_claims"]
        beta = 2 + data["total_claims"] - data["correct_claims"]
        data["trust_score"] = float(alpha) / (alpha + beta)
        self._save_user_data(user_id_hash, data)

    def get_risk_modifier(self, user_id_hash: str) -> float:
        trust_score = self.get_user_trust(user_id_hash)
        return 1.5 - trust_score

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
