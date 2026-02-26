import os
import time
from src.infra.postgres_db import SessionLocal, UserTrust, init_db
from src.core.evidence_retrieval import EvidenceRetrieval

def seed_database():
    print("[Seeding] Initializing Databases...")
    init_db()
    db = SessionLocal()
    
    # 1. Seed Postgres (Trust)
    # We'll create a few users with varied trust histories
    users = [
        {"user_id_hash": "trusted_reporter_789", "total_claims": 50, "correct_claims": 48, "trust_score": 0.94},
        {"user_id_hash": "known_bad_actor_404", "total_claims": 100, "correct_claims": 5, "trust_score": 0.07},
        {"user_id_hash": "neutral_user_001", "total_claims": 1, "correct_claims": 1, "trust_score": 0.60},
    ]
    
    for u_data in users:
        user = db.query(UserTrust).filter(UserTrust.user_id_hash == u_data["user_id_hash"]).first()
        if not user:
            user = UserTrust(**u_data)
            db.add(user)
        else:
            user.total_claims = u_data["total_claims"]
            user.correct_claims = u_data["correct_claims"]
            user.trust_score = u_data["trust_score"]
    
    db.commit()
    db.close()
    print("[Seeding] Postgres seeded.")

    # 2. Seed Qdrant (Evidence Retrieval)
    er = EvidenceRetrieval(use_neo4j=False)
    knowledge = [
        "The Earth revolves around the Sun every 365.25 days.",
        "Water boils at 100 degrees Celsius at sea level.",
        "DeepMind was founded in London in 2010.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "OSKAR 2.0 uses a hybrid zero-cost microservice architecture.",
        "The Digital Services Act (DSA) regulates online platforms in the EU."
    ]
    er.add_evidence(knowledge)
    print("[Seeding] Qdrant seeded.")

if __name__ == "__main__":
    # Give DBs a moment to spin up if running in Docker, 
    # but for local script execution we just run.
    seed_database()
