import datetime
import os

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://oskar_user:oskar_pass@localhost:5432/oskar_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserTrust(Base):
    __tablename__ = "user_trust"

    user_id_hash = Column(String, primary_key=True, index=True)
    total_claims = Column(Integer, default=0)
    correct_claims = Column(Integer, default=0)
    trust_score = Column(Float, default=0.5)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
