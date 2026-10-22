import os
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

Base = declarative_base()

class LoanPredictionLog(Base):
    """
    Database table storing real-time Bank Muscat loan default predictions,
    risk tiers, and local SHAP explainability factors.
    """
    __tablename__ = "loan_prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    person_age = Column(Integer)
    person_income = Column(Float)
    person_emp_length = Column(Float)
    loan_amnt = Column(Float)
    loan_int_rate = Column(Float)
    loan_grade = Column(String(10))
    loan_intent = Column(String(50))
    default_probability = Column(Float)
    risk_tier = Column(String(20))
    decision = Column(String(20))
    top_shap_factors = Column(Text)

def get_db_engine(db_url: str = None):
    raw_url = db_url or os.getenv("DATABASE_URL", "")
    if not raw_url or "PASTE_YOUR" in raw_url:
        db_path = os.path.join(DATA_DIR, "bank_muscat_audit.db").replace(os.sep, "/")
        raw_url = f"sqlite:///{db_path}"
    elif raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)

    connect_args = {"check_same_thread": False} if raw_url.startswith("sqlite") else {}
    engine = create_engine(raw_url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    return engine, raw_url

def log_prediction_to_db(engine, payload: dict, prediction_result: dict):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as session:
        record = LoanPredictionLog(
            person_age=payload.get("person_age"),
            person_income=payload.get("person_income"),
            person_emp_length=payload.get("person_emp_length"),
            loan_amnt=payload.get("loan_amnt"),
            loan_int_rate=payload.get("loan_int_rate"),
            loan_grade=payload.get("loan_grade"),
            loan_intent=payload.get("loan_intent"),
            default_probability=prediction_result.get("default_probability"),
            risk_tier=prediction_result.get("risk_tier"),
            decision=prediction_result.get("decision"),
            top_shap_factors=json.dumps(prediction_result.get("top_shap_drivers", []))
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id