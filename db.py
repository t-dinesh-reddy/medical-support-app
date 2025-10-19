# db.py
import os, json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db.sqlite")

engine = create_engine(DATABASE_URL, future=True)

def init_db():
    """Create predictions table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          features_json TEXT NOT NULL,
          prob REAL NOT NULL,
          label INTEGER NOT NULL,
          model_version TEXT NOT NULL
        );
        """))

def log_prediction(features: dict, prob: float, label: int, model_version: str):
    payload = json.dumps(features)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO predictions (ts, features_json, prob, label, model_version) "
                 "VALUES (:ts,:fj,:p,:l,:mv)"),
            {"ts": datetime.utcnow().isoformat(),
             "fj": payload,
             "p": float(prob),
             "l": int(label),
             "mv": model_version}
        )

def fetch_recent(limit=25):
    """Return most recent prediction rows; ensure table exists."""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT ts, prob, label, model_version, features_json "
                     "FROM predictions ORDER BY id DESC LIMIT :lim"),
                {"lim": limit}
            ).all()
        return [dict(r._mapping) for r in rows]
    except OperationalError:
        init_db()
        return []
