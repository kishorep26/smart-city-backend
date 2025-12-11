import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if "sslmode" not in DATABASE_URL and "supabase" in DATABASE_URL.lower():
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base = declarative_base()


class IncidentDB(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, index=True)
    lat = Column(Float)
    lon = Column(Float)
    description = Column(String)
    status = Column(String, default="active")
    assigned_agent_id = Column(Integer, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    created_at = Column(DateTime, default=datetime.now)


class AgentDB(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String, index=True)
    icon = Column(String, nullable=True)
    status = Column(String, default="available")
    current_incident = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    response_time = Column(Float, default=0.0)
    efficiency = Column(Float, default=90.0)
    
    # Advanced Simulation Stats
    fuel = Column(Float, default=100.0)  # 0-100%
    stress = Column(Float, default=0.0)  # 0-100%
    role = Column(String, default="standard") # expert, trainee, leader
    status_message = Column(String, nullable=True) # "Refuelling", "Patrolling", etc.
    
    total_responses = Column(Integer, default=0)
    successful_responses = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    lat = Column(Float, default=0.0)
    lon = Column(Float, default=0.0)


class StatsDB(Base):
    __tablename__ = "stats"

    id = Column(Integer, primary_key=True, index=True)
    total_incidents = Column(Integer, default=0)
    active_incidents = Column(Integer, default=0)
    resolved_incidents = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class IncidentHistoryDB(Base):
    __tablename__ = "incident_history"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, index=True, nullable=True)
    agent_id = Column(Integer, index=True, nullable=True)
    event_type = Column(String)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.now, index=True)


def create_tables():
    Base.metadata.create_all(bind=engine)
