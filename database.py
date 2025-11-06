import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
    timestamp = Column(DateTime, default=datetime.now, index=True)

class AgentDB(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    icon = Column(String)
    status = Column(String, default="Available")
    current_incident = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    response_time = Column(Float, default=0.0)
    efficiency = Column(Float, default=90.0)
    total_responses = Column(Integer, default=0)
    successful_responses = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    lat = Column(Float, default=0.0)  # 🔥 ADDED
    lon = Column(Float, default=0.0)  # 🔥 ADDED

# 🔥 REMOVED ResponseMetricDB - Not being used anywhere
# class ResponseMetricDB(Base):
#     __tablename__ = "response_metrics"
#     ...

class IncidentHistoryDB(Base):
    __tablename__ = "incident_history"
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, index=True)
    agent_id = Column(Integer, index=True, nullable=True)
    action = Column(String)
    detail = Column(String)
    timestamp = Column(DateTime, default=datetime.now, index=True)

def create_tables():
    Base.metadata.create_all(bind=engine)
