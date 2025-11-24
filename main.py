from fastapi import FastAPI, Depends, HTTPException, Query
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import httpx
import math
import random

from database import (
    get_session,
    IncidentDB,
    AgentDB,
    IncidentHistoryDB,
    create_tables,
    SessionLocal
)

# Global flag to track initialization
_initialized = False


def _lazy_init():
    """Initialize database only when first request comes in"""
    global _initialized
    if not _initialized:
        try:
            create_tables()
            _seed_agents_if_needed()
            _initialized = True
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"⚠️ Initialization error: {e}")
            # Don't raise - let the app start anyway


def _seed_agents_if_needed():
    """Seed agents if database is empty"""
    try:
        db = SessionLocal()
        if db.query(AgentDB).count() == 0:
            center_lat = 40.7580
            center_lon = -73.9855

            def random_position(center_lat, center_lon, radius_km=5):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius_km)
                lat_offset = (distance / 111) * math.cos(angle)
                lon_offset = (distance / (111 * math.cos(math.radians(center_lat)))) * math.sin(angle)
                return center_lat + lat_offset, center_lon + lon_offset

            fire_lat, fire_lon = random_position(center_lat, center_lon)
            police_lat, police_lon = random_position(center_lat, center_lon)
            ambulance_lat, ambulance_lon = random_position(center_lat, center_lon)

            db.add_all([
                AgentDB(name="Fire Agent", icon="🚒", lat=fire_lat, lon=fire_lon),
                AgentDB(name="Police Agent", icon="🚓", lat=police_lat, lon=police_lon),
                AgentDB(name="Ambulance Agent", icon="🚑", lat=ambulance_lat, lon=ambulance_lon),
            ])
            db.commit()
            print("✅ Agents seeded with random locations")
        db.close()
    except Exception as e:
        print(f"⚠️ Seeding error: {e}")


# Create FastAPI app
app = FastAPI(title="Smart City AI Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to initialize database on first request
@app.middleware("http")
async def init_middleware(request, call_next):
    _lazy_init()
    response = await call_next(request)
    return response


# --- Models ---
class IncidentLoc(BaseModel):
    lat: float
    lon: float


class IncidentIn(BaseModel):
    type: str
    location: IncidentLoc
    description: str
    status: Optional[str] = "active"


class IncidentOut(IncidentIn):
    id: int
    timestamp: datetime


class AgentOut(BaseModel):
    id: int
    name: str
    icon: str
    status: str
    current_incident: Optional[str] = None
    decision: Optional[str] = None
    response_time: float
    efficiency: float
    total_responses: int
    successful_responses: int
    updated_at: Optional[datetime]
    lat: float
    lon: float


class StatsOut(BaseModel):
    total_incidents: int
    active_incidents: int
    resolved_incidents: int
    total_agents: int
    active_agents: int
    average_response_time: float
    average_efficiency: float


class IncidentHistoryOut(BaseModel):
    id: int
    incident_id: int
    agent_id: Optional[int]
    action: str
    detail: str
    timestamp: datetime


# --- Helper functions ---
def to_incident_out(row) -> IncidentOut:
    return IncidentOut(
        id=int(row.id),
        type=str(row.type or ""),
        location=IncidentLoc(lat=float(row.lat or 0), lon=float(row.lon or 0)),
        description=str(row.description or ""),
        status=str(row.status or ""),
        timestamp=row.timestamp or datetime.now()
    )


def to_agent_out(row) -> AgentOut:
    return AgentOut(
        id=int(row.id),
        name=str(row.name or ""),
        icon=str(row.icon or ""),
        status=str(row.status or ""),
        current_incident=(str(row.current_incident) if row.current_incident is not None else None),
        decision=(str(row.decision) if row.decision is not None else None),
        response_time=float(row.response_time or 0),
        efficiency=float(row.efficiency or 0),
        total_responses=int(row.total_responses or 0),
        successful_responses=int(row.successful_responses or 0),
        updated_at=row.updated_at,
        lat=float(row.lat or 0),
        lon=float(row.lon or 0)
    )


def to_history_out(row) -> IncidentHistoryOut:
    return IncidentHistoryOut(
        id=int(row.id),
        incident_id=int(row.incident_id),
        agent_id=int(row.agent_id) if row.agent_id is not None else None,
        action=str(row.action or ""),
        detail=str(row.detail or ""),
        timestamp=row.timestamp
    )


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def log_event(db, incident_id, agent_id, action, detail):
    event = IncidentHistoryDB(
        incident_id=incident_id, agent_id=agent_id,
        action=action, detail=detail, timestamp=datetime.now()
    )
    db.add(event)
    db.commit()


def assign_agent_to_incident(incident_id: int, db: Session):
    """Automatically assign appropriate agent based on incident type and distance"""
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    agents = db.query(AgentDB).filter(AgentDB.status == "Available").all()

    if not incident or not agents:
        return None

    # Map incident types to preferred agent types
    incident_type_mapping = {
        "fire": ["Fire Agent"],
        "medical": ["Ambulance Agent"],
        "police": ["Police Agent"],
        "crime": ["Police Agent"],
        "accident": ["Ambulance Agent", "Police Agent"],
        "emergency": ["Ambulance Agent"],
        "theft": ["Police Agent"],
        "robbery": ["Police Agent"],
        "assault": ["Police Agent"],
        "hazard": ["Fire Agent"],
    }

    # Get preferred agent types for this incident
    incident_type_lower = incident.type.lower()
    preferred_agent_names = []

    for key, agent_types in incident_type_mapping.items():
        if key in incident_type_lower:
            preferred_agent_names.extend(agent_types)
            break

    # If no match, allow any agent
    if not preferred_agent_names:
        preferred_agent_names = [a.name for a in agents]

    # Filter agents by type preference
    preferred_agents = [a for a in agents if a.name in preferred_agent_names]

    # If no preferred agents available, use any available agent
    candidates = preferred_agents if preferred_agents else agents

    # Find closest agent among candidates
    min_dist = float('inf')
    chosen = None
    for agent in candidates:
        dist = haversine(agent.lat or 0, agent.lon or 0, incident.lat, incident.lon)
        if dist < min_dist:
            min_dist = dist
            chosen = agent

    if chosen is None:
        return None

    # Assign agent
    chosen.status = "Responding"
    chosen.current_incident = str(incident_id)
    chosen.decision = f"Assigned to {incident.type} incident {incident_id} at {datetime.now().isoformat()} (distance: {min_dist:.2f}km)"
    chosen.response_time = min_dist
    chosen.updated_at = datetime.now()
    chosen.total_responses += 1

    db.commit()
    log_event(db, incident_id, chosen.id, "ASSIGN", chosen.decision)

    return chosen


# --- REST Endpoints ---

@app.post("/classify-incident")
def classify_incident(data: dict):
    desc = data.get("desc", "")
    label = "police" if "theft" in desc.lower() else "fire" if "fire" in desc.lower() else "medical"
    return {"category": label, "confidence": 0.95}


@app.get("/search-address")
def search_address(query: str = Query(..., min_length=3)):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 5
    }
    resp = httpx.get(url, params=params, headers={"User-Agent": "smart-city-backend"})
    resp.raise_for_status()
    data = resp.json()
    results = [{
        "lat": float(item.get("lat", 0)),
        "lon": float(item.get("lon", 0)),
        "address": str(item.get("display_name", ""))
    } for item in data]
    return results


@app.get("/")
def root():
    return {"message": "Backend running", "status": "online"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/incidents", response_model=List[IncidentOut])
def get_incidents(db: Session = Depends(get_session)):
    return [to_incident_out(row) for row in db.query(IncidentDB).all()]


@app.post("/incidents", response_model=IncidentOut)
def create_incident(incident: IncidentIn, db: Session = Depends(get_session)):
    new_incident = IncidentDB(
        type=incident.type,
        lat=incident.location.lat,
        lon=incident.location.lon,
        description=incident.description,
        status=incident.status,
        timestamp=datetime.now()
    )
    db.add(new_incident)
    db.commit()
    db.refresh(new_incident)
    log_event(db, new_incident.id, None, "CREATE_INCIDENT", f"New incident created: {new_incident.description}")

    # Auto-assign agent immediately
    try:
        assigned_agent = assign_agent_to_incident(new_incident.id, db)
        if assigned_agent:
            print(f"✅ Auto-assigned {assigned_agent.name} to incident {new_incident.id}")
        else:
            print(f"⚠️ No available agent for incident {new_incident.id}")
    except Exception as e:
        print(f"❌ Failed to assign agent: {e}")

    return to_incident_out(new_incident)


@app.put("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(incident_id: int, db: Session = Depends(get_session)):
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "resolved"
    agent = db.query(AgentDB).filter(AgentDB.current_incident == str(incident_id),
                                     AgentDB.status == "Responding").first()
    if agent:
        agent.status = "Available"
        agent.current_incident = None
        agent.decision = f"Incident {incident_id} resolved at {datetime.now().isoformat()}"
        agent.successful_responses += 1
        agent.updated_at = datetime.now()
        db.commit()
        log_event(db, incident_id, agent.id, "RESOLVE", f"Incident resolved and agent {agent.name} released")
    db.commit()
    db.refresh(incident)
    return to_incident_out(incident)


@app.post("/assign-agent", response_model=AgentOut)
def assign_agent(incident_id: int, db: Session = Depends(get_session)):
    result = assign_agent_to_incident(incident_id, db)
    if not result:
        raise HTTPException(404, "No available agent or incident not found")
    return to_agent_out(result)


@app.get("/agents", response_model=List[AgentOut])
def get_agents(db: Session = Depends(get_session)):
    return [to_agent_out(row) for row in db.query(AgentDB).all()]


@app.get("/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_session)):
    incs = db.query(IncidentDB).all()
    agents = db.query(AgentDB).all()
    total = len(incs)
    active = len([i for i in incs if str(getattr(i, "status", "")).lower() == "active"])
    resolved = len([i for i in incs if str(getattr(i, "status", "")).lower() == "resolved"])
    t_agents = len(agents)
    a_agents = len([a for a in agents if str(getattr(a, "status", "")).lower() in ("responding", "active", "busy")])
    avg_resp = sum([float(getattr(a, "response_time", 0)) for a in agents]) / t_agents if t_agents else 0.0
    avg_eff = sum([float(getattr(a, "efficiency", 0)) for a in agents]) / t_agents if t_agents else 0.0
    return StatsOut(
        total_incidents=total,
        active_incidents=active,
        resolved_incidents=resolved,
        total_agents=t_agents,
        active_agents=a_agents,
        average_response_time=avg_resp,
        average_efficiency=avg_eff
    )


@app.get("/incident-history", response_model=List[IncidentHistoryOut])
def incident_history(db: Session = Depends(get_session)):
    return [to_history_out(row) for row in db.query(IncidentHistoryDB).order_]
