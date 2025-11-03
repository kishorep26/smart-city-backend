from fastapi import FastAPI, Depends, HTTPException, Query
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import httpx
from contextlib import asynccontextmanager

from database import (
    get_session,
    IncidentDB,
    AgentDB,
    create_tables
)

@asynccontextmanager
async def lifespan(_):
    create_tables()
    db = next(get_session())
    if db.query(AgentDB).count() == 0:
        db.add_all([
            AgentDB(name="Fire Agent", icon="🚒"),
            AgentDB(name="Police Agent", icon="🚓"),
            AgentDB(name="Ambulance Agent", icon="🚑"),
        ])
        db.commit()
    db.close()
    yield

app = FastAPI(title="Smart City AI Backend", lifespan=lifespan)

app.add_middleware(CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class StatsOut(BaseModel):
    total_incidents: int
    active_incidents: int
    resolved_incidents: int
    total_agents: int
    active_agents: int
    average_response_time: float
    average_efficiency: float

# Helper - always use instance fields, never class fields!
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
        updated_at=row.updated_at
    )

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
        "lat": float(item.get("lat",0)),
        "lon": float(item.get("lon",0)),
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
    return to_incident_out(new_incident)

@app.put("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(incident_id: int, db: Session = Depends(get_session)):
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "resolved"
    db.commit()
    db.refresh(incident)
    return to_incident_out(incident)

@app.get("/agents", response_model=List[AgentOut])
def get_agents(db: Session = Depends(get_session)):
    return [to_agent_out(row) for row in db.query(AgentDB).all()]

@app.get("/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_session)):
    incs = db.query(IncidentDB).all()
    agents = db.query(AgentDB).all()
    total = len(incs)
    active = len([i for i in incs if str(getattr(i, "status", "")) == "active"])
    resolved = len([i for i in incs if str(getattr(i, "status", "")) == "resolved"])
    t_agents = len(agents)
    a_agents = len([a for a in agents if str(getattr(a, "status", "")) == "Responding"])
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
