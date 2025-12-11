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
    SessionLocal,
    IncidentDB,
    AgentDB,
    StatsDB,
    IncidentHistoryDB,
    create_tables,
    get_session
)

# Global flag to track initialization
_initialized = False


def _ensure_initialized():
    """Lazy initialization - only runs on first request"""
    global _initialized
    if _initialized:
        return

    try:
        print("Initializing database...")
        create_tables()
        
        # Auto-migration: Ensure new columns exist (Self-Healing)
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS fuel FLOAT DEFAULT 100.0"))
                conn.execute(text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS stress FLOAT DEFAULT 0.0"))
                conn.execute(text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS role VARCHAR DEFAULT 'standard'"))
                conn.execute(text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS status_message VARCHAR"))
                conn.commit()
            print("✅ Schema migration verified")
        except Exception as e:
            print(f"⚠️ Migration warning (safe to ignore if columns exist): {e}")

        print("✅ Tables created")

        db = SessionLocal()
        try:
            # Seed agents if they don't exist
            if db.query(AgentDB).count() == 0:
                print("Seeding agents...")
                center_lat, center_lon = 40.7831, -73.9712  # NYC center

                agents_data = [
                    {"name": "Fire Engine 1", "type": "fire", "icon": "🚒", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                    {"name": "Fire Engine 2", "type": "fire", "icon": "🚒", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                    {"name": "Police Patrol 1", "type": "police", "icon": "🚓", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                    {"name": "Police Patrol 2", "type": "police", "icon": "🚓", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                    {"name": "Ambulance 1", "type": "medical", "icon": "🚑", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                    {"name": "Ambulance 2", "type": "medical", "icon": "🚑", "status": "available",
                     "lat": center_lat + random.uniform(-0.02, 0.02),
                     "lon": center_lon + random.uniform(-0.02, 0.02)},
                ]

                # Check individual existence to be safe against race conditions
                for agent_data in agents_data:
                    exists = db.query(AgentDB).filter(AgentDB.name == agent_data["name"]).first()
                    if not exists:
                         db.add(AgentDB(**agent_data))

                db.commit()
                print(f"✅ Seeding complete")

            # Initialize stats if not exists
            stats = db.query(StatsDB).first()
            if not stats:
                stats = StatsDB(
                    total_incidents=0,
                    active_incidents=0,
                    resolved_incidents=0,
                    average_response_time=0.0
                )
                db.add(stats)
                db.commit()
                print("✅ Stats initialized")

        finally:
            db.close()

        _initialized = True
        print("✅ Database initialization complete!")

    except Exception as e:
        print(f"⚠️ Initialization error: {e}")
        import traceback
        traceback.print_exc()


# Create FastAPI app
app = FastAPI(title="Smart City Emergency Response API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    _ensure_initialized()


# --- Pydantic Models ---

class IncidentLoc(BaseModel):
    lat: float
    lon: float


class IncidentIn(BaseModel):
    type: str
    location: IncidentLoc
    description: str
    status: Optional[str] = "active"


class IncidentOut(BaseModel):
    id: int
    type: str
    location: IncidentLoc
    description: str
    status: str
    timestamp: datetime

    class Config:
        from_attributes = True


class AgentOut(BaseModel):
    id: int
    name: str
    type: str
    icon: str
    status: str
    current_incident: Optional[str] = None
    decision: Optional[str] = None
    response_time: float
    efficiency: float
    total_responses: int
    successful_responses: int
    successful_responses: int
    updated_at: Optional[datetime] = None
    lat: float
    lon: float
    
    # Advanced Stats
    fuel: float
    stress: float
    role: str
    status_message: Optional[str] = None

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_incidents: int
    active_incidents: int
    resolved_incidents: int
    average_response_time: float

    class Config:
        from_attributes = True


class IncidentHistoryOut(BaseModel):
    id: int
    incident_id: Optional[int] = None
    agent_id: Optional[int] = None
    event_type: str
    description: str
    timestamp: datetime

    class Config:
        from_attributes = True


# --- Helper Functions ---

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Incident type to agent type mapping
incident_type_mapping = {
    "fire": "fire",
    "medical": "medical",
    "crime": "police",
    "accident": "police",
    "theft": "police",
    "robbery": "police",
    "assault": "police",
    "emergency": "medical",
    "hazard": "fire",
    "other": "police"
}


def to_incident_out(incident: IncidentDB) -> IncidentOut:
    """Convert IncidentDB to IncidentOut"""
    return IncidentOut(
        id=incident.id,
        type=incident.type,
        location=IncidentLoc(lat=incident.lat, lon=incident.lon),
        description=incident.description,
        status=incident.status,
        timestamp=incident.created_at or incident.timestamp
    )


def to_agent_out(agent: AgentDB) -> AgentOut:
    """Convert AgentDB to AgentOut"""
    return AgentOut(
        id=agent.id,
        name=agent.name,
        type=agent.type,
        icon=agent.icon or "",
        status=agent.status,
        current_incident=agent.current_incident,
        decision=agent.decision,
        response_time=agent.response_time,
        efficiency=agent.efficiency,
        total_responses=agent.total_responses,
        successful_responses=agent.successful_responses,
        updated_at=agent.updated_at,
        lat=agent.lat,
        lon=agent.lon,
        fuel=agent.fuel,
        stress=agent.stress,
        role=agent.role,
        status_message=agent.status_message
    )


def simulate_tick(db: Session):
    """Simulate agent states (fuel burn, stress, movement)"""
    agents = db.query(AgentDB).all()
    
    for agent in agents:
        # Recover if available
        if agent.status == "available":
            agent.stress = max(0.0, agent.stress - 0.5)
            agent.fuel = max(0.0, agent.fuel - 0.05) # Idling consumes fuel too
            agent.status_message = "Patrolling sector"
            
            # Auto-refuel if critical
            if agent.fuel < 20.0:
               agent.status = "refueling"
               agent.status_message = "Low fuel - Returning to base"
               
        # Burn resources if busy
        elif agent.status == "busy":
            agent.stress = min(100.0, agent.stress + 0.8)
            agent.fuel = max(0.0, agent.fuel - 0.2)
            agent.status_message = f"Responding to Incident #{agent.current_incident}"
            
        # Refueling state
        elif agent.status == "refueling":
            agent.fuel = min(100.0, agent.fuel + 5.0)
            agent.stress = max(0.0, agent.stress - 2.0)
            agent.status_message = "Refueling at station"
            
            if agent.fuel >= 99.0:
                agent.status = "available"
        
        # Random position drift for 'patrolling' effect
        if agent.status == "available" and agent.fuel > 0:
            agent.lat += random.uniform(-0.0005, 0.0005)
            agent.lon += random.uniform(-0.0005, 0.0005)
            
    db.commit()


# --- API Routes ---

@app.get("/")
def root():
    return {"message": "Smart City Emergency Response API", "status": "online"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/agents", response_model=List[AgentOut])
def get_agents(db: Session = Depends(get_session)):
    """Get all agents"""
    # Trigger simulation tick
    try:
        simulate_tick(db)
    except Exception as e:
        print(f"Simulation tick failed: {e}")
        
    agents = db.query(AgentDB).all()
    return [to_agent_out(agent) for agent in agents]


@app.get("/incidents", response_model=List[IncidentOut])
def get_incidents(db: Session = Depends(get_session)):
    """Get all incidents"""
    incidents = db.query(IncidentDB).all()
    return [to_incident_out(incident) for incident in incidents]


@app.post("/incidents", response_model=IncidentOut)
def create_incident(incident: IncidentIn, db: Session = Depends(get_session)):
    """Create a new incident and auto-assign agent"""
    try:
        # Create new incident
        new_incident = IncidentDB(
            type=incident.type,
            description=incident.description,
            status="active",
            lat=incident.location.lat,
            lon=incident.location.lon,
            timestamp=datetime.now(),
            created_at=datetime.now()
        )
        db.add(new_incident)
        db.flush()

        # Get preferred agent type for this incident
        prepared_type = incident.type.lower()
        preferred_type = incident_type_mapping.get(prepared_type, "police")

        # Find nearest available agent of preferred type
        available_agents = db.query(AgentDB).filter(
            AgentDB.status == "available",
            AgentDB.type == preferred_type
        ).all()

        # If no preferred type available, get any available agent
        if not available_agents:
            available_agents = db.query(AgentDB).filter(
                AgentDB.status == "available"
            ).all()

        if available_agents:
            # Calculate distances and find nearest
            nearest_agent = min(
                available_agents,
                key=lambda a: calculate_distance(
                    incident.location.lat, incident.location.lon, a.lat, a.lon
                )
            )

            distance = calculate_distance(
                incident.location.lat, incident.location.lon,
                nearest_agent.lat, nearest_agent.lon
            )

            # Assign agent
            new_incident.assigned_agent_id = nearest_agent.id
            nearest_agent.status = "busy"
            nearest_agent.current_incident = str(new_incident.id)
            nearest_agent.decision = f"Responding to {incident.type} incident"
            nearest_agent.response_time = distance * 2  # Estimate: 2 min per km
            nearest_agent.total_responses += 1
            nearest_agent.updated_at = datetime.now()
            
            # Reset simulation states for active duty
            nearest_agent.status_message = "En route to scene"
            if nearest_agent.fuel < 10: nearest_agent.fuel = 20.0 # Emergency reserve

            # Log assignment
            history = IncidentHistoryDB(
                incident_id=new_incident.id,
                agent_id=nearest_agent.id,
                event_type="incident_created",
                description=f"Incident created and assigned to {nearest_agent.name} ({distance:.2f}km away)"
            )
            db.add(history)

        # Update stats
        stats = db.query(StatsDB).first()
        if stats:
            stats.total_incidents += 1
            stats.active_incidents += 1

        db.commit()
        db.refresh(new_incident)
        return to_incident_out(new_incident)

    except Exception as e:
        db.rollback()
        print(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(incident_id: int, db: Session = Depends(get_session)):
    """Mark incident as resolved and free up agent"""
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.status = "resolved"

    # Free up assigned agent
    if incident.assigned_agent_id:
        agent = db.query(AgentDB).filter(AgentDB.id == incident.assigned_agent_id).first()
        if agent:
            agent.status = "available"
            agent.current_incident = None
            agent.decision = "Available for assignment"
            agent.successful_responses += 1
            agent.updated_at = datetime.now()

            # Log resolution
            history = IncidentHistoryDB(
                incident_id=incident_id,
                agent_id=agent.id,
                event_type="incident_resolved",
                description=f"Incident resolved by {agent.name}"
            )
            db.add(history)

    # Update stats
    stats = db.query(StatsDB).first()
    if stats:
        stats.active_incidents = max(0, stats.active_incidents - 1)
        stats.resolved_incidents += 1

    db.commit()
    db.refresh(incident)
    return to_incident_out(incident)


@app.get("/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_session)):
    """Get system statistics"""
    stats = db.query(StatsDB).first()
    if not stats:
        # Return default stats if none exist
        return StatsOut(
            total_incidents=0,
            active_incidents=0,
            resolved_incidents=0,
            average_response_time=0.0
        )
    return stats


@app.get("/incident-history", response_model=List[IncidentHistoryOut])
def get_incident_history(db: Session = Depends(get_session)):
    """Get incident history logs"""
    history = db.query(IncidentHistoryDB).order_by(
        IncidentHistoryDB.timestamp.desc()
    ).limit(50).all()
    return history


@app.get("/search-address")
def search_address(query: str = Query(..., min_length=3)):
    """Search for addresses using OpenStreetMap Nominatim"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 5}
        resp = httpx.get(url, params=params, headers={"User-Agent": "smart-city-backend"})
        resp.raise_for_status()
        data = resp.json()

        results = [{
            "lat": float(item.get("lat", 0)),
            "lon": float(item.get("lon", 0)),
            "address": str(item.get("display_name", ""))
        } for item in data]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Address search failed: {str(e)}")


@app.post("/classify-incident")
def classify_incident(data: dict):
    """Simple incident classification based on keywords"""
    desc = data.get("desc", "").lower()

    # Simple keyword-based classification
    if any(word in desc for word in ["fire", "smoke", "burning", "flames"]):
        return {"category": "fire", "confidence": 0.95}
    elif any(word in desc for word in ["medical", "injury", "accident", "ambulance", "emergency"]):
        return {"category": "medical", "confidence": 0.92}
    elif any(word in desc for word in ["theft", "robbery", "crime", "police", "assault", "break"]):
        return {"category": "crime", "confidence": 0.90}
    else:
        return {"category": "other", "confidence": 0.75}


@app.post("/assign-agent")
def assign_agent(incident_id: int, db: Session = Depends(get_session)):
    """Manually trigger agent assignment for an incident"""
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    if incident.assigned_agent_id:
        return {"message": "Agent already assigned", "agent_id": incident.assigned_agent_id}

    # reuse logic from create_incident ideally, but for now simple assignment
    preferred_type = incident_type_mapping.get(incident.type.lower(), "police")
    
    available_agents = db.query(AgentDB).filter(
        AgentDB.status == "available",
        AgentDB.type == preferred_type
    ).all()
    
    if not available_agents:
        available_agents = db.query(AgentDB).filter(AgentDB.status == "available").all()
        
    if not available_agents:
        raise HTTPException(status_code=404, detail="No available agents")

    nearest_agent = min(
        available_agents,
        key=lambda a: calculate_distance(incident.lat, incident.lon, a.lat, a.lon)
    )

    distance = calculate_distance(incident.lat, incident.lon, nearest_agent.lat, nearest_agent.lon)

    incident.assigned_agent_id = nearest_agent.id
    nearest_agent.status = "busy"
    nearest_agent.current_incident = str(incident.id)
    nearest_agent.decision = f"Manual assignment to {incident.type}"
    nearest_agent.response_time = distance * 2
    nearest_agent.total_responses += 1
    nearest_agent.updated_at = datetime.now()

    history = IncidentHistoryDB(
        incident_id=incident.id,
        agent_id=nearest_agent.id,
        event_type="agent_assigned",
        description=f"Agent {nearest_agent.name} manually assigned ({distance:.2f}km)"
    )
    db.add(history)
    db.commit()

    return {"status": "assigned", "agent": to_agent_out(nearest_agent)}
