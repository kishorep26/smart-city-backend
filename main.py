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
    total_agents: int = 0
    active_agents: int = 0
    
    class Config:
        from_attributes = True

# ... (omitted code) ...

@app.get("/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_session)):
    """Get system statistics"""
    stats = db.query(StatsDB).first()
    
    # calculate live agent stats
    total_agents = db.query(AgentDB).count()
    active_agents = db.query(AgentDB).filter(AgentDB.status != "available").count()
    
    if not stats:
        return StatsOut(
            total_incidents=0,
            active_incidents=0,
            resolved_incidents=0,
            average_response_time=0.0,
            total_agents=total_agents,
            active_agents=active_agents
        )
        
    # Return hybrid stats (persistent + live)
    return StatsOut(
        total_incidents=stats.total_incidents,
        active_incidents=stats.active_incidents,
        resolved_incidents=stats.resolved_incidents,
        average_response_time=stats.average_response_time,
        total_agents=total_agents,
        active_agents=active_agents
    )


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
    """Create a new incident with Dynamic Sector Deployment"""

# ... imports
import os
import json
from groq import Groq

# ...

@app.post("/incidents", response_model=IncidentOut)
def create_incident(incident: IncidentIn, db: Session = Depends(get_session)):
    """Create a new incident with AI Classification (Groq)"""
    
    # Defaults
    final_type = "police"
    description_summary = incident.description
    
    # 1. AI CLASSIFICATION (GROQ Llama3)
    groq_key = os.getenv("GROQ_API_KEY")
    ai_success = False

    if groq_key and incident.type == "auto":
        try:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI Dispatcher. Analyze the emergency description. Return ONLY a JSON object with keys: 'type' (must be one of: 'fire', 'medical', 'police', 'accident'), 'severity' (1-10 int), and 'risk_analysis' (short string). Do not include markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": incident.description
                    }
                ],
                temperature=0,
                max_tokens=100
            )
            
            # Parse AI Response
            content = completion.choices[0].message.content
            # Clean possible markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            
            ai_data = json.loads(content)
            final_type = ai_data.get("type", "police").lower()
            description_summary = f"{ai_data.get('risk_analysis', incident.description)} (Severity: {ai_data.get('severity')})"
            ai_success = True
            print(f"AI Classification: {final_type} | {description_summary}")

        except Exception as e:
            print(f"Groq AI Failed: {e}")
            ai_success = False

    # 2. FALLBACK LEGACY LOGIC (If AI fails or no key)
    if not ai_success and incident.type in ["auto", "unknown", ""]:
        desc = incident.description.lower()
        if any(x in desc for x in ["fire", "smoke", "burn", "flame", "explosion"]):
            final_type = "fire"
        elif any(x in desc for x in ["hurt", "injur", "blood", "medi", "pain", "collap", "heart"]):
            final_type = "medical"
        elif any(x in desc for x in ["crash", "accident", "smash", "collision", "car"]):
            final_type = "accident"
        else:
            final_type = "police"

    elif not ai_success and incident.type != "auto":
        final_type = incident.type

    try:
        # Create new incident
        new_incident = IncidentDB(
            type=final_type,
            description=description_summary if ai_success else incident.description,
            status="active",
            lat=incident.location.lat,
            lon=incident.location.lon,
            timestamp=datetime.now(),
            created_at=datetime.now()
        )
        db.add(new_incident)
        db.flush() # Get ID
        
        # ... logic continues (Dynamic sector check)
        # Check if we have agents nearby (within 50km)
        existing_agents = db.query(AgentDB).all()
        nearby_agents = [
            a for a in existing_agents 
            if calculate_distance(incident.location.lat, incident.location.lon, a.lat, a.lon) < 50
        ]

        if not nearby_agents:
            print(f"⚠️ No agents in this sector ({incident.location.lat}, {incident.location.lon}). Deploying new Task Force.")
            # Deploy new local agents for this city
            new_agents_data = [
                {"name": f"Local Fire {random.randint(10,99)}", "type": "fire", "icon": "🚒", "lat": incident.location.lat + 0.01, "lon": incident.location.lon + 0.01},
                {"name": f"Local Police {random.randint(10,99)}", "type": "police", "icon": "🚓", "lat": incident.location.lat - 0.01, "lon": incident.location.lon - 0.01},
                {"name": f"Local Medic {random.randint(10,99)}", "type": "medical", "icon": "🚑", "lat": incident.location.lat + 0.01, "lon": incident.location.lon - 0.01},
            ]
            for agent_data in new_agents_data:
                new_agent = AgentDB(
                    name=agent_data["name"], 
                    type=agent_data["type"], 
                    icon=agent_data["icon"],
                    status="available",
                    lat=agent_data["lat"],
                    lon=agent_data["lon"],
                    fuel=100.0,
                    stress=0.0,
                    role="local_unit",
                    status_message="Deployed to new sector"
                )
                db.add(new_agent)
            db.commit() # Commit new agents so we can use them immediately

        # Refetch agents including new ones
        prepared_type = incident.type.lower()
        preferred_type = incident_type_mapping.get(prepared_type, "police")

        available_agents = db.query(AgentDB).filter(
            AgentDB.status == "available",
            AgentDB.type == preferred_type
        ).all()

        # Fallback to any agent
        if not available_agents:
            available_agents = db.query(AgentDB).filter(AgentDB.status == "available").all()

        if available_agents:
            # Smart Assignment: Prefer closest, but avoid Stressed/LowFuel agents if possible
            candidates = []
            for agent in available_agents:
                dist = calculate_distance(incident.location.lat, incident.location.lon, agent.lat, agent.lon)
                # Cost function: Distance + (Stress * 0.1) + (100 - Fuel) * 0.1
                cost = dist + (agent.stress * 0.05) + ((100 - agent.fuel) * 0.05)
                candidates.append((cost, agent, dist))

            # Pick lowest cost
            candidates.sort(key=lambda x: x[0])
            best_match = candidates[0]
            nearest_agent = best_match[1]
            distance = best_match[2]

            # Assign
            new_incident.assigned_agent_id = nearest_agent.id
            
            # Smart Reasoning Log
            urgency = "HIGH" if incident.type in ["fire", "crime"] else "MED"
            reason = "OPTIMAL"
            if nearest_agent.stress > 60: reason = "STRESSED_ASSIGNMENT"
            if nearest_agent.fuel < 30: reason = "LOW_FUEL_EMERGENCY"
            
            decision_log = f"Sector Deployment | {nearest_agent.name} | PRIORITY:{urgency} | {reason}"

            nearest_agent.status = "busy"
            nearest_agent.current_incident = str(new_incident.id)
            nearest_agent.decision = decision_log
            nearest_agent.response_time = distance * 2
            nearest_agent.total_responses += 1
            nearest_agent.updated_at = datetime.now()
            nearest_agent.status_message = f"Responding to Incident #{new_incident.id}"
            
            # Log history
            history = IncidentHistoryDB(
                incident_id=new_incident.id,
                agent_id=nearest_agent.id,
                event_type="agent_dispatched",
                description=f"{decision_log} - Distance: {distance:.1f}km"
            )
            db.add(history)

        # Update stats
        stats = db.query(StatsDB).first()
        if stats:
            stats.total_incidents += 1
            stats.active_incidents += 1
            # Force update flag if needed (sqlalchemy usually handles change tracking)
            stats.updated_at = datetime.now() 

        db.commit()
        db.refresh(new_incident)
        return to_incident_out(new_incident)

    except Exception as e:
        db.rollback()
        print(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(incident_id: int, db: Session = Depends(get_session)):
    """Mark incident as resolved, free agent, update stats"""
    incident = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    if incident.status == "resolved":
        return to_incident_out(incident)

    incident.status = "resolved"

    # Free agent
    if incident.assigned_agent_id:
        agent = db.query(AgentDB).filter(AgentDB.id == incident.assigned_agent_id).first()
        if agent:
            agent.status = "available"
            agent.current_incident = None
            agent.decision = "Mission Complete - Returning to Patrol"
            agent.successful_responses += 1
            agent.status_message = "Patrolling Sector"
            agent.stress = max(0, agent.stress - 20) # Success reduces stress
            
            history = IncidentHistoryDB(
                incident_id=incident_id,
                agent_id=agent.id,
                event_type="mission_complete",
                description=f"Incident resolved by {agent.name}. Safety score increased."
            )
            db.add(history)

    # Update stats
    stats = db.query(StatsDB).first()
    if stats:
        stats.active_incidents = max(0, stats.active_incidents - 1)
        stats.resolved_incidents += 1
        
        # Recalculate global efficiency
        total = stats.total_incidents
        if total > 0:
            # Simple efficiency metric: Resolved / Total * 100
             # But let's make it 'smart': Based on average stress of agents
             avg_stress = 0
             agents = db.query(AgentDB).all()
             if agents:
                 avg_stress = sum([a.stress for a in agents]) / len(agents)
             stats.average_efficiency = max(0, 100 - avg_stress)

    db.commit()
    db.refresh(incident)
    return to_incident_out(incident)


@app.get("/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_session)):
    """Get system statistics and SIMULATE WORLD TICKS"""
    
    # --- SIMULATION TICK: REFUELING LOGIC ---
    # This runs every time the frontend polls stats (every 3s)
    agents = db.query(AgentDB).all()
    for agent in agents:
        if agent.status == "refueling":
             agent.fuel += 15.0 # Recover fuel
             if agent.fuel >= 100:
                 agent.fuel = 100.0
                 agent.status = "available"
                 agent.status_message = "Refueled and ready"
        
        elif agent.status == "available" and (agent.fuel < 20 or agent.stress > 80):
             agent.status = "refueling"
             agent.status_message = "Low fuel/High stress - Returning to base"
             # Unassign if for some reason assigned (shouldn't happen if logic is good)
    
    db.commit()
    # ----------------------------------------

    stats = db.query(StatsDB).first()
    
    # calculate live agent stats
    total_agents = db.query(AgentDB).count()
    active_agents = db.query(AgentDB).filter(AgentDB.status != "available").count()
    
    if not stats:
        return StatsOut(
            total_incidents=0,
            active_incidents=0,
            resolved_incidents=0,
            average_response_time=0.0,
            total_agents=total_agents,
            active_agents=active_agents
        )
        
    # Return hybrid stats (persistent + live)
    return StatsOut(
        total_incidents=stats.total_incidents,
        active_incidents=stats.active_incidents,
        resolved_incidents=stats.resolved_incidents,
        average_response_time=stats.average_response_time,
        total_agents=total_agents,
        active_agents=active_agents
    )


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


@app.get("/analytics/prediction")
def predict_risk_zones(db: Session = Depends(get_session)):
    """
    MASTERS LEVEL FEATURE: Custom Unsupervised Learning Algorithm
    (Scikit-Learn removed to optimize Serverless utilization; implemented internal K-Means)
    """
    # 1. Fetch Data
    incidents = db.query(IncidentDB).all()
    if len(incidents) < 5:
        return {"status": "insufficient_data", "message": "Need more incidents to train model"}

    # 2. Prepare Data Points
    points = [(i.lat, i.lon) for i in incidents]
    
    # --- CUSTOM K-MEANS IMPLEMENTATION (Lightweight) ---
    def custom_kmeans(data, k, max_iterations=10):
        # 1. Initialize random centroids
        centroids = random.sample(data, k)
        
        for _ in range(max_iterations):
            # 2. Assign clusters
            clusters = [[] for _ in range(k)]
            for point in data:
                # Find nearest centroid
                distances = [math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in centroids]
                closest_idx = distances.index(min(distances))
                clusters[closest_idx].append(point)
            
            # 3. Update centroids
            new_centroids = []
            for i in range(k):
                cluster_points = clusters[i]
                if not cluster_points:
                    new_centroids.append(centroids[i]) # Keep old if empty
                else:
                    avg_lat = sum(p[0] for p in cluster_points) / len(cluster_points)
                    avg_lon = sum(p[1] for p in cluster_points) / len(cluster_points)
                    new_centroids.append((avg_lat, avg_lon))
            
            # Check convergence (simplified)
            if new_centroids == centroids:
                break
            centroids = new_centroids
            
        return centroids, clusters

    # ---------------------------------------------------

    # Dynamic K based on density (roughly 1 cluster per 5 incidents)
    k = min(5, len(points) // 5) + 1
    centers, clusters = custom_kmeans(points, k)

    predictions = []
    for i, center in enumerate(centers):
        if not clusters[i]: continue
        
        # Calculate Risk Score (Density)
        risk_score = len(clusters[i]) / len(points)
        
        predictions.append({
            "id": i,
            "lat": center[0],
            "lon": center[1],
            "risk_score": risk_score,
            "radius": 500 + (risk_score * 2000), 
            "label": "HIGH RISK ZONE" if risk_score > 0.3 else "MODERATE RISK ZONE"
        })

    predictions.sort(key=lambda x: x['risk_score'], reverse=True)
    return {"status": "success", "zones": predictions, "algorithm": "custom_kmeans_v1"}
