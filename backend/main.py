"""
City Intelligence Platform — FastAPI Backend
============================================
Entry point. Mounts all routers, starts background pipelines,
seeds the database with demo data on first run.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import get_settings
from database.session import engine, init_db, AsyncSessionLocal, Base
from database.models import User, UserRole, Junction, Camera
from auth.utils import hash_password
from auth.routes import router as auth_router
from camera.routes import router as camera_router
from camera.ingestion import registry
from modules.emergency.routes import router as emergency_router, run_emergency_pipeline
from modules.violation.routes import router as violation_router, run_violation_pipeline

settings = get_settings()

# ─── Demo Data ────────────────────────────────────────────────────────────────

DEMO_USERS = [
    {"username": "admin", "password": "admin123", "full_name": "City Command Admin",
     "role": UserRole.SUPER_ADMIN, "department": "Central Command"},
    {"username": "traffic_hq", "password": "traffic123", "full_name": "Traffic Police HQ",
     "role": UserRole.TRAFFIC_POLICE_HQ, "department": "Bengaluru Traffic Police"},
    {"username": "municipal_hq", "password": "municipal123", "full_name": "BBMP Municipal HQ",
     "role": UserRole.MUNICIPAL_HQ, "department": "BBMP"},
    {"username": "operator1", "password": "op123", "full_name": "Junction Operator — MG Road",
     "role": UserRole.JUNCTION_OPERATOR, "department": "MG Road Junction"},
]

DEMO_CAMERAS = [
    {"id": "CAM_001", "name": "MG Road North", "location": "MG Road Junction",
     "lat": 12.9716, "lon": 77.5946, "source": "data/sample_videos/traffic1.mp4", "junction": "JCT_001"},
    {"id": "CAM_002", "name": "Brigade Road", "location": "Brigade Road",
     "lat": 12.9730, "lon": 77.5960, "source": "data/sample_videos/traffic2.mp4", "junction": "JCT_002"},
    {"id": "CAM_WEBCAM", "name": "Live Webcam", "location": "Demo Station",
     "lat": 12.9700, "lon": 77.5930, "source": "0", "junction": "JCT_005"},
]


async def seed_demo_data():
    """Insert demo users and cameras if they don't exist."""
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select

        for u in DEMO_USERS:
            result = await db.execute(select(User).where(User.username == u["username"]))
            if result.scalar_one_or_none() is None:
                user = User(
                    username=u["username"],
                    full_name=u["full_name"],
                    hashed_password=hash_password(u["password"]),
                    role=u["role"],
                    department=u["department"],
                )
                db.add(user)

        await db.commit()
        print("[Seed] Demo users created.")


# ─── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await seed_demo_data()
    os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)

    # Start camera feeds
    for cam_data in DEMO_CAMERAS:
        src = cam_data["source"]
        # Only start if source exists or is a webcam
        if src == "0" or os.path.exists(src):
            registry.add_camera(cam_data["id"], src)

    # Start AI pipelines as background tasks
    background_tasks = []
    if registry.get_camera("CAM_001"):
        background_tasks.append(
            asyncio.create_task(
                run_emergency_pipeline("CAM_001", "JCT_001", AsyncSessionLocal)
            )
        )
    if registry.get_camera("CAM_002"):
        background_tasks.append(
            asyncio.create_task(
                run_violation_pipeline("CAM_002", 12.9730, 77.5960, AsyncSessionLocal)
            )
        )

    print("[City AI] 🚀 All systems online.")
    yield

    # Shutdown
    for task in background_tasks:
        task.cancel()
    registry.stop_all()
    await engine.dispose()
    print("[City AI] Shutdown complete.")


# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="City Intelligence Platform API",
    description="Real-time AI-powered city management system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence images as static files
app.mount("/evidence", StaticFiles(directory=settings.EVIDENCE_DIR, check_dir=False), name="evidence")

# Include routers
app.include_router(auth_router)
app.include_router(camera_router)
app.include_router(emergency_router)
app.include_router(violation_router)


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "cameras": registry.list_cameras(),
        "demo_mode": settings.DEMO_MODE,
    }


@app.get("/api/dashboard/summary")
async def dashboard_summary():
    """Quick summary stats for the super admin dashboard."""
    from sqlalchemy import select, func
    from database.models import EmergencyEvent, CivicViolation

    async with AsyncSessionLocal() as db:
        emergency_count = await db.execute(select(func.count(EmergencyEvent.id)))
        violation_count = await db.execute(select(func.count(CivicViolation.id)))
        pending_count = await db.execute(
            select(func.count(CivicViolation.id)).where(CivicViolation.status == "PENDING")
        )

    return {
        "total_emergency_events": emergency_count.scalar(),
        "total_violations": violation_count.scalar(),
        "pending_violations": pending_count.scalar(),
        "active_cameras": len(registry.list_cameras()),
    }
