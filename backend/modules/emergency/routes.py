"""
Emergency Module Routes
-----------------------
Handles:
- WebSocket feed for emergency alerts
- API to get all junction signal states
- Manual signal override
- Recent emergency event log
- Background pipeline that processes camera frames
"""

import asyncio
import json
import os
import cv2
import time
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from database.models import EmergencyEvent, UserRole
from auth.utils import get_current_user, require_role
from camera.ingestion import registry
from websocket.manager import manager
from modules.emergency.detector import EmergencyDetector
from modules.emergency.routing import signal_manager, DEMO_JUNCTIONS
from config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/emergency", tags=["emergency"])

# Module-level detector (lazy init to avoid YOLO load on import)
_detector: EmergencyDetector | None = None


def get_detector() -> EmergencyDetector:
    global _detector
    if _detector is None:
        _detector = EmergencyDetector()
    return _detector


# ─── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/{role}")
async def emergency_ws(websocket: WebSocket, role: str):
    """Real-time emergency alerts over WebSocket."""
    await manager.connect(websocket, role)
    try:
        # Send recent alerts on connect
        recent = await manager.get_recent_alerts("emergency")
        if recent:
            await websocket.send_text(json.dumps({"type": "history", "events": recent}))

        # Keep alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, role)


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@router.get("/signals")
async def get_signal_states():
    """Get current signal state for all junctions."""
    states = await signal_manager.get_all_signal_states()
    return {"junctions": list(states.values())}


@router.post("/signals/{junction_id}/override")
async def override_signal(
    junction_id: str,
    state: str,
    current_user=Depends(require_role(UserRole.SUPER_ADMIN, UserRole.TRAFFIC_POLICE_HQ, UserRole.JUNCTION_OPERATOR)),
):
    if junction_id not in DEMO_JUNCTIONS:
        raise HTTPException(status_code=404, detail="Junction not found")
    if state not in ("GREEN", "RED", "YELLOW"):
        raise HTTPException(status_code=400, detail="Invalid state")

    await signal_manager.manual_override(junction_id, state)

    # Broadcast state change
    event = {
        "type": "signal_override",
        "junction_id": junction_id,
        "state": state,
        "by": current_user.username,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast_all(event)
    return {"ok": True}


@router.get("/events")
async def get_emergency_events(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    from sqlalchemy import select, desc
    result = await db.execute(
        select(EmergencyEvent).order_by(desc(EmergencyEvent.detected_at)).limit(limit)
    )
    events = result.scalars().all()
    return {"events": [
        {
            "id": e.id,
            "vehicle_type": e.vehicle_type,
            "plate_number": e.plate_number,
            "detected_at": e.detected_at.isoformat(),
            "junction_id": e.junction_id,
            "camera_id": e.camera_id,
            "confidence": e.confidence,
        }
        for e in events
    ]}


# ─── Background Processing Pipeline ──────────────────────────────────────────

async def run_emergency_pipeline(camera_id: str, junction_id: str, db_factory):
    """
    Long-running background task that reads frames from a camera,
    runs emergency detection, and fires alerts.
    """
    detector = get_detector()
    cam = registry.get_camera(camera_id)
    if cam is None:
        print(f"[EmergencyPipeline] Camera {camera_id} not found, aborting.")
        return

    print(f"[EmergencyPipeline] Started for camera {camera_id} at junction {junction_id}")
    last_alert_time = 0
    ALERT_COOLDOWN = 10  # seconds between alerts for same vehicle

    while True:
        frame = cam.get_frame()
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        # Run detection in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        detection = await loop.run_in_executor(None, detector.detect, frame)

        if detection.detected:
            now = time.time()
            if now - last_alert_time < ALERT_COOLDOWN:
                await asyncio.sleep(0.1)
                continue
            last_alert_time = now

            # Save snapshot
            snapshot_path = None
            if detection.snapshot is not None:
                os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"{settings.EVIDENCE_DIR}/emergency_{ts}_{camera_id}.jpg"
                cv2.imwrite(snapshot_path, detection.snapshot)

            # Preempt signals
            cleared = await signal_manager.preempt_route(junction_id, detection.vehicle_type, detection.plate_number)

            # Persist to DB
            async with db_factory() as db:
                event = EmergencyEvent(
                    vehicle_type=detection.vehicle_type,
                    plate_number=detection.plate_number,
                    camera_id=camera_id,
                    junction_id=junction_id,
                    route_cleared=json.dumps(cleared),
                    confidence=detection.confidence,
                    snapshot_path=snapshot_path,
                )
                db.add(event)
                await db.commit()
                await db.refresh(event)
                event_id = event.id

            # Build alert payload
            alert = {
                "type": "emergency_alert",
                "event_id": event_id,
                "vehicle_type": detection.vehicle_type,
                "plate_number": detection.plate_number,
                "camera_id": camera_id,
                "junction_id": junction_id,
                "junctions_cleared": cleared,
                "confidence": detection.confidence,
                "flash_detected": detection.flash_detected,
                "timestamp": datetime.utcnow().isoformat(),
                "snapshot_url": f"/evidence/{os.path.basename(snapshot_path)}" if snapshot_path else None,
            }

            # Broadcast to traffic police + super admin
            await manager.broadcast_to_roles(alert, [
                "super_admin", "traffic_police_hq", "junction_operator"
            ])
            await manager.store_recent_alert("emergency", alert)

            # Also push signal state update
            signal_update = {
                "type": "signal_update",
                "junctions": list((await signal_manager.get_all_signal_states()).values()),
            }
            await manager.broadcast_all(signal_update)

            print(f"[EmergencyPipeline] 🚨 {detection.vehicle_type} detected! Cleared: {cleared}")

        await asyncio.sleep(0.05)  # ~20 fps processing
