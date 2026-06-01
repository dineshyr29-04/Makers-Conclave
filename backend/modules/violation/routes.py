"""
Civic Violation Module Routes
------------------------------
Handles:
- WebSocket feed for violation alerts
- REST API for violation reports
- Background pipeline for littering detection
"""

import asyncio
import json
import os
import cv2
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from database.session import get_db
from database.models import CivicViolation, ViolationType, UserRole
from auth.utils import get_current_user, require_role
from camera.ingestion import registry
from websocket.manager import manager
from modules.violation.detector import LitteringDetector
from config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/violations", tags=["violations"])

_detector: LitteringDetector | None = None


def get_detector() -> LitteringDetector:
    global _detector
    if _detector is None:
        _detector = LitteringDetector()
    return _detector


# ─── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/{role}")
async def violation_ws(websocket: WebSocket, role: str):
    await manager.connect(websocket, role)
    try:
        recent = await manager.get_recent_alerts("violation")
        if recent:
            await websocket.send_text(json.dumps({"type": "history", "events": recent}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, role)


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@router.get("/")
async def get_violations(
    limit: int = 50,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    query = select(CivicViolation).order_by(desc(CivicViolation.detected_at)).limit(limit)
    if status:
        query = query.where(CivicViolation.status == status.upper())
    result = await db.execute(query)
    violations = result.scalars().all()

    return {"violations": [
        {
            "id": v.id,
            "type": v.violation_type,
            "detected_at": v.detected_at.isoformat(),
            "camera_id": v.camera_id,
            "latitude": v.latitude,
            "longitude": v.longitude,
            "plate_number": v.plate_number,
            "face_image_url": f"/evidence/{os.path.basename(v.face_image_path)}" if v.face_image_path else None,
            "body_image_url": f"/evidence/{os.path.basename(v.body_image_path)}" if v.body_image_path else None,
            "full_frame_url": f"/evidence/{os.path.basename(v.full_frame_path)}" if v.full_frame_path else None,
            "confidence": v.confidence,
            "status": v.status,
        }
        for v in violations
    ]}


@router.patch("/{violation_id}/status")
async def update_violation_status(
    violation_id: int,
    status: str,
    notes: str = "",
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role(UserRole.SUPER_ADMIN, UserRole.MUNICIPAL_HQ)),
):
    result = await db.execute(select(CivicViolation).where(CivicViolation.id == violation_id))
    v = result.scalar_one_or_none()
    if not v:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Violation not found")
    v.status = status.upper()
    v.notes = notes
    v.reviewed_by = current_user.id
    await db.commit()
    return {"ok": True}


# ─── Background Processing Pipeline ──────────────────────────────────────────

async def run_violation_pipeline(camera_id: str, camera_lat: float, camera_lon: float, db_factory):
    """Long-running background task for littering detection on a camera."""
    detector = get_detector()
    cam = registry.get_camera(camera_id)
    if cam is None:
        print(f"[ViolationPipeline] Camera {camera_id} not found, aborting.")
        return

    print(f"[ViolationPipeline] Started for camera {camera_id}")

    import time
    last_alert_time = 0
    ALERT_COOLDOWN = 15  # seconds

    while True:
        frame = cam.get_frame()
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        loop = asyncio.get_event_loop()
        detection = await loop.run_in_executor(None, detector.detect, frame)

        if detection.detected:
            now = time.time()
            if now - last_alert_time < ALERT_COOLDOWN:
                await asyncio.sleep(0.1)
                continue
            last_alert_time = now

            os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            face_path = body_path = frame_path = None

            if detection.face_crop is not None and detection.face_crop.size > 0:
                face_path = f"{settings.EVIDENCE_DIR}/face_{ts}_{camera_id}.jpg"
                cv2.imwrite(face_path, detection.face_crop)

            if detection.body_crop is not None and detection.body_crop.size > 0:
                body_path = f"{settings.EVIDENCE_DIR}/body_{ts}_{camera_id}.jpg"
                cv2.imwrite(body_path, detection.body_crop)

            if detection.full_frame is not None:
                frame_path = f"{settings.EVIDENCE_DIR}/frame_{ts}_{camera_id}.jpg"
                cv2.imwrite(frame_path, detection.full_frame)

            async with db_factory() as db:
                v = CivicViolation(
                    violation_type=ViolationType.LITTERING,
                    camera_id=camera_id,
                    latitude=camera_lat,
                    longitude=camera_lon,
                    plate_number=detection.plate_number,
                    face_image_path=face_path,
                    body_image_path=body_path,
                    full_frame_path=frame_path,
                    confidence=detection.confidence,
                )
                db.add(v)
                await db.commit()
                await db.refresh(v)
                vid = v.id

            alert = {
                "type": "violation_alert",
                "event_id": vid,
                "violation_type": "LITTERING",
                "camera_id": camera_id,
                "latitude": camera_lat,
                "longitude": camera_lon,
                "plate_number": detection.plate_number,
                "object_class": detection.object_class,
                "confidence": detection.confidence,
                "face_image_url": f"/evidence/{os.path.basename(face_path)}" if face_path else None,
                "body_image_url": f"/evidence/{os.path.basename(body_path)}" if body_path else None,
                "full_frame_url": f"/evidence/{os.path.basename(frame_path)}" if frame_path else None,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await manager.broadcast_to_roles(alert, ["super_admin", "municipal_hq"])
            await manager.store_recent_alert("violation", alert)

            print(f"[ViolationPipeline] ⚠ LITTERING detected! Plate: {detection.plate_number}")

        await asyncio.sleep(0.05)
