from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from camera.ingestion import registry

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/{camera_id}/stream")
async def stream_camera(camera_id: str):
    """MJPEG live stream for a specific camera."""
    cam = registry.get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or inactive")

    return StreamingResponse(
        cam.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/")
async def list_cameras():
    return {"cameras": registry.list_cameras()}
