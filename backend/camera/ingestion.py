import cv2
import asyncio
import threading
import time
from typing import Generator
from config import get_settings

settings = get_settings()


class CameraSource:
    """
    Unified camera source that handles:
    - Webcam (source = "0" or integer)
    - Pre-recorded video file (source = "path/to/video.mp4")
    - IP camera RTSP stream (source = "rtsp://...")
    """

    def __init__(self, camera_id: str, source: str, loop: bool = True):
        self.camera_id = camera_id
        self.source = source
        self.loop = loop  # Loop video file for demo mode
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _open_capture(self):
        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def _capture_loop(self):
        self._cap = self._open_capture()
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self.loop and not self.source.isdigit() and not self.source.startswith("rtsp"):
                    # Loop video file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue
            with self._lock:
                self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """Yield MJPEG frames for HTTP streaming."""
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.033)
                continue
            # Resize to reduce bandwidth
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30fps cap


# ─── Camera Registry ──────────────────────────────────────────────────────────

class CameraRegistry:
    """Manages all active camera sources."""

    def __init__(self):
        self._cameras: dict[str, CameraSource] = {}

    def add_camera(self, camera_id: str, source: str, loop: bool = True) -> CameraSource:
        if camera_id in self._cameras:
            return self._cameras[camera_id]
        cam = CameraSource(camera_id, source, loop=loop)
        cam.start()
        self._cameras[camera_id] = cam
        print(f"[Camera] Started {camera_id} → {source}")
        return cam

    def get_camera(self, camera_id: str) -> CameraSource | None:
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list[str]:
        return list(self._cameras.keys())

    def stop_all(self):
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()


registry = CameraRegistry()
