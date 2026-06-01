"""
Littering / Civic Violation Detector
-------------------------------------
Detects littering events using:
  1. YOLOv8 person + object detection
  2. Temporal analysis — object appears on ground that wasn't there before
  3. MediaPipe for face crop extraction
  4. EasyOCR for number plate extraction from nearby vehicles
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO
import mediapipe as mp
import easyocr
import re
from config import get_settings

settings = get_settings()

# COCO classes relevant to littering
PERSON_CLASS = 0
LITTER_OBJECTS = {
    39: "bottle", 41: "cup", 67: "cell phone",
    73: "book", 60: "dining table"
}
# Vehicle classes for plate extraction
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PLATE_PATTERN = re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}", re.IGNORECASE)


@dataclass
class ViolationDetection:
    detected: bool = False
    confidence: float = 0.0
    person_bbox: tuple = ()
    object_bbox: tuple = ()
    face_crop: Optional[np.ndarray] = None
    body_crop: Optional[np.ndarray] = None
    full_frame: Optional[np.ndarray] = None
    plate_number: Optional[str] = None
    object_class: str = ""


class LitteringDetector:
    def __init__(self, model_path: str = None):
        model_path = model_path or settings.YOLO_MODEL_PATH
        print(f"[LitteringDetector] Loading YOLO from {model_path}")
        self.model = YOLO(model_path)
        self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Frame history for temporal analysis (ground object tracking)
        # Store sets of (class, x_bin, y_bin) tuples for each frame
        self._ground_objects_history: deque = deque(maxlen=30)

    def detect(self, frame: np.ndarray) -> ViolationDetection:
        result = ViolationDetection()
        h, w = frame.shape[:2]

        # ── Step 1: YOLOv8 detection ──────────────────────────────────────────
        results = self.model(
            frame,
            conf=0.35,  # Lower threshold for small objects
            imgsz=settings.YOLO_IMG_SIZE,
            verbose=False,
        )

        persons = []
        ground_objects = []
        vehicles = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().astype(int)

                if cls == PERSON_CLASS:
                    persons.append((bbox, conf))
                elif cls in LITTER_OBJECTS:
                    ground_objects.append((bbox, cls, conf))
                elif cls in VEHICLE_CLASSES:
                    vehicles.append((bbox, conf))

        if not persons:
            self._update_history(ground_objects, w, h)
            return result

        # ── Step 2: Temporal analysis — new object appeared near person ───────
        current_set = self._objects_to_set(ground_objects, w, h)
        prev_set = self._ground_objects_history[-1] if self._ground_objects_history else set()
        new_objects_keys = current_set - prev_set
        self._update_history(ground_objects, w, h)

        if not new_objects_keys:
            return result  # No new objects on ground

        # Find which new object is closest to a person
        matched_person = None
        matched_obj = None
        min_dist = float("inf")

        for obj_bbox, obj_cls, obj_conf in ground_objects:
            key = self._object_key(obj_bbox, obj_cls, w, h)
            if key not in new_objects_keys:
                continue
            obj_cx = (obj_bbox[0] + obj_bbox[2]) // 2
            obj_cy = (obj_bbox[1] + obj_bbox[3]) // 2

            for person_bbox, person_conf in persons:
                px_cx = (person_bbox[0] + person_bbox[2]) // 2
                px_cy = person_bbox[3]  # bottom of person (feet)
                dist = ((obj_cx - px_cx) ** 2 + (obj_cy - px_cy) ** 2) ** 0.5

                if dist < min_dist and dist < w * 0.2:  # Within 20% of frame width
                    min_dist = dist
                    matched_person = (person_bbox, person_conf)
                    matched_obj = (obj_bbox, obj_cls, obj_conf)

        if matched_person is None:
            return result

        person_bbox, person_conf = matched_person
        obj_bbox, obj_cls, obj_conf = matched_obj
        confidence = (person_conf + obj_conf) / 2

        # ── Step 3: Face detection ────────────────────────────────────────────
        x1p, y1p, x2p, y2p = person_bbox
        person_crop = frame[max(0, y1p):y2p, max(0, x1p):x2p]
        face_crop = self._extract_face(person_crop)
        body_crop = person_crop.copy()

        # ── Step 4: Plate extraction from nearby vehicle ──────────────────────
        plate_text = None
        if vehicles:
            # Use the nearest vehicle
            obj_cx = (obj_bbox[0] + obj_bbox[2]) // 2
            nearest_vehicle = min(vehicles, key=lambda v: abs((v[0][0] + v[0][2]) // 2 - obj_cx))
            vx1, vy1, vx2, vy2 = nearest_vehicle[0]
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size > 0:
                ocr_results = self.ocr.readtext(vehicle_crop, detail=0, paragraph=False)
                all_text = " ".join(ocr_results)
                match = PLATE_PATTERN.search(all_text.upper())
                if match:
                    plate_text = match.group(0).upper().replace(" ", "")

        result.detected = True
        result.confidence = round(confidence, 3)
        result.person_bbox = tuple(person_bbox)
        result.object_bbox = tuple(obj_bbox)
        result.face_crop = face_crop
        result.body_crop = body_crop
        result.full_frame = frame.copy()
        result.plate_number = plate_text
        result.object_class = LITTER_OBJECTS.get(obj_cls, "object")

        return result

    def _extract_face(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        if person_crop.size == 0:
            return None
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb)
        if not face_results.detections:
            return None
        det = face_results.detections[0]
        bb = det.location_data.relative_bounding_box
        h, w = person_crop.shape[:2]
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        fw = int(bb.width * w)
        fh = int(bb.height * h)
        face = person_crop[y:y + fh, x:x + fw]
        return face if face.size > 0 else None

    def _objects_to_set(self, objects: list, frame_w: int, frame_h: int) -> set:
        return {self._object_key(bbox, cls, frame_w, frame_h) for bbox, cls, _ in objects}

    def _object_key(self, bbox: np.ndarray, cls: int, frame_w: int, frame_h: int) -> tuple:
        """Bin object position for stable comparison across frames."""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        # 20x20 grid bins
        return (cls, cx // (frame_w // 20), cy // (frame_h // 20))

    def _update_history(self, objects: list, frame_w: int, frame_h: int):
        self._ground_objects_history.append(self._objects_to_set(objects, frame_w, frame_h))

    def draw_overlay(self, frame: np.ndarray, detection: ViolationDetection) -> np.ndarray:
        if not detection.detected:
            return frame
        x1, y1, x2, y2 = detection.person_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
        label = f"⚠ LITTERING ({detection.confidence:.0%})"
        cv2.rectangle(frame, (x1, y1 - 36), (x2, y1), (0, 100, 200), -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if detection.plate_number:
            ox1, oy1, ox2, oy2 = detection.object_bbox
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 200, 100), 2)
        return frame
