"""
Emergency Vehicle Detector
--------------------------
Detects ambulances, fire trucks, and police vehicles using a 3-pronged approach:
  1. YOLOv8 vehicle detection → filters to car/truck/bus
  2. EasyOCR text recognition → looks for AMBULANCE, FIRE, POLICE on vehicle body
  3. Flashing light detection → brightness pulse analysis across consecutive frames

Returns EmergencyDetection dataclass with confidence and evidence.
"""

import cv2
import numpy as np
import time
import re
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO
import easyocr
from config import get_settings

settings = get_settings()

# Vehicle classes in COCO dataset that YOLOv8 uses
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
EMERGENCY_KEYWORDS = ["AMBULANCE", "FIRE", "POLICE", "108", "101", "100", "FIRETRUCK"]

# Regex pattern for Indian number plates: XX 00 XX 0000
PLATE_PATTERN = re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}", re.IGNORECASE)


@dataclass
class EmergencyDetection:
    detected: bool = False
    vehicle_type: str = ""
    plate_number: Optional[str] = None
    confidence: float = 0.0
    bbox: tuple = ()
    text_evidence: list[str] = field(default_factory=list)
    flash_detected: bool = False
    snapshot: Optional[np.ndarray] = None


class EmergencyDetector:
    def __init__(self, model_path: str = None):
        model_path = model_path or settings.YOLO_MODEL_PATH
        print(f"[EmergencyDetector] Loading YOLO from {model_path}")
        self.model = YOLO(model_path)
        self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        self._prev_brightness: list[float] = []
        self._flash_window = 10  # frames to analyse for flash

    def detect(self, frame: np.ndarray) -> EmergencyDetection:
        """Run full emergency detection pipeline on a single frame."""
        result = EmergencyDetection()

        # ── Step 1: YOLOv8 vehicle detection ─────────────────────────────────
        results = self.model(
            frame,
            conf=settings.YOLO_CONF_THRESHOLD,
            imgsz=settings.YOLO_IMG_SIZE,
            verbose=False,
        )

        best_vehicle = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in VEHICLE_CLASSES and conf > best_conf:
                    best_conf = conf
                    best_vehicle = (box.xyxy[0].cpu().numpy().astype(int), VEHICLE_CLASSES[cls])

        if best_vehicle is None:
            return result  # No vehicle found

        bbox, vehicle_class = best_vehicle
        x1, y1, x2, y2 = bbox
        vehicle_crop = frame[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return result

        # ── Step 2: OCR on vehicle body ───────────────────────────────────────
        ocr_results = self.ocr.readtext(vehicle_crop, detail=0, paragraph=False)
        found_keywords = []
        vehicle_type = ""

        for text in ocr_results:
            text_upper = text.upper().strip()
            for kw in EMERGENCY_KEYWORDS:
                if kw in text_upper:
                    found_keywords.append(text_upper)
                    vehicle_type = kw if kw not in ["108", "101", "100"] else "AMBULANCE"

        # ── Step 3: Plate extraction ──────────────────────────────────────────
        plate_text = None
        all_text = " ".join(ocr_results)
        match = PLATE_PATTERN.search(all_text.upper())
        if match:
            plate_text = match.group(0).upper().replace(" ", "")

        # ── Step 4: Flash detection ───────────────────────────────────────────
        flash = self._detect_flash(vehicle_crop)

        # ── Step 5: Confidence scoring ────────────────────────────────────────
        confidence = best_conf
        if found_keywords:
            confidence = min(1.0, confidence + 0.3)
        if flash:
            confidence = min(1.0, confidence + 0.2)

        # Emergency confirmed if: keyword found OR (flash + vehicle detected)
        is_emergency = bool(found_keywords) or (flash and best_conf > 0.5)

        if is_emergency:
            if not vehicle_type:
                vehicle_type = "EMERGENCY"
            result.detected = True
            result.vehicle_type = vehicle_type
            result.plate_number = plate_text
            result.confidence = round(confidence, 3)
            result.bbox = tuple(bbox)
            result.text_evidence = found_keywords
            result.flash_detected = flash
            result.snapshot = frame.copy()

        return result

    def _detect_flash(self, crop: np.ndarray) -> bool:
        """
        Detect flashing lights by tracking mean brightness over recent frames.
        A flash is detected when brightness oscillates rapidly (ambulance beacons).
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        self._prev_brightness.append(brightness)

        if len(self._prev_brightness) > self._flash_window:
            self._prev_brightness.pop(0)

        if len(self._prev_brightness) < 4:
            return False

        # Count alternating peaks and troughs
        diffs = np.diff(self._prev_brightness)
        sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs)))) // 2)
        return sign_changes >= 2  # 2+ oscillations in the window → flash

    def draw_overlay(self, frame: np.ndarray, detection: EmergencyDetection) -> np.ndarray:
        """Draw bounding box and labels on frame."""
        if not detection.detected or not detection.bbox:
            return frame

        x1, y1, x2, y2 = detection.bbox
        color = (0, 50, 255)  # Red-orange for emergency
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"🚨 {detection.vehicle_type}"
        if detection.plate_number:
            label += f" | {detection.plate_number}"
        label += f" ({detection.confidence:.0%})"

        cv2.rectangle(frame, (x1, y1 - 36), (x2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detection.flash_detected:
            cv2.putText(frame, "⚡ FLASH", (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return frame
