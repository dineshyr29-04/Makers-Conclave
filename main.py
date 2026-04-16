"""Smart Traffic Demo using YOLOv8.

Features:
- Real-time vehicle detection from webcam or video file
- Counts car, bus, truck, and motorcycle only
- Traffic density logic with signal timing
- On-screen bounding boxes, labels, total count, status, and FPS
- Optional line-crossing counter using a lightweight centroid tracker
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "car": (80, 220, 120),
    "motorcycle": (255, 180, 0),
    "bus": (255, 120, 80),
    "truck": (80, 180, 255),
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    centroid: Tuple[int, int]
    object_id: Optional[int] = None
    previous_centroid: Optional[Tuple[int, int]] = None


class SimpleCentroidTracker:
    """A small tracker to keep stable IDs for line-crossing counts."""

    def __init__(self, max_distance: float = 70.0, max_missing: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_object_id = 1
        self.objects: Dict[int, Dict[str, object]] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            self._mark_missing([])
            return detections

        if not self.objects:
            for detection in detections:
                self._register(detection)
            return detections

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[object_id]["centroid"] for object_id in object_ids], dtype=np.float32)
        detection_centroids = np.array([detection.centroid for detection in detections], dtype=np.float32)

        distances = np.linalg.norm(object_centroids[:, None, :] - detection_centroids[None, :, :], axis=2)

        matched_objects = set()
        matched_detections = set()

        while True:
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_index]
            object_index, detection_index = int(min_index[0]), int(min_index[1])

            if not np.isfinite(min_distance) or min_distance > self.max_distance:
                break
            if object_index in matched_objects or detection_index in matched_detections:
                distances[object_index, detection_index] = np.inf
                continue

            object_id = object_ids[object_index]
            tracked_object = self.objects[object_id]
            detection = detections[detection_index]

            if tracked_object["class_id"] != detection.class_id:
                distances[object_index, detection_index] = np.inf
                continue

            detection.object_id = object_id
            detection.previous_centroid = tracked_object["centroid"]  # type: ignore[assignment]
            self.objects[object_id]["centroid"] = detection.centroid
            self.objects[object_id]["class_id"] = detection.class_id
            self.objects[object_id]["missing"] = 0

            matched_objects.add(object_index)
            matched_detections.add(detection_index)
            distances[object_index, :] = np.inf
            distances[:, detection_index] = np.inf

        for index, detection in enumerate(detections):
            if detection.object_id is None:
                self._register(detection)

        self._mark_missing([detection.object_id for detection in detections if detection.object_id is not None])
        return detections

    def _register(self, detection: Detection) -> None:
        detection.object_id = self.next_object_id
        self.objects[self.next_object_id] = {
            "centroid": detection.centroid,
            "class_id": detection.class_id,
            "missing": 0,
        }
        self.next_object_id += 1

    def _mark_missing(self, active_object_ids: List[Optional[int]]) -> None:
        active_ids = {object_id for object_id in active_object_ids if object_id is not None}
        to_remove = []

        for object_id, tracked_object in self.objects.items():
            if object_id in active_ids:
                continue
            tracked_object["missing"] = int(tracked_object["missing"]) + 1
            if int(tracked_object["missing"]) > self.max_missing:
                to_remove.append(object_id)

        for object_id in to_remove:
            del self.objects[object_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Traffic Demo using YOLOv8")
    parser.add_argument("--source", default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--max-width", type=int, default=960, help="Resize frame width for speed")
    parser.add_argument("--line-ratio", type=float, default=0.6, help="Counting line position as a fraction of frame height")
    parser.add_argument("--tracker-distance", type=float, default=70.0, help="Maximum centroid distance for line tracking")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        camera_index = int(source)
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(camera_index)
        return capture
    return cv2.VideoCapture(source)


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def traffic_profile(vehicle_count: int) -> Tuple[str, int, Tuple[int, int, int]]:
    if vehicle_count < 5:
        return "LOW TRAFFIC", 20, (0, 200, 0)
    if vehicle_count <= 15:
        return "MEDIUM TRAFFIC", 40, (0, 180, 255)
    return "HIGH TRAFFIC", 60, (0, 0, 255)


def draw_panel(frame: np.ndarray, lines: List[str], bg_color: Tuple[int, int, int]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (400, 180), bg_color, 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 42
    for index, text in enumerate(lines):
        font_scale = 0.7 if index == 0 else 0.6
        thickness = 2 if index == 0 else 1
        color = (255, 255, 255) if index else bg_color
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += 30


def annotate_detection(frame: np.ndarray, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.bbox
    color = CLASS_COLORS.get(detection.class_name, (255, 255, 255))
    label = f"{detection.class_name} {detection.confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_top = max(y1 - label_size[1] - baseline - 6, 0)
    label_bottom = label_top + label_size[1] + baseline + 6
    cv2.rectangle(frame, (x1, label_top), (x1 + label_size[0] + 8, label_bottom), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def detect_vehicles(model: YOLO, frame: np.ndarray, device: str, img_size: int, conf: float) -> List[Detection]:
    results = model.predict(frame, imgsz=img_size, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    detections: List[Detection] = []

    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        if class_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        class_name = VEHICLE_CLASSES[class_id]
        confidence = float(box.conf.item())

        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                centroid=centroid,
            )
        )

    return detections


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"Downloading or loading weights: {args.weights}")

    model = YOLO(args.weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = SimpleCentroidTracker(max_distance=args.tracker_distance, max_missing=12)
    counted_ids = set()

    capture = open_capture(args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous_time = time.time()
    smoothed_fps = 0.0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = resize_frame(frame, args.max_width)
        height, width = frame.shape[:2]
        line_y = int(height * args.line_ratio)

        detections = detect_vehicles(model, frame, device, args.img_size, args.conf)
        detections = tracker.update(detections)

        for detection in detections:
            annotate_detection(frame, detection)

            if detection.object_id is not None and detection.previous_centroid is not None:
                prev_y = detection.previous_centroid[1]
                current_y = detection.centroid[1]
                crossed_line = (prev_y < line_y <= current_y) or (prev_y > line_y >= current_y)
                if crossed_line and detection.object_id not in counted_ids:
                    counted_ids.add(detection.object_id)

        total_vehicle_count = len(detections)
        traffic_label, green_time, traffic_color = traffic_profile(total_vehicle_count)

        current_time = time.time()
        frame_time = current_time - previous_time
        previous_time = current_time
        fps = 1.0 / max(frame_time, 1e-6)
        smoothed_fps = fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * fps)

        cv2.line(frame, (0, line_y), (width, line_y), traffic_color, 2)
        cv2.putText(frame, "COUNTING LINE", (12, max(line_y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, traffic_color, 2, cv2.LINE_AA)

        draw_panel(
            frame,
            [
                f"Vehicle Count: {total_vehicle_count}",
                f"Status: {traffic_label}",
                f"Green Time: {green_time}s",
                f"FPS: {smoothed_fps:.1f}",
                f"Line Crossings: {len(counted_ids)}",
            ],
            traffic_color,
        )

        cv2.imshow("Smart Traffic Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
