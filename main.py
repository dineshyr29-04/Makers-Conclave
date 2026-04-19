"""Smart Traffic Demo using YOLOv8.

Features:
- Real-time vehicle detection from webcam or video file
- Counts car, bus, truck, and motorcycle only
- Traffic density logic with signal timing
- On-screen bounding boxes, labels, total count, status, and FPS
- Line-crossing counter using YOLOv8 track IDs
"""

from __future__ import annotations

import argparse
import os

"""
Advanced Smart Traffic Management System Simulation using YOLOv8n
Author: GitHub Copilot

Features:
- Real-time vehicle detection (YOLOv8n)
- 2-lane simulation (LEFT, RIGHT)
- Per-lane vehicle counting, bounding boxes, and lane divider
- Adaptive traffic signal logic (GREEN/YELLOW/RED, per lane)
- Green time adapts to density: LOW (<5): 15s, MEDIUM (5–15): 30s, HIGH (>15): 50s
- Smooth transitions: Green → Yellow (3s) → Red
- Draws traffic lights, highlights active lane, shows countdown, FPS, and system status
- Optimized for real-time (frame resize, YOLOv8n)
"""

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO

 # --- CONFIG ---
# Add emergency vehicle class index if your model supports it (e.g., 21: "ambulance")
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 21: "ambulance"}  # Add/adjust index for emergency vehicle
CLASS_COLORS = {"car": (80, 220, 120), "motorcycle": (255, 180, 0), "bus": (255, 120, 80), "truck": (80, 180, 255), "ambulance": (255, 0, 255)}
LANE_NAMES = ["LEFT", "RIGHT"]
SIGNAL_COLORS = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 200, 0)}
LANE_STATUS = {"ACTIVE": (0, 200, 0), "WAITING": (0, 0, 255)}

# Set this to 0 for LEFT lane, 1 for RIGHT lane
SELECTED_LANE = 0  # Change to 1 for RIGHT lane only

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Traffic Management System (YOLOv8)")
    parser.add_argument("--source", default="sample_traffic.mp4", help="Video file or webcam index (default: sample_traffic.mp4)")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--img-size", type=int, default=1280, help="Inference image size (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--max-width", type=int, default=1600, help="Resize frame width for speed (default: 1600)")
    return parser.parse_args()

def open_capture(source):
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
        return cap
    return cv2.VideoCapture(source)

def resize_frame(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def get_lane_regions(width, height, num_lanes=2):
    # Returns list of (x1, y1, x2, y2) for each lane
    lane_width = width // num_lanes
    return [(i * lane_width, 0, (i + 1) * lane_width, height) for i in range(num_lanes)]

def get_density_label(count):
    if count < 5:
        return "LOW", 15
    elif count <= 15:
        return "MEDIUM", 30
    else:
        return "HIGH", 50

def draw_lane_dividers(frame, num_lanes=2):
    h, w = frame.shape[:2]
    lane_width = w // num_lanes
    for i in range(1, num_lanes):
        x = i * lane_width
        cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 2, cv2.LINE_AA)

def draw_traffic_light(frame, center, state, radius=18):
    # Draws a traffic light (red/yellow/green) at center
    for idx, color in enumerate(["RED", "YELLOW", "GREEN"]):
        offset = (idx - 1) * 2 * radius
        c = (center[0], center[1] + offset)
        fill = SIGNAL_COLORS[color] if state == color else (60, 60, 60)
        cv2.circle(frame, c, radius, fill, -1)
        cv2.circle(frame, c, radius, (40, 40, 40), 2)

def draw_panel(frame, lines, pos=(10, 10), width=420, height=240, color=(0, 0, 0)):
    overlay = frame.copy()
    cv2.rectangle(overlay, pos, (pos[0] + width, pos[1] + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, pos, (pos[0] + width, pos[1] + height), color, 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    y = pos[1] + 38
    for i, text in enumerate(lines):
        font_scale = 0.7 if i == 0 else 0.6
        thickness = 2 if i == 0 else 1
        txt_color = (255, 255, 255) if i == 0 else color
        cv2.putText(frame, text, (pos[0] + 18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness, cv2.LINE_AA)
        y += 32

def draw_detection(frame, box, class_name, conf, color, track_id=None):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {conf:.2f}"
    if track_id is not None:
        label += f" #{track_id}"
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_top = max(y1 - label_size[1] - baseline - 6, 0)
    label_bottom = label_top + label_size[1] + baseline + 6
    cv2.rectangle(frame, (x1, label_top), (x1 + label_size[0] + 8, label_bottom), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    args = parse_args()
    model = YOLO(args.weights)
    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    prev_time = time.time()
    smoothed_fps = 0.0
    lane_states = ["RED", "RED"]  # [LEFT, RIGHT]
    lane_timers = [0, 0]
    lane_green_times = [15, 15]
    lane_density_labels = ["LOW", "LOW"]
    active_lane = 0  # 0: LEFT, 1: RIGHT
    signal_state = "GREEN"  # GREEN, YELLOW, RED
    signal_timer = 0
    yellow_duration = 3
    min_red_time = 2  # Minimum red time to avoid rapid switching
    red_timer = 0
    switch_pending = False
    countdown = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame, args.max_width)
        h, w = frame.shape[:2]
        lane_regions = get_lane_regions(w, h, 2)
        draw_lane_dividers(frame, 2)

        # --- Detection (Single Lane, Emergency Logic) ---
        results = model.track(frame, persist=True, classes=list(VEHICLE_CLASSES.keys()), imgsz=args.img_size, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results and results[0].boxes is not None else []
        lane_ids = [set(), set()]
        emergency_present = False
        for box in boxes:
            class_id = int(box.cls.item())
            if class_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx = (x1 + x2) // 2
            lane_idx = 0 if cx < w // 2 else 1
            if lane_idx != SELECTED_LANE:
                continue  # Only process selected lane
            # Emergency vehicle logic
            if VEHICLE_CLASSES[class_id] == "ambulance":
                emergency_present = True
            # Use unique track IDs for perfect counting
            if box.id is not None:
                lane_ids[lane_idx].add(int(box.id.item()))
            draw_detection(frame, (x1, y1, x2, y2), VEHICLE_CLASSES[class_id], float(box.conf.item()), CLASS_COLORS[VEHICLE_CLASSES[class_id]], box.id.item() if box.id is not None else None)

        # Only keep counts and labels for the selected lane
        lane_counts = [0, 0]
        lane_counts[SELECTED_LANE] = len(lane_ids[SELECTED_LANE])
        total_count = lane_counts[SELECTED_LANE]
        for i in range(2):
            if i == SELECTED_LANE:
                lane_density_labels[i], lane_green_times[i] = get_density_label(lane_counts[i])
            else:
                lane_density_labels[i], lane_green_times[i] = ("N/A", 0)

        # --- Traffic Logic (Single Lane) ---
        now = time.time()
        dt = now - prev_time
        prev_time = now
        smoothed_fps = dt if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * dt)
        fps = 1.0 / max(smoothed_fps, 1e-6)

        # --- State Machine (Single Lane, Emergency Logic, 15-20 Vehicles for Green) ---
        most_congested = SELECTED_LANE
        # Emergency vehicle overrides all
        if emergency_present:
            signal_state = "GREEN"
            lane_states[SELECTED_LANE] = "GREEN"
            lane_states[1 - SELECTED_LANE] = "RED"
            lane_timers[SELECTED_LANE] = 0
            countdown = "EMERGENCY"
        elif 5 <= lane_counts[SELECTED_LANE] <= 10:
            # Only allow green if 15-20 vehicles
            if signal_state == "RED":
                # Insert yellow phase before green
                if signal_timer == 0:
                    signal_state = "YELLOW"
                    signal_timer = 0
                    lane_states[SELECTED_LANE] = "YELLOW"
                    lane_states[1 - SELECTED_LANE] = "RED"
                    countdown = yellow_duration
                else:
                    countdown = yellow_duration - int(signal_timer)
                    lane_states[SELECTED_LANE] = "YELLOW"
                    lane_states[1 - SELECTED_LANE] = "RED"
                    signal_timer += dt
                    if signal_timer >= yellow_duration:
                        signal_state = "GREEN"
                        lane_timers[SELECTED_LANE] = 0
                        signal_timer = 0
            elif signal_state == "YELLOW":
                countdown = yellow_duration - int(signal_timer)
                lane_states[SELECTED_LANE] = "YELLOW"
                lane_states[1 - SELECTED_LANE] = "RED"
                signal_timer += dt
                if signal_timer >= yellow_duration:
                    signal_state = "GREEN"
                    lane_timers[SELECTED_LANE] = 0
                    signal_timer = 0
            elif signal_state == "GREEN":
                countdown = lane_green_times[SELECTED_LANE] - int(lane_timers[SELECTED_LANE])
                lane_states[SELECTED_LANE] = "GREEN"
                lane_states[1 - SELECTED_LANE] = "RED"
                lane_timers[SELECTED_LANE] += dt
                if lane_timers[SELECTED_LANE] >= lane_green_times[SELECTED_LANE]:
                    signal_state = "YELLOW"
                    signal_timer = 0
                    switch_pending = True
            else:
                # Fallback to red
                signal_state = "RED"
                lane_states[SELECTED_LANE] = "RED"
                lane_states[1 - SELECTED_LANE] = "GREEN"
                lane_timers[SELECTED_LANE] = 0
                countdown = 0
        else:
            # Not in 15-20 range, always red
            signal_state = "RED"
            lane_states[SELECTED_LANE] = "RED"
            lane_states[1 - SELECTED_LANE] = "GREEN"
            lane_timers[SELECTED_LANE] = 0
            countdown = 0

        # --- UI Drawing (Single Lane) ---
        for i, (x1, y1, x2, y2) in enumerate(lane_regions):
            if i != SELECTED_LANE:
                continue
            center = (x1 + (x2 - x1) // 2, 60)
            draw_traffic_light(frame, center, lane_states[i])
            status = "ACTIVE" if lane_states[i] == "GREEN" else "WAITING"
            status_color = LANE_STATUS["ACTIVE"] if status == "ACTIVE" else LANE_STATUS["WAITING"]
            cv2.putText(frame, f"{LANE_NAMES[i]} LANE {status}", (x1 + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Count: {lane_counts[i]}", (x1 + 10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Density: {lane_density_labels[i]}", (x1 + 10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Highlight selected lane
        x1, y1, x2, y2 = lane_regions[SELECTED_LANE]
        cv2.rectangle(frame, (x1, 0), (x2, h), (0, 255, 255), 4)

        draw_panel(
            frame,
            [
                f"Total Vehicles: {total_count}",
                f"Lane: {LANE_NAMES[SELECTED_LANE]}",
                f"Signal: {lane_states[SELECTED_LANE]}",
                f"Countdown: {countdown}s",
                f"Count: {lane_counts[SELECTED_LANE]}",
                f"FPS: {fps:.1f}",
            ],
            pos=(10, h-250),
            width=420,
            height=220,
            color=SIGNAL_COLORS[lane_states[SELECTED_LANE]]
        )

        cv2.imshow("Smart Traffic Management System - Single Lane", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
            