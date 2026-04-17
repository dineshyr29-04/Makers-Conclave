# Smart Traffic Demo

A beginner-friendly Smart Traffic Demo built with YOLOv8 and OpenCV.

## Features

- Real-time vehicle detection with YOLOv8n
- Real-time vehicle detection and tracking with YOLOv8
- Supports webcam or a sample traffic video file
- Detects only `car`, `bus`, `truck`, and `motorcycle`
- Shows bounding boxes, labels, total vehicle count, traffic status, and recommended green time
- Includes a counting line based on YOLOv8 track IDs and FPS display
- Designed to be lightweight enough to try on a Raspberry Pi with the YOLOv8n model

## Installation

Open a terminal in VS Code inside this folder, then run:

```bash
pip install -r requirements.txt
```

If the model file is not already available, Ultralytics will download `yolov8n.pt` automatically the first time you run the program.

## How To Run

### In VS Code

1. Open this folder in VS Code.
2. Install the Python extension if it is not already installed.
3. Press `F5` and choose `Smart Traffic Demo: Webcam` or `Smart Traffic Demo: Video File`.

### Webcam input

```bash
python main.py --source 0
```

If your Windows setup uses the Python launcher, this also works:

```bash
py main.py --source 0
```

### Video file input

```bash
python main.py --source sample_traffic.mp4
```

If your video file is in another folder, use the full path:

```bash
python main.py --source C:\\path\\to\\traffic_video.mp4
```

## Sample Video Input Instructions

1. Put your traffic video file in this project folder, or keep note of its full path.
2. Rename it to something simple like `sample_traffic.mp4` if you want.
3. Run the video command above.
4. Press `q` or `Esc` to quit the window.

## Traffic Logic

- Fewer than 5 vehicles: `LOW TRAFFIC` and `Green Time: 20s`
- 5 to 15 vehicles: `MEDIUM TRAFFIC` and `Green Time: 40s`
- More than 15 vehicles: `HIGH TRAFFIC` and `Green Time: 60s`

## Useful Options

```bash
python main.py --source 0 --conf 0.4 --img-size 640 --max-width 960 --tracker bytetrack.yaml
```

- `--conf` controls detection confidence.
- `--img-size` controls inference resolution.
- `--max-width` controls resize speed for faster processing.
- `--line-ratio` changes the counting line position.
- `--tracker` selects YOLOv8 tracker config (`bytetrack.yaml` or `botsort.yaml`).
- `--device` sets inference device (`cpu`, `0`, `0,1`, etc.).

## Notes For Raspberry Pi

- Keep using `yolov8n.pt`
- Lower `--img-size` if the device feels slow
- Lower `--max-width` to reduce processing cost
- Use a USB webcam or a low-resolution video for better speed

## What To Expect

The display shows:

- Bounding boxes around vehicles
- Vehicle type labels
- Total vehicle count in the top-left panel
- LOW, MEDIUM, or HIGH traffic status
- Recommended green signal time
- FPS
- Counting line and line-crossing count
