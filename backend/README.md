# City Intelligence Platform - Backend

This directory contains the FastAPI-based backend and AI pipelines for the City Intelligence Platform.

## Architecture Overview

- **FastAPI**: Core API server providing both REST endpoints and real-time WebSockets.
- **AI Modules**: 
  - **Emergency Detection**: YOLOv8 (vehicle detection) + EasyOCR (text reading like "AMBULANCE", ANPR) + Flash detection.
  - **Civic Violations**: YOLOv8 (person/litter detection) + MediaPipe (face capture) + temporal action analysis.
- **Database**: Async SQLAlchemy with PostgreSQL for persisting events, violations, and user data.
- **Real-time**: Redis Pub/Sub for broadcasting events to connected WebSocket clients (dashboards).
- **Camera Ingestion**: Robust, thread-safe MJPEG streaming supporting webcams, RTSP, and local video files.

## Project Structure

- `/auth` - JWT authentication, password hashing, and role-based access logic.
- `/camera` - Frame buffer, camera sources, and MJPEG streaming.
- `/database` - SQLAlchemy models, async session management.
- `/modules` - The core AI logic:
  - `/emergency` - Emergency vehicle detection and signal preemption logic.
  - `/violation` - Littering and civic violation detection.
- `/websocket` - Connection manager for managing active WebSocket clients.

## Development Setup

1. **Prerequisites**: Python 3.10+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
   Copy `.env.example` to `.env`. Ensure your PostgreSQL and Redis instances are running (via Docker Compose in the root directory).
4. **Run the Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Adding New AI Modules

To add a new AI module, follow the structure in `modules/emergency` or `modules/violation`. Create a dedicated detector class, a router for its endpoints, and integrate it into `main.py` as a background task.
