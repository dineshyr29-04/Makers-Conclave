# City Intelligence Platform
### Makers Conclave Demo

A Government-Grade AI Platform for Unified City Monitoring, Traffic Preemption, and Civic Violation Detection.

---

## 🗺️ What This Project Is

This system acts as a **smart brain** for a city's camera network, processing footage in real-time to manage road conditions and public safety.

### Core Modules:
1. **🚨 Emergency Vehicle Routing**: Detects emergency vehicles using YOLOv8, extracts text ("AMBULANCE") and license plates using EasyOCR, and identifies flashing beacon lights. It then dynamically preempts traffic signals along the route, turning them green to clear traffic.
2. **⚠️ Civic Violation Detection**: Detects littering and other civic offenses using temporal frame analysis to detect "dropping" actions. Automatically captures face crops, body shots, and license plates of nearby vehicles for municipal reports.

---

## 🏗️ Architecture

- **Backend (`/backend`)**: FastAPI, Async PostgreSQL, Redis Pub/Sub, YOLOv8, MediaPipe, EasyOCR, OpenCV.
- **Frontend (`/frontend`)**: React 19, Vite, Leaflet, Custom Design System, WebSockets.
- **Infrastructure**: Docker for PostgreSQL and Redis.

---

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

### 1. Start Infrastructure
```bash
docker-compose up -d
```
*(Runs PostgreSQL on port 5433 and Redis on port 6380 to avoid local port conflicts).*

### 2. Setup Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open Dashboard
Navigate to **http://localhost:5173** (or the port specified by Vite, e.g., 5174).

---

## 🔑 Demo Login Credentials

The backend automatically creates demo users on its first run:

| Role | Username | Password |
|------|----------|----------|
| Super Admin | `admin` | `admin123` |
| Traffic Police HQ | `traffic_hq` | `traffic123` |
| Municipal Corp HQ | `municipal_hq` | `municipal123` |
| Junction Operator | `operator1` | `op123` |

---

## 📹 Providing Sample Videos

To test the AI pipelines effectively, place demo videos in `data/sample_videos/`:
- `traffic1.mp4` — Mapped to CAM_001 (Emergency Vehicle Module)
- `traffic2.mp4` — Mapped to CAM_002 (Littering Module)

*If these videos are not present, the system defaults to using the webcam (`CAM_WEBCAM`).*

---

## 🤝 Contribution Guidelines

We welcome contributions to expand the platform!

1. **Understand the Architecture**: Read `backend/README.md` and `frontend/README.md`.
2. **Code Style**:
   - Backend: Use type hints, async/await where possible, and document complex AI logic.
   - Frontend: Adhere to the CSS variables in `index.css` to maintain the dark, glassmorphism UI.
3. **Atomic Commits**: Group related changes logically and write descriptive commit messages.

*Built for Makers Conclave.*
