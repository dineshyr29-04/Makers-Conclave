# City Intelligence Platform
### Makers Conclave Demo

An AI-powered unified city monitoring system for emergency vehicle routing and civic violation detection.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

### 1. Start Databases
```bash
docker-compose up -d
```

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
npm run dev
```

### 4. Open Dashboard
Navigate to **http://localhost:5173**

---

## Demo Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Super Admin | `admin` | `admin123` |
| Traffic Police HQ | `traffic_hq` | `traffic123` |
| Municipal Corp HQ | `municipal_hq` | `municipal123` |
| Junction Operator | `operator1` | `op123` |

---

## Sample Videos
Place demo videos in `data/sample_videos/`:
- `traffic1.mp4` — Used by CAM_001 (Emergency Vehicle Module)
- `traffic2.mp4` — Used by CAM_002 (Littering Module)

If videos are not present, the system will use the webcam (CAM_WEBCAM).

---

## Modules

### 🚨 Emergency Vehicle Routing
- Detects ambulances, fire trucks, police vehicles
- 3-pronged detection: YOLOv8 + OCR text + flash detection
- Automatically preempts traffic signals along the route
- Real-time alerts to Traffic Police HQ

### ⚠️ Civic Violation Detection
- Detects littering via temporal frame analysis
- Captures face, body, and full-frame evidence
- ANPR for nearby vehicle plates
- Reports routed to Municipal Corp HQ

---

## Architecture
See `backend/` for the FastAPI AI pipeline and `frontend/` for the React dashboard.
