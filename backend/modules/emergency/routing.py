"""
Signal Preemption & Routing Engine
------------------------------------
When an emergency vehicle is detected at a junction, this module:
  1. Calculates the vehicle's trajectory direction
  2. Determines which junctions lie along that route
  3. Sets those junctions to PREEMPTED (green) state
  4. Schedules signal reversion after a timeout
"""

import asyncio
import json
import math
from datetime import datetime, timedelta
from typing import Optional
import redis.asyncio as aioredis
from config import get_settings

settings = get_settings()

# Demo city junction layout — will be replaced with DB data in production
# Format: id → {lat, lon, name, next_junctions: [ids]}
DEMO_JUNCTIONS = {
    "JCT_001": {"lat": 12.9716, "lon": 77.5946, "name": "MG Road Junction", "next": ["JCT_002", "JCT_005"]},
    "JCT_002": {"lat": 12.9730, "lon": 77.5960, "name": "Brigade Road", "next": ["JCT_003"]},
    "JCT_003": {"lat": 12.9745, "lon": 77.5975, "name": "Residency Road", "next": ["JCT_004"]},
    "JCT_004": {"lat": 12.9758, "lon": 77.5990, "name": "Richmond Circle", "next": []},
    "JCT_005": {"lat": 12.9700, "lon": 77.5930, "name": "Lalbagh Road", "next": ["JCT_006"]},
    "JCT_006": {"lat": 12.9685, "lon": 77.5915, "name": "DVG Road", "next": []},
}


class SignalManager:
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._preemption_tasks: dict[str, asyncio.Task] = {}

    async def get_redis(self):
        if self._redis is None:
            self._redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def get_all_signal_states(self) -> dict[str, dict]:
        """Return signal state for all junctions (from Redis or defaults)."""
        r = await self.get_redis()
        states = {}
        for jid, jdata in DEMO_JUNCTIONS.items():
            raw = await r.get(f"signal:{jid}")
            if raw:
                states[jid] = json.loads(raw)
            else:
                states[jid] = {
                    "id": jid,
                    "name": jdata["name"],
                    "lat": jdata["lat"],
                    "lon": jdata["lon"],
                    "state": "RED",
                    "is_preempted": False,
                }
        return states

    async def preempt_route(
        self,
        start_junction_id: str,
        vehicle_type: str,
        plate: Optional[str],
        depth: int = 3,
    ) -> list[str]:
        """
        Set the next `depth` junctions along route to GREEN (preempted).
        Returns list of cleared junction IDs.
        """
        cleared = []
        queue = [start_junction_id]
        visited = set()

        while queue and len(cleared) < depth:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            await self._set_preempted(current_id, vehicle_type, plate)
            cleared.append(current_id)

            jdata = DEMO_JUNCTIONS.get(current_id, {})
            for next_id in jdata.get("next", []):
                if next_id not in visited:
                    queue.append(next_id)

        # Schedule reversion
        task_key = f"revert_{start_junction_id}"
        if task_key in self._preemption_tasks:
            self._preemption_tasks[task_key].cancel()
        self._preemption_tasks[task_key] = asyncio.create_task(
            self._revert_after_delay(cleared, delay=90)
        )

        return cleared

    async def _set_preempted(self, junction_id: str, vehicle_type: str, plate: Optional[str]):
        """Set a single junction to preempted (GREEN) state in Redis."""
        r = await self.get_redis()
        jdata = DEMO_JUNCTIONS.get(junction_id, {})
        state = {
            "id": junction_id,
            "name": jdata.get("name", junction_id),
            "lat": jdata.get("lat", 0),
            "lon": jdata.get("lon", 0),
            "state": "PREEMPTED",
            "is_preempted": True,
            "preempted_by": vehicle_type,
            "plate": plate,
            "preempted_at": datetime.utcnow().isoformat(),
        }
        await r.set(f"signal:{junction_id}", json.dumps(state), ex=120)
        print(f"[SignalManager] Junction {junction_id} → PREEMPTED for {vehicle_type} ({plate})")

    async def _revert_after_delay(self, junction_ids: list[str], delay: int = 90):
        """Revert junctions back to normal after the vehicle has passed."""
        await asyncio.sleep(delay)
        r = await self.get_redis()
        for jid in junction_ids:
            jdata = DEMO_JUNCTIONS.get(jid, {})
            state = {
                "id": jid,
                "name": jdata.get("name", jid),
                "lat": jdata.get("lat", 0),
                "lon": jdata.get("lon", 0),
                "state": "RED",
                "is_preempted": False,
            }
            await r.set(f"signal:{jid}", json.dumps(state), ex=3600)
            print(f"[SignalManager] Junction {jid} → NORMAL (revert after {delay}s)")

    async def manual_override(self, junction_id: str, state: str):
        """Operator manual signal override."""
        r = await self.get_redis()
        jdata = DEMO_JUNCTIONS.get(junction_id, {})
        current_raw = await r.get(f"signal:{junction_id}")
        current = json.loads(current_raw) if current_raw else {}
        current.update({
            "id": junction_id,
            "name": jdata.get("name", junction_id),
            "state": state,
            "is_preempted": False,
            "manual_override": True,
        })
        await r.set(f"signal:{junction_id}", json.dumps(current), ex=3600)


signal_manager = SignalManager()
