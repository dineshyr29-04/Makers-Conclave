import asyncio
import json
from typing import Any
from fastapi import WebSocket
import redis.asyncio as aioredis
from config import get_settings

settings = get_settings()


class ConnectionManager:
    """
    Manages WebSocket connections per role.
    When an event is broadcast, it's sent to all connected clients
    that have access to that event type.
    """

    def __init__(self):
        # role -> list of active WebSocket connections
        self.connections: dict[str, list[WebSocket]] = {}
        self._redis: aioredis.Redis | None = None

    async def get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def connect(self, websocket: WebSocket, role: str):
        await websocket.accept()
        if role not in self.connections:
            self.connections[role] = []
        self.connections[role].append(websocket)

    def disconnect(self, websocket: WebSocket, role: str):
        if role in self.connections:
            self.connections[role].remove(websocket)

    async def broadcast_to_roles(self, message: dict[str, Any], roles: list[str]):
        """Send message to all connections of specified roles."""
        text = json.dumps(message)
        for role in roles:
            dead = []
            for ws in self.connections.get(role, []):
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.connections[role].remove(ws)

    async def broadcast_all(self, message: dict[str, Any]):
        """Send message to ALL connected clients (super admin events)."""
        all_roles = list(self.connections.keys())
        await self.broadcast_to_roles(message, all_roles)

    async def publish_to_redis(self, channel: str, message: dict[str, Any]):
        """Publish event to Redis pub/sub for cross-process delivery."""
        r = await self.get_redis()
        await r.publish(channel, json.dumps(message))

    async def store_recent_alert(self, alert_type: str, data: dict[str, Any], max_count: int = 50):
        """Store recent alerts in Redis list for dashboard on-load."""
        r = await self.get_redis()
        key = f"alerts:{alert_type}"
        await r.lpush(key, json.dumps(data))
        await r.ltrim(key, 0, max_count - 1)
        await r.expire(key, 86400)  # 24h TTL

    async def get_recent_alerts(self, alert_type: str, limit: int = 20) -> list[dict]:
        r = await self.get_redis()
        key = f"alerts:{alert_type}"
        items = await r.lrange(key, 0, limit - 1)
        return [json.loads(i) for i in items]


# Singleton instance
manager = ConnectionManager()
