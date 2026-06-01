from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    DATABASE_URL: str = "postgresql+asyncpg://cityai:cityai123@localhost:5433/cityai_db"
    REDIS_URL: str = "redis://localhost:6380"

    DEFAULT_CAMERA_SOURCE: str = "0"
    RTSP_URL: str = ""

    YOLO_MODEL_PATH: str = "../ai_models/yolov8n.pt"
    YOLO_CONF_THRESHOLD: float = 0.45
    YOLO_IMG_SIZE: int = 640

    EVIDENCE_DIR: str = "../data/evidence"
    MAPBOX_TOKEN: str = ""

    DEMO_MODE: bool = True

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
