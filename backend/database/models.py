import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, Float, Boolean, Text, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.session import Base


class UserRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    TRAFFIC_POLICE_HQ = "traffic_police_hq"
    MUNICIPAL_HQ = "municipal_hq"
    JUNCTION_OPERATOR = "junction_operator"


class TrafficStatus(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SignalState(str, enum.Enum):
    GREEN = "GREEN"
    RED = "RED"
    YELLOW = "YELLOW"
    PREEMPTED = "PREEMPTED"  # Emergency override


class ViolationType(str, enum.Enum):
    LITTERING = "LITTERING"
    RED_LIGHT = "RED_LIGHT"
    WRONG_WAY = "WRONG_WAY"


# ─── Users ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128))
    hashed_password: Mapped[str] = mapped_column(String(256))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.JUNCTION_OPERATOR)
    department: Mapped[str] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─── Cameras ──────────────────────────────────────────────────────────────────

class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g. "CAM_001"
    name: Mapped[str] = mapped_column(String(128))
    location_name: Mapped[str] = mapped_column(String(256))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(512))  # file path, 0, or rtsp://
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    junction_id: Mapped[str] = mapped_column(String(32), ForeignKey("junctions.id"), nullable=True)


# ─── Junctions ────────────────────────────────────────────────────────────────

class Junction(Base):
    __tablename__ = "junctions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g. "JCT_001"
    name: Mapped[str] = mapped_column(String(128))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    signal_state: Mapped[SignalState] = mapped_column(Enum(SignalState), default=SignalState.RED)
    is_preempted: Mapped[bool] = mapped_column(Boolean, default=False)
    preempted_until: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    cameras: Mapped[list["Camera"]] = relationship("Camera", backref="junction")


# ─── Emergency Events ─────────────────────────────────────────────────────────

class EmergencyEvent(Base):
    __tablename__ = "emergency_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    vehicle_type: Mapped[str] = mapped_column(String(64))  # AMBULANCE, FIRE, POLICE
    plate_number: Mapped[str] = mapped_column(String(32), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    camera_id: Mapped[str] = mapped_column(String(32), ForeignKey("cameras.id"))
    junction_id: Mapped[str] = mapped_column(String(32), ForeignKey("junctions.id"))
    route_cleared: Mapped[str] = mapped_column(Text, nullable=True)  # JSON list of junction IDs
    confidence: Mapped[float] = mapped_column(Float)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    snapshot_path: Mapped[str] = mapped_column(String(512), nullable=True)


# ─── Civic Violations ─────────────────────────────────────────────────────────

class CivicViolation(Base):
    __tablename__ = "civic_violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    violation_type: Mapped[ViolationType] = mapped_column(Enum(ViolationType))
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    camera_id: Mapped[str] = mapped_column(String(32), ForeignKey("cameras.id"))
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    plate_number: Mapped[str] = mapped_column(String(32), nullable=True)
    face_image_path: Mapped[str] = mapped_column(String(512), nullable=True)
    body_image_path: Mapped[str] = mapped_column(String(512), nullable=True)
    full_frame_path: Mapped[str] = mapped_column(String(512), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")  # PENDING, REVIEWED, FINED
    reviewed_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
