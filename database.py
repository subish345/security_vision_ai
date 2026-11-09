# ==============================================================================
# Security Vision Database (Async SQLite in WAL Mode / PostgreSQL Ready)
# Features Orbiter / Demo 11 Alert Lifecycle States & Operator Tracking
# ==============================================================================

import datetime
import uuid
import yaml
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, JSON, event, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

DB_URL = CONFIG["database"]["url"]

engine = create_async_engine(DB_URL, echo=False, future=True)

# Enable Write-Ahead Logging (WAL) for SQLite to provide high-concurrency read/write speed
if DB_URL.startswith("sqlite"):
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()


class SecurityAlertLog(Base):
    __tablename__ = "security_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(64), unique=True, index=True, nullable=False)
    module_name = Column(String(32), index=True, nullable=False)     # CONSTRUCTION | OFFICE | RESTAURANT
    alert_type = Column(String(64), index=True, nullable=False)      # PPE_VIOLATION | ACTIVE_PHONE_USAGE | etc.
    table_or_zone_id = Column(String(64), index=True, nullable=False)
    severity = Column(String(16), nullable=False)                    # CRITICAL | WARNING | NOTICE | INFO
    message = Column(String(512), nullable=False)
    snapshot_path = Column(String(256), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Orbiter / Demo 11 Alert Lifecycle States: RECEIVED -> ACKNOWLEDGED -> RESOLVED
    lifecycle_state = Column(String(24), default="RECEIVED", index=True)
    acknowledged_by = Column(String(64), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(64), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


async def init_db():
    """Initializes the database tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def log_alert_to_db(alert_dict: dict) -> str:
    """Inserts a new alert log asynchronously."""
    alert_id = alert_dict.get("alert_id") or f"ALT-{uuid.uuid4().hex[:8].upper()}"
    alert_dict["alert_id"] = alert_id
    alert_dict["lifecycle_state"] = "RECEIVED"

    async with AsyncSessionLocal() as session:
        record = SecurityAlertLog(
            alert_id=alert_id,
            module_name=alert_dict["module_name"],
            alert_type=alert_dict["alert_type"],
            table_or_zone_id=alert_dict["table_or_zone_id"],
            severity=alert_dict["severity"],
            message=alert_dict["message"],
            snapshot_path=alert_dict.get("snapshot_path"),
            metadata_json=alert_dict.get("metadata", {}),
            lifecycle_state="RECEIVED",
        )
        session.add(record)
        await session.commit()
    return alert_id


async def acknowledge_alert_in_db(alert_id: str, operator_name: str = "Operator") -> bool:
    """Transitions alert lifecycle: RECEIVED -> ACKNOWLEDGED."""
    now = datetime.datetime.utcnow()
    async with AsyncSessionLocal() as session:
        stmt = (
            update(SecurityAlertLog)
            .where(SecurityAlertLog.alert_id == alert_id)
            .values(
                lifecycle_state="ACKNOWLEDGED",
                acknowledged_by=operator_name,
                acknowledged_at=now,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def resolve_alert_in_db(alert_id: str, operator_name: str = "Operator") -> bool:
    """Transitions alert lifecycle: ACKNOWLEDGED -> RESOLVED."""
    now = datetime.datetime.utcnow()
    async with AsyncSessionLocal() as session:
        stmt = (
            update(SecurityAlertLog)
            .where(SecurityAlertLog.alert_id == alert_id)
            .values(
                lifecycle_state="RESOLVED",
                resolved_by=operator_name,
                resolved_at=now,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def get_recent_alerts(limit: int = 50, module: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves recent alerts sorted by descending timestamp."""
    async with AsyncSessionLocal() as session:
        stmt = select(SecurityAlertLog).order_by(SecurityAlertLog.timestamp.desc()).limit(limit)
        if module and module != "ALL":
            stmt = stmt.where(SecurityAlertLog.module_name == module)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "alert_id": r.alert_id,
                "module_name": r.module_name,
                "alert_type": r.alert_type,
                "table_or_zone_id": r.table_or_zone_id,
                "severity": r.severity,
                "message": r.message,
                "snapshot_path": r.snapshot_path,
                "metadata": r.metadata_json,
                "lifecycle_state": r.lifecycle_state,
                "timestamp": r.timestamp.strftime("%H:%M:%S") if r.timestamp else "",
            }
            for r in rows
        ]

