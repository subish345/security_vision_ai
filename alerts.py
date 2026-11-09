# ==============================================================================
# Real-Time Alert Manager & Snapshot Dispatcher
# Merges WebSocket broadcasting, WAL persistence, and Demo 11 Voice Guidance
# ==============================================================================

import os
import time
import datetime
import asyncio
import logging
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import WebSocket
from database import log_alert_to_db, acknowledge_alert_in_db, resolve_alert_in_db
from voice_guidance import voice_system

logger = logging.getLogger("AlertManager")
os.makedirs("snapshots", exist_ok=True)


class AlertManager:
    """
    Manages active dashboard WebSocket clients, captures violation JPEG snapshots,
    persists alerts to the database, emits priority voice guidance, and broadcasts real-time events.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.recent_alerts: List[Dict[str, Any]] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send recent history to newly connected dashboard
        for alert in self.recent_alerts[-30:]:
            try:
                await websocket.send_json(alert)
            except Exception:
                pass

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    def save_snapshot(self, frame: np.ndarray, module_name: str, alert_type: str) -> str:
        """Saves a timestamped violation frame to snapshots/ directory."""
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snapshots/{module_name}_{alert_type}_{ts_str}.jpg"
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return f"/{filename}"

    async def dispatch_alert(self, alert_data: Dict[str, Any], frame: Optional[np.ndarray] = None):
        """
        Dispatches an alert: saves snapshot, triggers voice guidance, persists to DB,
        and broadcasts to open WebSocket connections.
        """
        now_iso = datetime.datetime.now().strftime("%H:%M:%S")
        alert_data["timestamp"] = now_iso

        if frame is not None:
            alert_data["snapshot_path"] = await asyncio.to_thread(
                self.save_snapshot, frame.copy(), alert_data["module_name"], alert_data["alert_type"]
            )
        else:
            alert_data.setdefault("snapshot_path", None)

        # Trigger Priority Offline Voice Guidance (from Demo 11)
        voice_system.speak(
            text=alert_data["message"],
            severity=alert_data["severity"],
            phrase_key=f"{alert_data['module_name']}_{alert_data['alert_type']}"
        )

        # Persist to Database and obtain unique Alert ID
        alert_id = await log_alert_to_db(alert_data)
        alert_data["alert_id"] = alert_id
        alert_data["lifecycle_state"] = "RECEIVED"

        # Update in-memory ring buffer
        self.recent_alerts.append(alert_data)
        if len(self.recent_alerts) > 100:
            self.recent_alerts.pop(0)

        # Broadcast to all open WebSockets
        dead_sockets = []
        for ws in self.active_connections:
            try:
                await ws.send_json(alert_data)
            except Exception:
                dead_sockets.append(ws)

        for ws in dead_sockets:
            self.disconnect(ws)

    async def update_lifecycle(self, alert_id: str, new_state: str, operator_name: str = "Operator"):
        """Updates alert lifecycle state (ACKNOWLEDGED or RESOLVED) and broadcasts the update."""
        if new_state == "ACKNOWLEDGED":
            await acknowledge_alert_in_db(alert_id, operator_name)
        elif new_state == "RESOLVED":
            await resolve_alert_in_db(alert_id, operator_name)

        # Update in memory
        for a in self.recent_alerts:
            if a.get("alert_id") == alert_id:
                a["lifecycle_state"] = new_state
                break

        # Broadcast update message
        update_payload = {
            "type": "LIFECYCLE_UPDATE",
            "alert_id": alert_id,
            "lifecycle_state": new_state,
            "operator": operator_name,
        }
        for ws in list(self.active_connections):
            try:
                await ws.send_json(update_payload)
            except Exception:
                self.disconnect(ws)

