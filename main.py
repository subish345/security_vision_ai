import os
import uuid
import time
import asyncio
import logging
import subprocess
import cv2
import numpy as np
import yaml
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from database import init_db, get_recent_alerts
from detector import VisionDetector, HardwareManager
from alerts import AlertManager
from voice_guidance import voice_system
from scene_classifier import classify_video_scene

# Attempt to load compiled Rust PyO3 core; seamlessly fall back to Python timers if not built yet
try:
    from security_core_rs import RustSecurityEngine
    RUST_CORE_ACTIVE = True
except ImportError:
    RUST_CORE_ACTIVE = False
    from timers import ConstructionPPEMonitor, OfficePhoneMonitor, RestaurantTableMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecurityVisionVideoAnalyzer")

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

for folder in ["uploads", "processed", "snapshots", "static"]:
    os.makedirs(folder, exist_ok=True)

alert_manager = AlertManager()
detector = VisionDetector(CONFIG)


def create_rust_or_py_engine():
    if RUST_CORE_ACTIVE:
        rois = {
            k: v["polygon"]
            for k, v in CONFIG.get("restaurant_rois", {}).items()
        }
        return RustSecurityEngine(
            time_scale=float(CONFIG["system"]["time_scale"]),
            dropout_tolerance_sec=float(CONFIG["system"]["frame_drop_tolerance_sec"]),
            ppe_cooldown_sec=float(CONFIG["thresholds"]["construction"]["alert_cooldown_sec"]),
            phone_threshold_sec=float(CONFIG["thresholds"]["office"]["phone_continuous_threshold_sec"]),
            phone_repeat_sec=float(CONFIG["thresholds"]["office"]["phone_repeat_interval_sec"]),
            idle_warning_sec=float(CONFIG["thresholds"]["restaurant"]["idle_no_order_warning_sec"]),
            unattended_dispatch_sec=float(CONFIG["thresholds"]["restaurant"]["unattended_waiter_dispatch_sec"]),
            late_food_sec=float(CONFIG["thresholds"]["restaurant"]["late_food_delivery_sec"]),
            clear_grace_sec=float(CONFIG["thresholds"]["restaurant"]["table_clear_grace_sec"]),
            table_rois=rois,
        )
    return {
        "ppe": ConstructionPPEMonitor(CONFIG),
        "office": OfficePhoneMonitor(CONFIG),
        "restaurant": RestaurantTableMonitor(CONFIG),
    }


# Store active & completed video analysis jobs
video_jobs: Dict[str, Dict[str, Any]] = {}


async def broadcast_job_progress(job_data: Dict[str, Any]):
    payload = {"type": "JOB_PROGRESS", **job_data}
    for ws in list(alert_manager.active_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            alert_manager.disconnect(ws)


def process_uploaded_video_sync(
    job_id: str,
    input_path: str,
    output_path: str,
    requested_mode: str,
    loop: asyncio.AbstractEventLoop,
):
    """
    Runs on a background worker thread:
    1. Uses AI Scene Classifier if requested_mode == "AUTO" to identify:
       CONSTRUCTION, OFFICE, or RESTAURANT.
    2. Runs YOLO11 + ByteTrack on the locked Discrete NVIDIA RTX GPU (cuda:0, FP16).
    3. Executes spatial & timer state machines in compiled Rust (security_core_rs)
       strictly for the detected environment.
    4. Burns bounding boxes, Table ROIs, and telemetry HUD into the output MP4.
    5. Transcodes output via FFmpeg to web-compliant H.264 (yuv420p + faststart)
       for seamless HTML5 browser playback.
    6. Streams real-time progress and alerts to the WebSocket dashboard.
    """
    try:
        # 1. AI Auto-Detection of Scene / Environment
        if requested_mode == "AUTO":
            video_jobs[job_id]["status"] = "AI DETECTING SCENE..."
            asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)
            detected_mode, conf, details = classify_video_scene(input_path, detector)
            effective_mode = detected_mode
            video_jobs[job_id]["detected_scene"] = detected_mode
            video_jobs[job_id]["scene_confidence"] = conf
            logger.info(f"[JOB {job_id}] AI Classified Video Environment as: {detected_mode} ({conf}% confidence)")
        else:
            effective_mode = requested_mode
            video_jobs[job_id]["detected_scene"] = requested_mode
            video_jobs[job_id]["scene_confidence"] = 100.0

        channel_names = {
            "CONSTRUCTION": "CAM-01 [CONSTRUCTION YARD]",
            "OFFICE": "CAM-02 [OFFICE FLOOR 3]",
            "RESTAURANT": "CAM-03 [DINING & BAR AREA]",
        }
        video_jobs[job_id]["camera_channel"] = channel_names.get(effective_mode, f"CAM-01 [{effective_mode}]")
        video_jobs[job_id]["effective_mode"] = effective_mode
        asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            video_jobs[job_id]["status"] = "FAILED"
            video_jobs[job_id]["error"] = "Unable to open uploaded video file."
            asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # Intermediate raw OpenCV output written to temp file; will be transcoded to H.264
        temp_cv_path = output_path.replace(".mp4", "_temp_cv.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_cv_path, fourcc, fps, (width, height))

        engine = create_rust_or_py_engine()
        frame_idx = 0
        total_alerts_triggered = 0
        start_wall = time.perf_counter()

        video_jobs[job_id].update({
            "status": "PROCESSING",
            "fps_source": round(fps, 1),
            "total_frames": total_frames,
            "resolution": f"{width}x{height}",
            "engine_backend": "Rust PyO3 Native + CUDA FP16" if RUST_CORE_ACTIVE else "Python + CUDA FP16",
        })

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            # Exact video timestamp in seconds so timers are 100% accurate regardless of GPU speed
            video_ts = frame_idx / fps

            # 1. GPU YOLO Detection + Tracking (Strictly isolated to effective_mode)
            detections = detector.track_frame(frame, mode=effective_mode)

            # 2. Rust / Python Spatial ROI & State Machine Execution (Dedicated per Environment)
            if RUST_CORE_ACTIVE:
                triggered_alerts, box_overlays, roi_overlays = engine.process_frame(
                    detections, width, height, video_ts, effective_mode
                )
            else:
                triggered_alerts, box_overlays, roi_overlays = [], [], []
                if effective_mode == "CONSTRUCTION":
                    a, o = engine["ppe"].process(detections, video_ts)
                    triggered_alerts.extend(a)
                    box_overlays.extend(o)
                elif effective_mode == "OFFICE":
                    a, o = engine["office"].process(detections, video_ts)
                    triggered_alerts.extend(a)
                    box_overlays.extend(o)
                elif effective_mode == "RESTAURANT":
                    a, r = engine["restaurant"].process(detections, width, height, video_ts)
                    triggered_alerts.extend(a)
                    roi_overlays.extend(r)

            # 3. Render Restaurant Table ROI Polygons (Only in Restaurant Mode)
            if effective_mode == "RESTAURANT":
                for roi in roi_overlays:
                    pts = np.array(roi["polygon"], dtype=np.int32)
                    color = tuple(int(c) for c in roi["color"])
                    overlay_img = frame.copy()
                    cv2.fillPoly(overlay_img, [pts], color)
                    cv2.addWeighted(overlay_img, 0.22, frame, 0.78, 0, frame)
                    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

                    x0, y0 = pts[0]
                    status_txt = f"{roi['table_id']} | {roi['status']} ({roi['elapsed']:.1f}s)"
                    cv2.rectangle(frame, (x0, max(0, y0 - 24)), (x0 + 320, max(24, y0)), color, -1)
                    cv2.putText(frame, status_txt, (x0 + 8, max(16, y0 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Render Bounding Boxes
            for ov in box_overlays:
                x1, y1, x2, y2 = map(int, ov["bbox"])
                color = tuple(int(c) for c in ov["color"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = ov["label"]
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, lbl, (x1 + 4, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Burn Tactical Telemetry Stamp into Output Video
            elapsed_wall = max(1e-4, time.perf_counter() - start_wall)
            proc_fps = frame_idx / elapsed_wall
            conf_val = video_jobs[job_id].get("scene_confidence", 100.0)
            cv2.rectangle(frame, (10, 10), (640, 42), (11, 15, 25), -1)
            cv2.rectangle(frame, (10, 10), (640, 42), (30, 41, 59), 1)
            hud_str = f"AI SCENE: {effective_mode} ({conf_val:.0f}%) | GPU: {detector.device.upper()} | {proc_fps:.1f} FPS | T:{video_ts:.1f}s"
            cv2.putText(frame, hud_str, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

            writer.write(frame)

            # 6. Dispatch Triggered Alerts & Violation Snapshots
            for alert in triggered_alerts:
                total_alerts_triggered += 1
                alert["job_id"] = job_id
                alert["video_timestamp_sec"] = round(video_ts, 2)
                asyncio.run_coroutine_threadsafe(
                    alert_manager.dispatch_alert(alert, frame), loop
                )

            # Broadcast progress periodically
            if frame_idx % 15 == 0 or frame_idx == total_frames:
                pct = min(100.0, round((frame_idx / max(1, total_frames)) * 100.0, 1))
                video_jobs[job_id].update({
                    "processed_frames": frame_idx,
                    "progress_pct": pct,
                    "processing_fps": round(proc_fps, 1),
                    "alerts_count": total_alerts_triggered,
                })
                asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)

        cap.release()
        writer.release()

        # 7. Fast web-compatible H.264 transcode (required for HTML5 browser video playback)
        video_jobs[job_id]["status"] = "ENCODING_FOR_WEB"
        asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_cv_path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(temp_cv_path):
                os.remove(temp_cv_path)
        except Exception as ffmpeg_err:
            logger.warning(f"[JOB {job_id}] FFmpeg transcode warning ({ffmpeg_err}), falling back to raw output")
            if os.path.exists(temp_cv_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.replace(temp_cv_path, output_path)

        total_wall = round(time.perf_counter() - start_wall, 2)
        video_jobs[job_id].update({
            "status": "COMPLETED",
            "progress_pct": 100.0,
            "processed_frames": frame_idx,
            "total_wall_time_sec": total_wall,
            "output_video_url": f"/processed/{os.path.basename(output_path)}",
            "alerts_count": total_alerts_triggered,
        })
        asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)
        logger.info(f"[JOB {job_id}] Finished {frame_idx} frames in {total_wall}s for {effective_mode}")

    except Exception as exc:
        logger.exception(f"[JOB {job_id}] Video processing error: {exc}")
        video_jobs[job_id]["status"] = "FAILED"
        video_jobs[job_id]["error"] = str(exc)
        asyncio.run_coroutine_threadsafe(broadcast_job_progress(video_jobs[job_id]), loop)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    voice_system.stop()


app = FastAPI(title="Pramman Security Vision AI — Hybrid Rust + Python Video Analyzer", lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")


@app.get("/api/system/status")
async def system_status():
    device_name, is_gpu = HardwareManager.select_optimal_device()
    return {
        "status": "online",
        "device": device_name,
        "is_discrete_gpu": is_gpu,
        "rust_core_active": RUST_CORE_ACTIVE,
        "voice_guidance_enabled": voice_system.enabled,
        "is_custom_model": detector.is_custom_model,
        "model_name": "Custom Security Vision (Trained)" if detector.is_custom_model else "YOLO11s Base",
    }


@app.post("/api/video/analyze")
async def upload_and_analyze_video(
    file: UploadFile = File(...),
    mode: str = Form("AUTO"),
):
    """
    Accepts an uploaded video file, saves it to disk, and runs the
    AI Scene Classifier + dedicated Rust/GPU detection pipeline.
    Modes supported: AUTO (default) | CONSTRUCTION | OFFICE | RESTAURANT
    """
    if mode not in ("AUTO", "CONSTRUCTION", "OFFICE", "RESTAURANT"):
        raise HTTPException(status_code=400, detail="Invalid module mode. Allowed: AUTO, CONSTRUCTION, OFFICE, RESTAURANT")

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    input_path = os.path.join("uploads", f"{job_id}_raw{ext}")
    output_path = os.path.join("processed", f"{job_id}_annotated.mp4")

    with open(input_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            f_out.write(chunk)

    video_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "mode": mode,
        "status": "QUEUED",
        "detected_scene": None,
        "scene_confidence": 0.0,
        "progress_pct": 0.0,
        "processed_frames": 0,
        "total_frames": 0,
        "processing_fps": 0.0,
        "alerts_count": 0,
        "output_video_url": None,
    }

    loop = asyncio.get_running_loop()
    asyncio.get_running_loop().run_in_executor(
        None,
        process_uploaded_video_sync,
        job_id,
        input_path,
        output_path,
        mode,
        loop,
    )

    return video_jobs[job_id]


@app.get("/api/video/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return video_jobs[job_id]


@app.post("/api/voice/toggle")
async def toggle_voice():
    voice_system.enabled = not voice_system.enabled
    return {"status": "ok", "voice_guidance_enabled": voice_system.enabled}


@app.get("/api/alerts")
async def fetch_alerts(limit: int = 50, module: Optional[str] = None):
    return await get_recent_alerts(limit=limit, module=module)


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, operator: str = "Surveillance Operator"):
    await alert_manager.update_lifecycle(alert_id, "ACKNOWLEDGED", operator)
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, operator: str = "Surveillance Operator"):
    await alert_manager.update_lifecycle(alert_id, "RESOLVED", operator)
    return {"status": "resolved", "alert_id": alert_id}


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await alert_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
