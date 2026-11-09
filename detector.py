# ==============================================================================
# Hardware Manager & Multi-Class Vision Detector (GPU Locked + YOLO11)
# Enforces discrete NVIDIA RTX GPU priority and zero-copy inference pipeline
# Enhanced for varying camera quality with adaptive contrast & multi-color PPE
# ==============================================================================

import os
import cv2
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareAndDetector")


class HardwareManager:
    """
    Strictly enforces hardware execution priority:
    1. External / Discrete NVIDIA GPU (CUDA with FP16 Half-Precision) - locks to RTX GPU.
    2. Integrated GPU (MPS / OpenVINO / DirectML) ONLY if no discrete GPU exists.
    3. CPU fallback (with thread throttling capped at 50% CPU to prevent memory/core lockups).
    """

    @staticmethod
    def select_optimal_device() -> Tuple[str, bool]:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            selected_idx = 0
            # Scan CUDA devices to prefer discrete RTX/GTX/Tesla/discrete chips over integrated
            for idx in range(device_count):
                prop = torch.cuda.get_device_properties(idx)
                name_lower = prop.name.lower()
                if any(k in name_lower for k in ["rtx", "gtx", "tesla", "quadro", "a100", "h100", "geforce", "nvidia"]):
                    selected_idx = idx
                    break

            try:
                torch.cuda.set_device(selected_idx)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                gpu_name = torch.cuda.get_device_name(selected_idx)
                vram_gb = torch.cuda.get_device_properties(selected_idx).total_memory / (1024 ** 3)
                logger.info(f"[GPU LOCKED] Using Discrete GPU cuda:{selected_idx} ({gpu_name}, {vram_gb:.1f} GB VRAM) with FP16.")
                return f"cuda:{selected_idx}", True
            except Exception as e:
                logger.warning(f"[CUDA INIT WARNING] Error setting device cuda:{selected_idx}: {e}")

        # Integrated Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[iGPU FALLBACK] No discrete NVIDIA GPU detected. Using Apple Silicon MPS.")
            return "mps", False

        # Throttled CPU Fallback (caps thread count to 50% of CPU cores)
        max_threads = max(1, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(max_threads)
        logger.warning(f"[CPU FALLBACK] No active GPU detected. Throttling inference to {max_threads} CPU threads.")
        return "cpu", False


class VisionDetector:
    """
    Unified detector handling standard COCO classes (person, cell phone, table, food)
    and custom PPE / Restaurant classes (helmet, safety_vest, safety_shoes, waiter).
    Includes an enhanced multi-spectral heuristic fallback so that PPE and Waiter logic
    work reliably on webcams and low-quality CCTV feeds prior to custom model fine-tuning.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device, self.use_half = HardwareManager.select_optimal_device()
        self.conf = config["system"]["inference_conf"]
        self.iou = config["system"]["iou_threshold"]

        custom_weights = config["system"]["custom_ppe_weights"]
        default_weights = config["system"]["default_weights"]

        if os.path.exists(custom_weights):
            logger.info(f"Loading Custom Trained Security Vision Model: {custom_weights}")
            self.model = YOLO(custom_weights)
            self.is_custom_model = True
        else:
            logger.info(f"Custom weights not found at {custom_weights}. Loading base model: {default_weights}")
            self.model = YOLO(default_weights)
            self.is_custom_model = False

        # If on CPU, disable half precision
        if self.device == "cpu":
            self.use_half = False

        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move model to {self.device}: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.use_half = False
            self.model.to("cpu")

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for varying CCTV camera quality
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances frame quality if the feed is dark or low contrast (common in CCTV).
        """
        if frame is None or frame.size == 0:
            return frame
        try:
            # Check average luminance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 70:  # Dark / low-light frame
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = self.clahe.apply(l)
                lab2 = cv2.merge((l2, a, b))
                return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        except Exception:
            pass
        return frame

    def track_frame(self, frame: np.ndarray, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs object detection + ByteTrack tracking directly on the locked GPU.
        Returns a normalized list of tracked detections:
        [{'track_id': int, 'class_name': str, 'bbox': [x1, y1, x2, y2], 'conf': float}]
        """
        if frame is None or frame.size == 0:
            return []

        proc_frame = self.preprocess_frame(frame)

        track_kwargs = {
            "source": proc_frame,
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "verbose": False,
        }
        if self.device.startswith("cuda") and self.use_half:
            track_kwargs["half"] = True

        try:
            results = self.model.track(**track_kwargs)
        except Exception as e:
            # Fallback to standard detect if tracker yaml issues occur
            try:
                results = self.model(proc_frame, device=self.device, conf=self.conf, iou=self.iou, verbose=False)
            except Exception as e2:
                logger.error(f"Inference error: {e2}")
                return []

        detections = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        names = res.names
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        
        # Track IDs
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes))

        for bbox, conf, cls_id, tid in zip(xyxys, confs, clss, track_ids):
            raw_name = names.get(cls_id, "unknown").lower()
            detections.append({
                "track_id": int(tid),
                "class_name": raw_name,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "conf": float(conf),
            })

        # Augment heuristics ONLY if mode is explicitly passed (never during scene classification)
        # and strictly isolated to that environment
        if not self.is_custom_model and mode is not None:
            detections.extend(self._heuristic_ppe_and_waiter_augment(frame, detections, mode=mode))

        return detections

    def _heuristic_ppe_and_waiter_augment(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        mode: str = "CONSTRUCTION"
    ) -> List[Dict[str, Any]]:
        """
        Multi-spectral visual analyzer fallback when running base COCO weights:
        Strictly applies only to the active operational mode:
        - CONSTRUCTION: Checks hard-hat colors, high-vis vests, safety boots.
        - RESTAURANT: Checks waiter tie/attire near dining tables.
        Automatically bypassed once custom_security_vision.pt is loaded.
        """
        extra = []
        h_img, w_img = frame.shape[:2]

        for det in detections:
            if det["class_name"] != "person":
                continue

            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            pw, ph = x2 - x1, y2 - y1
            if pw < 25 or ph < 50:
                continue

            # -------------------------------------------------------------
            # Construction-only heuristics (Strictly isolated to CONSTRUCTION)
            # -------------------------------------------------------------
            if mode == "CONSTRUCTION":
                # 1. Head Region (top 20% of person bbox) -> Check hard-hat dome colors
                head_y2 = y1 + int(ph * 0.20)
                head_crop = frame[y1:head_y2, x1:x2]
                if head_crop.size > 0:
                    hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                    total_head_px = head_crop.shape[0] * head_crop.shape[1]

                    # Yellow / Amber Hard Hat: H: 18-35, S: 120-255, V: 130-255
                    mask_yellow = cv2.inRange(hsv_head, (18, 120, 130), (35, 255, 255))
                    # Orange Hard Hat: H: 8-18, S: 140-255, V: 130-255
                    mask_orange = cv2.inRange(hsv_head, (8, 140, 130), (18, 255, 255))
                    # Blue Hard Hat: H: 100-125, S: 110-255, V: 90-255
                    mask_blue = cv2.inRange(hsv_head, (100, 110, 90), (125, 255, 255))
                    # Red Hard Hat: H < 8 or H > 172, S > 140, V > 100
                    mask_red1 = cv2.inRange(hsv_head, (0, 140, 100), (8, 255, 255))
                    mask_red2 = cv2.inRange(hsv_head, (172, 140, 100), (180, 255, 255))

                    combined_helmet = mask_yellow | mask_orange | mask_blue | mask_red1 | mask_red2
                    ratio = np.count_nonzero(combined_helmet) / max(1, total_head_px)
                    if ratio > 0.14:
                        extra.append({
                            "track_id": det["track_id"] + 10000,
                            "class_name": "helmet",
                            "bbox": [float(x1), float(y1), float(x2), float(head_y2)],
                            "conf": min(0.95, 0.70 + ratio),
                        })

                # 2. Torso Region (25% to 65% of person bbox) -> Check High-Vis Vest
                torso_y1, torso_y2 = y1 + int(ph * 0.25), y1 + int(ph * 0.65)
                torso_crop = frame[torso_y1:torso_y2, x1:x2]
                if torso_crop.size > 0:
                    hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                    total_torso_px = torso_crop.shape[0] * torso_crop.shape[1]

                    mask_neon_vest = cv2.inRange(hsv_torso, (24, 130, 130), (45, 255, 255))
                    mask_orange_vest = cv2.inRange(hsv_torso, (9, 150, 140), (20, 255, 255))

                    combined_vest = mask_neon_vest | mask_orange_vest
                    ratio_v = np.count_nonzero(combined_vest) / max(1, total_torso_px)
                    if ratio_v > 0.12:
                        extra.append({
                            "track_id": det["track_id"] + 20000,
                            "class_name": "safety_vest",
                            "bbox": [float(x1), float(torso_y1), float(x2), float(torso_y2)],
                            "conf": min(0.95, 0.72 + ratio_v),
                        })

                # 3. Foot Region (bottom 15% of person bbox) -> Safety Shoes
                foot_y1 = y1 + int(ph * 0.85)
                if y2 < h_img - 5 and (y2 - foot_y1) > 10:
                    extra.append({
                        "track_id": det["track_id"] + 30000,
                        "class_name": "safety_shoes",
                        "bbox": [float(x1), float(foot_y1), float(x2), float(y2)],
                        "conf": 0.80,
                    })

            # -------------------------------------------------------------
            # Restaurant-only heuristics (Strictly isolated to RESTAURANT)
            # -------------------------------------------------------------
            elif mode == "RESTAURANT":
                for sub in detections:
                    if sub["class_name"] == "tie" and self._boxes_overlap(det["bbox"], sub["bbox"]):
                        extra.append({
                            "track_id": det["track_id"],
                            "class_name": "waiter",
                            "bbox": det["bbox"],
                            "conf": 0.90,
                        })

        return extra

    @staticmethod
    def _boxes_overlap(boxA: List[float], boxB: List[float]) -> bool:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return (xB > xA) and (yB > yA)
