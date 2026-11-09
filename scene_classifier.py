# ==============================================================================
# AI Scene & Environment Classifier
# Automatically classifies video into: CONSTRUCTION | OFFICE | RESTAURANT
# ==============================================================================

import cv2
import logging
from typing import Tuple, Dict, Any
from detector import VisionDetector

logger = logging.getLogger("SceneClassifier")

CONSTRUCTION_CLASSES = {
    "helmet", "hardhat", "safety_helmet", "safety_vest", "vest",
    "high_vis_vest", "safety_coat", "safety_shoes", "boots", "safety_boots",
    "traffic cone", "truck"
}

OFFICE_CLASSES = {
    "cell phone", "mobile_phone", "phone", "laptop", "keyboard",
    "mouse", "monitor", "tv", "desk", "book"
}

RESTAURANT_CLASSES = {
    "dining table", "chair", "couch", "cup", "bowl", "fork", "knife", "spoon",
    "plate", "dish", "food", "pizza", "sandwich", "cake", "donut", "apple",
    "orange", "bottle", "wine glass", "potted plant", "vase", "umbrella",
    "waiter", "staff", "server"
}


def classify_video_scene(
    video_path: str,
    detector: VisionDetector,
    max_sample_frames: int = 15
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Samples frames across the uploaded video to automatically detect the environment:
    - CONSTRUCTION (Helmet, Safety Vest, Safety Shoes, Traffic Cones, Work Trucks)
    - OFFICE (Mobile Phone, Laptop, Monitor, Keyboard, Desk)
    - RESTAURANT (Dining Tables, Chairs, Wine Glasses, Cups, Bottles, Food, Tableware)
    Returns: (detected_mode, confidence_pct, details_dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "CONSTRUCTION", 50.0, {"error": "Could not open video file"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total_frames // max_sample_frames)
    sample_indices = [i * step for i in range(min(max_sample_frames, total_frames))]

    scores = {"CONSTRUCTION": 0.0, "OFFICE": 0.0, "RESTAURANT": 0.0}
    tallies = {"CONSTRUCTION": 0, "OFFICE": 0, "RESTAURANT": 0}
    detected_classes = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Mode=None ensures pure model detections without false heuristic PPE injection
        detections = detector.track_frame(frame, mode=None)
        for d in detections:
            cname = d["class_name"].lower()
            detected_classes.append(cname)

            if cname in RESTAURANT_CLASSES:
                # Strong weight for distinctive restaurant furniture & tableware
                weight = 6.0 if cname in ("dining table", "chair", "wine glass", "cup", "bottle", "bowl") else 3.5
                scores["RESTAURANT"] += weight
                tallies["RESTAURANT"] += 1
            elif cname in OFFICE_CLASSES:
                weight = 6.0 if cname in ("laptop", "keyboard", "monitor", "cell phone") else 3.5
                scores["OFFICE"] += weight
                tallies["OFFICE"] += 1
            elif cname in CONSTRUCTION_CLASSES:
                weight = 6.0 if cname in ("helmet", "hardhat", "safety_vest", "traffic cone") else 3.5
                scores["CONSTRUCTION"] += weight
                tallies["CONSTRUCTION"] += 1

    cap.release()

    # Category domain boosters
    if any(c in detected_classes for c in ("dining table", "wine glass", "cup", "bottle", "bowl", "pizza", "food")):
        scores["RESTAURANT"] += 15.0
    if any(c in detected_classes for c in ("laptop", "keyboard", "mouse", "monitor")):
        scores["OFFICE"] += 15.0
    if any(c in detected_classes for c in ("helmet", "hardhat", "safety_vest", "traffic cone")):
        scores["CONSTRUCTION"] += 15.0

    total_score = sum(scores.values())
    if total_score <= 0.0:
        logger.info("[SCENE AI] No distinctive scene items detected; defaulting to RESTAURANT if chairs/people present else CONSTRUCTION.")
        if any(c in detected_classes for c in ("chair", "couch", "person")):
            return "RESTAURANT", 60.0, {"reason": "Dining/social seating layout detected"}
        return "CONSTRUCTION", 55.0, {"reason": "Undetermined scene layout"}

    best_mode = max(scores, key=scores.get)
    confidence = min(99.0, max(60.0, (scores[best_mode] / total_score) * 100.0))

    logger.info(f"[SCENE AI] Detected Environment: {best_mode} ({confidence:.1f}% confidence, Tallies: {tallies})")
    return best_mode, round(confidence, 1), {
        "scores": scores,
        "tallies": tallies,
        "samples": len(sample_indices),
    }

