# ==============================================================================
# Spatial Tracker & Geometric Association Engine
# Handles Bounding Box associations, IoU math, and Normalized Polygon ROIs
# ==============================================================================

import cv2
import numpy as np


class SpatialTracker:
    """
    Handles geometric and spatial associations:
    - Associates PPE (Helmet, Vest, Shoes) and Mobile Phones with a detected Person.
    - Tests whether a Person, Waiter, or Food item lies inside a Table ROI polygon.
    """

    @staticmethod
    def get_center(bbox: list[float]) -> tuple[float, float]:
        """Calculates center coordinate (cx, cy) of a bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def get_bottom_center(bbox: list[float]) -> tuple[float, float]:
        """Calculates bottom center coordinate (standing ground point) of a bounding box."""
        return ((bbox[0] + bbox[2]) / 2.0, bbox[3])

    @staticmethod
    def is_item_associated_with_person(
        person_bbox: list[float],
        item_bbox: list[float],
        expand_ratio: float = 0.15
    ) -> bool:
        """
        Checks if an item (helmet, vest, shoes, or phone) center falls inside
        the person's bounding box expanded by a proportional margin.
        """
        px1, py1, px2, py2 = person_bbox
        pw, ph = px2 - px1, py2 - py1
        ex1 = px1 - pw * expand_ratio
        ey1 = py1 - ph * expand_ratio
        ex2 = px2 + pw * expand_ratio
        ey2 = py2 + ph * expand_ratio

        cx, cy = SpatialTracker.get_center(item_bbox)
        return (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)

    @staticmethod
    def is_point_in_normalized_polygon(
        point: tuple[float, float],
        norm_polygon: list[list[float]],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Tests whether a pixel coordinate (x, y) is inside a normalized polygon [[x, y], ...].
        Works across any camera resolution (720p, 1080p, 4K).
        """
        pts = np.array(
            [[int(pt[0] * frame_w), int(pt[1] * frame_h)] for pt in norm_polygon],
            dtype=np.int32,
        )
        res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return res >= 0

    @staticmethod
    def get_pixel_polygon(
        norm_polygon: list[list[float]],
        frame_w: int,
        frame_h: int
    ) -> np.ndarray:
        """Converts normalized float coordinates [[0.05, 0.40], ...] to integer pixel coordinates."""
        return np.array(
            [[int(pt[0] * frame_w), int(pt[1] * frame_h)] for pt in norm_polygon],
            dtype=np.int32,
        )

    @staticmethod
    def calculate_iou(boxA: list[float], boxB: list[float]) -> float:
        """Calculates Intersection over Union (IoU) between two boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

