# ==============================================================================
# Detection State Machines & Timers
# Modules: 1. Construction PPE | 2. Office Mobile | 3. Restaurant Smart ROI
# ==============================================================================

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any
from tracker import SpatialTracker


# =====================================================================
# MODULE 1: CONSTRUCTION SITE PPE COMPLIANCE MONITOR
# =====================================================================
class ConstructionPPEMonitor:
    """
    Verifies that every detected worker is wearing:
    1. Safety Helmet ('helmet' / 'hardhat')
    2. Safety Shoes ('safety_shoes' / 'boots')
    3. Safety Coat / Vest ('safety_vest' / 'vest')
    """

    HELMET_CLASSES = {"helmet", "hardhat", "safety_helmet"}
    VEST_CLASSES = {"safety_vest", "vest", "safety_coat", "high_vis_vest"}
    SHOE_CLASSES = {"safety_shoes", "boots", "safety_boots"}

    def __init__(self, config: dict):
        self.cooldown_sec = config["thresholds"]["construction"]["alert_cooldown_sec"]
        self.last_alert_time: Dict[int, float] = {}  # track_id -> timestamp

    def process(self, detections: List[Dict[str, Any]], now: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        persons = [d for d in detections if d["class_name"] == "person"]
        helmets = [d for d in detections if d["class_name"] in self.HELMET_CLASSES]
        vests = [d for d in detections if d["class_name"] in self.VEST_CLASSES]
        shoes = [d for d in detections if d["class_name"] in self.SHOE_CLASSES]

        alerts = []
        overlays = []

        for p in persons:
            tid = p["track_id"]
            p_box = p["bbox"]

            has_helmet = any(SpatialTracker.is_item_associated_with_person(p_box, h["bbox"], expand_ratio=0.10) for h in helmets)
            has_vest = any(SpatialTracker.is_item_associated_with_person(p_box, v["bbox"], expand_ratio=0.12) for v in vests)
            has_shoes = any(SpatialTracker.is_item_associated_with_person(p_box, s["bbox"], expand_ratio=0.15) for s in shoes)

            missing = []
            if not has_helmet:
                missing.append("Safety Helmet")
            if not has_vest:
                missing.append("Safety Coat/Vest")
            if not has_shoes:
                missing.append("Safety Shoes")

            if not missing:
                overlays.append({
                    "bbox": p_box,
                    "label": f"Worker #{tid}: COMPLIANT",
                    "color": (46, 204, 113),  # Green (BGR)
                })
            else:
                missing_str = ", ".join(missing)
                overlays.append({
                    "bbox": p_box,
                    "label": f"Worker #{tid}: MISSING [{missing_str}]",
                    "color": (59, 59, 255),   # Red (BGR)
                })

                last_t = self.last_alert_time.get(tid, 0.0)
                if (now - last_t) >= self.cooldown_sec:
                    self.last_alert_time[tid] = now
                    alerts.append({
                        "module_name": "CONSTRUCTION",
                        "alert_type": "PPE_VIOLATION",
                        "table_or_zone_id": f"Worker-ID-{tid}",
                        "severity": "CRITICAL",
                        "message": f"Worker #{tid} missing required PPE: {missing_str}",
                        "metadata": {"worker_id": tid, "missing_items": missing},
                    })

        return alerts, overlays


# =====================================================================
# MODULE 2: WORKPLACE / OFFICE MOBILE PHONE USAGE MONITOR
# =====================================================================
@dataclass
class EmployeePhoneState:
    first_detected_at: float = 0.0
    last_seen_at: float = 0.0
    last_alert_sent_at: float = 0.0
    active_duration: float = 0.0


class OfficePhoneMonitor:
    """
    Tracks continuous mobile phone usage per employee ID.
    Features frame-drop tolerance so momentary blinks don't reset the timer.
    Triggers a Warning Alert when continuous phone usage reaches 5-10 seconds.
    """

    PHONE_CLASSES = {"cell phone", "mobile_phone", "phone"}

    def __init__(self, config: dict):
        self.threshold_sec = config["thresholds"]["office"]["phone_continuous_threshold_sec"]
        self.repeat_sec = config["thresholds"]["office"]["phone_repeat_interval_sec"]
        self.dropout_tolerance = config["system"]["frame_drop_tolerance_sec"]
        self.states: Dict[int, EmployeePhoneState] = {}

    def process(self, detections: List[Dict[str, Any]], now: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        persons = [d for d in detections if d["class_name"] == "person"]
        phones = [d for d in detections if d["class_name"] in self.PHONE_CLASSES]

        alerts = []
        overlays = []
        currently_using_ids = set()

        for p in persons:
            tid = p["track_id"]
            p_box = p["bbox"]

            using_phone = any(
                SpatialTracker.is_item_associated_with_person(p_box, ph["bbox"], expand_ratio=0.20)
                for ph in phones
            )

            if using_phone:
                currently_using_ids.add(tid)
                st = self.states.get(tid)
                if st is None or (now - st.last_seen_at) > self.dropout_tolerance:
                    st = EmployeePhoneState(first_detected_at=now, last_seen_at=now)
                    self.states[tid] = st
                else:
                    st.last_seen_at = now
                    st.active_duration = now - st.first_detected_at

                dur = st.active_duration
                is_violation = dur >= self.threshold_sec

                overlays.append({
                    "bbox": p_box,
                    "label": f"Emp #{tid}: PHONE ({dur:.1f}s)",
                    "color": (59, 59, 255) if is_violation else (0, 215, 255),  # Red if violation else Amber
                })

                if is_violation:
                    if st.last_alert_sent_at == 0.0 or (now - st.last_alert_sent_at) >= self.repeat_sec:
                        st.last_alert_sent_at = now
                        alerts.append({
                            "module_name": "OFFICE",
                            "alert_type": "ACTIVE_PHONE_USAGE",
                            "table_or_zone_id": f"Employee-ID-{tid}",
                            "severity": "WARNING",
                            "message": f"Employee #{tid} continuously using mobile phone for {dur:.1f}s.",
                            "metadata": {"employee_id": tid, "duration_seconds": round(dur, 1)},
                        })
            else:
                overlays.append({
                    "bbox": p_box,
                    "label": f"Emp #{tid}: Working",
                    "color": (160, 160, 160),
                })

        # Prune states that exceeded frame dropout tolerance
        stale_ids = [
            k for k, v in self.states.items()
            if k not in currently_using_ids and (now - v.last_seen_at) > self.dropout_tolerance
        ]
        for k in stale_ids:
            del self.states[k]

        return alerts, overlays


# =====================================================================
# MODULE 3: RESTAURANT SMART TABLE & SERVICE MONITOR
# =====================================================================
@dataclass
class TableROIState:
    table_id: str
    norm_polygon: List[List[float]]
    status: str = "FREE"  # FREE | OCCUPIED | ORDER_TAKEN_WAITING_FOOD | DINING
    occupied_since: float = 0.0
    last_person_seen: float = 0.0
    waiter_visited: bool = False
    waiter_currently_present: bool = False
    order_taken_at: float = 0.0
    food_delivered_at: float = 0.0

    # Debounce flags
    idle_warning_sent: bool = False
    unattended_alert_sent: bool = False
    late_food_alert_sent: bool = False


class RestaurantTableMonitor:
    """
    Tracks state machine and service deadlines for each Restaurant Table ROI:
    1. FREE -> OCCUPIED: Customer arrives; starts occupancy timer.
    2. Idle / No-Order Warning: Customer seated >= 5 mins without waiter or order.
    3. Unattended Customer Alert: Seated >= 10 mins without waiter -> CRITICAL Waiter Dispatch.
    4. Order Taken -> Food Tracking: Waiter departs -> ORDER_TAKEN_WAITING_FOOD.
       Starts food prep timer; if food not detected within threshold -> Late Food Alert.
    """

    FOOD_CLASSES = {"bowl", "cup", "pizza", "sandwich", "cake", "donut", "apple", "orange", "food", "plate", "dish", "bottle", "fork", "knife", "spoon"}
    WAITER_CLASSES = {"waiter", "staff", "server"}

    def __init__(self, config: dict):
        scale = config["system"]["time_scale"]
        t_cfg = config["thresholds"]["restaurant"]

        self.idle_warning_sec = t_cfg["idle_no_order_warning_sec"] * scale
        self.unattended_dispatch_sec = t_cfg["unattended_waiter_dispatch_sec"] * scale
        self.late_food_sec = t_cfg["late_food_delivery_sec"] * scale
        self.clear_grace_sec = t_cfg["table_clear_grace_sec"]

        self.tables: Dict[str, TableROIState] = {}
        for t_id, roi_data in config.get("restaurant_rois", {}).items():
            self.tables[t_id] = TableROIState(
                table_id=t_id,
                norm_polygon=roi_data["polygon"],
            )

    def process(
        self,
        detections: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        now: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        persons = [d for d in detections if d["class_name"] == "person"]
        waiters = [d for d in detections if d["class_name"] in self.WAITER_CLASSES]
        foods = [d for d in detections if d["class_name"] in self.FOOD_CLASSES]

        alerts = []
        roi_overlays = []

        for t_id, st in self.tables.items():
            poly = st.norm_polygon

            # 1. Customers present in table ROI
            customers_in_roi = [
                p for p in persons
                if SpatialTracker.is_point_in_normalized_polygon(
                    SpatialTracker.get_center(p["bbox"]), poly, frame_w, frame_h
                )
            ]

            # 2. Waiter present in table ROI
            waiter_in_roi = any(
                SpatialTracker.is_point_in_normalized_polygon(
                    SpatialTracker.get_center(w["bbox"]), poly, frame_w, frame_h
                )
                for w in waiters
            )

            # 3. Food / plates present in table ROI
            food_in_roi = any(
                SpatialTracker.is_point_in_normalized_polygon(
                    SpatialTracker.get_center(f["bbox"]), poly, frame_w, frame_h
                )
                for f in foods
            )

            if len(customers_in_roi) > 0:
                st.last_person_seen = now

                # State Transition: FREE -> OCCUPIED
                if st.status == "FREE":
                    st.status = "OCCUPIED"
                    st.occupied_since = now
                    st.waiter_visited = False
                    st.idle_warning_sent = False
                    st.unattended_alert_sent = False
                    st.late_food_alert_sent = False

                # State Transition: Food Arrived -> DINING
                if food_in_roi and st.status != "DINING":
                    st.status = "DINING"
                    st.food_delivered_at = now

                # Waiter Interaction: Waiter arrives & leaves -> ORDER_TAKEN_WAITING_FOOD
                if waiter_in_roi:
                    st.waiter_currently_present = True
                    st.waiter_visited = True
                else:
                    if st.waiter_currently_present and st.status == "OCCUPIED":
                        st.waiter_currently_present = False
                        st.status = "ORDER_TAKEN_WAITING_FOOD"
                        st.order_taken_at = now

                # Evaluate Timing Deadlines
                if st.status == "OCCUPIED" and not st.waiter_visited and not food_in_roi:
                    elapsed_occ = now - st.occupied_since

                    # 5-minute Idle / No-Order Warning
                    if elapsed_occ >= self.idle_warning_sec and not st.idle_warning_sent:
                        st.idle_warning_sent = True
                        alerts.append({
                            "module_name": "RESTAURANT",
                            "alert_type": "IDLE_NO_ORDER_WARNING",
                            "table_or_zone_id": t_id,
                            "severity": "WARNING",
                            "message": f"{t_id}: Customers sitting idle without ordering ({elapsed_occ:.0f}s elapsed).",
                            "metadata": {"table_id": t_id, "elapsed_sec": round(elapsed_occ, 1)},
                        })

                    # 10-minute Unattended Customer -> Waiter Dispatch
                    if elapsed_occ >= self.unattended_dispatch_sec and not st.unattended_alert_sent:
                        st.unattended_alert_sent = True
                        alerts.append({
                            "module_name": "RESTAURANT",
                            "alert_type": "UNATTENDED_CUSTOMER_DISPATCH_WAITER",
                            "table_or_zone_id": t_id,
                            "severity": "CRITICAL",
                            "message": f"{t_id}: Occupied for {elapsed_occ:.0f}s without waiter! Dispatch waiter to {t_id} immediately.",
                            "metadata": {"table_id": t_id, "elapsed_sec": round(elapsed_occ, 1)},
                        })

                elif st.status == "ORDER_TAKEN_WAITING_FOOD" and not food_in_roi:
                    wait_food_elapsed = now - st.order_taken_at
                    # Late Food Delivery Notification
                    if wait_food_elapsed >= self.late_food_sec and not st.late_food_alert_sent:
                        st.late_food_alert_sent = True
                        alerts.append({
                            "module_name": "RESTAURANT",
                            "alert_type": "LATE_FOOD_DELIVERY",
                            "table_or_zone_id": t_id,
                            "severity": "CRITICAL",
                            "message": f"{t_id}: Late Food Delivery! Waiting {wait_food_elapsed:.0f}s since order was taken.",
                            "metadata": {"table_id": t_id, "wait_food_sec": round(wait_food_elapsed, 1)},
                        })

            else:
                # Table is empty; reset to FREE after clear grace period
                if st.status != "FREE" and (now - st.last_person_seen) >= self.clear_grace_sec:
                    st.status = "FREE"
                    st.occupied_since = 0.0
                    st.waiter_visited = False
                    st.waiter_currently_present = False

            # Compute HUD polygon color and timer text
            elapsed_display = 0.0
            if st.status == "OCCUPIED":
                elapsed_display = now - st.occupied_since
                color = (0, 165, 255) if not st.unattended_alert_sent else (59, 59, 255)
            elif st.status == "ORDER_TAKEN_WAITING_FOOD":
                elapsed_display = now - st.order_taken_at
                color = (255, 191, 0) if not st.late_food_alert_sent else (59, 59, 255)
            elif st.status == "DINING":
                elapsed_display = now - st.food_delivered_at
                color = (46, 204, 113)
            else:
                color = (130, 130, 130)

            roi_overlays.append({
                "table_id": t_id,
                "polygon": SpatialTracker.get_pixel_polygon(poly, frame_w, frame_h),
                "status": st.status,
                "elapsed": elapsed_display,
                "color": color,
                "customers": len(customers_in_roi),
            })

        return alerts, roi_overlays

