use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
struct Detection {
    track_id: i64,
    class_name: String,
    bbox: [f64; 4],
}

#[derive(Clone, Debug)]
struct PhoneState {
    first_detected_at: f64,
    last_seen_at: f64,
    last_alert_sent_at: f64,
    active_duration: f64,
}

#[derive(Clone, Debug)]
struct TableROIState {
    table_id: String,
    norm_polygon: Vec<[f64; 2]>,
    status: String, // "FREE" | "OCCUPIED" | "ORDER_TAKEN_WAITING_FOOD" | "DINING"
    occupied_since: f64,
    last_person_seen: f64,
    waiter_visited: bool,
    waiter_currently_present: bool,
    order_taken_at: f64,
    food_delivered_at: f64,
    idle_warning_sent: bool,
    unattended_alert_sent: bool,
    late_food_alert_sent: bool,
}

/// Fast geometric helpers compiled to native SIMD-friendly machine code
#[inline(always)]
fn get_center(bbox: &[f64; 4]) -> (f64, f64) {
    ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
}

#[inline(always)]
fn is_item_associated_with_person(person_bbox: &[f64; 4], item_bbox: &[f64; 4], expand_ratio: f64) -> bool {
    let pw = person_bbox[2] - person_bbox[0];
    let ph = person_bbox[3] - person_bbox[1];
    let ex1 = person_bbox[0] - pw * expand_ratio;
    let ey1 = person_bbox[1] - ph * expand_ratio;
    let ex2 = person_bbox[2] + pw * expand_ratio;
    let ey2 = person_bbox[3] + ph * expand_ratio;

    let (cx, cy) = get_center(item_bbox);
    cx >= ex1 && cx <= ex2 && cy >= ey1 && cy <= ey2
}

/// Ray-casting Point-in-Polygon algorithm in normalized [0.0, 1.0] space
#[inline(always)]
fn is_point_in_norm_polygon(px: f64, py: f64, poly: &[[f64; 2]], frame_w: f64, frame_h: f64) -> bool {
    if poly.len() < 3 || frame_w <= 0.0 || frame_h <= 0.0 {
        return false;
    }
    let nx = px / frame_w;
    let ny = py / frame_h;
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0];
        let yi = poly[i][1];
        let xj = poly[j][0];
        let yj = poly[j][1];

        let intersect = ((yi > ny) != (yj > ny))
            && (nx < (xj - xi) * (ny - yi) / ((yj - yi).abs().max(1e-9) * if yj >= yi { 1.0 } else { -1.0 }) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[pyclass]
pub struct RustSecurityEngine {
    // Construction PPE Config & State
    ppe_cooldown_sec: f64,
    ppe_last_alert: HashMap<i64, f64>,

    // Office Phone Config & State
    phone_threshold_sec: f64,
    phone_repeat_sec: f64,
    dropout_tolerance_sec: f64,
    phone_states: HashMap<i64, PhoneState>,

    // Restaurant Config & State
    idle_warning_sec: f64,
    unattended_dispatch_sec: f64,
    late_food_sec: f64,
    clear_grace_sec: f64,
    tables: Vec<TableROIState>,
}

#[pymethods]
impl RustSecurityEngine {
    #[new]
    #[pyo3(signature = (
        time_scale,
        dropout_tolerance_sec,
        ppe_cooldown_sec,
        phone_threshold_sec,
        phone_repeat_sec,
        idle_warning_sec,
        unattended_dispatch_sec,
        late_food_sec,
        clear_grace_sec,
        table_rois
    ))]
    pub fn new(
        time_scale: f64,
        dropout_tolerance_sec: f64,
        ppe_cooldown_sec: f64,
        phone_threshold_sec: f64,
        phone_repeat_sec: f64,
        idle_warning_sec: f64,
        unattended_dispatch_sec: f64,
        late_food_sec: f64,
        clear_grace_sec: f64,
        table_rois: HashMap<String, Vec<[f64; 2]>>,
    ) -> Self {
        let mut tables = Vec::new();
        let mut sorted_keys: Vec<_> = table_rois.keys().cloned().collect();
        sorted_keys.sort();

        for t_id in sorted_keys {
            if let Some(poly) = table_rois.get(&t_id) {
                tables.push(TableROIState {
                    table_id: t_id,
                    norm_polygon: poly.clone(),
                    status: "FREE".to_string(),
                    occupied_since: 0.0,
                    last_person_seen: 0.0,
                    waiter_visited: false,
                    waiter_currently_present: false,
                    order_taken_at: 0.0,
                    food_delivered_at: 0.0,
                    idle_warning_sent: false,
                    unattended_alert_sent: false,
                    late_food_alert_sent: false,
                });
            }
        }

        RustSecurityEngine {
            ppe_cooldown_sec,
            ppe_last_alert: HashMap::new(),
            phone_threshold_sec,
            phone_repeat_sec,
            dropout_tolerance_sec,
            phone_states: HashMap::new(),
            idle_warning_sec: idle_warning_sec * time_scale,
            unattended_dispatch_sec: unattended_dispatch_sec * time_scale,
            late_food_sec: late_food_sec * time_scale,
            clear_grace_sec,
            tables,
        }
    }

    pub fn reset_state(&mut self) {
        self.ppe_last_alert.clear();
        self.phone_states.clear();
        for t in &mut self.tables {
            t.status = "FREE".to_string();
            t.occupied_since = 0.0;
            t.last_person_seen = 0.0;
            t.waiter_visited = false;
            t.waiter_currently_present = false;
            t.idle_warning_sent = false;
            t.unattended_alert_sent = false;
            t.late_food_alert_sent = false;
        }
    }

    /// Processes a single video frame at timestamp `video_ts` across the requested `mode`
    /// ("ALL", "CONSTRUCTION", "OFFICE", or "RESTAURANT").
    pub fn process_frame<'py>(
        &mut self,
        py: Python<'py>,
        py_detections: &Bound<'py, PyList>,
        frame_w: i32,
        frame_h: i32,
        video_ts: f64,
        mode: &str,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>)> {
        let mut detections = Vec::with_capacity(py_detections.len());
        for item in py_detections.iter() {
            if let Ok(d) = item.downcast::<PyDict>() {
                let track_id: i64 = d.get_item("track_id")?.map(|v| v.extract().unwrap_or(0)).unwrap_or(0);
                let class_name: String = d.get_item("class_name")?.map(|v| v.extract().unwrap_or_default()).unwrap_or_default();
                let bbox_vec: Vec<f64> = d.get_item("bbox")?.map(|v| v.extract().unwrap_or_default()).unwrap_or_default();
                if bbox_vec.len() == 4 {
                    detections.push(Detection {
                        track_id,
                        class_name,
                        bbox: [bbox_vec[0], bbox_vec[1], bbox_vec[2], bbox_vec[3]],
                    });
                }
            }
        }

        let alerts_out = PyList::empty_bound(py);
        let boxes_out = PyList::empty_bound(py);
        let rois_out = PyList::empty_bound(py);

        let persons: Vec<&Detection> = detections.iter().filter(|d| d.class_name == "person").collect();

        // 1. MODULE 1: CONSTRUCTION PPE COMPLIANCE
        if mode == "CONSTRUCTION" {
            let helmets: Vec<&Detection> = detections
                .iter()
                .filter(|d| matches!(d.class_name.as_str(), "helmet" | "hardhat" | "safety_helmet"))
                .collect();
            let vests: Vec<&Detection> = detections
                .iter()
                .filter(|d| matches!(d.class_name.as_str(), "safety_vest" | "vest" | "safety_coat" | "high_vis_vest"))
                .collect();
            let shoes: Vec<&Detection> = detections
                .iter()
                .filter(|d| matches!(d.class_name.as_str(), "safety_shoes" | "boots" | "safety_boots"))
                .collect();

            for p in &persons {
                let tid = p.track_id;
                let has_helmet = helmets.iter().any(|h| is_item_associated_with_person(&p.bbox, &h.bbox, 0.10));
                let has_vest = vests.iter().any(|v| is_item_associated_with_person(&p.bbox, &v.bbox, 0.12));
                let has_shoes = shoes.iter().any(|s| is_item_associated_with_person(&p.bbox, &s.bbox, 0.15));

                let mut missing = Vec::new();
                if !has_helmet {
                    missing.push("Safety Helmet");
                }
                if !has_vest {
                    missing.push("Safety Coat/Vest");
                }
                if !has_shoes {
                    missing.push("Safety Shoes");
                }

                let ov = PyDict::new_bound(py);
                ov.set_item("bbox", p.bbox.to_vec())?;
                if missing.is_empty() {
                    ov.set_item("label", format!("Worker #{}: COMPLIANT", tid))?;
                    ov.set_item("color", vec![46, 204, 113])?;
                } else {
                    let missing_str = missing.join(", ");
                    ov.set_item("label", format!("Worker #{}: MISSING [{}]", tid, missing_str))?;
                    ov.set_item("color", vec![59, 59, 255])?;

                    let last_t = *self.ppe_last_alert.get(&tid).unwrap_or(&-999.0);
                    if (video_ts - last_t) >= self.ppe_cooldown_sec {
                        self.ppe_last_alert.insert(tid, video_ts);
                        let alert = PyDict::new_bound(py);
                        alert.set_item("module_name", "CONSTRUCTION")?;
                        alert.set_item("alert_type", "PPE_VIOLATION")?;
                        alert.set_item("table_or_zone_id", format!("Worker-ID-{}", tid))?;
                        alert.set_item("severity", "CRITICAL")?;
                        alert.set_item(
                            "message",
                            format!("Worker #{} missing required PPE: {} (at {:.1}s)", tid, missing_str, video_ts),
                        )?;
                        alerts_out.append(alert)?;
                    }
                }
                boxes_out.append(ov)?;
            }
        }

        // 2. MODULE 2: OFFICE MOBILE PHONE USAGE
        if mode == "OFFICE" {
            let phones: Vec<&Detection> = detections
                .iter()
                .filter(|d| matches!(d.class_name.as_str(), "cell phone" | "mobile_phone" | "phone"))
                .collect();

            let mut active_ids = HashSet::new();

            for p in &persons {
                let tid = p.track_id;
                let using_phone = phones
                    .iter()
                    .any(|ph| is_item_associated_with_person(&p.bbox, &ph.bbox, 0.20));

                let ov = PyDict::new_bound(py);
                ov.set_item("bbox", p.bbox.to_vec())?;

                if using_phone {
                    active_ids.insert(tid);
                    let reset_needed = match self.phone_states.get(&tid) {
                        Some(st) => (video_ts - st.last_seen_at) > self.dropout_tolerance_sec,
                        None => true,
                    };

                    if reset_needed {
                        self.phone_states.insert(
                            tid,
                            PhoneState {
                                first_detected_at: video_ts,
                                last_seen_at: video_ts,
                                last_alert_sent_at: -999.0,
                                active_duration: 0.0,
                            },
                        );
                    }

                    let st = self.phone_states.get_mut(&tid).unwrap();
                    st.last_seen_at = video_ts;
                    st.active_duration = (video_ts - st.first_detected_at).max(0.0);
                    let dur = st.active_duration;
                    let is_violation = dur >= self.phone_threshold_sec;

                    ov.set_item("label", format!("Emp #{}: PHONE ({:.1}s)", tid, dur))?;
                    ov.set_item(
                        "color",
                        if is_violation {
                            vec![59, 59, 255]
                        } else {
                            vec![0, 215, 255]
                        },
                    )?;

                    if is_violation && (st.last_alert_sent_at < 0.0 || (video_ts - st.last_alert_sent_at) >= self.phone_repeat_sec) {
                        st.last_alert_sent_at = video_ts;
                        let alert = PyDict::new_bound(py);
                        alert.set_item("module_name", "OFFICE")?;
                        alert.set_item("alert_type", "ACTIVE_PHONE_USAGE")?;
                        alert.set_item("table_or_zone_id", format!("Employee-ID-{}", tid))?;
                        alert.set_item("severity", "WARNING")?;
                        alert.set_item(
                            "message",
                            format!("Employee #{} continuously using mobile phone for {:.1}s.", tid, dur),
                        )?;
                        alerts_out.append(alert)?;
                    }
                } else {
                    ov.set_item("label", format!("Emp #{}: Working", tid))?;
                    ov.set_item("color", vec![160, 160, 160])?;
                }
                boxes_out.append(ov)?;
            }

            let tol = self.dropout_tolerance_sec;
            self.phone_states
                .retain(|k, v| active_ids.contains(k) || (video_ts - v.last_seen_at) <= tol);
        }

        // 3. MODULE 3: RESTAURANT SMART TABLE & SERVICE MONITOR
        if mode == "RESTAURANT" {
            let waiters: Vec<&Detection> = detections
                .iter()
                .filter(|d| matches!(d.class_name.as_str(), "waiter" | "staff" | "server"))
                .collect();
            let foods: Vec<&Detection> = detections
                .iter()
                .filter(|d| {
                    matches!(
                        d.class_name.as_str(),
                        "bowl" | "cup" | "pizza" | "sandwich" | "cake" | "donut" | "apple" | "orange" | "food" | "plate" | "dish" | "bottle"
                    )
                })
                .collect();

            let fw = frame_w as f64;
            let fh = frame_h as f64;

            for st in &mut self.tables {
                let customers_count = persons
                    .iter()
                    .filter(|p| {
                        let (cx, cy) = get_center(&p.bbox);
                        is_point_in_norm_polygon(cx, cy, &st.norm_polygon, fw, fh)
                    })
                    .count();

                let waiter_in_roi = waiters.iter().any(|w| {
                    let (cx, cy) = get_center(&w.bbox);
                    is_point_in_norm_polygon(cx, cy, &st.norm_polygon, fw, fh)
                });

                let food_in_roi = foods.iter().any(|f| {
                    let (cx, cy) = get_center(&f.bbox);
                    is_point_in_norm_polygon(cx, cy, &st.norm_polygon, fw, fh)
                });

                if customers_count > 0 {
                    st.last_person_seen = video_ts;
                    if st.status == "FREE" {
                        st.status = "OCCUPIED".to_string();
                        st.occupied_since = video_ts;
                        st.waiter_visited = false;
                        st.idle_warning_sent = false;
                        st.unattended_alert_sent = false;
                        st.late_food_alert_sent = false;
                    }

                    if food_in_roi && st.status != "DINING" {
                        st.status = "DINING".to_string();
                        st.food_delivered_at = video_ts;
                    }

                    if waiter_in_roi {
                        st.waiter_currently_present = true;
                        st.waiter_visited = true;
                    } else if st.waiter_currently_present && st.status == "OCCUPIED" {
                        st.waiter_currently_present = false;
                        st.status = "ORDER_TAKEN_WAITING_FOOD".to_string();
                        st.order_taken_at = video_ts;
                    }

                    if st.status == "OCCUPIED" && !st.waiter_visited && !food_in_roi {
                        let elapsed = (video_ts - st.occupied_since).max(0.0);
                        if elapsed >= self.idle_warning_sec && !st.idle_warning_sent {
                            st.idle_warning_sent = true;
                            let alert = PyDict::new_bound(py);
                            alert.set_item("module_name", "RESTAURANT")?;
                            alert.set_item("alert_type", "IDLE_NO_ORDER_WARNING")?;
                            alert.set_item("table_or_zone_id", st.table_id.clone())?;
                            alert.set_item("severity", "WARNING")?;
                            alert.set_item(
                                "message",
                                format!("{}: Customers sitting idle without ordering ({:.0}s elapsed).", st.table_id, elapsed),
                            )?;
                            alerts_out.append(alert)?;
                        }
                        if elapsed >= self.unattended_dispatch_sec && !st.unattended_alert_sent {
                            st.unattended_alert_sent = true;
                            let alert = PyDict::new_bound(py);
                            alert.set_item("module_name", "RESTAURANT")?;
                            alert.set_item("alert_type", "UNATTENDED_CUSTOMER_DISPATCH_WAITER")?;
                            alert.set_item("table_or_zone_id", st.table_id.clone())?;
                            alert.set_item("severity", "CRITICAL")?;
                            alert.set_item(
                                "message",
                                format!("{}: Occupied for {:.0}s without waiter! Dispatch waiter immediately.", st.table_id, elapsed),
                            )?;
                            alerts_out.append(alert)?;
                        }
                    } else if st.status == "ORDER_TAKEN_WAITING_FOOD" && !food_in_roi {
                        let wait_elapsed = (video_ts - st.order_taken_at).max(0.0);
                        if wait_elapsed >= self.late_food_sec && !st.late_food_alert_sent {
                            st.late_food_alert_sent = true;
                            let alert = PyDict::new_bound(py);
                            alert.set_item("module_name", "RESTAURANT")?;
                            alert.set_item("alert_type", "LATE_FOOD_DELIVERY")?;
                            alert.set_item("table_or_zone_id", st.table_id.clone())?;
                            alert.set_item("severity", "CRITICAL")?;
                            alert.set_item(
                                "message",
                                format!("{}: Late Food Delivery! Waiting {:.0}s since order was taken.", st.table_id, wait_elapsed),
                            )?;
                            alerts_out.append(alert)?;
                        }
                    }
                } else if st.status != "FREE" && (video_ts - st.last_person_seen) >= self.clear_grace_sec {
                    st.status = "FREE".to_string();
                    st.occupied_since = 0.0;
                    st.waiter_visited = false;
                    st.waiter_currently_present = false;
                }

                let (elapsed_display, color) = match st.status.as_str() {
                    "OCCUPIED" => (
                        (video_ts - st.occupied_since).max(0.0),
                        if st.unattended_alert_sent {
                            vec![59, 59, 255]
                        } else {
                            vec![0, 165, 255]
                        },
                    ),
                    "ORDER_TAKEN_WAITING_FOOD" => (
                        (video_ts - st.order_taken_at).max(0.0),
                        if st.late_food_alert_sent {
                            vec![59, 59, 255]
                        } else {
                            vec![255, 191, 0]
                        },
                    ),
                    "DINING" => ((video_ts - st.food_delivered_at).max(0.0), vec![46, 204, 113]),
                    _ => (0.0, vec![130, 130, 130]),
                };

                let pixel_poly: Vec<[i32; 2]> = st
                    .norm_polygon
                    .iter()
                    .map(|pt| [(pt[0] * fw) as i32, (pt[1] * fh) as i32])
                    .collect();

                let roi_dict = PyDict::new_bound(py);
                roi_dict.set_item("table_id", st.table_id.clone())?;
                roi_dict.set_item("polygon", pixel_poly)?;
                roi_dict.set_item("status", st.status.clone())?;
                roi_dict.set_item("elapsed", elapsed_display)?;
                roi_dict.set_item("color", color)?;
                roi_dict.set_item("customers", customers_count)?;
                rois_out.append(roi_dict)?;
            }
        }

        Ok((alerts_out, boxes_out, rois_out))
    }
}

#[pymodule]
fn security_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSecurityEngine>()?;
    Ok(())
}

