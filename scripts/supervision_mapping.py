import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATAROOT = "datasets/nuscenes"
DEFAULT_VERSION = "v1.0-mini"
CAMERA_CHANNEL = "CAM_FRONT"
FRONT_REGION_X_METERS = 2.5
IMAGE_BORDER_MARGIN_RATIO = 0.03


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_global_to_sensor(
    point_global: Sequence[float],
    ego_translation: Sequence[float],
    ego_rotation: Sequence[float],
    sensor_translation: Sequence[float],
    sensor_rotation: Sequence[float],
) -> np.ndarray:
    point = np.asarray(point_global, dtype=np.float64)
    ego_t = np.asarray(ego_translation, dtype=np.float64)
    sensor_t = np.asarray(sensor_translation, dtype=np.float64)
    ego_r = quaternion_to_rotation_matrix(ego_rotation)
    sensor_r = quaternion_to_rotation_matrix(sensor_rotation)

    point_ego = ego_r.T @ (point - ego_t)
    point_sensor = sensor_r.T @ (point_ego - sensor_t)
    return point_sensor


def project_point(point_sensor: Sequence[float], intrinsic: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    point = np.asarray(point_sensor, dtype=np.float64)
    if point[2] <= 1e-3:
        return None

    k = np.asarray(intrinsic, dtype=np.float64)
    projected = k @ point
    return projected[:2] / projected[2]


def compute_box_corners(size: Sequence[float], rotation: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    width, length, height = [float(v) for v in size]
    x_corners = np.array([length / 2, length / 2, length / 2, length / 2, -length / 2, -length / 2, -length / 2, -length / 2])
    y_corners = np.array([width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2])
    z_corners = np.array([height / 2, height / 2, -height / 2, -height / 2, height / 2, height / 2, -height / 2, -height / 2])

    corners_local = np.vstack([x_corners, y_corners, z_corners])
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    translation_vec = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return (rotation_matrix @ corners_local) + translation_vec


def project_3d_box_to_image(
    annotation: Dict[str, Any],
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, Any]]:
    corners_global = compute_box_corners(annotation["size"], annotation["rotation"], annotation["translation"])
    corners_sensor = []
    corners_image = []
    for idx in range(corners_global.shape[1]):
        corner_sensor = transform_global_to_sensor(
            corners_global[:, idx],
            ego_pose["translation"],
            ego_pose["rotation"],
            calibrated_sensor["translation"],
            calibrated_sensor["rotation"],
        )
        corners_sensor.append(corner_sensor)
        corner_image = project_point(corner_sensor, calibrated_sensor["camera_intrinsic"])
        if corner_image is not None:
            corners_image.append(corner_image)

    if not corners_image:
        return None

    corners_image_np = np.asarray(corners_image, dtype=np.float64)
    x1_raw = float(np.min(corners_image_np[:, 0]))
    y1_raw = float(np.min(corners_image_np[:, 1]))
    x2_raw = float(np.max(corners_image_np[:, 0]))
    y2_raw = float(np.max(corners_image_np[:, 1]))

    x1_clip = float(np.clip(x1_raw, 0.0, float(image_width)))
    y1_clip = float(np.clip(y1_raw, 0.0, float(image_height)))
    x2_clip = float(np.clip(x2_raw, 0.0, float(image_width)))
    y2_clip = float(np.clip(y2_raw, 0.0, float(image_height)))

    raw_area = max(x2_raw - x1_raw, 0.0) * max(y2_raw - y1_raw, 0.0)
    clipped_area = max(x2_clip - x1_clip, 0.0) * max(y2_clip - y1_clip, 0.0)
    visible_area_ratio = clipped_area / raw_area if raw_area > 0 else 0.0

    return {
        "bbox_xyxy": [x1_clip, y1_clip, x2_clip, y2_clip],
        "bbox_center_x": (x1_clip + x2_clip) / 2.0,
        "bbox_area": clipped_area,
        "visible_area_ratio": visible_area_ratio,
        "is_truncated": (x1_raw < 0.0) or (y1_raw < 0.0) or (x2_raw > image_width) or (y2_raw > image_height),
        "raw_bbox_xyxy": [x1_raw, y1_raw, x2_raw, y2_raw],
        "corners_sensor": np.asarray(corners_sensor, dtype=np.float64),
    }


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_area = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def overlap_ratio(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_area = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    if area_a <= 0.0:
        return 0.0
    return inter_area / area_a


class NuScenesTableLoader:
    def __init__(self, dataroot: str = DEFAULT_DATAROOT, version: str = DEFAULT_VERSION):
        self.dataroot = Path(dataroot)
        self.version = version
        self.table_root = self.dataroot / version
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.token_index: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for name in [
            "sample",
            "sample_data",
            "sample_annotation",
            "scene",
            "log",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "visibility",
            "attribute",
            "instance",
            "category",
            "map",
        ]:
            table = json.loads((self.table_root / f"{name}.json").read_text(encoding="utf-8"))
            self.tables[name] = table
            self.token_index[name] = {row["token"]: row for row in table}

        self.sample_to_annotations: Dict[str, List[Dict[str, Any]]] = {}
        for ann in self.tables["sample_annotation"]:
            self.sample_to_annotations.setdefault(ann["sample_token"], []).append(ann)

        self.sample_to_sample_data: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.tables["sample_data"]:
            self.sample_to_sample_data.setdefault(row["sample_token"], []).append(row)

        self.visibility_levels = {row["token"]: row["level"] for row in self.tables["visibility"]}
        self.attribute_names = {row["token"]: row["name"] for row in self.tables["attribute"]}
        self.category_names = {
            row["token"]: row["name"] for row in self.tables["category"]
        }

    def get(self, table: str, token: str) -> Dict[str, Any]:
        return self.token_index[table][token]

    def get_sample(self, sample_token: str) -> Dict[str, Any]:
        return self.get("sample", sample_token)

    def get_annotations(self, sample_token: str) -> List[Dict[str, Any]]:
        return list(self.sample_to_annotations.get(sample_token, []))

    def get_category_name(self, annotation: Dict[str, Any]) -> str:
        instance = self.get("instance", annotation["instance_token"])
        return self.category_names[instance["category_token"]]

    def get_attribute_names(self, annotation: Dict[str, Any]) -> List[str]:
        return [self.attribute_names[token] for token in annotation.get("attribute_tokens", []) if token in self.attribute_names]

    def get_camera_sample_data(self, sample_token: str, channel: str = CAMERA_CHANNEL) -> Dict[str, Any]:
        for row in self.sample_to_sample_data.get(sample_token, []):
            calibrated = self.get("calibrated_sensor", row["calibrated_sensor_token"])
            sensor = self.get("sensor", calibrated["sensor_token"])
            if sensor["channel"] == channel and row["is_key_frame"]:
                return row
        raise KeyError(f"No key-frame {channel} sample_data found for sample {sample_token}")

    def get_scene_context(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        scene = self.get("scene", sample["scene_token"])
        log = self.get("log", scene["log_token"])
        map_row = next((row for row in self.tables["map"] if scene["log_token"] in row["log_tokens"]), None)
        return {"scene": scene, "log": log, "map": map_row}


def map_horizontal_position(x_coord: float, image_width: int) -> str:
    # Rule: use projected image x coordinate to bucket objects into left / center / right.
    normalized_x = x_coord / float(image_width)
    if normalized_x < 1.0 / 3.0:
        return "left"
    if normalized_x > 2.0 / 3.0:
        return "right"
    return "center"


def map_depth_bucket(distance_m: Optional[float], bbox_area: Optional[float], image_area: float) -> str:
    # Rule: prefer annotation distance when available and use area-based scale as a fallback proxy.
    if distance_m is not None:
        if distance_m < 12.0:
            return "near"
        if distance_m < 30.0:
            return "mid"
        return "far"

    area_ratio = (bbox_area or 0.0) / image_area if image_area > 0 else 0.0
    if area_ratio > 0.06:
        return "near"
    if area_ratio > 0.015:
        return "mid"
    return "far"


def map_occlusion_label(
    visibility_level: Optional[str],
    is_truncated: bool,
    visible_area_ratio: float,
    max_overlap_ratio: float,
    lidar_points: int,
    radar_points: int,
    bbox_xyxy: Optional[Sequence[float]] = None,
    image_width: Optional[int] = None,
) -> str:
    # Rule: use nuScenes visibility levels directly when present because they are the strongest supervision signal.
    if visibility_level is not None:
        if visibility_level in {"v0-40", "v40-60"}:
            return "occluded"
        return "visible"

    border_touch = False
    if bbox_xyxy is not None and image_width is not None:
        margin = image_width * IMAGE_BORDER_MARGIN_RATIO
        border_touch = bbox_xyxy[0] <= margin or bbox_xyxy[2] >= (image_width - margin)

    # Fallback rule: low visible ratio, truncation, strong overlaps, or sparse points imply proxy occlusion.
    if visible_area_ratio < 0.65 or is_truncated or border_touch:
        return "occluded"
    if max_overlap_ratio > 0.35:
        return "occluded"
    if (lidar_points + radar_points) <= 1:
        return "occluded"
    return "visible"


def estimate_importance(
    category_name: str,
    horizontal_position: str,
    depth_bucket: str,
    occlusion_label: str,
    distance_m: Optional[float],
    front_x_m: Optional[float],
    lane_path_state: Optional[str] = None,
) -> float:
    # Rule: importance is higher for dynamic road users, near objects, center/front objects, and occluded hazards.
    score = 0.2

    if category_name.startswith("human.") or category_name.startswith("vehicle.") or category_name.startswith("cycle."):
        score += 0.25
    if depth_bucket == "near":
        score += 0.3
    elif depth_bucket == "mid":
        score += 0.15

    if horizontal_position == "center":
        score += 0.2
    elif horizontal_position in {"left", "right"}:
        score += 0.1

    if occlusion_label == "occluded":
        score += 0.15

    if distance_m is not None and distance_m < 8.0:
        score += 0.1

    if front_x_m is not None and abs(front_x_m) < FRONT_REGION_X_METERS:
        score += 0.1

    if lane_path_state == "blocked":
        score += 0.1
    elif lane_path_state == "partially_blocked":
        score += 0.05

    return round(float(np.clip(score, 0.0, 1.0)), 3)


def infer_lane_direction(scene_description: str) -> str:
    description = scene_description.lower()

    # Rule: use scene description keywords when no explicit lane-graph API is wired into the current baseline.
    if "turn left" in description or "left turn" in description:
        return "left"
    if "turn right" in description or "right turn" in description:
        return "right"
    return "straight"


def build_lane_proxy(sample_bundle: Dict[str, Any], object_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    scene_description = sample_bundle["scene"]["description"]

    direction = infer_lane_direction(scene_description)

    front_blockers = [
        obj for obj in object_labels
        if obj["depth"] in {"near", "mid"}
        and obj["horizontal_position"] == "center"
        and obj["category_name"].startswith(("vehicle.", "human.", "cycle.", "movable_object."))
    ]
    occluded_front = [obj for obj in front_blockers if obj["occlusion"] == "occluded"]

    # Rule: without direct drivable-space masks, use front-object occupancy as a path-state proxy.
    if len(front_blockers) >= 2 or any(obj["depth"] == "near" for obj in front_blockers):
        path_state = "blocked"
    elif front_blockers or occluded_front:
        path_state = "partially_blocked"
    else:
        path_state = "clear"

    return {
        "direction": direction,
        "path_state": path_state,
        "source": "scene.description + front-object proxy",
    }


def compute_risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def build_risk_proxy(
    sample_bundle: Dict[str, Any],
    object_labels: List[Dict[str, Any]],
    lane_proxy: Dict[str, Any],
) -> Dict[str, Any]:
    scene_description = sample_bundle["scene"]["description"].lower()
    best_score = 0.1
    best_region = "front"
    best_source = "visible"
    reasons: List[str] = []

    for obj in object_labels:
        score = 0.1

        # Rule: object ahead and near contributes strong risk.
        if obj["horizontal_position"] == "center" and obj["depth"] == "near":
            score += 0.45
        elif obj["horizontal_position"] == "center" and obj["depth"] == "mid":
            score += 0.2

        # Rule: near occluded objects near the path are risky even if not fully visible.
        if obj["occlusion"] == "occluded" and obj["horizontal_position"] in {"center", "left", "right"} and obj["depth"] != "far":
            score += 0.25

        # Rule: human and cycle classes receive an extra caution prior.
        if obj["category_name"].startswith(("human.", "cycle.")):
            score += 0.15

        # Rule: front-area objects near likely intersections get extra risk weight.
        if obj["horizontal_position"] == "center" and "intersection" in scene_description:
            score += 0.15

        if score > best_score:
            best_score = score
            best_region = obj["horizontal_position"] if obj["horizontal_position"] in {"left", "right"} else "front"
            best_source = "hidden" if obj["occlusion"] == "occluded" else "visible"
            reasons = [
                f"object:{obj['category_name']}",
                f"depth:{obj['depth']}",
                f"occlusion:{obj['occlusion']}",
            ]

    # Rule: blocked drivable path escalates risk even if no single object dominates.
    if lane_proxy["path_state"] == "blocked":
        blocked_score = 0.85
        if blocked_score > best_score:
            best_score = blocked_score
            best_region = "front"
            best_source = "visible"
            reasons = ["path_state:blocked"]
    elif lane_proxy["path_state"] == "partially_blocked":
        blocked_score = 0.55
        if blocked_score > best_score:
            best_score = blocked_score
            best_region = "front"
            best_source = "visible"
            reasons = ["path_state:partially_blocked"]

    return {
        "region": best_region,
        "risk_level": compute_risk_level(best_score),
        "risk_source": best_source,
        "score": round(float(np.clip(best_score, 0.0, 1.0)), 3),
        "reasons": reasons,
    }


def build_sample_bundle(loader: NuScenesTableLoader, sample_token: str) -> Dict[str, Any]:
    sample = loader.get_sample(sample_token)
    sample_data = loader.get_camera_sample_data(sample_token, CAMERA_CHANNEL)
    ego_pose = loader.get("ego_pose", sample_data["ego_pose_token"])
    calibrated_sensor = loader.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    context = loader.get_scene_context(sample)
    return {
        "sample": sample,
        "sample_data": sample_data,
        "ego_pose": ego_pose,
        "calibrated_sensor": calibrated_sensor,
        "scene": context["scene"],
        "log": context["log"],
        "map": context["map"],
    }


def build_object_proxy(
    loader: NuScenesTableLoader,
    annotation: Dict[str, Any],
    sample_bundle: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    sample_data = sample_bundle["sample_data"]
    ego_pose = sample_bundle["ego_pose"]
    calibrated_sensor = sample_bundle["calibrated_sensor"]

    center_sensor = transform_global_to_sensor(
        annotation["translation"],
        ego_pose["translation"],
        ego_pose["rotation"],
        calibrated_sensor["translation"],
        calibrated_sensor["rotation"],
    )

    if center_sensor[2] <= 0.1:
        return None

    projected_center = project_point(center_sensor, calibrated_sensor["camera_intrinsic"])
    if projected_center is None:
        return None

    projection = project_3d_box_to_image(
        annotation=annotation,
        ego_pose=ego_pose,
        calibrated_sensor=calibrated_sensor,
        image_width=sample_data["width"],
        image_height=sample_data["height"],
    )
    if projection is None:
        return None

    distance_m = float(np.linalg.norm(center_sensor))
    category_name = loader.get_category_name(annotation)
    attribute_names = loader.get_attribute_names(annotation)
    visibility_level = loader.visibility_levels.get(annotation.get("visibility_token"))

    return {
        "annotation_token": annotation["token"],
        "category_name": category_name,
        "attribute_names": attribute_names,
        "distance_m": round(distance_m, 3),
        "center_sensor_xyz": [round(float(v), 3) for v in center_sensor.tolist()],
        "projected_center_xy": [round(float(v), 2) for v in projected_center.tolist()],
        "bbox_xyxy": [round(float(v), 2) for v in projection["bbox_xyxy"]],
        "bbox_area": round(float(projection["bbox_area"]), 2),
        "visibility_level": visibility_level,
        "visible_area_ratio": round(float(projection["visible_area_ratio"]), 3),
        "is_truncated": bool(projection["is_truncated"]),
        "num_lidar_pts": int(annotation.get("num_lidar_pts", 0)),
        "num_radar_pts": int(annotation.get("num_radar_pts", 0)),
    }


def finalize_object_proxies(
    object_proxies: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    image_area = float(image_width * image_height)

    for obj in object_proxies:
        max_overlap = 0.0
        for other in object_proxies:
            if other["annotation_token"] == obj["annotation_token"]:
                continue
            max_overlap = max(max_overlap, overlap_ratio(obj["bbox_xyxy"], other["bbox_xyxy"]))

        obj["max_overlap_ratio"] = round(float(max_overlap), 3)
        obj["horizontal_position"] = map_horizontal_position(obj["projected_center_xy"][0], image_width)
        obj["depth"] = map_depth_bucket(obj["distance_m"], obj["bbox_area"], image_area)
        obj["occlusion"] = map_occlusion_label(
            visibility_level=obj["visibility_level"],
            is_truncated=obj["is_truncated"],
            visible_area_ratio=obj["visible_area_ratio"],
            max_overlap_ratio=max_overlap,
            lidar_points=obj["num_lidar_pts"],
            radar_points=obj["num_radar_pts"],
            bbox_xyxy=obj["bbox_xyxy"],
            image_width=image_width,
        )

    lane_stub = {"path_state": "clear"}
    for obj in object_proxies:
        obj["importance_proxy_score"] = estimate_importance(
            category_name=obj["category_name"],
            horizontal_position=obj["horizontal_position"],
            depth_bucket=obj["depth"],
            occlusion_label=obj["occlusion"],
            distance_m=obj["distance_m"],
            front_x_m=obj["center_sensor_xyz"][0],
            lane_path_state=lane_stub["path_state"],
        )

    return sorted(object_proxies, key=lambda item: item["importance_proxy_score"], reverse=True)


def build_proxy_labels_for_sample(loader: NuScenesTableLoader, sample_token: str) -> Dict[str, Any]:
    sample_bundle = build_sample_bundle(loader, sample_token)
    sample_annotations = loader.get_annotations(sample_token)

    object_proxies: List[Dict[str, Any]] = []
    for ann in sample_annotations:
        object_proxy = build_object_proxy(loader, ann, sample_bundle)
        if object_proxy is not None:
            object_proxies.append(object_proxy)

    object_proxies = finalize_object_proxies(
        object_proxies,
        image_width=sample_bundle["sample_data"]["width"],
        image_height=sample_bundle["sample_data"]["height"],
    )

    lane_proxy = build_lane_proxy(sample_bundle, object_proxies)

    # Refresh importance with the final lane-path proxy because blocked path should increase supervision weight.
    for obj in object_proxies:
        obj["importance_proxy_score"] = estimate_importance(
            category_name=obj["category_name"],
            horizontal_position=obj["horizontal_position"],
            depth_bucket=obj["depth"],
            occlusion_label=obj["occlusion"],
            distance_m=obj["distance_m"],
            front_x_m=obj["center_sensor_xyz"][0],
            lane_path_state=lane_proxy["path_state"],
        )
    object_proxies = sorted(object_proxies, key=lambda item: item["importance_proxy_score"], reverse=True)

    risk_proxy = build_risk_proxy(sample_bundle, object_proxies, lane_proxy)

    return {
        "sample_token": sample_token,
        "scene_name": sample_bundle["scene"]["name"],
        "scene_description": sample_bundle["scene"]["description"],
        "camera_channel": CAMERA_CHANNEL,
        "image_path": str(Path(loader.dataroot) / sample_bundle["sample_data"]["filename"]),
        "location": sample_bundle["log"]["location"],
        "map_filename": sample_bundle["map"]["filename"] if sample_bundle["map"] else None,
        "object_proxy_labels": object_proxies,
        "lane_proxy_token": lane_proxy,
        "risk_proxy_token": risk_proxy,
        "annotation_fields_used": [
            "sample_annotation.translation",
            "sample_annotation.size",
            "sample_annotation.rotation",
            "sample_annotation.visibility_token",
            "sample_annotation.attribute_tokens",
            "sample_annotation.num_lidar_pts",
            "sample_annotation.num_radar_pts",
            "instance.category_token -> category.name",
            "sample_data.width",
            "sample_data.height",
            "sample_data.filename",
            "calibrated_sensor.translation",
            "calibrated_sensor.rotation",
            "calibrated_sensor.camera_intrinsic",
            "ego_pose.translation",
            "ego_pose.rotation",
            "scene.description",
            "log.location",
            "map.filename",
        ],
    }


def describe_proxy_rules() -> Dict[str, Any]:
    return {
        "horizontal_position": "Projected object center x / image width < 1/3 => left, > 2/3 => right, else center.",
        "depth": "Use camera-frame distance when available: <12m near, 12-30m mid, >30m far; fallback to bbox area ratio if distance is unavailable.",
        "occlusion": "Prefer nuScenes visibility_token: v0-40 and v40-60 => occluded, v60-80 and v80-100 => visible. Fallback uses truncation, border touch, low visible-area ratio, overlap ratio, and sparse lidar/radar support.",
        "importance_proxy_score": "Weighted heuristic over class type, depth bucket, horizontal position, occlusion, very short distance, front-path alignment, and blocked/partially blocked path state.",
        "lane_proxy_token": "Direction comes from scene.description keywords when lane API is not wired in; path_state comes from front-center near/mid dynamic objects as a drivable-path proxy.",
        "risk_proxy_token": "Higher risk for near front objects, near occluded path-adjacent objects, blocked path state, pedestrians/cyclists, and objects in front near an intersection scene.",
    }


def sample_tokens(loader: NuScenesTableLoader, limit: int) -> List[str]:
    return [row["token"] for row in loader.tables["sample"][:limit]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak/proxy supervision labels for nuScenes scene tokens.")
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    parser.add_argument("--sample-token", type=str, default=None)
    parser.add_argument("--num-demo-samples", type=int, default=2)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = NuScenesTableLoader(dataroot=args.dataroot, version=args.version)

    if args.sample_token:
        tokens = [args.sample_token]
    else:
        tokens = sample_tokens(loader, args.num_demo_samples)

    outputs = [build_proxy_labels_for_sample(loader, token) for token in tokens]

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        print(f"Saved proxy labels to: {output_path.resolve()}")

    print("Annotation fields used:")
    for field_name in outputs[0]["annotation_fields_used"]:
        print(f"- {field_name}")

    print("\nProxy rules:")
    for key, description in describe_proxy_rules().items():
        print(f"- {key}: {description}")

    print("\nExample proxy-label output:")
    print(json.dumps(outputs[0], indent=2))


if __name__ == "__main__":
    main()
