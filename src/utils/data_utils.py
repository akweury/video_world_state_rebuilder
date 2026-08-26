import json
import csv
from functools import lru_cache
import numpy as np
from pathlib import Path
import cv2

def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)



def _normalize_rotation_value(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"clockwise", "cw", "right", "rotate_cw", "90", "90cw", "+90", "r90"}:
        return "clockwise"
    if text in {"anticlockwise", "counterclockwise", "ccw", "left", "rotate_ccw", "-90", "270", "90ccw", "l90"}:
        return "anticlockwise"
    return None


def _rotation_from_row(row):
    for key in (
        "rotation",
        "rotate",
        "rotationDirection",
        "rotation_direction",
        "direction",
        "orientation",
    ):
        rotation = _normalize_rotation_value(row.get(key))
        if rotation is not None:
            return rotation
    return None


@lru_cache(maxsize=8)
def _load_label_rotation_map(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return {}

    rotation_map = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rotation_map
        for row in reader:
            rotation = _rotation_from_row(row)
            if rotation is None:
                continue
            name = (row.get("name") or "").strip()
            if name:
                rotation_map[name] = rotation
            video_name = (row.get("videoName") or "").strip()
            frame_index = (row.get("frameIndex") or "").strip()
            if video_name and frame_index.isdigit():
                frame_name = f"{video_name}-frame_{int(frame_index):05d}.jpg"
                rotation_map.setdefault(frame_name, rotation)
                rotation_map.setdefault(f"{video_name}-frame_{int(frame_index):05d}.png", rotation)
    return rotation_map


def _find_labels_csv(frame_path):
    frame_path = Path(frame_path)
    for parent in [frame_path.parent, *frame_path.parents]:
        labels_path = parent / "labels.csv"
        if labels_path.exists():
            return labels_path
    return None


def load_image_bgr(frame_path):
    frame_path = Path(frame_path)
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    labels_path = _find_labels_csv(frame_path)
    if labels_path is None:
        raise ValueError(f"Could not find labels.csv for frame: {frame_path}")

    rotation_map = _load_label_rotation_map(labels_path)
    rotation = rotation_map.get(frame_path.name)
    if rotation is None:
        rotation = rotation_map.get(frame_path.with_suffix(".jpg").name)
    if rotation == "clockwise":
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "anticlockwise":
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_bgr


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_npz_dict(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}

def load_json_list(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise ValueError(f"JSON file does not exist: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)