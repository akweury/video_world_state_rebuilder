
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import json

from src.pipeline.step01_od import DetectionTier, ObjectCandidate, load_od_model
from src.pipeline.step01_mask import load_mask_model
from src.pipeline.step01_depth import load_depth_model
from src.pipeline.step01_flow import load_flow_model
from src.pipeline.step01_tensor import load_packing_model
from src.utils import data_utils


def _mask_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (70, 130, 180),
        (238, 130, 238),
        (255, 215, 0),
        (0, 206, 209),
        (255, 140, 0),
        (154, 205, 50),
        (123, 104, 238),
        (255, 182, 193),
        (255, 255, 255),
    ]
    return palette[index % len(palette)]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    left_area = float(max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1))
    right_area = float(max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1))
    union_area = left_area + right_area - inter_area
    return 0.0 if union_area <= 0.0 else inter_area / union_area


def _mask_label_candidates(output, detections, mask_index: int, top_k: int) -> list[dict]:
    prompt_detection_id = str(getattr(output, "prompt_detection_id", ""))
    if prompt_detection_id.startswith("semseg:"):
        label = prompt_detection_id.split(":", 1)[1].replace("_", " ")
        score = float(getattr(output, "confidence", 0.0))
        return [
            {
                "label": label,
                "score": score,
                "mask_label_score": score,
                "source": "semseg",
                "rank": 1,
            }
        ]

    if prompt_detection_id.startswith("prompt:"):
        try:
            detection_index = int(prompt_detection_id.split(":", 1)[1])
        except (ValueError, IndexError):
            detection_index = -1
        if 0 <= detection_index < len(detections):
            detection = detections[detection_index]
            score = float(detection.confidence)
            candidates = [
                {
                    "label": str(detection.class_name),
                    "score": score,
                    "mask_label_score": score,
                    "source": "prompt",
                    "rank": 1,
                }
            ]
            return candidates[: max(1, int(top_k))]

    mask_bbox = _mask_bbox(np.asarray(output.mask, dtype=bool))
    if mask_bbox is None:
        fallback_label = prompt_detection_id or f"mask_{mask_index:04d}"
        return [
            {
                "label": fallback_label,
                "score": 0.0,
                "mask_label_score": 0.0,
                "source": "fallback",
                "rank": 1,
            }
        ]

    candidates: list[dict] = []
    for detection in detections:
        detection_bbox = tuple(int(round(value)) for value in detection.bbox_xyxy)
        overlap_score = _bbox_iou(mask_bbox, detection_bbox)
        if overlap_score <= 0.0:
            continue
        label_score = float(detection.confidence) * overlap_score
        candidates.append(
            {
                "label": str(detection.class_name),
                "score": round(label_score, 6),
                "mask_label_score": round(label_score, 6),
                "iou": round(overlap_score, 6),
                "confidence": float(detection.confidence),
                "source": "detection_overlap",
                "rank": 0,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("confidence", 0.0)),
            -float(item.get("iou", 0.0)),
            str(item.get("label", "")),
        )
    )

    if not candidates:
        fallback_label = prompt_detection_id if prompt_detection_id else f"proposal {mask_index:04d}"
        return [
            {
                "label": fallback_label,
                "score": 0.0,
                "mask_label_score": 0.0,
                "source": "fallback",
                "rank": 1,
            }
        ]

    top_k = max(1, int(top_k))
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate["rank"] = rank
    return candidates[:top_k]


def _infer_mask_label(output, detections, mask_index: int, top_k: int) -> str:
    candidates = _mask_label_candidates(output, detections, mask_index, top_k)
    if not candidates:
        return f"proposal {mask_index:04d}"
    return str(candidates[0].get("label", f"proposal {mask_index:04d}"))


def _draw_frame_mask_visual(frame_bgr, detections, outputs, output_path, top_k: int = 3):
    """Render a single visualization image with colored masks, boxes, and labels."""
    canvas = frame_bgr.copy()
    overlay = frame_bgr.copy()
    alpha = 0.45

    for mask_index, output in enumerate(outputs):
        mask = np.asarray(output.mask, dtype=bool)
        if mask.shape[:2] != canvas.shape[:2]:
            continue

        color = np.array(_mask_color(mask_index), dtype=np.float32)
        mask_pixels = mask.astype(bool)
        overlay_pixels = overlay[mask_pixels].astype(np.float32)
        overlay[mask_pixels] = np.clip(
            overlay_pixels * (1.0 - alpha) + color * alpha,
            0,
            255,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

        ys, xs = np.where(mask_pixels)
        if xs.size and ys.size:
            center_x = int(xs.mean())
            center_y = int(ys.mean())
            label = _infer_mask_label(output, detections, mask_index, top_k)
            cv2.putText(
                overlay,
                label,
                (center_x + 4, center_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color.tolist(),
                1,
                cv2.LINE_AA,
            )

    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        box_color = _mask_color(detection_index + 32)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f} {detection.tier.value}"
        label_origin = (x1, max(0, y1 - 6))
        cv2.putText(
            overlay,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    composed = cv2.addWeighted(overlay, 1.0, canvas, 0.25, 0.0)
    cv2.imwrite(str(output_path), composed)


def _video_to_frames(video_path, output_dir):
    """
    Convert a video into frames and save them as images in the output directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
    Returns:
        All frames path in the output directory.
    """
    if os.path.exists(output_dir):
        # return all the frame file paths in the output directory
        return sorted(
            path for path in Path(output_dir).iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()
    return sorted(
        str(path) for path in Path(output_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

def _frames_to_flows(frame_dir, output_dir, flow_model):
    print(f"- frames to flows for video: {Path(frame_dir).name}")
    if os.path.exists(output_dir):
        # return all the frame file paths in the output directory
        return sorted(
            str(path) for path in Path(output_dir).iterdir()
            if path.suffix.lower() in {".npz"}
        )
        

    os.makedirs(output_dir, exist_ok=True)
    flow_model.warmup()

    frame_paths = sorted(
        path for path in Path(frame_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if len(frame_paths) < 2:
        return

    class Frame:
        def __init__(self, video_id, frame_index, image_bgr):
            self.video_id = video_id
            self.frame_index = frame_index
            self.timestamp_s = 0.0
            self.source_frame_index = frame_index
            self.source_timestamp_s = 0.0
            self.image_bgr = image_bgr

        @property
        def image_rgb(self):
            return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

    def read_frame(frame_index, frame_path):
        image_bgr = data_utils.load_image_bgr(frame_path)
        if image_bgr is None:
            return None
        return Frame(Path(output_dir).name, frame_index, image_bgr)

    results = []
    previous = read_frame(0, frame_paths[0])
    if previous is None:
        return

    for frame_index, frame_path in enumerate(frame_paths[1:], start=1):
        current = read_frame(frame_index, frame_path)
        if current is None:
            continue

        pair = flow_model.predict_pair(previous, current)

        forward_path = Path(output_dir) / f"{frame_paths[frame_index - 1].stem}_to_{frame_path.stem}_forward.npz"
        backward_path = Path(output_dir) / f"{frame_path.stem}_to_{frame_paths[frame_index - 1].stem}_backward.npz"

        np.savez_compressed(
            forward_path,
            flow=pair.forward.flow.astype(np.float32, copy=False),
            domain_valid=pair.forward.domain_valid.astype(np.uint8, copy=False),
            consistency_valid=pair.forward.consistency_valid.astype(np.uint8, copy=False),
            fb_error=pair.forward.fb_error.astype(np.float32, copy=False),
        )
        np.savez_compressed(
            backward_path,
            flow=pair.backward.flow.astype(np.float32, copy=False),
            domain_valid=pair.backward.domain_valid.astype(np.uint8, copy=False),
            consistency_valid=pair.backward.consistency_valid.astype(np.uint8, copy=False),
            fb_error=pair.backward.fb_error.astype(np.float32, copy=False),
        )

        results.append(
            {
                "source_frame": frame_paths[frame_index - 1].name,
                "target_frame": frame_path.name,
                "forward_path": forward_path.name,
                "backward_path": backward_path.name,
            }
        )
        previous = current

    import json

    with open(Path(output_dir) / "flows.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

def _frames_to_depths(frame_dir, output_dir, depth_model):
    """
    use depth anything v3 to estimate the depth maps for each frame in the given directory.
    
    Args:
        frame_dir (str): Directory containing the frames for which depth maps will be estimated.
        output_dir (str): Directory where the estimated depth maps will be saved.
        depth_model: The depth estimation model to be used for predicting depth maps.
    """
    print(f"- frames to depths for video: {Path(frame_dir).name}")
    if os.path.exists(output_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    depth_model.warmup()

    frame_paths = sorted(
        path for path in Path(frame_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not frame_paths:
        return

    class Frame:
        def __init__(self, video_id, frame_index, image_bgr):
            self.video_id = video_id
            self.frame_index = frame_index
            self.timestamp_s = 0.0
            self.source_frame_index = frame_index
            self.source_timestamp_s = 0.0
            self.image_bgr = image_bgr

    def read_frame(frame_index, frame_path):
        image_bgr = data_utils.load_image_bgr(frame_path)
        if image_bgr is None:
            return None
        return Frame(Path(output_dir).name, frame_index, image_bgr)

    for frame_index, frame_path in enumerate(frame_paths):
        frame = read_frame(frame_index, frame_path)
        if frame is None:
            continue

        output = depth_model.predict_frame(frame)
        depth_path = Path(output_dir) / f"{frame_path.stem}.npz"
        np.savez_compressed(
            depth_path,
            depth=output.depth.astype(np.float32, copy=False),
            valid=output.valid.astype(np.uint8, copy=False),
            confidence=(
                output.confidence.astype(np.float32, copy=False)
                if output.confidence is not None
                else np.array([], dtype=np.float32)
            ),
        )

def _frames_to_objects(video_id, frame_dir, output_dir, od_model, batch_size=8):
    """
    use the object detection model to detect objects in each frame and save the results as json file.
    
    Args:
        frame_dir (str): Directory containing the frames for object detection.
        output_dir (str): Directory where the detected objects will be saved.
        od_model: The object detection model to be used for detecting objects.
    """
    print(f"- frames to objects for video: {video_id}")
    obj_file = Path(output_dir) / f"{video_id}_objects.json"
    if os.path.exists(obj_file):
        return
    od_model.warmup()

    frame_paths = sorted(
        path for path in Path(frame_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not frame_paths:
        return

    class Frame:
        def __init__(self, image_bgr):
            self.image_bgr = image_bgr

    def serialize_candidate(candidate):
        return {
            "bbox_xyxy": [float(value) for value in candidate.bbox_xyxy],
            "class_name": candidate.class_name,
            "confidence": float(candidate.confidence),
            "tier": candidate.tier.value,
        }

    detections_by_frame = []
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        frames = []
        valid_paths = []
        for frame_path in batch_paths:
            image_bgr = data_utils.load_image_bgr(frame_path)
            if image_bgr is None:
                continue
            frames.append(Frame(image_bgr))
            valid_paths.append(frame_path)

        if not frames:
            continue

        candidates_batch = od_model.predict_batch(frames)
        if len(candidates_batch) != len(frames):
            raise RuntimeError("object backend returned the wrong batch length")

        for frame_path, candidates in zip(valid_paths, candidates_batch):
            detections_by_frame.append(
                {
                    "frame": frame_path.name,
                    "objects": [serialize_candidate(candidate) for candidate in candidates],
                }
            )

    os.makedirs(Path(obj_file).parent, exist_ok=True)
    with open(obj_file, "w", encoding="utf-8") as handle:
        import json
        json.dump(detections_by_frame, handle, indent=2)

def _frames_to_masks(video_id, frame_paths, output_dir, mask_model, label_top_k: int = 3):
    """
    use the mask model to detect masks in each frame and save the results as json file.
    
    Args:
        frame_dir (str): Directory containing the frames for mask detection.
        output_dir (str): Directory where the detected masks will be saved.
        mask_model: The mask model to be used for detecting masks.
    """
    print(f"- frames to masks for video: {video_id}")
    mask_file = Path(output_dir) / f"{video_id}_masks.json"
    if os.path.exists(mask_file):
        return
    mask_model.warmup()
    objects_json = data_utils.load_json(Path(output_dir).parent / "objects" / f"{video_id}_objects.json")
    detections_by_frame = {entry["frame"]: entry.get("objects", []) for entry in objects_json}
    
    class Frame:
        def __init__(self, frame_name, image_bgr, frame_index):
            self.video_id = video_id
            self.frame_index = frame_index
            self.timestamp_s = 0.0
            self.source_frame_index = frame_index
            self.source_timestamp_s = 0.0
            self.image_bgr = image_bgr
            self.frame_name = frame_name

        @property
        def image_rgb(self):
            return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

    def restore_candidate(raw_candidate):
        return ObjectCandidate(
            bbox_xyxy=tuple(float(value) for value in raw_candidate["bbox_xyxy"]),
            class_name=str(raw_candidate["class_name"]),
            confidence=float(raw_candidate["confidence"]),
            tier=DetectionTier(raw_candidate["tier"]),
        )

    mask_output_dir = Path(output_dir) / video_id
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for frame_index, frame_path in enumerate(frame_paths):
        image_bgr = data_utils.load_image_bgr(frame_path)
        if image_bgr is None:
            continue

        frame = Frame(frame_path.name, image_bgr, frame_index)
        detections = [restore_candidate(candidate) for candidate in detections_by_frame.get(frame_path.name, [])]
        outputs = mask_model.predict_frame(frame, detections)

        frame_mask_dir = mask_output_dir / frame_path.stem
        frame_mask_dir.mkdir(parents=True, exist_ok=True)

        frame_results = []
        for mask_index, output in enumerate(outputs):
            mask_path = frame_mask_dir / f"mask_{mask_index:04d}.png"
            cv2.imwrite(str(mask_path), (output.mask.astype(np.uint8) * 255))
            label_candidates = _mask_label_candidates(output, detections, mask_index, label_top_k)
            frame_results.append(
                {
                    "prompt_detection_id": output.prompt_detection_id,
                    "confidence": float(output.confidence),
                    "label": label_candidates[0]["label"] if label_candidates else None,
                    "label_candidates": label_candidates,
                    "mask_path": str(mask_path.relative_to(Path(output_dir).parent)),
                    "mask_pixels": int(np.count_nonzero(output.mask)),
                }
            )

        visual_path = frame_mask_dir / "masks_visual.png"
        _draw_frame_mask_visual(image_bgr, detections, outputs, visual_path, label_top_k)
        for entry in frame_results:
            entry["visual_path"] = str(visual_path.relative_to(Path(output_dir).parent))

        results.append({"frame": frame_path.name, "masks": frame_results})
    data_utils.save_json(results, mask_file)

def _frames_to_records(video_id, frame_dir, depth_dir, flow_dir, obj_dir, mask_dir, output_dir, tensor_model):
    print(f"- frames to dicts for video: {video_id}")
    """
    Convert the processed frames, depth maps, flow maps, object detections, 
    and masks into a dict format
    """
    tensor_model.warmup()
    context = tensor_model.prepare_video(
        video_id=video_id,
        frame_dir=frame_dir,
        depth_dir=depth_dir,
        flow_dir=flow_dir,
        obj_dir=obj_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
    )

    if context.get("already_packed"):
        return []

    frame_tensors = []
    for frame_index, frame_path in enumerate(context.get("frame_paths", [])):
        frame_record = tensor_model.pack_frame(
            frame_index=frame_index,
            frame_path=frame_path,
            depth_dir=context.get("depth_dir", depth_dir),
            flow_dir=context.get("flow_dir", flow_dir),
            obj_dir=context.get("obj_dir", obj_dir),
            mask_dir=context.get("mask_dir", mask_dir),
            output_dir=output_dir,
            objects_by_frame=context.get("objects_by_frame", {}),
            masks_by_frame=context.get("masks_by_frame", {}),
            flow_frames=context.get("flow_frames", []),
        )
        if frame_record is not None:
            frame_tensors.append(frame_record)

    return tensor_model.finalize_video(context["dict_path"], frame_tensors)



def main(input_data):
    print("\n------- Step 01 -------\n")
    od_model = load_od_model(input_data)
    mask_model = load_mask_model(input_data)
    depth_model = load_depth_model(input_data)
    flow_model = load_flow_model(input_data)
    packing_model = load_packing_model(input_data)
    mask_label_top_k = int(input_data.get("mask_label_top_k", 3))
    all_video_ids = input_data["video_ids"]
    all_video_paths = input_data["video_path"]
    all_frame_paths = input_data["frame_path"]
    all_depth_paths = input_data["depth_path"]
    all_flow_paths = input_data["flow_path"]
    output_dir = input_data["output_dir"]

    obj_dir = output_dir / "objects"
    mask_dir = output_dir / "masks"
    record_dir = output_dir / "records"
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(record_dir, exist_ok=True)

    print(f"- Total Processed Videos: {len(all_video_paths)}")
    for vid, v_path, f_path, d_path, flow_path in tqdm(zip(all_video_ids, all_video_paths, all_frame_paths, all_depth_paths, all_flow_paths),
                                            desc="frames to objects/masks/depths/flows", total=len(all_video_ids)):
        frame_paths = _video_to_frames(v_path, f_path)
        _frames_to_objects(vid, f_path, obj_dir, od_model)
        _frames_to_masks(vid, frame_paths, mask_dir, mask_model, mask_label_top_k)
        _frames_to_depths(f_path, d_path, depth_model)
        _frames_to_flows(f_path, flow_path, flow_model)
        _frames_to_records(vid, f_path, d_path, flow_path, obj_dir, mask_dir, record_dir, packing_model)


     
    print("\n--------- Step 01 Done ---------------\n")

    