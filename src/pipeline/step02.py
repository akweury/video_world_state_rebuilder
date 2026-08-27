import os 
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from src.utils.association import FrameDetection, FrameDetections, track_objects_across_frames
from src.pipeline.step02_tracker import load_tracker_model
from src.utils.data_utils import _to_numpy

def load_step01_output(step01_output_dir):
    """ 
    Load the output data from step 1,
    which includes masks, object labels, depth maps, flow maps,
    and bounding boxes for each frame in the video.
    The returned data structure is a list of dicts,
    where each dict corresponds to a frame and contains:
    - frame_id: the identifier of the frame
    - objects: a list of detected objects in the frame
    - masks: a list of masks corresponding to the detected objects
    - mask_label_candidates: a list of mask label candidates corresponding to the detected objects
    - depths: a list of depth maps corresponding to the detected objects
    - flows: a list of flow maps corresponding to the detected objects
    - bboxes: a list of bounding boxes corresponding to the detected objects
    """
    step01_output_dir = Path(step01_output_dir)
    record_dir = step01_output_dir / "records"
    record_paths = sorted(record_dir.glob("*_step01.pkl")) if record_dir.exists() else sorted(step01_output_dir.glob("*_step01.pkl"))

    step01_output_data = []
    for record_path in record_paths:
        with record_path.open("rb") as handle:
            frame_records = pickle.load(handle)
        step01_output_data.append(
            {
                "video_id": record_path.stem.replace("_step01", ""),
                "frames": frame_records,
            }
        )

    return step01_output_data


def _frame_record_to_detections(frame_record, frame_index):
    objects = frame_record["objects"]
    masks = frame_record["masks"]
    detections = []

    def load_mask(mask_path):
        mask_path = Path(mask_path)
        if mask_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                return None
            return mask_image > 0
        if mask_path.suffix.lower() in {".npy", ".npz"}:
            loaded = np.load(mask_path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if not loaded.files:
                    return None
                first_key = loaded.files[0]
                return np.asarray(loaded[first_key])
            return np.asarray(loaded)
        return None

    for detection_index, obj in enumerate(objects):
        bbox_xyxy = obj["bbox_xyxy"]
        class_name = obj["class_name"]
        if bbox_xyxy is None or class_name is None:
            continue
        mask_path = masks[detection_index]["mask_path"]
        mask = load_mask(mask_path)
        if mask is None:
            continue
        detections.append(
            FrameDetection(
                detection_id=f"frame:{frame_index:06d}:det:{detection_index:04d}",
                bbox_xyxy=tuple(float(value) for value in bbox_xyxy),
                mask_array=mask,
                class_name=str(class_name),
                confidence=float(obj["confidence"]),
            )
        )
    return FrameDetections(frame_index=frame_index, detections=tuple(detections))

def _serialize_tracks(tracks, frame_name_by_index, track_path):
    serialized_tracks = []
    for track in tracks:
        observations = []
        for observation in track.observations:
            observations.append(
                {
                    "frame_index": observation.frame_index,
                    "frame_name": frame_name_by_index.get(observation.frame_index),
                    "detection_id": observation.detection_id,
                    "mask": _to_numpy(observation.mask).astype(bool) if observation.mask is not None else None,
                    "class_name": observation.class_name,
                    "confidence": observation.confidence,
                    "depth_score": observation.depth_score,
                    "flow_score": observation.flow_score,
                    "mask_pixels": observation.mask_pixels,
                    "beam_candidate_ids": list(getattr(observation, "beam_candidate_ids", ())),
                    "narrowed_candidate_masks": [
                        _to_numpy(mask).astype(bool) if mask is not None else None
                        for mask in getattr(observation, "narrowed_candidate_masks", ())
                    ],
                }
            )

        serialized_tracks.append(
            {
                "track_id": track.track_id,
                "first_frame_index": track.first_frame_index,
                "last_frame_index": track.last_frame_index,
                "cumulative_score": float(getattr(track, "cumulative_score", 0.0)),
                "first_frame_name": frame_name_by_index.get(track.first_frame_index),
                "last_frame_name": frame_name_by_index.get(track.last_frame_index),
                "observations": observations,
            }
        )
    with track_path.open("wb") as handle:
        pickle.dump(serialized_tracks, handle)


    return serialized_tracks


def _save_tracks(video_id, tracks, output_dir):
    track_path = output_dir / f"{video_id}_tracks.pkl"
    serialized_tracks = []
    for track in tracks:
        observations = []
        for observation in track.observations:
            observations.append(
                {
                    "frame_index": observation.frame_index,
                    "frame_name": frame_name_by_index.get(observation.frame_index),
                    "detection_id": observation.detection_id,
                    "mask": _to_numpy(observation.mask).astype(bool) if observation.mask is not None else None,
                    "class_name": observation.class_name,
                    "confidence": observation.confidence,
                    "depth_score": observation.depth_score,
                    "flow_score": observation.flow_score,
                    "mask_pixels": observation.mask_pixels,
                    "beam_candidate_ids": list(observation.beam_candidate_ids),
                    "narrowed_candidate_masks": [
                        _to_numpy(mask).astype(bool) if mask is not None else None
                        for mask in observation.narrowed_candidate_masks
                    ],
                }
            )

        serialized_tracks.append(
            {
                "track_id": track.track_id,
                "first_frame_index": track.first_frame_index,
                "last_frame_index": track.last_frame_index,
                "first_frame_name": frame_name_by_index.get(track.first_frame_index),
                "last_frame_name": frame_name_by_index.get(track.last_frame_index),
                "cumulative_score": track.cumulative_score,
                "observation_count": len(observations),
                "observations": observations,
            }
        )

    with track_path.open("wb") as handle:
        pickle.dump(serialized_tracks, handle)

    _save_tracks_summary(video_id, serialized_tracks, output_dir)
    return serialized_tracks


def _mask_from_value(mask_value):
    if mask_value is None:
        return None
    return np.asarray(_to_numpy(mask_value), dtype=bool)


def _overlay_mask(canvas, mask, color, alpha=0.45, thickness=2):
    if mask is None:
        return
    if mask.shape[:2] != canvas.shape[:2]:
        return
    pixels = mask.astype(bool)
    if not np.any(pixels):
        return
    color_array = np.asarray(color, dtype=np.float32)
    blended = canvas[pixels].astype(np.float32)
    canvas[pixels] = np.clip(blended * (1.0 - alpha) + color_array * alpha, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, tuple(int(value) for value in color), thickness)


def _frame_index_from_name(frame_name):
    stem = Path(frame_name).stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return None
    try:
        return int(stem)
    except ValueError:
        return None


def _flow_to_numpy(flow_tensor):
    flow = _to_numpy(flow_tensor)
    if flow is None:
        return None
    flow = np.asarray(flow)
    if flow.ndim == 3 and flow.shape[0] >= 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow[:2], 0, -1)
    elif flow.ndim == 3 and flow.shape[-1] >= 2:
        flow = flow[..., :2]
    else:
        return None
    return flow.astype(np.float32, copy=False)


def _warp_mask_with_flow(mask, flow_tensor):
    if mask is None:
        return None
    flow = _flow_to_numpy(flow_tensor)
    if flow is None or flow.shape[:2] != mask.shape[:2]:
        return mask

    height, width = mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    warped = cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)


def _visual_tracks(video_id, serialized_tracks, output_dir, frames, visual_fps=30):
    visual_dir = output_dir / "tracks_visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    frame_by_index = {int(frame["frame_index"]): frame for frame in frames}
    if not frame_by_index:
        return []

    sample_frame = next(iter(frame_by_index.values()))
    sample_canvas = np.asarray(sample_frame["frame"])
    height, width = sample_canvas.shape[:2]
    timeline_height = max(54, height // 10)
    output_height = height + timeline_height

    source_frame_dir = Path(sample_frame["frame_path"]).parent
    source_frame_paths = sorted(
        path for path in source_frame_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_paths = []

    for track in serialized_tracks:
        track_path = visual_dir / f"{video_id}_{track['track_id']}.mp4"
        writer = cv2.VideoWriter(str(track_path), fourcc, float(visual_fps), (width, output_height))
        if not writer.isOpened():
            continue

        observations = sorted(track["observations"], key=lambda item: int(item["frame_index"]))
        observations_by_frame = {int(item["frame_index"]): item for item in observations}
        appearance_frames = sorted(observations_by_frame.keys())
        total_frame_count = max(1, len(source_frame_paths))
        current_frame_color = (0, 165, 255)
        appearance_color = (0, 220, 0)
        timeline_bg_color = (28, 28, 28)
        timeline_line_color = (220, 220, 220)

        latest_observation = None

        for frame_path in source_frame_paths:
            frame_index = _frame_index_from_name(frame_path.name)
            if frame_index is None:
                continue

            canvas = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if canvas is None:
                continue

            frame_record = frame_by_index.get(frame_index)
            current_observation = observations_by_frame.get(frame_index)
            if current_observation is not None:
                latest_observation = current_observation

            if latest_observation is not None:
                candidate_masks = [
                    _mask_from_value(mask)
                    for mask in latest_observation.get("narrowed_candidate_masks", [])
                ]
                if candidate_masks:
                    for candidate_mask in candidate_masks:
                        if candidate_mask is not None:
                            _overlay_mask(canvas, candidate_mask, (190, 190, 190), alpha=0.30, thickness=1)
                elif frame_record is not None:
                    candidate_map = {
                        entry["prompt_detection_id"]: _mask_from_value(entry["mask"])
                        for entry in frame_record["masks"]
                    }
                    for candidate_id in latest_observation.get("beam_candidate_ids", []):
                        candidate_mask = candidate_map.get(candidate_id)
                        if candidate_mask is not None:
                            _overlay_mask(canvas, candidate_mask, (190, 190, 190), alpha=0.30, thickness=1)

                moving_mask = _mask_from_value(latest_observation.get("mask"))
                if moving_mask is not None:
                    _overlay_mask(canvas, moving_mask, (0, 255, 0), alpha=0.45, thickness=2)

            timeline = np.full((timeline_height, width, 3), timeline_bg_color, dtype=np.uint8)
            timeline_y = timeline_height // 2
            cv2.line(timeline, (0, timeline_y), (width - 1, timeline_y), timeline_line_color, 2)

            for appearance_frame_index in appearance_frames:
                appearance_x = int(round((appearance_frame_index / max(1, total_frame_count - 1)) * (width - 1)))
                cv2.line(timeline, (appearance_x, 8), (appearance_x, timeline_height - 8), appearance_color, 2)

            current_x = int(round((frame_index / max(1, total_frame_count - 1)) * (width - 1)))
            cv2.line(timeline, (current_x, 0), (current_x, timeline_height - 1), current_frame_color, 3)

            cv2.putText(
                timeline,
                f"appeared: {len(appearance_frames)}",
                (10, timeline_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            composed = np.zeros((output_height, width, 3), dtype=np.uint8)
            composed[:height] = canvas
            composed[height:] = timeline

            cv2.putText(
                composed,
                f"{track['track_id']}  frame {frame_index:05d}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(composed)

        writer.release()
        output_paths.append(track_path)

    return output_paths

def _save_tracks_summary(video_id, serialized_tracks, output_dir):
    summary_path = output_dir / f"{video_id}_tracks.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_id": video_id,
                "tracks": [
                    {
                        "track_id": track["track_id"],
                        "first_frame_index": track["first_frame_index"],
                        "last_frame_index": track["last_frame_index"],
                        "first_frame_name": track["first_frame_name"],
                        "last_frame_name": track["last_frame_name"],
                        "observation_count": len(track["observations"]),
                    }
                    for track in serialized_tracks
                ],
            },
            handle,
            indent=2,
        )

def _track_video(tracker_model, video_data, output_dir, window_size=5, visual_fps=30):
    video_id = video_data["video_id"]
    frames = video_data["frames"]
    
    track_file = output_dir / f"{video_id}_tracks.pkl"
    if os.path.exists(track_file):
        print(f"Tracks for video {video_id} already exist. Skipping tracking.")
        # load the tracks from the existing file
        with open(track_file, "rb") as handle:
            serialized_tracks = pickle.load(handle)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        objs = [frame["objects"] for frame in frames]
        masks = [frame["masks"] for frame in frames]
        indices = [frame["frame_index"] for frame in frames]
        depths = [frame["depth"]["depth"] for frame in frames]
        flows = [frame["flows"] for frame in frames]
        # Track objects across frames using the tracker model
        for start in tqdm(range(0, len(frames), window_size)):
            end = min(start + window_size, len(frames))
            # given window size of frames, 
            # but only the objects in the next frame will be tracked, other frames are used to provide context for the tracker model to track the objects in the next frame., 
            # for each track, at most top_k candidates will be kept for the next frame, and the rest will be discarded.
            tracker_model.track(indices[start:end],objs[start:end], masks[start:end], depths[start:end], flows[start:end])

        serialized_tracks = tracker_model.finalize_tracks()

        # frame_name_by_index = {
        #     frame_record["frame_index"]: frame_record.get("frame_name", frame_record.get("frame_id", f"frame_{frame_record['frame_index']:05d}"))
        #     for frame_record in frames
        # }

        serialized_tracks = _save_tracks(video_id, serialized_tracks, output_dir)

    _visual_tracks(video_id, serialized_tracks, output_dir, frames, visual_fps=visual_fps)

    return serialized_tracks
    
def main(input_data):
    print("\n------- Step 02 -------\n")
    output_dir = input_data["output_dir"]
    device = input_data["device"]
    step01_output_dir = input_data["step01_output_dir"]
    tracker_model = load_tracker_model(input_data["top_k"], input_data["window_size"])
    visual_fps = input_data.get("visual_fps", 30)

    # Load the step 1 output data
    step01_output_data = load_step01_output(step01_output_dir)

    # id connections based on the step 1 output data
    for video_data in step01_output_data:
        _track_video(tracker_model, video_data, output_dir, input_data["window_size"], visual_fps=visual_fps)
    print(f"\n------- Step 02 Finished! -------\n")