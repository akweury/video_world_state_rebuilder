import os 
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.step02_tracker import load_tracker_model


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


def _save_tracks(video_id, tracks, output_dir, frame_name_by_index=None):
    track_path = output_dir / f"{video_id}_tracks.pkl"
    serialized_tracks = []
    for track_id, track_candidates in enumerate(tracks):
        serialized_track_candidates = []
        for track in track_candidates:
            track_nodes = []
            for frameMaskNode in track:
                track_nodes.append(
                    {
                        "frame_id": frameMaskNode.frame_id,
                        "mask_id": frameMaskNode.mask_id
                    }
                )
            serialized_track_candidates.append(
                {
                    "track_id": track_id,
                    "track_nodes": track_nodes
                }
            )
        serialized_tracks.append(serialized_track_candidates)
    with track_path.open("wb") as handle:
        pickle.dump(serialized_tracks, handle)
        
    return serialized_tracks


def _mask_from_value(mask_value):
    if mask_value is None:
        return None
    return np.asarray(mask_value, dtype=bool)


def _mask_iou_numpy(left, right) -> float:
    if left is None or right is None:
        return 0.0
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        return 0.0
    union = np.logical_or(left, right)
    union_count = int(union.sum())
    if union_count == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union_count)


def _mask_centroid(mask):
    if mask is None:
        return None
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _overlay_mask(canvas, mask, color, alpha=0.45, thickness=2):
    if mask is None:
        return
    if mask.shape[:2] != canvas.shape[:2]:
        return
    pixels = np.asarray(mask, dtype=bool)
    if not np.any(pixels):
        return
    color_array = np.asarray(color, dtype=np.float32)
    blended = canvas[pixels].astype(np.float32)
    canvas[pixels] = np.clip(blended * (1.0 - alpha) + color_array * alpha, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, tuple(int(value) for value in color), thickness)


def _frame_mask_lookup(frame_record):
    return {
        mask_entry["prompt_detection_id"]: _mask_from_value(mask_entry.get("mask"))
        for mask_entry in frame_record.get("masks", [])
    }


def visual_tracks(video_id, serialized_tracks, output_dir, frames, visual_fps=30):
    visual_dir = output_dir / "tracks_visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    frame_order = [frame for frame in frames if frame.get("frame_id") is not None and frame.get("frame") is not None]
    frame_by_id = {frame["frame_id"]: frame for frame in frame_order}
    frame_by_name = {frame["frame_name"]: frame for frame in frame_order if frame.get("frame_name") is not None}
    if not frame_order:
        return []

    sample_frame_path = Path(frame_order[0].get("frame_path", ""))
    source_frame_dir = sample_frame_path.parent if sample_frame_path.parent.exists() else None
    if source_frame_dir is not None:
        source_frame_paths = sorted(
            path for path in source_frame_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    else:
        source_frame_paths = []

    if not source_frame_paths:
        source_frame_paths = [Path(frame["frame_path"]) for frame in frame_order if frame.get("frame_path") is not None]

    def _frame_sort_key(frame_path):
        stem = Path(frame_path).stem
        if stem.startswith("frame_"):
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return stem
        try:
            return int(stem)
        except ValueError:
            return stem

    source_frame_paths = sorted(source_frame_paths, key=_frame_sort_key)
    if not source_frame_paths:
        return []

    source_frame_names = {path.name: index for index, path in enumerate(source_frame_paths)}
    sample_frame = np.asarray(frame_order[0]["frame"])
    height, width = sample_frame.shape[:2]
    timeline_height = max(54, height // 10)
    footer_row_height = 22
    candidate_palette = [
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
    ]

    max_candidate_count = max((len(track_candidates) for track_candidates in serialized_tracks), default=0)
    footer_height = max(80, 28 + footer_row_height * max(1, max_candidate_count))
    output_height = height + timeline_height + footer_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_paths = []

    for track_id, track_candidates in enumerate(serialized_tracks):
        track_path = visual_dir / f"{video_id}_{track_id}.mp4"
        writer = cv2.VideoWriter(str(track_path), fourcc, float(visual_fps), (width, output_height))
        if not writer.isOpened():
            continue

        candidate_paths = []
        for track_candidate in track_candidates:
            nodes = track_candidate.get("track_nodes", [])
            node_map = {node.get("frame_id"): (index, node) for index, node in enumerate(nodes) if node.get("frame_id") is not None}
            candidate_paths.append({
                "track_id": track_candidate.get("track_id", track_id),
                "nodes": nodes,
                "node_map": node_map,
            })

        candidate_states = [
            {
                "last_mask": None,
                "last_iou": None,
                "last_node_frame_id": None,
            }
            for _ in candidate_paths
        ]

        appearance_frame_id = None
        for candidate_path in candidate_paths:
            if candidate_path["nodes"]:
                appearance_frame_id = candidate_path["nodes"][0].get("frame_id")
                if appearance_frame_id is not None:
                    break

        for frame_path in source_frame_paths:
            frame_name = Path(frame_path).name
            frame_record = frame_by_name.get(frame_name)
            if frame_record is None:
                frame_record = frame_by_id.get(Path(frame_path).stem)
            if frame_record is None:
                canvas = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if canvas is None:
                    continue
                frame_id = Path(frame_path).stem
                frame_index = source_frame_names.get(frame_name, 0)
            else:
                canvas = np.asarray(frame_record["frame"]).copy()
                frame_id = frame_record["frame_id"]
                frame_index = int(frame_record.get("frame_index", source_frame_names.get(frame_name, 0)))

            frame_masks = _frame_mask_lookup(frame_record) if frame_record is not None else {}
            candidate_rows = []

            for candidate_index, candidate_path in enumerate(candidate_paths):
                candidate_color = candidate_palette[candidate_index % len(candidate_palette)]
                candidate_state = candidate_states[candidate_index]
                current_entry = candidate_path["node_map"].get(frame_id)
                if current_entry is not None:
                    current_node_index, current_node = current_entry
                    current_mask = frame_masks.get(current_node.get("mask_id"))
                    if current_mask is not None:
                        if current_node_index > 0:
                            previous_node = candidate_path["nodes"][current_node_index - 1]
                            previous_frame = frame_by_id.get(previous_node.get("frame_id"))
                            previous_mask = None if previous_frame is None else _frame_mask_lookup(previous_frame).get(previous_node.get("mask_id"))
                            candidate_state["last_iou"] = _mask_iou_numpy(previous_mask, current_mask)
                        else:
                            candidate_state["last_iou"] = None
                        candidate_state["last_mask"] = current_mask
                        candidate_state["last_node_frame_id"] = frame_id

                current_mask = candidate_state["last_mask"]
                if current_mask is not None:
                    _overlay_mask(canvas, current_mask, candidate_color, alpha=0.32, thickness=1)
                    centroid = _mask_centroid(current_mask)
                    if centroid is not None:
                        cv2.putText(
                            canvas,
                            str(candidate_index + 1),
                            (centroid[0] + 4, centroid[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                candidate_rows.append(
                    {
                        "candidate_id": candidate_path["track_id"],
                        "iou": candidate_state["last_iou"],
                        "mask": current_mask,
                    }
                )

            timeline = np.full((timeline_height, width, 3), (28, 28, 28), dtype=np.uint8)
            timeline_y = timeline_height // 2
            cv2.line(timeline, (0, timeline_y), (width - 1, timeline_y), (220, 220, 220), 2)

            if appearance_frame_id is not None:
                appearance_frame = frame_by_id.get(appearance_frame_id)
                if appearance_frame is not None:
                    appearance_name = appearance_frame.get("frame_name")
                    appearance_index = source_frame_names.get(appearance_name, int(appearance_frame.get("frame_index", 0)))
                else:
                    appearance_index = 0
                appearance_x = int(round((appearance_index / max(1, len(frame_order) - 1)) * (width - 1)))
                cv2.line(timeline, (appearance_x, 8), (appearance_x, timeline_height - 8), (0, 220, 0), 2)
                cv2.putText(
                    timeline,
                    "appeared",
                    (max(4, appearance_x - 26), 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )

            current_index = source_frame_names.get(frame_name, frame_index)
            current_x = int(round((current_index / max(1, len(source_frame_paths) - 1)) * (width - 1)))
            cv2.line(timeline, (current_x, 0), (current_x, timeline_height - 1), (0, 165, 255), 3)
            cv2.putText(
                timeline,
                f"frame {current_index + 1}/{len(source_frame_paths)}",
                (10, timeline_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            footer = np.full((footer_height, width, 3), (20, 20, 20), dtype=np.uint8)
            cv2.putText(
                footer,
                f"track {track_id}  frame {frame_id}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                footer,
                f"appeared: {appearance_frame_id if appearance_frame_id is not None else 'unknown'}",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            row_y = 64
            if not candidate_rows:
                cv2.putText(
                    footer,
                    "no candidates",
                    (10, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
            else:
                for candidate_index, row in enumerate(candidate_rows):
                    candidate_color = candidate_palette[candidate_index % len(candidate_palette)]
                    cv2.rectangle(footer, (10, row_y - 12), (24, row_y + 2), candidate_color, -1)
                    iou_text = "n/a" if row["iou"] is None else f"{row['iou']:.3f}"
                    cv2.putText(
                        footer,
                        f"{candidate_index + 1}. {row['candidate_id']}   IoU(parent): {iou_text}",
                        (32, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                    row_y += footer_row_height

            composed = np.zeros((output_height, width, 3), dtype=np.uint8)
            composed[:height] = canvas
            composed[height:height + timeline_height] = timeline
            composed[height + timeline_height:] = footer

            writer.write(composed)

        writer.release()
        output_paths.append(track_path)

    return output_paths

def _track_video(tracker_model, video_data, output_dir, window_size=5, visual_fps=30):
    video_id = video_data["video_id"]
    frames = video_data["frames"]
    frame_name_by_index = {
        int(frame["frame_index"]): frame["frame_name"]
        for frame in frames
        if frame.get("frame_index") is not None and frame.get("frame_name") is not None
    }
    
    track_file = output_dir / f"{video_id}_tracks.pkl"
    if os.path.exists(track_file):
        print(f"Tracks for video {video_id} already exist. Skipping tracking.")
        # load the tracks from the existing file
        with open(track_file, "rb") as handle:
            serialized_tracks = pickle.load(handle)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        serialized_tracks = tracker_model.run(frames)
        _save_tracks(video_id, serialized_tracks, output_dir, frame_name_by_index=frame_name_by_index)

    visual_tracks(video_id, serialized_tracks, output_dir, frames, visual_fps=visual_fps)

    return serialized_tracks
    
def main(input_data):
    print("\n------- Step 02 -------\n")
    output_dir = input_data["output_dir"]
    device = input_data["device"]
    step01_output_dir = input_data["step01_output_dir"]
    tracker_model = load_tracker_model(input_data["mask_iou_th"], input_data["top_k"], input_data["window_size"])
    visual_fps = input_data.get("visual_fps", 30)

    # Load the step 1 output data
    step01_output_data = load_step01_output(step01_output_dir)

    # id connections based on the step 1 output data
    for video_data in step01_output_data:
        _track_video(tracker_model, video_data, output_dir, input_data["window_size"], visual_fps=visual_fps)
    print(f"\n------- Step 02 Finished! -------\n")