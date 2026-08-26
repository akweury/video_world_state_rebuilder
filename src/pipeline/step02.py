import os 
import json
import pickle
from pathlib import Path

import numpy as np

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
    objects = frame_record.get("objects", [])
    masks = frame_record.get("masks", [])
    detections = []
    for detection_index, obj in enumerate(objects):
        bbox_xyxy = obj.get("bbox_xyxy")
        class_name = obj.get("class_name")
        if bbox_xyxy is None or class_name is None:
            continue
        mask = None
        if detection_index < len(masks):
            mask = _to_numpy(masks[detection_index].get("mask"))
            if mask is not None:
                mask = mask.astype(bool)
        detections.append(
            FrameDetection(
                detection_id=f"frame:{frame_index:06d}:det:{detection_index:04d}",
                bbox_xyxy=tuple(float(value) for value in bbox_xyxy),
                mask=mask,
                class_name=str(class_name),
                confidence=float(obj.get("confidence", 0.0)),
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
                    "bbox_xyxy": [float(value) for value in observation.bbox_xyxy],
                    "mask": _to_numpy(observation.mask).astype(bool) if observation.mask is not None else None,
                    "class_name": observation.class_name,
                    "confidence": observation.confidence,
                }
            )

        serialized_tracks.append(
            {
                "track_id": track.track_id,
                "first_frame_index": track.first_frame_index,
                "last_frame_index": track.last_frame_index,
                "first_frame_name": frame_name_by_index.get(track.first_frame_index),
                "last_frame_name": frame_name_by_index.get(track.last_frame_index),
                "observations": observations,
            }
        )
    with track_path.open("wb") as handle:
        pickle.dump(serialized_tracks, handle)


    return serialized_tracks

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

def _track_video(tracker_model, video_data, output_dir, window_size=5):
    video_id = video_data["video_id"]
    frames = video_data["frames"]
    
    track_path = output_dir / f"{video_id}_tracks.pkl"
    if os.path.exists(track_path):
        print(f"Tracks for video {video_id} already exist. Skipping tracking.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_detections = [_frame_record_to_detections(frame_record, frame_index) for frame_index, frame_record in enumerate(frames)]
    frame_depth = [frame_record.get("depths", []) for frame_record in frames]   
    frame_flows = [frame_record.get("flows", []) for frame_record in frames]
    # Track objects across frames using the tracker model
    for start in range(0, len(frame_detections), window_size):
        end = min(start + window_size, len(frame_detections))
        tracker_model.track(frame_detections[start:end], frame_depth[start:end], frame_flows[start:end])
        
        
    serialized_tracks = tracker_model.finalize_tracks()
    # tracks = track_objects_across_frames(frame_detections, frame_depth, frame_flows)
    # frame_name_by_index = {frame_index: frame_record.get("frame_name", frame_record.get("frame_id", f"frame_{frame_index:05d}")) for frame_index, frame_record in enumerate(frames)}
    # serialized_tracks = _serialize_tracks(tracks, frame_name_by_index, track_path)
    # _save_tracks_summary(video_id, serialized_tracks, output_dir)

    return serialized_tracks
    
def main(input_data):
    print("\n------- Step 02 -------\n")
    output_dir = input_data["output_dir"]
    device = input_data["device"]
    step01_output_dir = input_data["step01_output_dir"]
    tracker_model = load_tracker_model(input_data["top_k"], 
                                       input_data["window_size"])
    # Load the step 1 output data
    step01_output_data = load_step01_output(step01_output_dir)

    # id connections based on the step 1 output data
    for video_data in step01_output_data:
        _track_video(tracker_model, video_data, output_dir, input_data["window_size"])
        
    print(f"\n ------------------ Step 02 Finished! ------------------ \n")