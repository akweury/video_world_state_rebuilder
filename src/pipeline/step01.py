
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

from src.pipeline.step01_od import DetectionTier, ObjectCandidate, load_od_model
from src.pipeline.step01_mask import load_mask_model
from src.pipeline.step01_depth import load_depth_model
from src.pipeline.step01_flow import load_flow_model
from src.pipeline.step01_tensor import load_packing_model
from src.utils import data_utils
def _video_to_frames(video_path, output_dir):
    """
    Convert a video into frames and save them as images in the output directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
    """
    if os.path.exists(output_dir):
        return

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

def _frames_to_flows(frame_dir, output_dir, flow_model):
    print(f"- frames to flows for video: {Path(frame_dir).name}")
    if os.path.exists(output_dir):
        return

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

def _frames_to_masks(video_id,frame_dir, output_dir, mask_model):
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

    object_file = Path(output_dir).parent / "objects" / f"{video_id}_objects.json"
    if not object_file.exists():
        return

    import json

    with object_file.open("r", encoding="utf-8") as handle:
        object_frames = json.load(handle)

    detections_by_frame = {entry["frame"]: entry.get("objects", []) for entry in object_frames}
    frame_paths = sorted(
        path for path in Path(frame_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not frame_paths:
        return

    class Frame:
        def __init__(self, frame_name, image_bgr, frame_index):
            self.video_id = video_id
            self.frame_index = frame_index
            self.timestamp_s = 0.0
            self.source_frame_index = frame_index
            self.source_timestamp_s = 0.0
            self.image_bgr = image_bgr
            self.frame_name = frame_name

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
            frame_results.append(
                {
                    "prompt_detection_id": output.prompt_detection_id,
                    "confidence": float(output.confidence),
                    "mask_path": str(mask_path.relative_to(Path(output_dir).parent)),
                    "mask_pixels": int(np.count_nonzero(output.mask)),
                }
            )

        results.append({"frame": frame_path.name, "masks": frame_results})

    os.makedirs(Path(mask_file).parent, exist_ok=True)
    with open(mask_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

def _frames_to_dicts(video_id, frame_dir, depth_dir, flow_dir, obj_dir, mask_dir, output_dir, tensor_model):
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
        _video_to_frames(v_path, f_path)
        _frames_to_objects(vid, f_path, obj_dir, od_model)
        _frames_to_masks(vid, f_path, mask_dir, mask_model)
        _frames_to_depths(f_path, d_path, depth_model)
        _frames_to_flows(f_path, flow_path, flow_model)
        _frames_to_dicts(vid, f_path, d_path, flow_path, obj_dir, mask_dir, record_dir, packing_model)


     
    print("\n--------- Step 01 Done ---------------\n")

    