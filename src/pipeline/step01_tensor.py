
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils import data_utils


class TensorPackingModel:
    """Pack the already-generated step 01 artifacts into frame-aligned tensors."""

    backend_name = "tensor_packer"
    available = True
    unavailable_reason = None

    def __init__(self, device: str = "auto") -> None:
        if device is None or str(device).strip().lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested_device = torch.device(str(device))
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = requested_device

    def warmup(self) -> None:
        return

    @staticmethod
    def _to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: TensorPackingModel._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [TensorPackingModel._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(TensorPackingModel._to_cpu(item) for item in value)
        return value

    @staticmethod
    def _frame_paths(frame_dir: str | Path) -> list[Path]:
        frame_dir = Path(frame_dir)
        return sorted(
            path for path in frame_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

    @staticmethod
    def _frame_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).to(device=device, dtype=torch.float32)
        return tensor.permute(2, 0, 1).div_(255.0)

    def _load_mask(self, mask_path: str | Path) -> torch.Tensor | None:
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            return None
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_image > 0)).to(device=self.device)
        return mask_tensor.bool()

    def _npz_to_tensors(self, npz_data: dict | None) -> dict | None:
        if npz_data is None:
            return None
        packed = {}
        for key, value in npz_data.items():
            if isinstance(value, np.ndarray):
                packed[key] = torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            else:
                packed[key] = value
        return packed

    def prepare_video(
        self,
        video_id: str,
        frame_dir: str | Path,
        depth_dir: str | Path,
        flow_dir: str | Path,
        obj_dir: str | Path,
        mask_dir: str | Path,
        output_dir: str | Path,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dict_path = output_dir / f"{video_id}_step01.pkl"
        frame_paths = self._frame_paths(frame_dir)
        object_file = Path(obj_dir) / f"{video_id}_objects.json"
        mask_file = Path(mask_dir) / f"{video_id}_masks.json"
        flow_file = Path(flow_dir) / "flows.json"

        object_frames = data_utils.load_json_list(object_file)
        mask_frames = data_utils.load_json_list(mask_file)
        flow_frames = data_utils.load_json_list(flow_file)

        objects_by_frame = {
            entry.get("frame"): entry.get("objects", [])
            for entry in object_frames
            if entry.get("frame")
        }
        masks_by_frame = {
            entry.get("frame"): entry.get("masks", [])
            for entry in mask_frames
            if entry.get("frame")
        }

        return {
            "dict_path": dict_path,
            "frame_paths": frame_paths,
            "depth_dir": Path(depth_dir),
            "flow_dir": Path(flow_dir),
            "obj_dir": Path(obj_dir),
            "mask_dir": Path(mask_dir),
            "objects_by_frame": objects_by_frame,
            "masks_by_frame": masks_by_frame,
            "flow_frames": flow_frames,
            "already_packed": dict_path.exists(),
        }

    def pack_frame(
        self,
        *,
        frame_index: int,
        frame_path: str | Path,
        depth_dir: str | Path,
        flow_dir: str | Path,
        obj_dir: str | Path,
        mask_dir: str | Path,
        output_dir: str | Path,
        objects_by_frame: dict[str, list],
        masks_by_frame: dict[str, list],
        flow_frames: list[dict],
    ) -> dict | None:
        frame_path = Path(frame_path)
        frame_bgr = data_utils.load_image_bgr(frame_path)
        if frame_bgr is None:
            return None

        depth_path = Path(depth_dir) / f"{frame_path.stem}_depth.npz"
        depth_data = self._npz_to_tensors(data_utils.load_npz_dict(depth_path))

        objects = objects_by_frame.get(frame_path.name, [])
        mask_entries = masks_by_frame.get(frame_path.name, [])

        incoming_flows = []
        outgoing_flows = []
        for entry in flow_frames:
            source_frame = entry.get("source_frame")
            target_frame = entry.get("target_frame")
            if source_frame == frame_path.name:
                forward_path = Path(flow_dir) / entry.get("forward_path", "")
                outgoing_flows.append(
                    {
                        "source_frame": source_frame,
                        "target_frame": target_frame,
                        "direction": "forward",
                        "flow_path": entry.get("forward_path"),
                        "flow": self._npz_to_tensors(data_utils.load_npz_dict(forward_path)),
                    }
                )
            if target_frame == frame_path.name:
                backward_path = Path(flow_dir) / entry.get("backward_path", "")
                incoming_flows.append(
                    {
                        "source_frame": source_frame,
                        "target_frame": target_frame,
                        "direction": "backward",
                        "flow_path": entry.get("backward_path"),
                        "flow": self._npz_to_tensors(data_utils.load_npz_dict(backward_path)),
                    }
                )

        frame_masks = []
        for mask_entry in mask_entries:
            mask_path = Path(mask_dir).parent / mask_entry.get("mask_path", "")
            frame_masks.append(
                {
                    **mask_entry,
                    "mask": self._load_mask(mask_path),
                }
            )

        return {
            "frame_id": frame_path.stem,
            "frame_index": frame_index,
            "frame_name": frame_path.name,
            "frame_path": str(frame_path),
            "frame": frame_bgr,
            "frame_tensor": self._frame_tensor(frame_bgr, self.device),
            "depth": depth_data,
            "depth_tensor": None if depth_data is None else depth_data.get("depth"),
            "flows": {
                "incoming": incoming_flows,
                "outgoing": outgoing_flows,
            },
            "objects": objects,
            "masks": frame_masks,
        }

    def finalize_video(self, dict_path: str | Path, frame_records: list[dict]) -> list[dict]:
        dict_path = Path(dict_path)
        cpu_records = self._to_cpu(frame_records)
        with dict_path.open("wb") as handle:
            pickle.dump(cpu_records, handle)
        return cpu_records


def load_packing_model(input_data):
    """
    Load the tensor model based on the input data.
    """
    return TensorPackingModel(device=input_data.get("device", "auto"))