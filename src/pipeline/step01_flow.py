
import gc
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch        
import torchvision
from torchvision.models.optical_flow import Raft_Small_Weights


_RAFT_SMALL_RUNTIME = {}

@dataclass(frozen=True)
class CanonicalFrame:
    video_id: str
    frame_index: int
    timestamp_s: float
    source_frame_index: int
    source_timestamp_s: float
    image_bgr: np.ndarray

    @property
    def image_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class DirectionalFlowOutput:
    flow: np.ndarray
    domain_valid: np.ndarray
    consistency_valid: np.ndarray
    fb_error: np.ndarray


@dataclass(frozen=True)
class FlowPairOutput:
    forward: DirectionalFlowOutput
    backward: DirectionalFlowOutput

def _resolve_torch_device(device=None):
    if torch is None:
        raise ModuleNotFoundError("torch is required for optical flow computation")
    if device is None or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_optical_flow(frame1, frame2, device=None):
    """
    Compute dense optical flow between two RGB frames using torchvision's
    pretrained RAFT-Small model.

    Parameters
    ----------
    frame1 : np.ndarray
        First frame, shape (H, W, 3), dtype uint8, RGB.
    frame2 : np.ndarray
        Second frame, same shape/dtype as frame1.
    device : str or torch.device or None
        Target device. Defaults to 'cuda' if available, else 'cpu'.

    Returns
    -------
    flow : np.ndarray
        Shape (H, W, 2), float32.
        flow[y, x, 0] = horizontal displacement (dx),
        flow[y, x, 1] = vertical displacement   (dy).
    """
    import torch.nn.functional as F

    runtime = _get_raft_small_runtime(device=device)
    device = runtime["device"]
    model = runtime["model"]
    transforms = runtime["transforms"]

    def to_tensor(img):
        # (H, W, 3) uint8 -> (1, 3, H, W) uint8
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    t1, t2 = transforms(to_tensor(frame1), to_tensor(frame2))
    t1, t2 = t1.to(device), t2.to(device)

    # RAFT requires spatial dims divisible by 8
    H, W = t1.shape[-2], t1.shape[-1]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        t1 = F.pad(t1, (0, pad_w, 0, pad_h))
        t2 = F.pad(t2, (0, pad_w, 0, pad_h))

    with torch.inference_mode():
        # returns list of iteratively refined predictions; last is finest
        flow_predictions = model(t1, t2)
        flow_tensor = flow_predictions[-1]  # (1, 2, H_pad, W_pad)

    # crop padding back to original size, convert to (H, W, 2) numpy float32
    flow_tensor = flow_tensor[:, :, :H, :W]
    return flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _get_raft_small_runtime(device=None):
    device_obj = _resolve_torch_device(device)
    device_key = str(device_obj)
    cached = _RAFT_SMALL_RUNTIME.get(device_key)
    if cached is not None:
        return cached

    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device_obj).eval()
    transforms = weights.transforms()
    runtime = {
        "device": device_obj,
        "model": model,
        "transforms": transforms,
    }
    _RAFT_SMALL_RUNTIME[device_key] = runtime
    return runtime



def _consistency(
    flow: np.ndarray,
    reverse_flow: np.ndarray,
    threshold_px: float,
) -> DirectionalFlowOutput:
    flow = np.asarray(flow, dtype=np.float32)
    reverse_flow = np.asarray(reverse_flow, dtype=np.float32)
    if flow.shape != reverse_flow.shape or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("forward/backward flow must share HxWx2 shape")
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    domain_valid = (
        np.isfinite(flow).all(axis=2)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    sampled_reverse = cv2.remap(
        reverse_flow,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    error = np.linalg.norm(flow + sampled_reverse, axis=2).astype(np.float32)
    domain_valid &= np.isfinite(sampled_reverse).all(axis=2) & np.isfinite(error)
    error[~domain_valid] = np.nan
    consistency_valid = domain_valid & (error <= float(threshold_px))
    return DirectionalFlowOutput(
        flow=flow,
        domain_valid=domain_valid,
        consistency_valid=consistency_valid,
        fb_error=error,
    )

class RaftFlowEvidenceBackend:
    backend_name = "raft_small_bidirectional"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        device: str = "auto",
        consistency_threshold_px: float = 1.5,
        allow_model_download: bool = False,
    ) -> None:
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.consistency_threshold_px = float(consistency_threshold_px)
        if self.consistency_threshold_px <= 0.0:
            raise ValueError("flow consistency threshold must be positive")
        weights = Raft_Small_Weights.DEFAULT
        weight_name = Path(urlparse(weights.url).path).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
        if not checkpoint.is_file() and not allow_model_download:
            raise FileNotFoundError(
                f"RAFT weights are not cached at {checkpoint}; enable model download explicitly"
            )
        self.model_name = "torchvision/raft_small:Raft_Small_Weights.DEFAULT"
        self.model_id = f"raft_small@{weight_name}"

    def warmup(self) -> None:
        _get_raft_small_runtime(device=self.device)

    def predict_pair(
        self, earlier: CanonicalFrame, later: CanonicalFrame
    ) -> FlowPairOutput:
        earlier_shape = earlier.image_bgr.shape
        later_shape = later.image_bgr.shape
        if earlier_shape != later_shape:
            raise ValueError(
                "RAFT frame pairs must have identical image shapes; "
                f"got {earlier_shape} and {later_shape}"
            )
        height, width = earlier_shape[:2]
        padded_height = height + (8 - height % 8) % 8
        padded_width = width + (8 - width % 8) % 8
        if padded_height < 128 or padded_width < 128:
            raise ValueError(
                "torchvision RAFT requires each padded image dimension to be at "
                f"least 128 pixels; got {height}x{width} before padding"
            )

        forward = compute_optical_flow(
            earlier.image_rgb,
            later.image_rgb,
            device=self.device,
        )
        backward = compute_optical_flow(
            later.image_rgb,
            earlier.image_rgb,
            device=self.device,
        )
        return FlowPairOutput(
            forward=_consistency(forward, backward, self.consistency_threshold_px),
            backward=_consistency(backward, forward, self.consistency_threshold_px),
        )

    def teardown(self) -> None:
        try:
            _RAFT_SMALL_RUNTIME.pop(str(self.device), None)
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

def load_flow_model(input_data):
    """
    Load the flow detection model from the specified path.
    
    Args:
        input_data (dict): Dictionary containing the input data, including the flow model path.
    """
    flow_backend = RaftFlowEvidenceBackend(
            device=input_data.get("device", "auto"),
            consistency_threshold_px=input_data.get("flow_consistency_threshold_px", 1.0),
            allow_model_download=input_data.get("allow_model_download", True),
    )
    return flow_backend


