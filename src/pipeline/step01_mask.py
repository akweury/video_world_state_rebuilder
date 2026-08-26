import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.pipeline.step01_od import DetectionTier, ObjectCandidate

@dataclass(frozen=True)
class MaskCandidateOutput:
    prompt_detection_id: str
    mask: np.ndarray
    confidence: float

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



class Sam2MaskEvidenceBackend:
    """Generate independent masks from frame-local YOLO box prompts."""

    backend_name = "sam2_frame_local"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        model_name: str = "weights/sam2/sam2_t.pt",
        device: str = "auto",
        prompt_candidates: bool = False,
        allow_model_download: bool = False,
    ) -> None:
        import torch

        model_path = Path(model_name).expanduser()
        if model_path.is_file():
            self.model_source = str(model_path.resolve())
            self.model_name = str(model_name)
        elif allow_model_download:
            self.model_source = model_name
            self.model_name = model_name
            self.model_id = f"{model_name}@download-resolved-at-runtime"
        else:
            raise FileNotFoundError(
                f"SAM 2 model is not local: {model_path}; enable model download explicitly"
            )
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.prompt_candidates = bool(prompt_candidates)
        self._model = None

    def warmup(self) -> None:
        from ultralytics import SAM

        self._model = SAM(self.model_source)

    def predict_frame(
        self,
        frame: CanonicalFrame,
        detections: Sequence[ObjectCandidate],
    ) -> tuple[MaskCandidateOutput, ...]:
        if self._model is None:
            self.warmup()
        height, width = frame.image_bgr.shape[:2]
        outputs: list[MaskCandidateOutput] = []
        seen_masks: set[bytes] = set()
        proposal_results = self._model.predict(
            source=frame.image_bgr,
            device=self.device,
            retina_masks=True,
            conf=0.0,
            verbose=False,
            stream=False,
        )

        def append_results(result_prefix: str, result_list):
            if len(result_list) != 1 or result_list[0].masks is None:
                return False
            result = result_list[0]
            masks = result.masks.data.detach().cpu().numpy()
            scores = (
                result.boxes.conf.detach().cpu().tolist()
                if result.boxes is not None and result.boxes.conf is not None
                else [1.0] * len(masks)
            )
            for index, (mask, score) in enumerate(zip(masks, scores)):
                binary = np.asarray(mask, dtype=bool)
                if binary.shape != (height, width):
                    binary = cv2.resize(
                        binary.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                if not np.any(binary):
                    continue
                mask_key = binary.tobytes()
                if mask_key in seen_masks:
                    continue
                seen_masks.add(mask_key)
                outputs.append(
                    MaskCandidateOutput(
                        prompt_detection_id=f"{result_prefix}:{index}",
                        mask=binary,
                        confidence=max(0.0, min(1.0, float(score))),
                    )
                )
            return True

        append_results("proposal", proposal_results)

        prompts = tuple(
            detection
            for detection in detections
            if self.prompt_candidates or detection.tier == DetectionTier.PRIMARY
        )
        if prompts:
            boxes = [
                [
                    prompt.bbox_xyxy[0],
                    prompt.bbox_xyxy[1],
                    prompt.bbox_xyxy[2],
                    prompt.bbox_xyxy[3],
                ]
                for prompt in prompts
            ]
            prompt_results = self._model.predict(
                source=frame.image_bgr,
                bboxes=boxes,
                device=self.device,
                retina_masks=True,
                conf=0.0,
                verbose=False,
                stream=False,
            )
            append_results("prompt", prompt_results)

        if not outputs:
            outputs.append(
                MaskCandidateOutput(
                    prompt_detection_id="proposal:full_frame",
                    mask=np.ones((height, width), dtype=bool),
                    confidence=1.0,
                )
            )
        return tuple(outputs)

    def teardown(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass




def load_mask_model(input_data):
    """
    Load the mask detection model from the specified path.
    
    Args:
        input_data (dict): Dictionary containing the input data, including the mask model path.
    """
    mask_backend = Sam2MaskEvidenceBackend(
            model_name=input_data.get("mask_model_path", "weights/sam2/sam2_t.pt"),
            device=input_data.get("device", "auto"),
            prompt_candidates=input_data.get("sam_prompt_candidates", False),
            allow_model_download=input_data.get("allow_model_download", True),
    )

    return mask_backend