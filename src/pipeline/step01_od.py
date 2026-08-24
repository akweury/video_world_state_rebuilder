
import os
from tqdm import tqdm
from typing import Protocol, Sequence,Iterator
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np
from enum import Enum

from src import config




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



class DetectionTier(str, Enum):
    PRIMARY = "primary"
    CANDIDATE = "candidate"

@dataclass(frozen=True)
class ObjectCandidate:
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    tier: DetectionTier



class YoloWorldEvidenceBackend:
    """High-recall YOLO-World extraction without a persistent JPEG frame cache."""

    backend_name = "yolo_world"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        model_name: str,
        classes: Sequence[str],
        primary_confidence: float,
        candidate_confidence: float,
        nms_iou: float,
        inference_size: int,
        device: str = "auto",
        allow_model_download: bool = False,
    ) -> None:
        import torch
        import ultralytics

        candidate_path = Path(model_name).expanduser()
        if candidate_path.is_file():
            self.model_source = str(candidate_path.resolve())
            self.model_name = str(model_name)
            
        elif allow_model_download:
            self.model_source = model_name
            self.model_name = model_name
            self.model_id = f"{model_name}@download-resolved-at-runtime"
        else:
            raise FileNotFoundError(
                f"YOLO model is not local: {candidate_path}; "
                "pass allow_model_download=True to permit runtime resolution"
            )
        if not 0.0 <= candidate_confidence <= primary_confidence <= 1.0:
            raise ValueError("expected 0 <= candidate confidence <= primary confidence <= 1")
        self.classes = tuple(str(value) for value in classes)
        self.primary_confidence = float(primary_confidence)
        self.candidate_confidence = float(candidate_confidence)
        self.nms_iou = float(nms_iou)
        self.inference_size = int(inference_size)
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self._model = None

    def warmup(self) -> None:
        from ultralytics import YOLOWorld

        self._model = YOLOWorld(self.model_source)
        if self.classes:
            self._model.set_classes(list(self.classes))

    def predict_batch(
        self, frames: Sequence[CanonicalFrame]
    ) -> tuple[tuple[ObjectCandidate, ...], ...]:
        if self._model is None:
            self.warmup()
        results = self._model.predict(
            source=[frame.image_bgr for frame in frames],
            conf=self.candidate_confidence,
            iou=self.nms_iou,
            imgsz=self.inference_size,
            device=self.device,
            verbose=False,
            stream=False,
        )
        batch: list[tuple[ObjectCandidate, ...]] = []
        for frame, result in zip(frames, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                batch.append(())
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_indices = boxes.cls.detach().cpu().tolist()
            names = result.names
            candidates: list[ObjectCandidate] = []
            height, width = frame.image_bgr.shape[:2]
            for coordinates, confidence, class_index in zip(
                xyxy, confidences, class_indices
            ):
                x1, y1, x2, y2 = (float(value) for value in coordinates[:4])
                x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
                y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(class_index)
                class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                score = max(0.0, min(1.0, float(confidence)))
                candidates.append(
                    ObjectCandidate(
                        bbox_xyxy=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=score,
                        tier=(
                            DetectionTier.PRIMARY
                            if score >= self.primary_confidence
                            else DetectionTier.CANDIDATE
                        ),
                    )
                )
            candidates.sort(key=lambda item: (-item.confidence, item.class_name, item.bbox_xyxy))
            batch.append(tuple(candidates))
        if len(batch) != len(frames):
            raise RuntimeError("YOLO returned a different number of results than input frames")
        return tuple(batch)

    def teardown(self) -> None:
        self._model = None


class ObjectEvidenceBackend(Protocol):
    backend_name: str
    model_name: str
    model_id: str
    available: bool
    unavailable_reason: str | None

    def warmup(self) -> None: ...

    def predict_batch(
        self, frames: Sequence[CanonicalFrame]
    ) -> tuple[tuple[ObjectCandidate, ...], ...]: ...

    def teardown(self) -> None: ...




def load_od_model(input_data):
    """
    Load the object detection model from the specified path.
    
    Args:
        od_model_path (str): Path to the object detection model file.
    """
    od_model =  YoloWorldEvidenceBackend(
            model_name=input_data["od_model_path"],
            classes=input_data["classes"],
            primary_confidence=input_data["primary_confidence"],
            candidate_confidence=input_data["candidate_confidence"],
            nms_iou=input_data["nms_iou"],
            inference_size=input_data["inference_size"],
            device=input_data.get("device", "cuda:0"),
            allow_model_download=input_data.get("allow_model_download", True),
    )
    return od_model