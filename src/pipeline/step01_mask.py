import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation  # type: ignore[import-not-found]
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

        def append_results(result_prefix: str, result_list) -> None:
            if len(result_list) != 1 or result_list[0].masks is None:
                return
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
            # Prompt masks are computed as proposals, but the public output remains
            # the full-frame mask set from the automatic pass above.
            _ = prompt_results

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


class SegformerSemanticMaskEvidenceBackend:
    """Generate semantic region masks for stuff classes such as sky and road."""

    backend_name = "segformer_semantic"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str = "auto",
        target_labels: Sequence[str] = (),
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
        else:
            raise FileNotFoundError(
                f"Semantic segmentation model is not local: {model_path}; enable model download explicitly"
            )
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.target_labels = tuple(
            str(label).strip().lower()
            for label in target_labels
            if str(label).strip()
        )
        self._processor = None
        self._model = None

    def warmup(self) -> None:
        

        self._processor = AutoImageProcessor.from_pretrained(self.model_source)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_source)
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _normalize_label(label: str) -> str:
        return " ".join(str(label).strip().lower().replace("_", " ").split())

    def predict_frame(
        self,
        frame: CanonicalFrame,
        detections: Sequence[ObjectCandidate],
    ) -> tuple[MaskCandidateOutput, ...]:
        del detections
        if self._model is None or self._processor is None:
            self.warmup()

        import torch
        import torch.nn.functional as F

        height, width = frame.image_bgr.shape[:2]
        inputs = self._processor(images=frame.image_rgb, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        resized_logits = F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        probabilities = resized_logits.softmax(dim=1)[0]
        segmentation = probabilities.argmax(dim=0)
        confidence_map = probabilities.max(dim=0).values

        id2label = getattr(self._model.config, "id2label", {}) or {}
        label_to_ids: dict[str, list[int]] = {}
        for class_id, label in id2label.items():
            label_to_ids.setdefault(self._normalize_label(label), []).append(int(class_id))

        if self.target_labels:
            target_labels = self.target_labels
        else:
            target_labels = tuple(
                label for label in label_to_ids if label not in {"background", "unlabeled", "other"}
            )

        outputs_list: list[MaskCandidateOutput] = []
        seen_masks: set[bytes] = set()
        for target_label in target_labels:
            matching_ids = [
                class_id
                for label_name, class_ids in label_to_ids.items()
                if label_name == target_label or target_label in label_name or label_name in target_label
                for class_id in class_ids
            ]
            if not matching_ids:
                continue
            mask = torch.zeros_like(segmentation, dtype=torch.bool)
            for class_id in matching_ids:
                mask |= segmentation == class_id
            if not torch.any(mask):
                continue
            mask_cpu = mask.detach().cpu().numpy().astype(bool)
            mask_key = mask_cpu.tobytes()
            if mask_key in seen_masks:
                continue
            seen_masks.add(mask_key)
            confidence = float(confidence_map[mask].mean().item()) if torch.any(mask) else 0.0
            outputs_list.append(
                MaskCandidateOutput(
                    prompt_detection_id=f"semseg:{target_label}",
                    mask=mask_cpu,
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )

        return tuple(outputs_list)

    def teardown(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class CombinedMaskEvidenceBackend:
    """Combine multiple mask backends and deduplicate identical masks."""

    def __init__(self, backends: Sequence[object]) -> None:
        self.backends = tuple(backend for backend in backends if backend is not None)
        self.backend_name = "+".join(
            str(getattr(backend, "backend_name", backend.__class__.__name__))
            for backend in self.backends
        )
        self.available = all(bool(getattr(backend, "available", True)) for backend in self.backends)
        self.unavailable_reason = None

    def warmup(self) -> None:
        for backend in self.backends:
            backend.warmup()

    def predict_frame(
        self,
        frame: CanonicalFrame,
        detections: Sequence[ObjectCandidate],
    ) -> tuple[MaskCandidateOutput, ...]:
        outputs: list[MaskCandidateOutput] = []
        seen_masks: set[bytes] = set()
        for backend in self.backends:
            for output in backend.predict_frame(frame, detections):
                mask = np.asarray(output.mask, dtype=bool)
                mask_key = mask.tobytes()
                if mask_key in seen_masks:
                    continue
                seen_masks.add(mask_key)
                outputs.append(output)
        return tuple(outputs)

    def teardown(self) -> None:
        for backend in self.backends:
            backend.teardown()


def load_mask_model(input_data):
    """
    Load the mask detection model from the specified path.

    Args:
        input_data (dict): Dictionary containing the input data, including the mask model path.
    """
    sam2_backend = Sam2MaskEvidenceBackend(
        model_name=input_data.get("mask_model_path", "weights/sam2/sam2_t.pt"),
        device=input_data.get("device", "auto"),
        prompt_candidates=input_data.get("sam_prompt_candidates", False),
        allow_model_download=input_data.get("allow_model_download", True),
    )
    if input_data.get("semseg_enabled", False):
        semseg_backend = SegformerSemanticMaskEvidenceBackend(
            model_name=input_data.get(
                "semseg_model_path",
                "nvidia/segformer-b5-finetuned-ade-640-640",
            ),
            device=input_data.get("device", "auto"),
            target_labels=input_data.get("semseg_target_labels", ()),
            allow_model_download=input_data.get("allow_model_download", True),
        )
        return CombinedMaskEvidenceBackend((semseg_backend, sam2_backend))

    return sam2_backend