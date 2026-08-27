from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass(frozen=True)
class TrackObservation:
    """A single detection attached to one object track."""
    frame_index: int
    mask: np.ndarray | None = None
    class_name: str | None = None
    confidence: float | None = None
    depth_score: float | None = None
    flow_score: float | None = None



@dataclass
class TrackHypothesis:
    """One branch in the beam-search tree for a single object track."""

    track_id: str
    first_frame_index: int
    last_frame_index: int
    last_mask: np.ndarray | None = None
    cumulative_score: float = 0.0
    missed_count: int = 0
    observations: list[TrackObservation] = field(default_factory=list)
    children: list[TrackHypothesis] = field(default_factory=list)
    is_completed: bool = False

    def add_observation(self, observation: TrackObservation, score_delta: float) -> None:
        self.last_frame_index = observation.frame_index
        self.last_mask = observation.mask
        self.observations.append(observation)
        self.cumulative_score += float(score_delta)
        self.missed_count = 0

    def mark_missed(self) -> None:
        self.missed_count += 1

    def branch(self, child_track_id: str, score_delta: float = 0.0) -> TrackHypothesis:
        child = TrackHypothesis(
            track_id=child_track_id,
            first_frame_index=self.first_frame_index,
            last_frame_index=self.last_frame_index,
            last_mask=self.last_mask,
            cumulative_score=self.cumulative_score + float(score_delta),
            missed_count=self.missed_count,
            observations=list(self.observations),
        )
        self.children.append(child)
        return child


@dataclass(frozen=True)
class CompletedTrack:
    """Final output for a resolved object track."""

    track_id: str
    first_frame_index: int
    last_frame_index: int
    cumulative_score: float
    observations: tuple[TrackObservation, ...]


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    if left.shape != right.shape:
        return 0.0
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def select_corresponding_full_frame_mask_from_bbox(object_detection, frame_masks):
    obj_bbox_xyxy = object_detection['bbox_xyxy']
    obj_label = object_detection['class_name']

    mask_ious = []
    mask_labels = []
    # firstly rank the iou between the object_detection and each mask in frame_masks
    for mask in frame_masks:
        mask_ious.append(compute_bbox_mask_iou(obj_bbox_xyxy, mask['mask']))
        mask_labels.append(mask['label'])

    # firstly select the mask with same label and highest iou
    best_mask = None
    best_iou = 0.0
    for i, (iou, label) in enumerate(zip(mask_ious, mask_labels)):
        if label == obj_label and iou > best_iou:
            best_iou = iou
            best_mask = frame_masks[i]['mask']

    # if no mask with the same label is found, select the mask with the highest iou regardless of label
    if best_mask is None:
        for i, iou in enumerate(mask_ious):
            if iou > best_iou:
                best_iou = iou
                best_mask = frame_masks[i]['mask']
    return best_mask


def rank_corresponding_full_frame_mask(last_mask, frame_detections, top_k: int):
    """
    calculate the iou between the last mask and each of the next masks, 
    and return them sorted by their iou scores in descending order.
    """

    if last_mask is None:
        return []
    iou_scores = []
    for index, detection in enumerate(frame_detections):
        iou = _mask_iou(last_mask, detection['mask']['mask'])
        if iou <= 0.0:
            continue
        iou_scores.append((iou, index, detection))

    iou_scores.sort(key=lambda item: -item[0])
    return iou_scores[:top_k]


def compute_bbox_mask_iou(bbox: tuple[float, float, float, float], mask: torch.Tensor | None) -> float:
    if mask is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    bbox_mask = torch.zeros_like(mask, dtype=torch.bool)
    bbox_mask[int(y1):int(y2), int(x1):int(x2)] = True
    union = torch.logical_or(bbox_mask, mask).sum().item()
    if union == 0:
        return 0.0
    return float(torch.logical_and(bbox_mask, mask).sum().item())

def merge_detections(frame_objs, frame_masks):
    merged = []
    for obj in frame_objs:
        mask = select_corresponding_full_frame_mask_from_bbox(obj, frame_masks)
        merged.append({'obj': obj,'mask': mask})
    return merged