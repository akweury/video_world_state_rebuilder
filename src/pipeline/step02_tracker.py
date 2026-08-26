from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    """A single detection attached to one object track."""

    frame_index: int
    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
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
    last_bbox_xyxy: tuple[float, float, float, float]
    last_mask: np.ndarray | None = None
    cumulative_score: float = 0.0
    missed_count: int = 0
    observations: list[TrackObservation] = field(default_factory=list)
    children: list[TrackHypothesis] = field(default_factory=list)
    is_completed: bool = False

    def add_observation(self, observation: TrackObservation, score_delta: float) -> None:
        self.last_frame_index = observation.frame_index
        self.last_bbox_xyxy = observation.bbox_xyxy
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
            last_bbox_xyxy=self.last_bbox_xyxy,
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

class BeamSearchTracker:
    """
    Beam search tracker for tracking objects across video frames.
    for each frame, it takes the next N frames into account,
    and uses beam search to find the best tracking candidates, where N is the window size.
    
    The beam search works by maintaining a set of the top K tracking candidates at each step, where K is the beam width.
    
    Each detection is one of the tracking targets.
    For each tracking target,
    instead of only use a single detection in each frame,
    we can maintain a tree of the top K candidates for each target
    across the next N frames, 
    and use a scoring function to evaluate 
    the quality of each candidate.
    
    Concretely, the following steps are performed:
    1. For each target at frame t,
    we need to locate the corresponding candidate detections M
    in the frame t+1 in the region of R, where R is the region of interest (ROI)
    around the target's predicted position in frame t+1.
    
    2. For each candidate mask in M,
    we compute a score based on the following factors:
        - The IoU (Intersection over Union) between the candidate mask and the target's predicted mask.
        - The confidence score of the candidate detection.
        - The label similarity 
        - The depth similarity
        - The flow similarity
    the scoring function can be a weighted sum of these factors, 
    where the weights can be tuned based on the specific application.
    
    3. We select the top K candidates based on the computed scores,
    and add them to the beam for the next frame.
    
    
    """
    
    
    def __init__(self,
                 top_k=5,
                 window_size=5,
                 eot_num=3):
        # Initialize the tracker model here
        self.top_k = top_k  # Number of top candidates to keep in the beam search
        self.window_size = window_size
        self.eot_num = eot_num # end of track number, if a track has no corresponding detection for eot_num consecutive frames, it will be considered as completed.
        self.active_tracks: list[TrackHypothesis] = []  # List to hold active tracks
        self.completed_tracks: list[CompletedTrack] = []  # List to hold completed tracks
        self._next_track_index = 1
        self._frame_index_offset = 0
        self._last_input_frame_index: int | None = None
        self._last_effective_frame_index: int | None = None

    def _resolve_frame_index(self, frame_index: int) -> int:
        if self._last_input_frame_index is not None and frame_index <= self._last_input_frame_index:
            self._frame_index_offset = (self._last_effective_frame_index or -1) + 1
        effective_frame_index = self._frame_index_offset + frame_index
        self._last_input_frame_index = frame_index
        self._last_effective_frame_index = effective_frame_index
        return effective_frame_index

    def _frame_detections(self, frame_entry) -> Sequence:
        detections = getattr(frame_entry, "detections", frame_entry)
        return tuple(detections)

    def _make_observation(self, frame_index: int, detection, depth_score: float = 0.0, flow_score: float = 0.0) -> TrackObservation:
        return TrackObservation(
            frame_index=frame_index,
            detection_id=str(getattr(detection, "detection_id", f"frame:{frame_index:06d}")),
            bbox_xyxy=tuple(float(value) for value in getattr(detection, "bbox_xyxy")),
            mask=getattr(detection, "mask", None),
            class_name=getattr(detection, "class_name", None),
            confidence=getattr(detection, "confidence", None),
            depth_score=depth_score,
            flow_score=flow_score,
        )

    def _score_detection(self, track: TrackHypothesis, detection, depth_frame=None, flow_frame=None) -> tuple[float, float, float, float]:
        box_score = _box_iou(track.last_bbox_xyxy, tuple(float(value) for value in getattr(detection, "bbox_xyxy")))
        mask_score = _mask_iou(track.last_mask, getattr(detection, "mask", None))
        confidence = float(getattr(detection, "confidence", 0.0) or 0.0)
        confidence_score = max(0.0, min(1.0, confidence))
        class_name = getattr(detection, "class_name", None)
        label_score = 1.0 if track.observations and track.observations[-1].class_name == class_name else 0.0

        depth_score = 0.0
        if isinstance(depth_frame, np.ndarray) and depth_frame.size > 0:
            finite = depth_frame[np.isfinite(depth_frame)]
            if finite.size > 0:
                depth_score = float(np.clip(1.0 - abs(float(np.mean(finite))) / (abs(float(np.mean(finite))) + 1.0), 0.0, 1.0))

        flow_score = 0.0
        if isinstance(flow_frame, np.ndarray) and flow_frame.ndim >= 3 and flow_frame.shape[-1] >= 2:
            flow_magnitude = np.linalg.norm(flow_frame[..., :2], axis=-1)
            finite = flow_magnitude[np.isfinite(flow_magnitude)]
            if finite.size > 0:
                flow_score = float(np.clip(1.0 / (1.0 + float(np.mean(finite))), 0.0, 1.0))

        score = (
            0.45 * box_score
            + 0.20 * mask_score
            + 0.15 * confidence_score
            + 0.10 * label_score
            + 0.05 * depth_score
            + 0.05 * flow_score
        )
        return float(score), float(depth_score), float(flow_score), float(mask_score)

    def _finalize_active_track(self, track: TrackHypothesis) -> CompletedTrack:
        track.is_completed = True
        return CompletedTrack(
            track_id=track.track_id,
            first_frame_index=track.first_frame_index,
            last_frame_index=track.last_frame_index,
            cumulative_score=float(track.cumulative_score),
            observations=tuple(track.observations),
        )

    def _spawn_track(self, frame_index: int, detection) -> TrackHypothesis:
        observation = self._make_observation(frame_index, detection)
        track = TrackHypothesis(
            track_id=f"track:{self._next_track_index:06d}",
            first_frame_index=frame_index,
            last_frame_index=frame_index,
            last_bbox_xyxy=observation.bbox_xyxy,
            last_mask=observation.mask,
            cumulative_score=float(observation.confidence or 0.0),
            observations=[observation],
        )
        self._next_track_index += 1
        return track

    def _spawn_unassigned_tracks(
        self,
        frame_index: int,
        detections: Sequence,
        assigned_detection_indices: set[int],
    ) -> list[TrackHypothesis]:
        return [
            self._spawn_track(frame_index, detection)
            for detection_index, detection in enumerate(detections)
            if detection_index not in assigned_detection_indices
        ]
    def _miss_all_active_tracks(self):
        still_active: list[TrackHypothesis] = []
        for track in self.active_tracks:
            track.mark_missed()
            if track.missed_count >= self.eot_num:
                self.completed_tracks.append(self._finalize_active_track(track))
            else:
                still_active.append(track)
        self.active_tracks = still_active   
        
        
    def finalize_tracks(self):
        # Implement logic to finalize tracks
        for track in self.active_tracks:
            self.completed_tracks.append(self._finalize_active_track(track))
        self.active_tracks = []
        return tuple(self.completed_tracks)
    
    def track(self, frame_detections, frame_depth, frame_flows):
        """
        Track objects across from current frame t to the next frame t+1 based on
        a series of frames.
        
        1. If any active track does not have a corresponding detection 
        for a eot_num of consecutive frames,
        we mark it as completed and move it to the completed tracks list.
        
        2. If the active tracks are empty,
        we initialize the active tracks with the detections from the first frame.
        Otherwise, we update the active tracks with the new detections 
        from the current frame. 

        3. For each active track in frame t, 
        we predict its position in the frame t+1 using the depth and flow information.
        Then, we find the corresponding detections in the next frame
        that are within a certain region of interest (ROI) around the predicted position.
        We select the top K candidates of the detections in frame t+1.
        For each candidate in the frame t+1, 
        we repeat the process for the next N frames, 
        where N is the window size.
        
        4. Based on the search tree of candidates across the next N frames,
        we select the best candidate for each active track based on a scoring function.
        then we update the active tracks for the next frame with the selected candidates.
        
        We also keep the search tree of each active track across the next N frames,
        so the next frame can use it and getting the first N-1 frames results
        and only need to search the N-th frame.
        
        5. for each frame detections of frame t, 
        if it is not assigned to any active track,
        we initialize a new track for it.
        Args:
            frame_detections (list): A list of detections for each frame.
            frame_depth (list): A list of depth maps for each frame.
            frame_flows (list): A list of optical flow maps for each frame.

        Returns:
            list: A list of tracking evidence or results.
        """

        for local_index, frame_entry in enumerate(frame_detections):
            frame_index = self._resolve_frame_index(int(getattr(frame_entry, "frame_index", local_index)))
            detections = self._frame_detections(frame_entry)
            depth_frame = frame_depth[local_index] if local_index < len(frame_depth) else None
            flow_frame = frame_flows[local_index] if local_index < len(frame_flows) else None

            # If there are no active tracks,
            # we initialize them with the detections from the first frame.
            if not self.active_tracks:
                self.active_tracks.extend(self._spawn_unassigned_tracks(frame_index, detections, set()))
                continue

            # If there are no detections in the current frame, 
            # we mark all active tracks as missed.
            if not detections:
                self._miss_all_active_tracks()
                continue
            
            # Score each detection against each active track and store the results
            scored_pairs: list[tuple[float, int, int, float, float, float]] = []
            score_cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
            for track_index, track in enumerate(self.active_tracks):
                for detection_index, detection in enumerate(detections):
                    score = self._score_detection(track, detection, depth_frame, flow_frame)
                    score_cache[(track_index, detection_index)] = score
                    scored_pairs.append((score[0], track_index, detection_index, score[1], score[2], score[3]))
            # Sort scored pairs by score in descending order, 
            # then by track index and detection index
            scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            
            # Assign detections to tracks based on the scored pairs
            assigned_tracks: set[int] = set()
            assigned_detections: set[int] = set()
            assignments: dict[int, int] = {}
            for score, track_index, detection_index, _, _, _ in scored_pairs:
                if score <= 0.0:
                    continue
                if track_index in assigned_tracks or detection_index in assigned_detections:
                    continue
                assigned_tracks.add(track_index)
                assigned_detections.add(detection_index)
                assignments[track_index] = detection_index

            # Update active tracks based on the assignments
            next_active_tracks: list[TrackHypothesis] = []
            for track_index, track in enumerate(self.active_tracks):
                
                matched_detection_index = assignments.get(track_index)
                # if no detection is assigned to this track, we mark it as missed
                if matched_detection_index is None:
                    track.mark_missed()
                    if track.missed_count >= self.eot_num:
                        self.completed_tracks.append(self._finalize_active_track(track))
                    else:
                        next_active_tracks.append(track)
                    continue

                top_candidates = [
                    (candidate_index, score_cache[(track_index, candidate_index)][0])
                    for candidate_index in range(len(detections))
                ]
                top_candidates.sort(key=lambda item: (-item[1], item[0]))
                top_candidates = top_candidates[: max(1, int(self.top_k))]


                # For each top candidate, we create a new child track hypothesis
                track.children = []
                chosen_child: TrackHypothesis | None = None
                for candidate_index, candidate_score in top_candidates:
                    detection = detections[candidate_index]
                    depth_score = score_cache[(track_index, candidate_index)][1]
                    flow_score = score_cache[(track_index, candidate_index)][2]
                    child = track.branch(track.track_id)
                    child.add_observation(
                        self._make_observation(frame_index, detection, depth_score=depth_score, flow_score=flow_score),
                        candidate_score,
                    )
                    if chosen_child is None or child.cumulative_score > chosen_child.cumulative_score:
                        chosen_child = child

                # Update the active tracks with the chosen child track
                if chosen_child is None:
                    track.mark_missed()
                    if track.missed_count >= self.eot_num:
                        self.completed_tracks.append(self._finalize_active_track(track))
                    else:
                        next_active_tracks.append(track)
                    continue

                next_active_tracks.append(chosen_child)

            next_active_tracks.extend(
                self._spawn_unassigned_tracks(frame_index, detections, assigned_detections)
            )

            self.active_tracks = next_active_tracks

        return tuple(self.completed_tracks)

def load_tracker_model(top_k=5, window_size=5, eot_num=3):
    """
    Load a tracker model from the specified path.
    Returns:
        BeamSearchTracker: An instance of the tracker model.
    """
    return BeamSearchTracker(
        top_k=top_k, 
        window_size=window_size,
        eot_num=eot_num
    )