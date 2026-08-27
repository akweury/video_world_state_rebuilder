from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


from src.utils.tracker_utils import *


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

"""


class BeamSearchTracker:
    
    
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

    # def _frame_masks(self, mask_entry) -> Sequence:
    #     detections = getattr(mask_entry, "detections", mask_entry)
    #     return tuple(detections)

    def _make_observation(
        self,
        frame_index, obj, mask,
        depth_score: float = 0.0,
        flow_score: float = 0.0,
    ) -> TrackObservation:
        return TrackObservation(
            frame_index=frame_index,
            mask=mask, class_name=obj["class_name"],
            confidence=obj["confidence"],
            depth_score=depth_score,
            flow_score=flow_score
        )
    def _score_frame_masks(self, frame_detections, frame_depth, frame_flow):
        """
        return the assignments for the current frame based on the ranked masks.
        The assignments dictionary maps track IDs to a list of top candidate masks for that track.
        """
        assignments = {}
        assigned_detections = set()
        for track in self.active_tracks:
            new_assignment = rank_corresponding_full_frame_mask(track.last_mask, frame_detections, self.top_k)
            if new_assignment and len(new_assignment) > 0:
                assignments[track.track_id] = new_assignment
                assigned_detections.update(index for _, index, _ in new_assignment)
        return assignments, assigned_detections



    def _build_assignments(
        self,
        scored_pairs: list[tuple[float, int, int, float, float, float]],
    ) -> tuple[set[int], set[int], dict[int, int]]:
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

        return assigned_tracks, assigned_detections, assignments

    def _advance_track(self, track, frame_index, frame_detections, assignments):
        # If the track is not assigned to any detection in the current frame, mark it as missed.
        if track.track_id not in assignments:
            track.mark_missed()
            if track.missed_count >= self.eot_num:
                self.completed_tracks.append(self._finalize_active_track(track))
                return None
            return track

        iou_scores = assignments[track.track_id]
        advanced_track = self._choose_best_child(iou_scores, track, frame_index, frame_detections)
        return advanced_track

    def _top_candidate_mask_indices(
        self,
        track: TrackHypothesis,
        object_detection,
        frame_masks,
    ) -> list[tuple[int, float]]:
        ranked_candidates: list[tuple[int, float]] = []
        for candidate_index, candidate in enumerate(frame_masks):
            candidate_mask = candidate.get("mask")
            mask_score = _mask_iou(track.last_mask, candidate_mask)
            confidence_score = max(0.0, min(1.0, float(candidate.get("confidence", 0.0) or 0.0)))
            score = 0.80 * mask_score + 0.20 * confidence_score if track.last_mask is not None else confidence_score
            ranked_candidates.append((candidate_index, float(score)))
        ranked_candidates.sort(key=lambda item: (-item[1], item[0]))
        return ranked_candidates[: max(1, int(self.top_k))]

    def _choose_best_child(self, iou_scores, track, frame_index, frame_detections):
        """
        Choose the best child track hypothesis based on the given IOU scores.
        """
        track.children = []
        chosen_child: TrackHypothesis | None = None

        beam_candidate_ids = tuple(
            frame_detections[candidate_index]["prompt_detection_id"]
            for candidate_index, _ in iou_scores
        )
        narrowed_candidate_masks = tuple(
            frame_detections[candidate_index]["mask"]
            for candidate_index, _ in iou_scores
        )

        for candidate_index, candidate_score in iou_scores:
            candidate_mask = frame_detections[candidate_index]["mask"]
            child = track.branch(track.track_id)
            child.add_observation(
                self._make_observation(
                    frame_index,
                    frame_detections[candidate_index],
                    candidate_mask,
                    beam_candidate_ids=beam_candidate_ids,
                    narrowed_candidate_masks=narrowed_candidate_masks,
                ),
                candidate_score,
            )
            if chosen_child is None or child.cumulative_score > chosen_child.cumulative_score:
                chosen_child = child

        return chosen_child

    def _advance_active_tracks(self, frame_index, frame_detections, assignments):
        next_active_tracks: list[TrackHypothesis] = []
        
        for track in self.active_tracks:
            advanced_track = self._advance_track(track, frame_index, frame_detections, assignments)
            if advanced_track is not None:
                next_active_tracks.append(advanced_track)
        return next_active_tracks


    def _track_frame(self, frame_index: int, frame_objs, frame_masks, frame_depth, frame_flow) -> None:
        
        frame_detections = merge_detections(frame_objs, frame_masks)
        # Score the current frame's detections against active tracks
        assignments, assigned_detections = self._score_frame_masks(frame_detections, frame_depth, frame_flow)

        # Advance active tracks and choose their full-frame mask candidates
        next_active_tracks = self._advance_active_tracks(frame_index, frame_detections, assignments)

        # Spawn new tracks for any unassigned detections
        next_active_tracks.extend(self._spawn_unassigned_tracks(frame_index, frame_objs, frame_masks, assigned_detections))

        # Update the list of active tracks for the next frame
        self.active_tracks = next_active_tracks


    def _score_detection(self, track: TrackHypothesis, detection, depth_frame=None, flow_frame=None) -> tuple[float, float, float, float]:
        mask_score = _mask_iou(track.last_mask, detection["mask"])
        confidence = float(detection["confidence"] or 0.0)
        confidence_score = max(0.0, min(1.0, confidence))
        class_name = detection["class_name"]
        label_score = 1.0 if track.observations and track.observations[-1].class_name == class_name else 0.0

        depth_score = 0.0
        finite = depth_frame[torch.isfinite(depth_frame)]
        if finite.numel() > 0:
            mean_depth = torch.mean(finite).abs()
            depth_score = float(torch.clamp(1.0 - mean_depth / (mean_depth + 1.0), 0.0, 1.0).item())

        # todo:
        flow_score = 0.0
        flow_entry = (flow_frame["incoming"] or flow_frame["outgoing"])[0]
        flow_tensor = flow_entry["flow"]["flow"]
        flow_sample = flow_tensor.reshape(-1)[:5]
        flow_score = float(torch.clamp(1.0 / (1.0 + torch.mean(flow_sample.abs())), 0.0, 1.0).item())

        score = (
            0.60 * mask_score
            + 0.20 * confidence_score
            + 0.20 * label_score
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

    def _spawn_track(self, frame_index: int, detection, mask) -> TrackHypothesis:
        observation = self._make_observation(frame_index, detection, mask)
        track = TrackHypothesis(
            track_id=f"track:{self._next_track_index:06d}",
            first_frame_index=frame_index,
            last_frame_index=frame_index,
            last_mask=observation.mask,
            cumulative_score=float(observation.confidence or 0.0),
            observations=[observation],
        )
        self._next_track_index += 1
        return track

    def _spawn_unassigned_tracks(self,frame_index, frame_objs, frame_masks,assigned_indices):
        spawned_tracks = []
        for detection_index in range(len(frame_objs)):
            if detection_index in assigned_indices:
                continue
            object_detection = frame_objs[detection_index]
            obj_mask =  select_corresponding_full_frame_mask_from_bbox(object_detection, frame_masks)
            spawned_track = self._spawn_track(frame_index, object_detection, obj_mask)
            spawned_tracks.append(spawned_track)
        
        return spawned_tracks


        

    
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
    
    def track(self, range_frame_indices,range_frame_objs, range_frame_masks, range_frame_depths, range_frame_flows):
        for local_index in range(len(range_frame_masks)):
            frame_objs = range_frame_objs[local_index]
            frame_masks = range_frame_masks[local_index]
            frame_index = range_frame_indices[local_index]
            frame_depth = range_frame_depths[local_index]
            frame_flow = range_frame_flows[local_index]
            self._track_frame(frame_index, frame_objs, frame_masks, frame_depth, frame_flow)
        return tuple(self.completed_tracks)



def load_tracker_model(top_k=5, window_size=5, eot_num=3):
    return BeamSearchTracker( top_k=top_k,  window_size=window_size, eot_num=eot_num)