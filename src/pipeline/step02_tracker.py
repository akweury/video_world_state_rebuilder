from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.utils.tracker_utils import *
from src.utils.tracker_utils import _mask_iou


"""

Beam search tracker for tracking objects across video frames.
for each frame, it takes the next N frames into account,
and uses beam search to find the best tracking candidates, where N is the window size.

The beam search works by maintaining a set of the top K tracking candidates at each step, where K is the beam width.


"""


@dataclass(eq=False)
class FrameMaskNode:
    """One branch in the beam-search tree for a single object track."""
    frame_id: str
    mask_id: str  
    pos: tuple[int, int]
    parents: list[FrameMaskNode] = field(default_factory=list)
    children: list[FrameMaskNode] = field(default_factory=list)
    iou_with_parents: dict[FrameMaskNode, float] = field(default_factory=dict)
    iou_with_children: dict[FrameMaskNode, float] = field(default_factory=dict)

    # def mask_iou(self, other_mask) -> float:
    #     return _mask_iou(self._mask_value(self.mask), self._mask_value(other_mask))
    def get_pos(self):
        return self.pos  
    def get_frame_id(self) -> str:
        return self.frame_id
    def get_mask_id(self) -> str:
        return self.mask_id
    
    def link_child(self, child_id, iou: float) -> None:
        if child_id not in self.children:
            self.children.append(child_id)
        self.iou_with_children[child_id] = float(iou)

    def link_parent(self, parent_id, iou: float) -> None:
        if parent_id not in self.parents:
            self.parents.append(parent_id)
        self.iou_with_parents[parent_id] = float(iou)


class BSL_DAG_Tracker:
    nodes_by_id: dict[str, list]
    layers: list[list[FrameMaskNode]]
    root_node_ids: list
    frontier_layer_index: int
    next_node_id: int
    top_k: int
    eot_num: int

    def __init__(self,mask_iou_th, top_k: int = 5, eot_num: int = 3):
        self.iou_th = mask_iou_th
        self.root_node_ids = []
        self.layers = []
        self.frontier_layer_index = 0
        self.next_node_id = 0
        self.top_k = top_k
        self.eot_num = eot_num
        self.frames = []
    def add_root(self, node_id):
        self.root_node_ids.append(node_id)

    def add_layer(self, layer_index):
        layer_frame = self.frames[layer_index]
        frameMaskNodes = []
        for mask_index, mask in enumerate(layer_frame['masks']):
            node = FrameMaskNode(pos=[layer_index, mask_index], frame_id=layer_frame["frame_id"], mask_id=mask['prompt_detection_id'])
            frameMaskNodes.append(node)
        self.layers.append(frameMaskNodes)

    def advance(self):
        unassigned_frameMaskNode_ids = []
        leaf_node_ids = []
        for mask_index, frameMaskNode in enumerate(self.layers[self.frontier_layer_index]):
            node_id = frameMaskNode.get_pos()
            if self.frontier_layer_index == 0:
                parent_node_ids, parent_ious = [], []
            else:
                parent_node_ids, parent_ious = find_parent_ids(frameMaskNode, self.layers[self.frontier_layer_index - 1], self.iou_th, self.frames)

            if len(parent_node_ids) == 0:
                unassigned_frameMaskNode_ids.append(node_id)
            else:
                leaf_node_ids.append(node_id)
                for parent_node_id, parent_iou in zip(parent_node_ids, parent_ious):
                    self.layers[self.frontier_layer_index - 1][parent_node_id].link_child(mask_index, parent_iou)
                    self.layers[self.frontier_layer_index][mask_index].link_parent(parent_node_id, parent_iou)
        print(f"Found {len(leaf_node_ids)} leaf nodes and {len(unassigned_frameMaskNode_ids)} unassigned nodes")
        # if no parent is found for a mask, set it as a new root node
        for unassigned_node_id in unassigned_frameMaskNode_ids:
            self.add_root(unassigned_node_id)
        self.frontier_layer_index += 1
        
    def run(self, frames):
        self.frames = frames
        # build the DAG by advancing through each frame
        for layer_index in range(len(frames)):
            self.add_layer(layer_index)
            self.advance()

        print(f"Total Root Nodes: {len(self.root_node_ids)}")
        # serialize the tracks from the root nodes, one root node contains at most top_k tracks
        tracks = []
        for root_node_id in self.root_node_ids:
            best_tracks = get_best_tracks(self.layers, root_node_id, self.top_k) 
            tracks.append(best_tracks)
        print(f"Total Tracks: {len(tracks)}")
        return tracks
    

def get_best_tracks(layers, root_node_id, top_k):
    root_node = layers[root_node_id[0]][root_node_id[1]]
    beam: list[tuple[float, tuple[FrameMaskNode, ...]]] = [(0.0, (root_node,))]
    completed: list[tuple[float, tuple[FrameMaskNode, ...]]] = []
    while beam:
        next_beam: list[tuple[float, tuple[FrameMaskNode, ...]]] = []

        for score, path in beam:
            node = path[-1]
            if not node.children:
                completed.append((score, path))
                continue

            for child_id in node.children:
                edge_score = node.iou_with_children.get(child_id)
                if edge_score is None:
                    raise ValueError(f"Edge score not found for child_id {child_id} in node {node}")
                    edge_score = child.iou_with_parents.get(node, 0.0)
                child_node = layers[node.pos[0]+1][child_id]
                next_beam.append((score + float(edge_score), path + (child_node,)))

        if not next_beam:
            break

        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:top_k]

    if not completed:
        completed = beam

    completed.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in completed[:top_k]]



def load_tracker_model(mask_iou_th, top_k=5, window_size=5, eot_num=3):
    return BSL_DAG_Tracker(mask_iou_th, top_k=top_k, eot_num=eot_num)


    
def visual_tracks(video_id, serialized_tracks, output_dir, frames, visual_fps=30):
    raise NotImplementedError("The visual_tracks function is not yet implemented.")

