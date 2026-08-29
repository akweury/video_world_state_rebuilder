from __future__ import annotations

from dataclasses import dataclass, field

import cv2

import numpy as np
import torch 
from src.utils.tracker_utils import *


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
        
    def run(self, frames, output_dir):
        self.frames = frames
        # build the DAG by advancing through each frame
        for layer_index in range(len(frames)):
            self.add_layer(layer_index)
            self.advance()

        print(f"Total Root Nodes: {len(self.root_node_ids)}")
        visual_dag(self.layers, output_dir)
        # serialize the tracks from the root nodes, one root node contains at most top_k tracks
        tracks = []
        for root_node_id in self.root_node_ids:
            best_tracks = get_best_tracks(self.layers, root_node_id, self.top_k) 
            track_obj = best_tracks[0][0].mask_id
            print(f"Track object: {track_obj},Candidates:{len(best_tracks)}, Frame Num:{len(best_tracks[0])}")
            visual_tracks(self.frames,best_tracks,root_node_id, output_dir=output_dir, visual_fps=1)
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



def visual_dag(layers, output_dir):
    dag_output_dir = output_dir / "visual_tracks"
    dag_output_dir.mkdir(parents=True, exist_ok=True)

    if not layers:
        return

    column_width = 220
    top_margin = 80
    bottom_margin = 60
    node_radius = 12
    x_positions = [column_width * index + column_width // 2 for index in range(len(layers))]

    layer_counts = [len(layer) for layer in layers]
    max_nodes = max(layer_counts) if layer_counts else 0
    row_spacing = 80
    canvas_height = max(220, top_margin + bottom_margin + max_nodes * row_spacing)
    canvas_width = max(400, column_width * len(layers))
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    # Draw frame columns first so edges and nodes sit on top of them.
    for layer_index, x in enumerate(x_positions):
        cv2.line(canvas, (x, top_margin // 2), (x, canvas_height - bottom_margin // 2), (210, 210, 210), 1)
        cv2.putText(canvas, f"frame {layer_index}", (x - 44, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)

    node_positions = {}
    for layer_index, layer in enumerate(layers):
        x = x_positions[layer_index]
        layer_height = max(1, len(layer))
        total_span = (layer_height - 1) * row_spacing
        start_y = top_margin + (max_nodes * row_spacing - total_span) // 2
        for mask_index, node in enumerate(layer):
            y = start_y + mask_index * row_spacing
            node_positions[node] = (x, y)

    for layer in layers:
        for node in layer:
            start_pos = node_positions[node]
            child_layer_index = node.pos[0] + 1
            if child_layer_index >= len(layers):
                continue
            child_layer = layers[child_layer_index]
            for child_id in node.children:
                if child_id >= len(child_layer):
                    continue
                child_node = child_layer[child_id]
                end_pos = node_positions.get(child_node)
                if end_pos is None:
                    continue
                edge_score = node.iou_with_children.get(child_id)
                cv2.line(canvas, start_pos, end_pos, (90, 120, 255), 2)
                if edge_score is not None:
                    mid_x = (start_pos[0] + end_pos[0]) // 2
                    mid_y = (start_pos[1] + end_pos[1]) // 2
                    label = f"{edge_score:.2f}"
                    text_origin = (mid_x + 6, mid_y - 6)
                    cv2.rectangle(canvas, (text_origin[0] - 2, text_origin[1] - 16), (text_origin[0] + 44, text_origin[1] + 4), (245, 245, 245), -1)
                    cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    for layer_index, layer in enumerate(layers):
        x = x_positions[layer_index]
        for mask_index, node in enumerate(layer):
            y = node_positions[node][1]
            cv2.circle(canvas, (x, y), node_radius, (30, 30, 30), -1)
            cv2.circle(canvas, (x, y), node_radius - 3, (255, 255, 255), -1)
            label = str(node.mask_id)
            cv2.putText(canvas, label, (x + 18, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    output_path = dag_output_dir / "dag_visualization.png"
    cv2.imwrite(str(output_path), canvas)
    print(f"DAG visualization saved to {output_path}")


    
def visual_tracks(frames, track_candidates,node_id, output_dir, visual_fps=1):
    node_id_str = f"frame_{node_id[0]:04d}_mask_{node_id[1]:02d}"
    img_output_dir = output_dir / "visual_tracks" / node_id_str
    single_node_output_dir =output_dir / "visual_tracks" / "single_node_tracks"
    img_output_dir.mkdir(parents=True, exist_ok=True)
    single_node_output_dir.mkdir(parents=True, exist_ok=True)

    frame_dict = {
        frame["frame_id"]: frame for frame in frames
    }
    mask_visual_files = [frame['masks'][0]['visual_path'] for frame in frames]
    for mask_index, mask_visual_file in enumerate(mask_visual_files):
        visual_mask_next_frame = cv2.imread(str(mask_visual_file))
        output_mask_name = single_node_output_dir / f"mask_{mask_index:02d}.png"
        cv2.imwrite(str(output_mask_name), visual_mask_next_frame)
    save_frame_count = 0
    for candidate_index, track_candidate in enumerate(track_candidates):
        if candidate_index > 0:
            continue
        for node in track_candidate:
            mask_id = node.mask_id
            frame_img = frame_dict[node.frame_id]["frame"]
            frame_masks = frame_dict[node.frame_id]["masks"]
            visual_img = np.zeros_like(frame_img)
            visual_img += frame_img
            frame_id = node.pos[0]
            visual_mask_next = min(frame_id, len(mask_visual_files) - 1)
            
            for mask in frame_masks:
                if mask['prompt_detection_id'] == mask_id:
                    mask_tensor = mask['mask']
                    mask_label = mask['label']        
                    visual_img[mask_tensor.squeeze()] = torch.tensor([255, 0, 0]).reshape(1,3)
                    # add text label to the top left corner of the image
                    cv2.putText(visual_img, mask_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    break
            if len(track_candidate) == 1:
                output_name = single_node_output_dir / f"{node_id_str}_{candidate_index:02d}_{node.frame_id}.png"
            else:
                output_name = img_output_dir / f"{node_id_str}_{candidate_index:02d}_{node.frame_id}.png"
            cv2.imwrite(str(output_name), visual_img)
            
            save_frame_count += 1
    print(f"Total frames saved: {save_frame_count}")
    

