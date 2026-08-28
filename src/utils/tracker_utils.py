from __future__ import annotations

import torch


def _mask_iou(left, right) -> float:
    if left is None or right is None:
        return 0.0
    if left.shape != right.shape:
        return 0.0
    union = torch.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(torch.logical_and(left, right).sum() / union)

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


def get_node_mask(node, frames):
    for frame in frames:
        if frame["frame_id"] == node.frame_id:
            for mask in frame["masks"]:
                if mask['prompt_detection_id'] == node.mask_id:
                    return mask['mask']
    return None

def find_parent_ids(child_node, frontier_layer, iou_th, frames):
    parent_node_ids = []
    parent_ious = []
    for node_index, node in enumerate(frontier_layer):
        iou = _mask_iou(get_node_mask(node, frames), get_node_mask(child_node, frames))
        if iou > iou_th:
            parent_node_ids.append(node_index)
            parent_ious.append(iou)
    return parent_node_ids, parent_ious





