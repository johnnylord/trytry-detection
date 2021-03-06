import torch
import numpy as np


def iou_bbox_mask(bboxes, masks):
    """Copmute IoU between bboxes and masks

    Arguments:
        bboxes  (tensor): tensor of shape (N, 4)
        masks   (tensor): tensor of shape (M, H, W)

    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values between
        bbox and mask

    NOTES:
        box is in normalized format (x, y, w, h).
        mask is a boolean map in pixel coordinate (H, W).
    """
    epsilon = 1e-16
    img_width, img_height = masks.size(2), masks.size(1)

    # Convert bboxes to (xmin, ymin, xmax, ymax) format
    bboxes[..., 0] *= img_width     # x
    bboxes[..., 1] *= img_height    # y
    bboxes[..., 2] *= img_width     # w
    bboxes[..., 3] *= img_height    # h
    bboxes[..., 0] -= (bboxes[..., 2]/2)
    bboxes[..., 1] -= (bboxes[..., 3]/2)
    bboxes[..., 2] = bboxes[..., 0]+(bboxes[..., 2])
    bboxes[..., 3] = bboxes[..., 1]+(bboxes[..., 3])

    # Convert masks to bboxes (xmin, ymin, xmax, ymax) format
    mboxes = []
    for mask in masks:
        booleans = (mask > 0.9).long()
        indices = torch.nonzero(booleans)
        if indices.size(0) == 0:
            mboxes.append((0, 0, 0, 0))
            continue
        ymin = indices[..., 0].min().item()
        ymax = indices[..., 0].max().item()
        xmin = indices[..., 1].min().item()
        xmax = indices[..., 1].max().item()
        mboxes.append((xmin, ymin, xmax, ymax))
    mboxes = torch.tensor(mboxes, dtype=torch.float64)

    # Compute iou between bboxes & masks
    return iou(bboxes, mboxes)

def iou_wh(box1, box2):
    """Copmute IoU between two bbox sets with width & height only

    Arguments:
        box1 (tensor): tensor of shape (N, 2)
        box2 (tensor): tensor of shape (M, 2)

    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values
        between two bbox sets.

    NOTES: box format is (w, h)
    """
    N = box1.size(0)
    M = box2.size(0)
    # Computer intersection area
    min_w = torch.min(
            box1[..., 0].unsqueeze(1).expand(N, M), # (N,) -> (N, M)
            box2[..., 0].unsqueeze(0).expand(N, M), # (M,) -> (N, M)
            )
    min_h = torch.min(
            box1[..., 1].unsqueeze(1).expand(N, M), # (N,) -> (N, M)
            box2[..., 1].unsqueeze(0).expand(N, M), # (M,) -> (N, M)
            )
    inter = min_w * min_h # (N, M)
    area1 = box1[..., 0]*box1[..., 1]       # (N,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N, M)
    area2 = box2[..., 0]*box2[..., 1]       # (M,)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N, M)
    iou = inter / (area1+area2-inter)
    return iou

def iou(box1, box2):
    """Compute IoU between two bbox sets

    Arguments:
        box1 (tensor): tensor of shape (N, 4)
        box2 (tensor): tensor of shape (M, 4)

    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values
        between two bbox sets.

    NOTES: box format (x1, y1, x2, y2)
    """
    epsilon = 1e-16
    N = box1.size(0)
    M = box2.size(0)
    # Compute intersection area
    lt = torch.max(
            box1[..., :2].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., :2].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    rb = torch.min(
            box1[..., 2:].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., 2:].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    wh = rb - lt                    # (N, M, 2)
    wh[wh<0] = 0                    # Non-overlapping conditions
    inter = wh[..., 0] * wh[..., 1] # (N, M)
    # Compute respective areas of boxes
    area1 = (box1[..., 2]-box1[..., 0]) * (box1[..., 3]-box1[..., 1]) # (N,)
    area2 = (box2[..., 2]-box2[..., 0]) * (box2[..., 3]-box2[..., 1]) # (M,)
    area1 = area1.unsqueeze(1).expand(N,M) # (N, M)
    area2 = area2.unsqueeze(0).expand(N,M) # (N, M)
    # Compute IoU
    iou = inter / (area1+area2-inter+epsilon)
    return iou.clamp(0)
