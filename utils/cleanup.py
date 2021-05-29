import torch

from .convert import xywh2tlbr


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression on bbox set

    Arguments:
        boxes (tensor): tensor of shape (N, 4)
        scores (tensor): tensor of shape (N, 1)
        iou_threshold (float): minimum valid box confidence threshold

    Returns:
        a tuple of tensors (boxes, scores) after doing nms filtering

    NOTES:
        box format (x1, y1, x2, y2)
        score is (objectness*class probability) => P(Class|Obj)
    """
    if boxes.size(0) <= 1:
        return boxes, scores
    # Coordinate of shape (N,)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    # Area of shape (N,)
    areas = (x2-x1)*(y2-y1)

    keep = []
    order = scores.sort(0, descending=True)[1].squeeze(1)
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
        else:
            i = order[0].item()
            keep.append(i)
        # Compute IoU with remaining boxes (N-1,)
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0)*(yy2-yy1).clamp(min=0)
        iou = (inter / (areas[i]+areas[order[1:]]-inter))
        idx = (iou <= iou_threshold).nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        order = order[idx+1]

    return boxes[keep], scores[keep]


def nms_by_class(bboxes, target, iou_threshold):
    if bboxes.size(0) == 0:
        return []
    # Split the fields
    boxes = xywh2tlbr(bboxes[..., :4])
    scores = bboxes[..., 4:5]
    classes = bboxes[..., 5]
    # Filter out target class
    mask = (classes == target)
    if torch.sum(mask) == 0:
        return []
    # Perform nms on objects
    boxes, scores = nms(boxes=boxes[mask],
                        scores=scores[mask],
                        iou_threshold=iou_threshold)
    # Merge fields back
    bboxes = torch.cat([
                boxes,
                scores,
                torch.tensor([[target]]).repeat(boxes.size(0), 1)
                ], dim=1)
    return bboxes.tolist()
