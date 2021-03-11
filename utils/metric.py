import torch


def iou_width_height(boxes1, boxes2):
    """Compute iou between two bboxes set with width and height only

    Arguments:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes

    Returns:
        tensor: intersection over unoin of the corresponding obxes
    """
    intersection = (
            torch.min(boxes1[..., 0], boxes2[..., 0])*
            torch.min(boxes1[..., 1], boxes2[..., 1])
            )
    union = (
            (boxes1[..., 0]*boxes1[..., 1])+
            (boxes2[..., 0]*boxes2[..., 1])-
            intersection
            )
    return intersection / union


def intersection_over_union(preds, labels, box_format='tlbr'):
    """Compute iou value between two bboxes set

    Arguements:
        preds (tensor): tensor of shape (N, 4)
        labels (tensor): tensor of shape (N, 4)
        box_format (str): format of bounding boxes (xywh or tlbr)

    Returns:
        a tensor of iou values of shape (N, 1)

    NOTES:
        bbox format is (x1, y1, x2, y2)
    """
    if box_format=='tlbr':
        # Pred boxes
        box1_x1 = preds[..., 0:1]
        box1_y1 = preds[..., 1:2]
        box1_x2 = preds[..., 2:3]
        box1_y2 = preds[..., 3:4]
        # Label boxes
        box2_x1 = labels[..., 0:1]
        box2_y1 = labels[..., 1:2]
        box2_x2 = labels[..., 2:3]
        box2_y2 = labels[..., 3:4]

    if box_format=='xywh':
        # Pred boxes
        box1_x1 = preds[..., 0:1] - preds[..., 2:3] / 2
        box1_y1 = preds[..., 1:2] - preds[..., 3:4] / 2
        box1_x2 = preds[..., 0:1] + preds[..., 2:3] / 2
        box1_y2 = preds[..., 1:2] + preds[..., 3:4] / 2
        # Label boxes
        box2_x1 = labels[..., 0:1] - labels[..., 2:3] / 2
        box2_y1 = labels[..., 1:2] - labels[..., 3:4] / 2
        box2_x2 = labels[..., 0:1] + labels[..., 2:3] / 2
        box2_y2 = labels[..., 1:2] + labels[..., 3:4] / 2

    # Intersection boxes
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) is for the edge case when there are no intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    return intersection / (box1_area+box2_area-intersection + 1e-6)
