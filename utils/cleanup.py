import torch
from .metric import intersection_over_union


def non_max_suppression(bboxes, iou_threshold, prob_threshold, classes, box_format='tlbr'):
    """Non-maximum suppression to filter out redundant bounding boxes

    Arguements:
        bboxes (list): list of lists containing all bboxes with each bbox is
            represented as [class_id, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where bbox is determined as redundant
        prob_threshold (float): threhsold where bbox is determined as noise
        classes (list): list of integer representing valid class ids
        box_format (str): format of bounding boxes (xywh or tlbr)

    Returns:
        list: bboxes after performing NMS
    """
    assert type(bboxes) == list
    # Filter bboxes with prob_threshold and valid classes
    bboxes = [ box for box in bboxes if box[1] > prob_threshold ]
    bboxes = [ box for box in bboxes if box[0] in classes ]

    # Split bboxes based on class id
    categories = {}
    for cid in classes:
        groups = [ box for box in bboxes if box[0] == cid ]
        bboxes = [ box for box in bboxes if box[0] != cid ]
        categories[cid] = groups

    # Perform nms category-by-category
    bboxes_after_nms = []
    for cid, groups in categories.items():
        bboxes = sorted(groups, key=lambda x: x[1], reverse=True)
        while bboxes:
            chosen_box = bboxes.pop(0)
            bboxes_after_nms.append(chosen_box)
            # Remove redundant bboxes related to chosen_box
            bboxes = [  box
                        for box in bboxes
                        if intersection_over_union(
                            torch.tensor(chosen_box[2:]),
                            torch.tensor(box[2:]),
                            box_format=box_format,
                            ).item() < iou_threshold
                        ]
    return bboxes_after_nms
