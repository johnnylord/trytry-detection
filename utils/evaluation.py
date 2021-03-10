from collections import Counter

import torch

from .metric import intersection_over_union


def mean_average_precision(
                        pred_boxes,
                        true_boxes,
                        iou_threshold=0.5,
                        box_format='tlbr',
                        n_classes=20):
    """Return mean average precision statisitcs in dictionary

    Arguments:
        pred_boxes (list): [[img_id, class_id, prob_score, x1, y1, x2, y2], ...]

    """
    epsilon = 1e-6
    average_precisions = []
    # Caculate average precision class-by-class
    for c in range(n_classes):
        gts = [ b for b in true_boxes if b[1] == c]
        dets = [ b for b in pred_boxes if b[1] == c ]
        # Ground truth lookup table
        # ==============================================================
        # statistics_boxes[img_id] = torch.tensor([0, 0, .., 0])
        # the number of zeros depends on number of gt boxes in the image
        stat_boxes = Counter([ gt[0] for gt in gts ])
        stat_boxes = dict([
                        (img_id, torch.zeros(count))
                        for img_id, count in stat_boxes.items()
                        ])
        # Placeholder for TP & FP
        TP = torch.zeros(len(dets))
        FP = torch.zeros(len(dets))
        total_true_bboxes = len(gts)
        # Determine whether detection is TP & FP
        dets.sort(key=lambda x: x[2], reverse=True)
        for det_idx, det in enumerate(dets):
            # Copmare with gt boxes in the same image
            gts_img = [ gt for gt in gts if gt[0] == det[0] ]
            num_gts = len(gts_img)
            # Assign most likey gt box paired with current det
            best_iou = 0
            for gt_idx, gt in enumerate(gts_img):
                iou = intersection_over_union(
                        torch.tensor(det[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format
                        )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            # Determine TP | FP
            if (
                best_iou > iou_threshold
                and stat_boxes[det[0]][best_gt_idx] == 0
            ):
                TP[det_idx] = 1
                stat_boxes[det[0]][best_gt_idx] == 1
            else:
                FP[det_idx] = 1

        # Compute average precision
        TP_cumsum = torch.cumsum(TP, dim=0) # e.g. [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        recalls = torch.cat([torch.tensor([0]), recalls])
        precisions = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))
        precisions = torch.cat([torch.tensor([1]), precisions])
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
