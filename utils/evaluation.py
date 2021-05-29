from collections import Counter

import numpy as np
import torch

from utils.metric import iou as intersection_over_union


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, n_classes=20):
    """Compute average precision related to all classes

    Arguments:
        pred_boxes (tensor): tensor of shape (N, 7)
        true_boxes (tensor): tensor of shape (M, 7)
        iou_threshold (float): true positive criterion threshold
        n_classes (int): number of classes

    Returns:
        a dictionary of key "mAP", "recall", and "precision"

    NOTE: The format of boxes is (idx, x1, y1, x2, y2, conf, class)
    """
    epsilon = 1e-6
    mean_recalls = []
    mean_precisions = []
    average_precisions = [] # Save a list of average precision of each class
    # Caculate average precision class-by-class
    for c in range(n_classes):
        # Filter out boxes of class 'c'
        detections = true_boxes[true_boxes[..., 6] == c]
        ground_truths = pred_boxes[pred_boxes[..., 6] == c]

        # Exception handling
        total_true_bboxes = ground_truths.size(0)
        if total_true_bboxes == 0 or detections.size(0) == 0:
            continue

        # Lookup table
        amount_bboxes = Counter([ gt[0].item() for gt in ground_truths ])
        for sample_idx, count in amount_bboxes.items():
            amount_bboxes[sample_idx] = torch.zeros(count)
        # Placeholder to keep information where a pred box is TP/FP
        TP = torch.zeros(detections.size(0))
        FP = torch.zeros(detections.size(0))

        # Descending detections by confidence score
        order = detections[..., 5].argsort(descending=True)
        detections = detections[order]

        for sample_idx in amount_bboxes.keys():
            offsets = torch.where(detections[..., 0] == sample_idx)[0]
            preds = detections[detections[..., 0] == sample_idx]
            gts = ground_truths[ground_truths[..., 0] == sample_idx]

            # Exception Handling
            if preds.size(0) != 0 and gts.size(0) == 0:
                for offset in offsets:
                    FP[offset] = 1
            elif preds.size(0) == 0 or gts.size(0) == 0:
                continue

            iou_mat = intersection_over_union(preds[:, 1:5], gts[:, 1:5])
            for pred_idx, ious in enumerate(iou_mat):
                best_idx = -1
                best_iou = 0
                for gt_idx, iou in enumerate(ious):
                    if (
                        iou > best_iou
                        and iou > iou_threshold
                        and amount_bboxes[sample_idx][gt_idx] == 0
                    ):
                        best_iou = iou
                        best_idx = gt_idx
                if best_idx != -1:
                    amount_bboxes[sample_idx][best_idx] = 1
                    TP[offsets[pred_idx]] = 1
                else:
                    FP[offsets[pred_idx]] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        mean_recalls.append((torch.sum(recalls)/recalls.size(0)).item())
        mean_precisions.append((torch.sum(precisions)/precisions.size(0)).item())
        average_precisions.append(torch.trapz(precisions, recalls))

    recall = sum(mean_recalls)/len(mean_recalls)
    precision = sum(mean_precisions)/len(mean_precisions)
    mAP = sum(average_precisions) / len(average_precisions)
    return { "mAP": mAP, "recall": recall, "precision": precision }
