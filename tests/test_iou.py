import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
from utils.metric import intersection_over_union

def test_iou_with_flat_tensor_in_tlbr_format():
    # 'tlbr' version bounding boxes
    # =============================================
    bbox1 = torch.tensor([0, 0, 2, 2])
    bbox2 = torch.tensor([3, 0, 5, 2])
    pred_iou = intersection_over_union(bbox1, bbox2)
    assert tuple(pred_iou.shape) == (1,)
    assert abs(pred_iou.item()-0) < 1e-6

    bbox1 = torch.tensor([0, 0, 2, 2])
    bbox2 = torch.tensor([1, 1, 3, 3])
    pred_iou = intersection_over_union(bbox1, bbox2)
    assert tuple(pred_iou.shape) == (1,)
    assert abs(pred_iou.item()-1/7) < 1e-6

def test_iou_with_flat_tensor_in_xywh_format():
    # 'tlbr' version bounding boxes
    # =============================================
    bbox1 = torch.tensor([1, 1, 2, 2])
    bbox2 = torch.tensor([4, 1, 2, 2])
    pred_iou = intersection_over_union(bbox1, bbox2, box_format='xywh')
    assert tuple(pred_iou.shape) == (1,)
    assert abs(pred_iou.item()-0) < 1e-6

    bbox1 = torch.tensor([1, 1, 2, 2])
    bbox2 = torch.tensor([2, 2, 2, 2])
    pred_iou = intersection_over_union(bbox1, bbox2, box_format='xywh')
    assert tuple(pred_iou.shape) == (1,)
    assert abs(pred_iou.item()-1/7) < 1e-6

def test_iou_with_tensor():
    bboxes1 = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 3, 2],
        ]
    )
    bboxes2 = torch.tensor(
        [
            [3, 0, 5, 2],
            [3, 0, 5, 2],
            [0, 3, 2, 5],
            [2, 0, 5, 2],
            [1, 1, 3, 3],
            [1, 1, 3, 3],
        ]
    )
    pred_ious = intersection_over_union(bboxes1, bboxes2)
    assert tuple(pred_ious.shape) == (len(bboxes1), 1)

    correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])
    true = (torch.all(torch.abs(correct_ious - pred_ious.squeeze(1)) < 1e-6))
    assert true == True
