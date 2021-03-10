import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.cleanup import non_max_suppression


def test_nms_with_same_classes():
    bboxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [1, 0.8, 0.5, 0.5, 0.2, 0.4],
        [1, 0.7, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]
    gt_bboxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [1, 0.7, 0.25, 0.35, 0.3, 0.1]
    ]
    new_bboxes = non_max_suppression(bboxes, 0.5, 0.5, classes=[2])
    assert len(new_bboxes) == 0
    new_bboxes = non_max_suppression(bboxes,
                                    iou_threshold=7/20,
                                    prob_threshold=0.2,
                                    classes=[1],
                                    box_format='xywh')
    assert sorted(new_bboxes) == sorted(gt_bboxes)

def test_nms_with_different_classes():
    bboxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [2, 0.9, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]
    gt_bboxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [2, 0.9, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
    ]
    new_bboxes = non_max_suppression(bboxes,
                                    iou_threshold=7/20,
                                    prob_threshold=0.2,
                                    classes=[1, 2],
                                    box_format='xywh')
    assert sorted(new_bboxes) == sorted(gt_bboxes)
