import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.evaluation import mean_average_precision

def test_map_all_correct_one_class():
    preds = [
        [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    targets = [
        [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    correct_mAP = 1
    mean_avg_prec = mean_average_precision(preds, targets,
                                        iou_threshold=0.5,
                                        box_format='xywh',
                                        n_classes=1)
    assert abs(mean_avg_prec.item()-correct_mAP) < 1e-6

def test_map_all_correct_batch():
    preds = [
        [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    targets = [
        [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    correct_mAP = 1
    mean_avg_prec = mean_average_precision(preds, targets,
                                        iou_threshold=0.5,
                                        box_format='xywh',
                                        n_classes=1)
    assert abs(mean_avg_prec.item()-correct_mAP) < 1e-6

def test_map_all_wrong_class():
    preds = [
        [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    targets = [
        [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    correct_mAP = 0
    mean_avg_prec = mean_average_precision(preds, targets,
                                        iou_threshold=0.5,
                                        box_format='xywh',
                                        n_classes=1)
    assert abs(mean_avg_prec.item()-correct_mAP) < 1e-6

def test_map_one_inaccurate_box():
    preds = [
        [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    targets = [
        [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
        [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
        [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
    ]
    correct_mAP = 5 / 18
    mean_avg_prec = mean_average_precision(preds, targets,
                                        iou_threshold=0.5,
                                        box_format='xywh',
                                        n_classes=1)
    assert abs(mean_avg_prec.item()-correct_mAP) < 1e-6
