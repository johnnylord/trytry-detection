import math
import random
import torch
import torch.nn as nn


def bbox_iou(box1, box2, mode='iou'):
    """Compute iou(variant) between two bbox sets

    Arguments:
        box1 (tensor): shape (N, 4)
        box2 (tensor): shape (N, 4)

    Returns:
        tensor of shape (N, 4) containing iou metrics

    NOTE:
        The format of bounding box is (x_offset, y_offset, w_cell, h_cell)
    """
    mode = mode.lower()
    eplison = 1e-9
    # Pred boxes
    box1_x1 = box1[..., 0:1]-(box1[..., 2:3]/2)
    box1_y1 = box1[..., 1:2]-(box1[..., 3:4]/2)
    box1_x2 = box1[..., 0:1]+(box1[..., 2:3]/2)
    box1_y2 = box1[..., 1:2]+(box1[..., 3:4]/2)
    box1_w = (box1_x2-box1_x1).clamp(0)
    box1_h = (box1_y2-box1_y1).clamp(0)
    box1_area = (box1_w*box1_h) + eplison
    # True boxes
    box2_x1 = box2[..., 0:1]-(box2[..., 2:3]/2)
    box2_y1 = box2[..., 1:2]-(box2[..., 3:4]/2)
    box2_x2 = box2[..., 0:1]+(box2[..., 2:3]/2)
    box2_y2 = box2[..., 1:2]+(box2[..., 3:4]/2)
    box2_w = (box2_x2-box2_x1).clamp(0)
    box2_h = (box2_y2-box2_y1).clamp(0)
    box2_area = (box2_w*box2_h) + eplison
    # Intersection boxes
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_w = (inter_x2-inter_x1).clamp(0)
    inter_h = (inter_y2-inter_y1).clamp(0)
    inter_area = (inter_w*inter_h) + eplison
    union = (box1_area+box2_area-inter_area+eplison)
    # Computer IoU
    iou = inter_area / union

    if mode == 'iou':
        return iou

    if mode == 'giou':
        # Convex diagnal length
        convex_w = torch.max(box1_x2, box2_x2)-torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2)-torch.min(box1_y1, box2_y1)
        convex_area = convex_w*convex_h + eplison
        giou = iou - ((convex_area-union)/convex_area)
        return giou

    raise RuntimeError(f"Cannot compute '{mode}' metric")


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'
        self.gamma = gamma

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(nn.BCEWithLogitsLoss())
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, preds, target, anchors):
        """Copmute yolo loss

        Arguements:
            preds (tensor): tensor of shape (N, 3, S, S, 5+C)
            target (tensor): tensor of shape (N, 3, S, S, 6)
            anchors (tensor): tensor of shape (3, 2)

        Prediction format:
            (x_raw, y_raw, w_raw, h_raw, conf, [classes...])

        Target format:
            (x_offset, y_offset, w_cell, h_cell, conf, class)
        """
        device = preds.device
        # Normalize factor
        scale = preds.size(2)
        # target with -1 is ignored
        obj_mask = target[..., 4] == 1      # (N, 3, S, S)
        noobj_mask = target[..., 4] == 0    # (N, 3, S, S)
        # NO OBJECT LOSS
        # ==========================================
        noobj_loss = self.bce(
                        preds[..., 4:5][noobj_mask],
                        target[..., 4:5][noobj_mask]
                        )
        # Exception Handling
        if torch.sum(obj_mask) == 0:
            loss = {
                'box_loss': 0,
                'obj_loss': 0,
                'class_loss': 0,
                'noobj_loss': self.lambda_noobj * noobj_loss,
                'total_loss': self.lambda_noobj * noobj_loss,
            }
            return loss
        # OBJECT LOSS
        # ===========================================
        anchors = anchors.reshape(1, 3, 1, 1, 2)            # (1, 3, 1, 1, 2)
        # Prediction bboxes
        xy_offset = self.sigmoid(preds[..., 0:2])           # (N, 3, S, S, 2)
        wh_cell = torch.exp(preds[..., 2:4])*anchors        # (N, 3, S, S, 2)
        pred_bboxes = torch.cat([xy_offset, wh_cell], dim=-1) # (N, 3, S, S, 4)
        # Groundtruth bboxes
        xy_offset = target[..., 0:2]
        wh_cell = target[..., 2:4]
        true_bboxes = torch.cat([xy_offset, wh_cell], dim=-1) # (N, 3, S, S, 4)
        # Compute IoU Metric
        iou = bbox_iou(
                pred_bboxes[obj_mask],
                true_bboxes[obj_mask],
                mode='iou'
                )
        # Compute Objectness Loss
        probs = iou.detach()
        probs[probs < 0] = 0.
        obj_loss = self.mse(
                        self.sigmoid(preds[..., 4:5][obj_mask]),
                        target[..., 4:5][obj_mask]*probs
                        )
        # BOX COORDINATEDS LOSS
        # ===========================================
        preds[..., 0:2] = self.sigmoid(preds[..., 0:2])
        target[..., 2:4] = torch.log(1e-16 + target[..., 2:4]/anchors)
        box_loss = self.mse(preds[..., 0:4][obj_mask], target[..., 0:4][obj_mask])
        # box_loss += (1-giou).mean()

        # CLASS LOSS
        # ===========================================
        pred_labels = preds[..., 5:][obj_mask]
        target_labels = target[..., 5][obj_mask].long()
        class_loss = self.entropy(pred_labels, target_labels)
        # Aggregate Loss
        loss = {
            'box_loss': self.lambda_box * box_loss,
            'obj_loss': self.lambda_obj * obj_loss,
            'noobj_loss': self.lambda_noobj * noobj_loss,
            'class_loss': self.lambda_class * class_loss,
            'total_loss': (
                self.lambda_box * box_loss
                + self.lambda_obj * obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_class * class_loss
            )
        }
        return loss
