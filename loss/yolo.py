import math
import random
import torch
import torch.nn as nn

from utils.metric import intersection_over_union


def bbox_iou(bboxes1, bboxes2, mode='iou'):
    """Compute iou(variant) between two set of bounding boxes

    Arguments:
        bboxes1 (tensor): shape (N, 4)
        bboxes2 (tensor): shape (N, 4)

    Returns:
        tensor of shape (N, 4) containing iou metrics

    NOTE:
        The format of bounding box is (x, y, w, h)
    """
    mode = mode.lower()
    eplison = 1e-9
    # Pred boxes
    bboxes1_x1 = bboxes1[..., 0:1] - bboxes1[..., 2:3] / 2
    bboxes1_y1 = bboxes1[..., 1:2] - bboxes1[..., 3:4] / 2
    bboxes1_x2 = bboxes1[..., 0:1] + bboxes1[..., 2:3] / 2
    bboxes1_y2 = bboxes1[..., 1:2] + bboxes1[..., 3:4] / 2
    bboxes1_w = bboxes1_x2-bboxes1_x1
    bboxes1_h = bboxes1_y2-bboxes1_y1
    bboxes1_area = (bboxes1_w * bboxes1_h) + eplison
    # True boxes
    bboxes2_x1 = bboxes2[..., 0:1] - bboxes2[..., 2:3] / 2
    bboxes2_y1 = bboxes2[..., 1:2] - bboxes2[..., 3:4] / 2
    bboxes2_x2 = bboxes2[..., 0:1] + bboxes2[..., 2:3] / 2
    bboxes2_y2 = bboxes2[..., 1:2] + bboxes2[..., 3:4] / 2
    bboxes2_w = bboxes2_x2-bboxes2_x1
    bboxes2_h = bboxes2_y2-bboxes2_y1
    bboxes2_area = (bboxes2_w * bboxes2_h) + eplison
    # Intersection boxes
    inter_x1 = torch.max(bboxes1_x1, bboxes2_x1)
    inter_y1 = torch.max(bboxes1_y1, bboxes2_y1)
    inter_x2 = torch.min(bboxes1_x2, bboxes2_x2)
    inter_y2 = torch.min(bboxes1_y2, bboxes2_y2)
    inter_w = (inter_x2-inter_x1).clamp(0)
    inter_h = (inter_y2-inter_y1).clamp(0)
    inter_area = (inter_w * inter_h) + eplison
    union = (bboxes1_area + bboxes2_area - inter_area + eplison)
    # Computer IoU
    iou = inter_area / union
    if mode == 'iou':
        return iou
    elif (
        mode == 'giou'
        or mode == 'ciou'
        or mode == 'diou'
    ):
        # Convex diagnal length
        convex_w = torch.max(bboxes1_x2, bboxes2_x2) - torch.min(bboxes1_x1, bboxes2_x1)
        convex_h = torch.max(bboxes1_y2, bboxes2_y2) - torch.min(bboxes1_y1, bboxes2_y1)
        convex_diag = torch.clamp(convex_w, min=0)**2 + torch.clamp(convex_h, min=0)**2
        # Center diagnal length
        center_w = (bboxes2[..., 0:1]-bboxes1[..., 0:1])
        center_h = (bboxes2[..., 1:2]-bboxes1[..., 1:2])
        center_diag = center_w**2 + center_h**2
        # Distance ratio
        u = center_diag / convex_diag
        # Aspect ratio
        v = torch.pow((
                torch.atan(bboxes2_w/bboxes2_h)-
                torch.atan(bboxes1_w/bboxes1_h)
                ), 2) * (4/(math.pi**2))
        with torch.no_grad():
            S = (iou>0.5).float()
            alpha= (S*v)/((1-iou+v)+eplison)
        # Compute final loss
        if mode == 'diou':
            return iou - u
        elif mode == 'ciou':
            return torch.clamp(iou - u - alpha * v, min=-1.0, max=1.0)
        else:
            convex_area = convex_w * convex_h + eplison
            return iou - ((convex_area-union)/convex_area)
    else:
        raise RuntimeError(f"Cannot compute '{mode}' metric")


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = FocalLoss(nn.BCEWithLogitsLoss())
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 0.5
        self.lambda_coord = 0.05
        self.lambda_obj = 1.0
        self.lambda_noobj = 1.0

    def forward(self, preds, target, anchors):
        """Copmute yolo loss

        Arguements:
            preds (tensor): tensor of shape (N, 3, S, S, 5+C)
            target (tensor): tensor of shape (N, 3, S, S, 6)
            anchors (tensor): tensor of shape (3, 2)

        Prediction format:
            (conf, x, y, w, h, [classes:...])

        Target format:
            (conf, x_cell, y_cell, w_cell, h_cell, class_id)
        """
        device = preds.device
        # Normalize anchor size
        scale = preds.size(2)
        # target with -1 is ignored
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        # NO OBJECT LOSS
        # ==========================================
        noobj_loss = self.bce(
                        preds[..., 0:1][noobj_mask],
                        target[..., 0:1][noobj_mask]
                        )

        # OBJECT LOSS
        # ===========================================
        anchors = anchors.reshape(1, 3, 1, 1, 2)                            # (1, 3, 1, 1, 2)
        # Prediction bboxes
        xy_cell = 2.0*self.sigmoid(preds[..., 1:3])-0.5                     # (N, 3, S, S, 2)
        wh_cell = ((2.0*self.sigmoid(preds[..., 3:5]))**2)*anchors          # (N, 3, S, S, 2)
        pred_bboxes = torch.cat([xy_cell, wh_cell], dim=-1)                 # (N, 3, S, S, 4)
        # Groundtruth bboxes
        xy_cell = target[..., 1:3]
        wh_cell = target[..., 3:5]
        true_bboxes = torch.cat([xy_cell, wh_cell], dim=-1)
        # Compute IoU Metric
        ciou = bbox_iou(pred_bboxes[obj_mask],
                        true_bboxes[obj_mask],
                        mode='ciou')
        # Compute Objectness Loss
        obj_loss = self.bce(
                        preds[..., 0:1][obj_mask],
                        target[..., 0:1][obj_mask]*ciou.detach().clamp(0)
                        )
        # BOX COORDINATEDS LOSS
        # ===========================================
        coord_loss = (1.0-ciou).mean()

        # CLASS LOSS
        # ===========================================
        logits = torch.full_like(preds[..., 5:], 0., device=device)
        labels = logits[obj_mask]
        labels[range(len(labels)), target[..., 5][obj_mask].long()] = 1.0
        class_loss = self.bce(
                            preds[..., 5:][obj_mask],
                            labels.long()
                            )
        loss = {
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'coord_loss': coord_loss,
            'class_loss': class_loss,
            'total_loss': (
                self.lambda_coord * coord_loss
                + self.lambda_obj * obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_class * class_loss
            )
        }
        return loss
