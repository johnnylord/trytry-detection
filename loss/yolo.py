import random
import torch
import torch.nn as nn

from utils.metric import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_coord = 10
        self.lambda_noobj = 10
        self.lambda_obj = 1

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
        anchors = anchors.reshape(1, 3, 1, 1, 2)        # (1, 3, S, S, 1)
        x_cell = self.sigmoid(preds[..., 1:2])          # (N, 3, S, S, 1)
        y_cell = self.sigmoid(preds[..., 2:3])          # (N, 3, S, S, 1)
        wh_cell = torch.exp(preds[..., 3:5])*anchors    # (N, 3, S, S, 2)
        pred_bboxes = torch.cat([x_cell, y_cell, wh_cell], dim=-1) # (N, 3, S, S, 4)
        ious = intersection_over_union(
                    pred_bboxes[obj_mask],
                    target[..., 1:5][obj_mask]).detach()
        obj_loss = self.bce(
                preds[..., 0:1][obj_mask],
                ious*target[..., 0:1][obj_mask]
                )

        # BOX COORDINATEDS LOSS
        # ===========================================
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        target[..., 3:5] = torch.log(1e-16+target[..., 3:5]/anchors)
        coord_loss = self.mse(preds[..., 1:5][obj_mask], target[..., 1:5][obj_mask])

        # CLASS LOSS
        # ===========================================
        class_loss = self.entropy(
            preds[..., 5:][obj_mask],
            target[..., 5][obj_mask].long()
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
