import random
import torch
import torch.nn as nn

from utils.cleanup import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
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
        # target with -1 is ignored
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        # NO OBJECT LOSS
        # ==========================================
        noobj_loss = self.bce(
                preds[..., 0:1][noobj],
                target[..., 0:1][noobj]
                )

        # OBJECT LOSS
        # ===========================================
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        x_cell = self.sigmoid(preds[..., 1:2])
        y_cell = self.sigmoid(preds[..., 2:3])
        wh_cell = torch.exp(preds[..., 3:5])*anchors
        pred_bboxes = torch.cat([x_cell, y_cell, wh_cell], dim=-1)
        ious = intersection_over_union(pred_bboxes[obj],
                                    target[..., 1:5][obj]).detach()
        obj_loss = self.bce(
                preds[..., 0:1][obj],
                ious*target[..., 0:1][obj]
                )

        # BOX COORDINATEDS LOSS
        # ===========================================
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        target[..., 3:5] = torch.log(1e-16+target[..., 3:5]/anchors)
        coord_loss = self.mse(preds[..., 1:5][obj], target[..., 1:5][obj])

        # CLASS LOSS
        # ===========================================
        class_loss = self.entropy(
            preds[..., 5:][obj],
            target[..., 5][obj].long()
            )

        return (
                self.lambda_coord * coord_loss
                + self.lambda_obj * obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_class * class_loss
                )
