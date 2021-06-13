import os
import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.optimize import linear_sum_assignment

import torch
from torch.utils.data import Dataset

from utils.metric import iou, iou_wh, iou_bbox_mask
from utils.cleanup import nms


ImageFile.LOAD_TRUNCATED_IMAGES = True

PASCAL_CLASSES = (
    [ "aeroplane", "bicycle", "bird", "boat", "bottle" ] +
    [ "bus", "car", "cat", "chair", "cow" ] +
    [ "diningtable", "dog", "horse", "motorbike", "person" ] +
    [ "pottedplant", "sheep", "sofa", "train", "tvmonitor" ]
    )

COCO_CLASSES = (
    [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane' ] +
    [ 'bus', 'train', 'truck', 'boat', 'traffic light' ] +
    [ 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird' ] +
    [ 'cat', 'dog', 'horse', 'sheep', 'cow' ] +
    [ 'elephant', 'bear', 'zebra', 'giraffe', 'backpack' ] +
    [ 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee' ] +
    [ 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat' ] +
    [ 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle' ] +
    [ 'wine glass', 'cup', 'fork', 'knife', 'spoon' ] +
    [ 'bowl', 'banana', 'apple', 'sandwich', 'orange' ] +
    [ 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut' ] +
    [ 'cake', 'chair', 'couch', 'potted plant', 'bed' ] +
    [ 'dining table', 'toilet', 'tv', 'laptop', 'mouse' ] +
    [ 'remote', 'keyboard', 'cell phone', 'microwave', 'oven' ] +
    [ 'toaster', 'sink', 'refrigerator', 'book', 'clock' ] +
    [ 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]
    )

class YOLODataset:

    def __init__(self, csv_file, img_dir, label_dir, anchors,
                scales=[13, 26, 52],
                n_classes=20,
                transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.scales = scales
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2]) # (3*3, 2)
        self.n_anchors_per_scale = self.anchors.size(0) // len(scales)
        self.n_classes = n_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get metadata
        img_path = osp.join(self.img_dir, self.annotations.iloc[idx, 0])
        label_path = osp.join(self.label_dir, self.annotations.iloc[idx, 1])
        # Load img and bboxes
        img = np.array(Image.open(img_path).convert('RGB'))
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        bboxes = np.roll(bboxes, 4, axis=1).tolist() # (x, y, w, h, class)
        # Apply albumentation transformation
        if self.transform is not None:
            augmentations = self.transform(image=img, bboxes=bboxes)
            img = augmentations['image']
            bboxes = augmentations['bboxes']
        # Groundtruth cells placeholder
        targets = [ torch.zeros((self.n_anchors_per_scale, scale, scale, 6))
                    for scale in self.scales ]
        if len(bboxes) == 0:
            return img, targets
        # Create groundtruth across three scales
        # ====================================================================
        # - At least one cell will be assigned to an object
        # - No more than one cell can be assigned to an object in the same scale
        # - Cells in different scale might get assigned to an objects
        # - Cell Format: (x_cell, y_cell, w_cell, h_cell, conf, class)
        iou_anchors = iou_wh( # (N, 9)
                            torch.tensor(bboxes)[..., 2:4], # (N, 2)
                            self.anchors                    # (9, 2)
                            )
        # Assign cells to each object from the most likely one (highest IoU)
        anchor_orders = iou_anchors.argsort(descending=True, dim=-1)
        for idx, anchor_order in enumerate(anchor_orders):
            x, y, w, h, cls = bboxes[idx]
            # Check Flag
            has_obj_in_scale = [False]*len(self.scales)
            for anchor_idx in anchor_order:
                scale_idx = anchor_idx // self.n_anchors_per_scale
                scale_anchor_idx = anchor_idx % self.n_anchors_per_scale
                scale = self.scales[scale_idx]
                i, j = int(scale*y), int(scale*x)
                cell_has_obj = targets[scale_idx][scale_anchor_idx, i, j, 4]
                if (
                    not cell_has_obj
                    and not has_obj_in_scale[scale_idx]
                ):
                    targets[scale_idx][scale_anchor_idx, i, j, 4] = 1
                    x_offset, y_offset = (x*scale-j), (y*scale-i)
                    w_cell, h_cell = (w*scale), (h*scale)
                    coord = torch.tensor([x_offset, y_offset, w_cell, h_cell])
                    targets[scale_idx][scale_anchor_idx, i, j, :4] = coord
                    targets[scale_idx][scale_anchor_idx, i, j, 5] = int(cls)
                    has_obj_in_scale[scale_idx] = True
                elif (
                    not cell_has_obj
                    and has_obj_in_scale[scale_idx]
                    and iou_anchors[idx, anchor_idx] > self.ignore_iou_thresh
                ):
                    # For ignoring prediction (not count for loss)
                    targets[scale_idx][scale_anchor_idx, i, j, 4] = -1
        return img, targets


class YOLOMaskDataset:

    def __init__(self, csv_file, img_dir, mask_dir,
                anchors, scales=[13, 26, 52],
                n_classes=80,
                transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scales = scales
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2]) # (3*3, 2)
        self.n_anchors_per_scale = self.anchors.size(0) // len(scales)
        self.n_classes = n_classes
        self.transform = transform
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get metadata
        img_path = osp.join(self.img_dir, self.annotations.iloc[idx, 0])
        mask_dir = osp.join(self.mask_dir, self.annotations.iloc[idx, 0])

        # Load img, instance masks, and instance bboxes
        img = np.array(Image.open(img_path).convert('RGB'))

        # Load masks
        mask_files = sorted([
                    osp.join(mask_dir, f)
                    for f in os.listdir(mask_dir)
                    if 'txt' not in f
                    ])
        masks = []
        for mfile in mask_files:
            mask = np.uint8(Image.open(mfile))/255
            masks.append(mask)

        # Load bboxes
        bbox_files = sorted([
                    osp.join(mask_dir, f)
                    for f in os.listdir(mask_dir)
                    if 'txt' in f
                    ])
        bboxes = []
        for bfile in bbox_files:
            bbox = np.loadtxt(fname=bfile, delimiter=" ").tolist()
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        bboxes = np.roll(bboxes, 4, axis=1).tolist() # (x, y, w, h, class)
        bboxes = [ box + [idx] for idx, box in enumerate(bboxes) ] # (x, y, w, h, class, idx)

        # Apply albumentation transformation
        if self.transform is not None:
            augmentations = self.transform(image=img, masks=masks, bboxes=bboxes)
            img = augmentations['image']
            masks = augmentations['masks']
            bboxes = augmentations['bboxes']

        # Create ground truth targets
        # - 7 stands for (x_offset, y_offset, w_cell, h_cell, prob, class, mask)
        targets = [ torch.zeros((self.n_anchors_per_scale, scale, scale, 7))
                    for scale in self.scales ]

        # Filter masks with valid bboxes
        indices = np.array([ int(bbox[-1]) for bbox in bboxes ], dtype=np.uint8)
        bboxes = np.array([ bbox[:5] for bbox in bboxes ])
        if len(indices) > 0:
            masks = np.array(masks)[indices]

        if len(indices) == 0:
            return img, torch.tensor(masks), targets

        # Compute iou between groundtruth bboxes & anchor boxes
        iou_anchors = iou_wh( # (N, 9)
                            torch.tensor(bboxes)[..., 2:4], # (N, 2)
                            self.anchors                    # (9, 2)
                            )
        # Assign groundtruth object to the cell with the highest IoU value
        anchor_orders = iou_anchors.argsort(descending=True, dim=-1)
        for idx, anchor_order in enumerate(anchor_orders):
            x, y, w, h, cls = bboxes[idx]
            # Check Flag
            has_obj_in_scale = [False]*len(self.scales)
            for anchor_idx in anchor_order:
                scale_idx = anchor_idx // self.n_anchors_per_scale
                scale_anchor_idx = anchor_idx % self.n_anchors_per_scale
                scale = self.scales[scale_idx]
                i, j = int(scale*y), int(scale*x)
                cell_has_obj = targets[scale_idx][scale_anchor_idx, i, j, 4]
                if (
                    not cell_has_obj
                    and not has_obj_in_scale[scale_idx]
                ):
                    targets[scale_idx][scale_anchor_idx, i, j, 4] = 1
                    x_offset, y_offset = (x*scale-j), (y*scale-i)
                    w_cell, h_cell = (w*scale), (h*scale)
                    coord = torch.tensor([x_offset, y_offset, w_cell, h_cell])
                    targets[scale_idx][scale_anchor_idx, i, j, :4] = coord
                    targets[scale_idx][scale_anchor_idx, i, j, 5] = int(cls)
                    targets[scale_idx][scale_anchor_idx, i, j, 6] = int(idx)
                    has_obj_in_scale[scale_idx] = True
                elif (
                    not cell_has_obj
                    and has_obj_in_scale[scale_idx]
                    and iou_anchors[idx, anchor_idx] > self.ignore_iou_thresh
                ):
                    # For ignoring prediction (not count for loss)
                    targets[scale_idx][scale_anchor_idx, i, j, 4] = -1

        return img, torch.tensor(masks), targets
