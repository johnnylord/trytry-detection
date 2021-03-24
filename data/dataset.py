import os
import os.path as osp

from PIL import Image, ImageFile
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.metric import intersection_over_union as iou
from utils.metric import iou_width_height as iou_wh
from utils.cleanup import non_max_suppression as nms


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
        self.n_anchors = self.anchors.size(0)
        self.n_anchors_per_scale = self.n_anchors // len(scales)
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

        if self.transform is not None:
            augmentations = self.transform(image=img, bboxes=bboxes)
            img = augmentations['image']
            bboxes = augmentations['bboxes']

        # Assign bboxes to their proper grid cell (Might assign to cross scale)
        # Each cell: (prob, x, y, w, h, c)
        targets = [ torch.zeros((self.n_anchors_per_scale, scale, scale, 6))
                    for scale in self.scales ]
        for bbox in bboxes:
            iou_anchors = iou_wh(torch.tensor(bbox[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = bbox

            # At most one anchor will be assigned to bbox in each scale
            # At least one anchor will be assigned to bbox (the most likely one)
            has_anchor_in_scale = [False]*len(self.scales)
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.n_anchors_per_scale
                anchor_on_scale = anchor_idx % self.n_anchors_per_scale

                scale = self.scales[scale_idx]
                i, j = int(scale*y), int(scale*x)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor_in_scale[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1.
                    x_cell, y_cell = (x*scale-j), (y*scale-i) # [0 - 1]
                    width_cell, height_cell = (
                        width*scale,
                        height*scale
                    ) # can be greater than 1 since it's relative to cell
                    coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor_in_scale[scale_idx] = True

                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    # For ignoring prediction (not count for loss)
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return img, tuple(targets)
