import os
import os.path as osp

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.dataset import YOLODataset
from data.transform import get_yolo_transform
from model.yolov3 import YOLOv3
from loss.yolo import YOLOLoss
from utils.convert import cells_to_bboxes
from utils.cleanup import non_max_suppression
from utils.evaluation import mean_average_precision


class YOLOv3Agent:

    def __init__(self, config):
        self.config = config

        # Train on device
        target_device = config['train']['device']
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.device = target_device
        else:
            self.device = "cpu"

        # Load dataset
        train_transform = get_yolo_transform(config['dataset']['size'], mode='train')
        valid_transform = get_yolo_transform(config['dataset']['size'], mode='test')
        train_dataset = YOLODataset(
                            csv_file=config['dataset']['train']['csv'],
                            img_dir=config['dataset']['train']['img_root'],
                            label_dir=config['dataset']['train']['label_root'],
                            anchors=config['dataset']['anchors'],
                            scales=config['dataset']['scales'],
                            n_classes=config['dataset']['n_classes'],
                            transform=train_transform)
        valid_dataset = YOLODataset(
                            csv_file=config['dataset']['valid']['csv'],
                            img_dir=config['dataset']['valid']['img_root'],
                            label_dir=config['dataset']['valid']['label_root'],
                            anchors=config['dataset']['anchors'],
                            scales=config['dataset']['scales'],
                            n_classes=config['dataset']['n_classes'],
                            transform=valid_transform)
        # DataLoader
        self.train_loader = DataLoader(
                                dataset=train_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                pin_memory=True, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(
                                dataset=valid_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                pin_memory=True, shuffle=False, drop_last=False)
        # Model
        model = YOLOv3(
                    in_channels=config['model']['in_channels'],
                    num_classes=config['model']['num_classes'],
                    )
        self.model = model.to(self.device)
        # Faciliated Anchor boxes with model
        torch_anchors = torch.tensor(config['dataset']['anchors'])
        torch_scales = torch.tensor(config['dataset']['scales'])
        scaled_anchors = (
                torch_anchors * (
                    torch_scales
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, 3, 2)
                    )
                )
        self.scaled_anchors = scaled_anchors.to(self.device)

        # Optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = optim.Adam(
                            params=self.model.parameters(),
                            lr=config['optimizer']['lr'],
                            weight_decay=config['optimizer']['weight_decay']
                            )
        # Loss function
        self.loss_fn = YOLOLoss()

        # Tensorboard
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=config['train']['logdir'])

        # Training State
        self.current_epoch = 0
        self.current_map = 0

    def resume(self):
        pass

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']+1):
            self.current_epoch = epoch
            self._train_one_epoch()

            if self.current_epoch % self.config['train']['valid_cycle'] == 0:
                self._validate()

    def finalize(self):
        """Compute mAP value of object detector"""
        pass
        # self.model.eval()
        # loop = tqdm(self.valid_loader)
        # n_classes = self.config['dataset']['n_classes']
        # conf_threshold = self.config['valid']['conf_threshold']
        # nms_iou_threshold = self.config['valid']['nms_iou_threshold']
        # map_iou_threshold = self.config['valid']['map_iou_threshold']

        # train_idx = 0
        # pred_bboxes = []
        # true_bboxes = []
        # for batch_idx, (imgs, targets) in enumerate(loop):
            # batch_size = imgs.size(0)
            # imgs = imgs.to(self.device)
            # with torch.no_grad():
                # out = self.model(imgs)
            # # Extract predicted bounding boxes (for each scale)
            # batch_bboxes = [[] for _ in range(batch_size)]
            # for i in range(len(out)):
                # scale = out[i].size(2)
                # anchors = self.scaled_anchors[i]
                # scale_bboxes = cells_to_bboxes(preds=out[i],
                                            # anchors=anchors,
                                            # scale=scale)
                # scale_bboxes = torch.tensor(scale_bboxes)
                # for idx, bboxes in enumerate(scale_bboxes):
                    # obj_mask = bboxes[..., 1] > conf_threshold
                    # bboxes = bboxes[obj_mask].tolist()
                    # batch_bboxes[idx] += bboxes

            # # Extract ground truth bounding boxes
            # gt_bboxes = cells_to_bboxes(preds=targets[len(out)-1],
                                        # anchors=anchors,
                                        # scale=scale,
                                        # is_preds=False)
            # for i in range(batch_size):
                # nms_bboxes = non_max_suppression(bboxes=batch_bboxes[i],
                                # iou_threshold=nms_iou_threshold,
                                # prob_threshold=conf_threshold,
                                # classes=list(range(n_classes)),
                                # box_format='xywh')
                # for nms_box in nms_bboxes:
                    # pred_bboxes.append([train_idx] + nms_box)
                # for box in gt_bboxes:
                    # if box[1] > conf_threshold:
                        # true_bboxes.append([train_idx] + box)
                # train_idx += 1

        # # Compute mAP
        # print("MAP:", len(pred_bboxes), len(true_bboxes))
        # mAP = mean_average_precision(pred_boxes=pred_bboxes,
                                    # true_boxes=true_bboxes,
                                    # iou_threshold=map_iou_threshold,
                                    # box_format='xywh',
                                    # n_classes=n_classes)
        # print(f"mAP[{map_iou_threshold:.1f}]: {mAP.item()}")
        # if self.current_map < mAP.item():
            # self.current_map = mAP.item()
            # self._save_checkpoint()

    def _train_one_epoch(self):
        self.model.train()
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=f"Train Epoch {self.current_epoch}/{self.config['train']['n_epochs']}")
        losses = []
        for batch_idx, (imgs, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)

            with torch.cuda.amp.autocast():
                out = self.model(imgs)
                loss = (
                    self.loss_fn(out[0], target_s1, self.scaled_anchors[0])
                    + self.loss_fn(out[1], target_s2, self.scaled_anchors[1])
                    + self.loss_fn(out[2], target_s3, self.scaled_anchors[2])
                    )
            losses.append(loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Upadte progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

        # Logging (epoch)
        epoch_loss = sum(losses) / len(losses)
        self.board.add_scalar('Train Loss', epoch_loss, global_step=self.current_epoch)

    def _validate(self):
        self.model.eval()

        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        correct_class = 0

        sample_idx = 0
        all_pred_bboxes = []
        all_true_bboxes = []

        loop = tqdm(self.valid_loader,
                    leave=True,
                    desc=f"Valid Epoch {self.current_epoch}/{self.config['train']['n_epochs']}")

        for batch_idx, (imgs, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)     # (N, 3, 416, 416)
            labels = [
                targets[0].to(self.device), # (N, 3, 13, 13, 6)
                targets[1].to(self.device), # (N, 3, 26, 26, 6)
                targets[2].to(self.device), # (N, 3, 52, 52, 6)
                ]

            with torch.no_grad():
                out = self.model(imgs)

            batch_size = imgs.size(0)
            batch_bboxes = [ [] for _ in range(batch_size) ]
            for i in range(3):
                # Current processed prediction and target
                pred, target = out[i], labels[i]
                # Extract obj & noobj mask
                obj_mask = target[..., 0] == 1
                noobj_mask = target[..., 0] == 0
                # Count number of true objects and true non-objects
                tot_obj += torch.sum(obj_mask)
                tot_noobj += torch.sum(noobj_mask)
                # Count number of correct classified object
                correct_class += torch.sum(
                    torch.argmax(pred[..., 5:][obj_mask], dim=-1) == target[..., 5][obj_mask]
                    )
                # Count number of correct objectness & non-objectness
                obj_pred = torch.sigmoid(pred[..., 0]) > self.config['valid']['conf_threshold']
                correct_obj += torch.sum(obj_pred[obj_mask] == target[..., 0][obj_mask])
                correct_noobj += torch.sum(obj_pred[noobj_mask] == target[..., 0][noobj_mask])
                # ============================================================================
                curr_scale = pred.size(2) # pred: (N, 3, S, S, 5+classes)
                curr_anchor = self.scaled_anchors[i] # curr_anchor: (3, 2)
                bboxes = cells_to_bboxes(preds=pred,
                                        anchors=curr_anchor,
                                        scale=curr_scale)
                for idx, bbox in enumerate(bboxes):
                    batch_bboxes[idx] += bbox

            true_bboxes = cells_to_bboxes(preds=target,
                                        anchors=curr_anchor,
                                        scale=curr_scale,
                                        is_preds=False)
            for idx in range(batch_size):
                gt_bboxes = true_bboxes[idx]
                pred_bboxes = batch_bboxes[idx]
                nms_bboxes = non_max_suppression(
                                bboxes=pred_bboxes,
                                classes=list(range(self.config['dataset']['n_classes'])),
                                iou_threshold=self.config['valid']['nms_iou_threshold'],
                                prob_threshold=self.config['valid']['conf_threshold'],
                                box_format='xywh',
                                )
                for nms_bbox in nms_bboxes:
                    all_pred_bboxes.append([sample_idx]+nms_bbox)
                for bbox in gt_bboxes:
                    if bbox[0] > self.config['valid']['conf_threshold']:
                        all_true_bboxes.append([sample_idx]+bbox)

                sample_idx += 1

        acc_cls = (correct_class/(tot_obj+1e-6))*100
        acc_obj = (correct_obj/(tot_obj+1e-6))*100
        acc_noobj = (correct_noobj/(tot_noobj+1e-6))*100
        mapval = mean_average_precision(
                    all_pred_bboxes,
                    all_true_bboxes,
                    iou_threshold=self.config['valid']['map_iou_threshold'],
                    n_classes=self.config['dataset']['n_classes'],
                    box_format="xywh",
                    )
        print(acc_cls.item(), acc_obj.item(), acc_noobj.item(), mapval.item())

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.load_state_dict(),
            'optimizer': self.optimizer.load_state_dict(),
        }
