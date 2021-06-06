import os
import os.path as osp

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.dataset import YOLOMaskDataset
from data.transform import get_yolo_transform
from model.yolov3 import Maskv3
from loss.yolo import YOLOMaskLoss
from utils.cleanup import nms_by_class
from utils.convert import (
        cells_to_boxes,
        xywh2tlbr,
        tlbr2xywh,
        )
from utils.evaluation import mean_average_precision


class Maskv3Agent:

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
        train_dataset = YOLOMaskDataset(
                            csv_file=config['dataset']['train']['csv'],
                            img_dir=config['dataset']['train']['img_root'],
                            mask_dir=config['dataset']['train']['mask_root'],
                            label_dir=config['dataset']['train']['label_root'],
                            anchors=config['dataset']['anchors'],
                            scales=config['dataset']['scales'],
                            n_classes=config['dataset']['n_classes'],
                            transform=train_transform)
        valid_dataset = YOLOMaskDataset(
                            csv_file=config['dataset']['valid']['csv'],
                            img_dir=config['dataset']['valid']['img_root'],
                            mask_dir=config['dataset']['valid']['mask_root'],
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
        model = Maskv3(
                    # Detection Branch
                    in_channels=config['model']['in_channels'],
                    num_classes=config['model']['num_classes'],
                    # Prototype Branch
                    num_masks=config['model']['num_masks'],
                    num_features=config['model']['num_features'],
                    )
        self.model = model.to(self.device)
        # Faciliated Anchor boxes with model
        torch_anchors = torch.tensor(config['dataset']['anchors']) # (3, 3, 2)
        torch_scales = torch.tensor(config['dataset']['scales']) # (3,)
        scaled_anchors = (  # (3, 3, 2)
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
                            weight_decay=config['optimizer']['weight_decay'],
                            )
        # Scheduler
        self.scheduler = OneCycleLR(self.optimizer,
                                    max_lr=config['optimizer']['lr'],
                                    epochs=config['train']['n_epochs'],
                                    steps_per_epoch=len(self.train_loader),
                                    )
        # Loss function
        self.loss_fn = YOLOMaskLoss(num_classes=config['model']['num_classes'],
                                    num_masks=config['model']['num_masks'])

        # Tensorboard
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=config['train']['logdir'])

        # Training State
        self.current_epoch = 0
        self.current_map = 0

    def resume(self):
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.current_map = checkpoint['current_map']
        self.current_epoch = checkpoint['current_epoch']
        print("Restore checkpoint at '{}'".format(self.current_epoch))

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']+1):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._validate()
            accs = self._check_accuracy()

            if (
                self.current_epoch >= self.config['valid']['when']
                and self.current_epoch % 5 == 0
            ):
                mAP50 = self._check_map()
                if mAP50 > self.current_map:
                    self.current_map = mAP50
                    self._save_checkpoint()

    def finalize(self):
        self._check_map()

    def _train_one_epoch(self):
        n_epochs = self.config['train']['n_epochs']
        current_epoch = self.current_epoch
        current_lr = self.optimizer.param_groups[0]['lr']
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=(
                        f"Train Epoch:{current_epoch}/{n_epochs}"
                        f", LR: {current_lr:.5f}"
                        )
                    )
        obj_losses = []
        box_losses = []
        noobj_losses = []
        class_losses = []
        total_losses = []
        segment_losses = []
        self.model.train()
        for batch_idx, (imgs, masks, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            masks = [ m.to(self.device) for m in masks ]# (nM_g, H, W)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            # Model prediction
            with torch.cuda.amp.autocast():
                outs, prototypes = self.model(imgs)
                s1_loss = self.loss_fn(
                                    outs[0], target_s1, self.scaled_anchors[0], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
                s2_loss = self.loss_fn(
                                    outs[1], target_s2, self.scaled_anchors[1], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
                s3_loss = self.loss_fn(
                                    outs[2], target_s3, self.scaled_anchors[2], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
            # Aggregate loss
            obj_loss = s1_loss['obj_loss'] + s2_loss['obj_loss'] + s3_loss['obj_loss']
            box_loss = s1_loss['box_loss'] + s2_loss['box_loss'] + s3_loss['box_loss']
            noobj_loss = s1_loss['noobj_loss'] + s2_loss['noobj_loss'] + s3_loss['noobj_loss']
            class_loss = s1_loss['class_loss'] + s2_loss['class_loss'] + s3_loss['class_loss']
            segment_loss = s1_loss['segment_loss'] + s2_loss['segment_loss'] + s3_loss['segment_loss']
            total_loss = s1_loss['total_loss'] + s2_loss['total_loss'] + s3_loss['total_loss']
            # Moving average loss
            total_losses.append(total_loss.item())
            obj_losses.append(obj_loss.item())
            noobj_losses.append(noobj_loss.item())
            box_losses.append(box_loss.item())
            class_losses.append(class_loss.item())
            segment_losses.append(segment_loss.item())
            # Update Parameters
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            # Upadte progress bar
            mean_total_loss = sum(total_losses)/len(total_losses)
            mean_obj_loss = sum(obj_losses)/len(obj_losses)
            mean_noobj_loss = sum(noobj_losses)/len(noobj_losses)
            mean_box_loss = sum(box_losses)/len(box_losses)
            mean_class_loss = sum(class_losses)/len(class_losses)
            mean_segment_loss = sum(segment_losses)/len(segment_losses)
            loop.set_postfix(
                            loss=mean_total_loss,
                            cls=mean_class_loss,
                            box=mean_box_loss,
                            obj=mean_obj_loss,
                            noobj=mean_noobj_loss,
                            segment=mean_segment_loss,
                            )
        # Logging (epoch)
        epoch_total_loss = sum(total_losses)/len(total_losses)
        epoch_obj_loss = sum(obj_losses)/len(obj_losses)
        epoch_noobj_loss = sum(noobj_losses)/len(noobj_losses)
        epoch_box_loss = sum(box_losses)/len(box_losses)
        epoch_class_loss = sum(class_losses)/len(class_losses)
        epoch_segment_loss = sum(segment_losses)/len(segment_losses)
        self.board.add_scalar('Epoch Train Loss', epoch_total_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Train BOX Loss', epoch_box_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Train OBJ Loss', epoch_obj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Train NOOBJ Loss', epoch_noobj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Train CLASS Loss', epoch_class_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Train SEGMENT Loss', epoch_segment_loss, global_step=self.current_epoch)

    def _validate(self):
        n_epochs = self.config['train']['n_epochs']
        current_epoch = self.current_epoch
        current_lr = self.optimizer.param_groups[0]['lr']
        loop = tqdm(self.valid_loader,
                    leave=True,
                    desc=(
                        f"Valid Epoch:{current_epoch}/{n_epochs}"
                        f", LR: {current_lr:.5f}"
                        )
                    )
        obj_losses = []
        box_losses = []
        noobj_losses = []
        class_losses = []
        total_losses = []
        segment_losses = []
        self.model.eval()
        for batch_idx, (imgs, masks, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            masks = [ m.to(self.device) for m in masks ]# (nM_g, H, W)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            # Model Prediction
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outs, prototypes = self.model(imgs)
                    s1_loss = self.loss_fn(
                                    outs[0], target_s1, self.scaled_anchors[0], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
                    s2_loss = self.loss_fn(
                                    outs[1], target_s2, self.scaled_anchors[1], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
                    s3_loss = self.loss_fn(
                                    outs[2], target_s3, self.scaled_anchors[2], # Detection Branch
                                    prototypes, masks,                          # Prototype Branch
                                    )
            # Aggregate loss
            obj_loss = s1_loss['obj_loss'] + s2_loss['obj_loss'] + s3_loss['obj_loss']
            box_loss = s1_loss['box_loss'] + s2_loss['box_loss'] + s3_loss['box_loss']
            noobj_loss = s1_loss['noobj_loss'] + s2_loss['noobj_loss'] + s3_loss['noobj_loss']
            class_loss = s1_loss['class_loss'] + s2_loss['class_loss'] + s3_loss['class_loss']
            segment_loss = s1_loss['segment_loss'] + s2_loss['segment_loss'] + s3_loss['segment_loss']
            total_loss = s1_loss['total_loss'] + s2_loss['total_loss'] + s3_loss['total_loss']
            # Moving average loss
            obj_losses.append(obj_loss.item())
            box_losses.append(box_loss.item())
            noobj_losses.append(noobj_loss.item())
            class_losses.append(class_loss.item())
            total_losses.append(total_loss.item())
            segment_losses.append(segment_loss.item())
            # Upadte progress bar
            mean_total_loss = sum(total_losses)/len(total_losses)
            mean_obj_loss = sum(obj_losses)/len(obj_losses)
            mean_noobj_loss = sum(noobj_losses)/len(noobj_losses)
            mean_box_loss = sum(box_losses)/len(box_losses)
            mean_class_loss = sum(class_losses)/len(class_losses)
            mean_segment_loss = sum(segment_losses)/len(segment_losses)
            loop.set_postfix(
                            loss=mean_total_loss,
                            cls=mean_class_loss,
                            box=mean_box_loss,
                            obj=mean_obj_loss,
                            noobj=mean_noobj_loss,
                            segment=mean_segment_loss,
                            )
        # Logging (epoch)
        epoch_total_loss = sum(total_losses)/len(total_losses)
        epoch_obj_loss = sum(obj_losses)/len(obj_losses)
        epoch_noobj_loss = sum(noobj_losses)/len(noobj_losses)
        epoch_box_loss = sum(box_losses)/len(box_losses)
        epoch_class_loss = sum(class_losses)/len(class_losses)
        epoch_segment_loss = sum(segment_losses)/len(segment_losses)
        self.board.add_scalar('Epoch Valid Loss', epoch_total_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Valid BOX Loss', epoch_box_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Valid OBJ Loss', epoch_obj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Valid NOOBJ Loss', epoch_noobj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Valid CLASS Loss', epoch_class_loss, global_step=self.current_epoch)
        self.board.add_scalar('Epoch Valid SEGMENT Loss', epoch_segment_loss, global_step=self.current_epoch)

    def _check_accuracy(self):
        tot_obj = 0
        tot_noobj = 0
        correct_obj = 0
        correct_noobj = 0
        correct_class = 0
        self.model.eval()
        loop = tqdm(self.valid_loader, leave=True, desc=f"Check ACC")
        for batch_idx, (imgs, masks, targets) in enumerate(loop):
            batch_size = imgs.size(0)
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            targets = [ target_s1, target_s2, target_s3 ]
            # Model Prediction
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outs, prototypes = self.model(imgs)
            for scale_idx in range(len(outs)):
                # Get output
                pred = outs[scale_idx]
                target = targets[scale_idx]
                # Get mask
                obj_mask = target[..., 4] == 1
                noobj_mask = target[..., 4] == 0
                # Count objects
                tot_obj += torch.sum(obj_mask)
                tot_noobj += torch.sum(noobj_mask)
                # Exception Handling
                if torch.sum(obj_mask) == 0:
                    obj_pred = torch.sigmoid(pred[..., 4]) > self.config['valid']['conf_threshold']
                    correct_noobj += torch.sum(obj_pred[noobj_mask] == target[..., 4][noobj_mask])
                    continue
                # Count number of correct classified object
                correct_class += torch.sum((
                    torch.argmax(pred[..., 5:][obj_mask], dim=-1)
                    == target[..., 5][obj_mask]
                    ))
                # Count number of correct objectness & non-objectness
                obj_pred = torch.sigmoid(pred[..., 4]) > self.config['valid']['conf_threshold']
                correct_obj += torch.sum(obj_pred[obj_mask] == target[..., 4][obj_mask])
                correct_noobj += torch.sum(obj_pred[noobj_mask] == target[..., 4][noobj_mask])
        # Aggregation Result
        acc_obj = (correct_obj/(tot_obj+1e-6))*100
        acc_cls = (correct_class/(tot_obj+1e-6))*100
        acc_noobj = (correct_noobj/(tot_noobj+1e-6))*100
        accs = { 'cls': acc_cls.item(), 'obj': acc_obj.item(), 'noobj': acc_noobj.item() }
        print(f"Epoch {self.current_epoch} [Accs]: {accs}")
        return accs

    def _check_map(self):
        sample_idx = 0
        all_pred_bboxes = []
        all_true_bboxes = []
        self.model.eval()
        loop = tqdm(self.valid_loader, leave=True, desc="Check mAP")
        for batch_idx, (imgs, masks, targets) in enumerate(loop):
            batch_size = imgs.size(0)
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            targets = [ target_s1, target_s2, target_s3 ]
            # Model Forward
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    preds, prototypes = self.model(imgs)
            # Convert cells to bboxes
            # =================================================================
            true_bboxes = [ [] for _ in range(batch_size) ]
            pred_bboxes = [ [] for _ in range(batch_size) ]
            for scale_idx, (pred, target) in enumerate(zip(preds, targets)):
                scale = pred.size(2)
                anchors = self.scaled_anchors[scale_idx] # (3, 2)
                anchors = anchors.reshape(1, 3, 1, 1, 2) # (1, 3, 1, 1, 2)
                # Convert prediction to correct format
                pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])      # (N, 3, S, S, 2)
                pred[..., 2:4] = torch.exp(pred[..., 2:4])*anchors  # (N, 3, S, S, 2)
                pred[..., 4:5] = torch.sigmoid(pred[..., 4:5])      # (N, 3, S, S, 1)
                pred_cls_probs = F.softmax(pred[..., 5:], dim=-1)   # (N, 3, S, S, C)
                _, indices = torch.max(pred_cls_probs, dim=-1)      # (N, 3, S, S)
                indices = indices.unsqueeze(-1)                     # (N, 3, S, S, 1)
                pred = torch.cat([ pred[..., :5], indices ], dim=-1)# (N, 3, S, S, 6)
                # Convert coordinate system to normalized format (xywh)
                pboxes = cells_to_boxes(cells=pred, scale=scale)    # (N, 3, S, S, 6)
                tboxes = cells_to_boxes(cells=target, scale=scale)  # (N, 3, S, S, 6)
                # Filter out bounding boxes from all cells
                for idx, cell_boxes in enumerate(pboxes):
                    obj_mask = cell_boxes[..., 4] > self.config['valid']['conf_threshold']
                    boxes = cell_boxes[obj_mask]
                    pred_bboxes[idx] += boxes.tolist()
                # Filter out bounding boxes from all cells
                for idx, cell_boxes in enumerate(tboxes):
                    obj_mask = cell_boxes[..., 4] > 0.99
                    boxes = cell_boxes[obj_mask]
                    true_bboxes[idx] += boxes.tolist()
            # Perform NMS batch-by-batch
            # =================================================================
            for batch_idx in range(batch_size):
                pbboxes = torch.tensor(pred_bboxes[batch_idx])
                tbboxes = torch.tensor(true_bboxes[batch_idx])
                # Perform NMS class-by-class
                for c in range(self.config['model']['num_classes']):
                    # Filter pred boxes of specific class
                    nms_pred_boxes = nms_by_class(target=c,
                                                bboxes=pbboxes,
                                                iou_threshold=self.config['valid']['nms_iou_threshold'])
                    nms_true_boxes = nms_by_class(target=c,
                                                bboxes=tbboxes,
                                                iou_threshold=self.config['valid']['nms_iou_threshold'])
                    all_pred_bboxes.extend([[sample_idx]+box
                                            for box in nms_pred_boxes])
                    all_true_bboxes.extend([[sample_idx]+box
                                            for box in nms_true_boxes])
                sample_idx += 1
        # Compute mAP@0.5 & mAP@0.75
        # =================================================================
        # The format of the bboxes is (idx, x1, y1, x2, y2, conf, class)
        all_pred_bboxes = torch.tensor(all_pred_bboxes) # (J, 7)
        all_true_bboxes = torch.tensor(all_true_bboxes) # (K, 7)
        eval50 = mean_average_precision(
                        all_pred_bboxes,
                        all_true_bboxes,
                        iou_threshold=0.5,
                        n_classes=self.config['dataset']['n_classes'])
        eval75 = mean_average_precision(
                        all_pred_bboxes,
                        all_true_bboxes,
                        iou_threshold=0.75,
                        n_classes=self.config['dataset']['n_classes'])
        print((
            f"Epoch {self.current_epoch}:\n"
            f"\t-[mAP@0.5]={eval50['mAP']:.3f}, [Recall]={eval50['recall']:.3f}, [Precision]={eval50['precision']:.3f}\n"
            f"\t-[mAP@0.75]={eval75['mAP']:.3f}, [Recall]={eval75['recall']:.3f}, [Precision]={eval75['precision']:.3f}\n"
            ))
        return eval50['mAP']

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'current_map': self.current_map,
            'current_epoch': self.current_epoch
        }
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("Save checkpoint at '{}'".format(checkpoint_path))
