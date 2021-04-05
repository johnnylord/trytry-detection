import os
import os.path as osp

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

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
        torch_anchors = torch.tensor(config['dataset']['anchors']) # (3, 3, 2)
        torch_scales = torch.tensor(config['dataset']['scales']) # (3
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
        scheduler = MultiStepLR(self.optimizer,
                                milestones=config['scheduler']['milestones'],
                                gamma=config['scheduler']['gamma'])
        self.scheduler = GradualWarmupScheduler(self.optimizer,
                                                multiplier=1.,
                                                total_epoch=10,
                                                after_scheduler=scheduler)
        # Loss function
        self.loss_fn = YOLOLoss()

        # Tensorboard
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=config['train']['logdir'])

        # Training State
        self.current_epoch = 0
        self.current_map = 0
        self.current_acc = {}

    def resume(self):
        pass

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']+1):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._validate()
            self.scheduler.step()
            if (
                epoch > self.config['train']['valid_epoch']
                and epoch % self.config['train']['valid_cycle'] == 0
            ):
                acc = self._check_accuracy()
                mAP = self._check_map()
                if mAP > self.current_map:
                    self.current_map = mAP
                    self.current_acc = acc
                    self._save_checkpoint()

    def finalize(self):
        pass

    def _train_one_epoch(self):
        n_epochs = self.config['train']['n_epochs']
        current_epoch = self.current_epoch
        current_lr = self.optimizer.param_groups[0]['lr']
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=(
                        f"Train Epoch:{current_epoch}/{n_epochs}"
                        f", LR: {current_lr:.4f}"
                        )
                    )
        total_losses = []
        obj_losses = []
        noobj_losses = []
        box_losses = []
        class_losses = []
        self.model.train()
        for batch_idx, (imgs, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            # Model prediction
            with torch.cuda.amp.autocast():
                out = self.model(imgs)
                s1_loss = self.loss_fn(out[0], target_s1, self.scaled_anchors[0])
                s2_loss = self.loss_fn(out[1], target_s2, self.scaled_anchors[1])
                s3_loss = self.loss_fn(out[2], target_s3, self.scaled_anchors[2])
                # Aggregate loss
                total_loss = (
                    s1_loss['total_loss']
                    + s2_loss['total_loss']
                    + s3_loss['total_loss']
                    )
                obj_loss = (
                    s1_loss['obj_loss']
                    + s2_loss['obj_loss']
                    + s3_loss['obj_loss']
                    )
                noobj_loss = (
                    s1_loss['noobj_loss']
                    + s2_loss['noobj_loss']
                    + s3_loss['noobj_loss']
                    )
                box_loss = (
                    s1_loss['box_loss']
                    + s2_loss['box_loss']
                    + s3_loss['box_loss']
                    )
                class_loss = (
                    s1_loss['class_loss']
                    + s2_loss['class_loss']
                    + s3_loss['class_loss']
                    )
            # Moving average loss
            total_losses.append(total_loss.item())
            obj_losses.append(obj_loss.item())
            noobj_losses.append(noobj_loss.item())
            box_losses.append(box_loss.item())
            class_losses.append(class_loss.item())
            # Update Parameters
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Upadte progress bar
            mean_total_loss = sum(total_losses)/len(total_losses)
            mean_obj_loss = sum(obj_losses)/len(obj_losses)
            mean_noobj_loss = sum(noobj_losses)/len(noobj_losses)
            mean_box_loss = sum(box_losses)/len(box_losses)
            mean_class_loss = sum(class_losses)/len(class_losses)
            loop.set_postfix(
                            loss=mean_total_loss,
                            cls=mean_class_loss,
                            box=mean_box_loss,
                            obj=mean_obj_loss,
                            noobj=mean_noobj_loss,
                            )
        # Logging (epoch)
        epoch_total_loss = sum(total_losses)/len(total_losses)
        epoch_obj_loss = sum(obj_losses)/len(obj_losses)
        epoch_noobj_loss = sum(noobj_losses)/len(noobj_losses)
        epoch_box_loss = sum(box_losses)/len(box_losses)
        epoch_class_loss = sum(class_losses)/len(class_losses)
        self.board.add_scalar('Train Loss', epoch_total_loss, global_step=self.current_epoch)
        self.board.add_scalar('Train BOX Loss', epoch_box_loss, global_step=self.current_epoch)
        self.board.add_scalar('Train OBJ Loss', epoch_obj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Train NOOBJ Loss', epoch_noobj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Train CLASS Loss', epoch_class_loss, global_step=self.current_epoch)

    def _validate(self):
        n_epochs = self.config['train']['n_epochs']
        current_epoch = self.current_epoch
        current_lr = self.optimizer.param_groups[0]['lr']
        loop = tqdm(self.valid_loader,
                    leave=True,
                    desc=(
                        f"Valid Epoch:{current_epoch}/{n_epochs}"
                        f", LR: {current_lr:.4f}"
                        )
                    )
        total_losses = []
        obj_losses = []
        noobj_losses = []
        box_losses = []
        class_losses = []
        self.model.eval()
        for batch_idx, (imgs, targets) in enumerate(loop):
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            # Model Prediction
            with torch.no_grad():
                out = self.model(imgs)
                # Copmute Loss
                s1_loss = self.loss_fn(out[0], target_s1, self.scaled_anchors[0])
                s2_loss = self.loss_fn(out[1], target_s2, self.scaled_anchors[1])
                s3_loss = self.loss_fn(out[2], target_s3, self.scaled_anchors[2])
                # Aggregate loss
                total_loss = (
                    s1_loss['total_loss']
                    + s2_loss['total_loss']
                    + s3_loss['total_loss']
                    )
                obj_loss = (
                    s1_loss['obj_loss']
                    + s2_loss['obj_loss']
                    + s3_loss['obj_loss']
                    )
                noobj_loss = (
                    s1_loss['noobj_loss']
                    + s2_loss['noobj_loss']
                    + s3_loss['noobj_loss']
                    )
                box_loss = (
                    s1_loss['box_loss']
                    + s2_loss['box_loss']
                    + s3_loss['box_loss']
                    )
                class_loss = (
                    s1_loss['class_loss']
                    + s2_loss['class_loss']
                    + s3_loss['class_loss']
                    )
            # Moving average loss
            total_losses.append(total_loss.item())
            obj_losses.append(obj_loss.item())
            noobj_losses.append(noobj_loss.item())
            box_losses.append(box_loss.item())
            class_losses.append(class_loss.item())
            # Upadte progress bar
            mean_total_loss = sum(total_losses)/len(total_losses)
            mean_obj_loss = sum(obj_losses)/len(obj_losses)
            mean_noobj_loss = sum(noobj_losses)/len(noobj_losses)
            mean_box_loss = sum(box_losses)/len(box_losses)
            mean_class_loss = sum(class_losses)/len(class_losses)
            loop.set_postfix(
                            loss=mean_total_loss,
                            cls=mean_class_loss,
                            box=mean_box_loss,
                            obj=mean_obj_loss,
                            noobj=mean_noobj_loss,
                            )
        # Logging (epoch)
        epoch_total_loss = sum(total_losses)/len(total_losses)
        epoch_obj_loss = sum(obj_losses)/len(obj_losses)
        epoch_noobj_loss = sum(noobj_losses)/len(noobj_losses)
        epoch_box_loss = sum(box_losses)/len(box_losses)
        epoch_class_loss = sum(class_losses)/len(class_losses)
        self.board.add_scalar('Valid Loss', epoch_total_loss, global_step=self.current_epoch)
        self.board.add_scalar('Valid BOX Loss', epoch_box_loss, global_step=self.current_epoch)
        self.board.add_scalar('Valid OBJ Loss', epoch_obj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Valid NOOBJ Loss', epoch_noobj_loss, global_step=self.current_epoch)
        self.board.add_scalar('Valid CLASS Loss', epoch_class_loss, global_step=self.current_epoch)

    def _check_accuracy(self):
        self.model.eval()
        loop = tqdm(self.valid_loader, leave=True, desc=f"Check ACC")
        tot_obj, tot_noobj = 0, 0
        correct_class, correct_obj, correct_noobj = 0, 0, 0
        for batch_idx, (imgs, targets) in enumerate(loop):
            batch_size = imgs.size(0)
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            labels = [ target_s1, target_s2, target_s3 ]
            # Model Prediction
            with torch.no_grad():
                out = self.model(imgs)
            # Check accuracy
            for scale_idx in range(len(out)):
                pred, label = out[scale_idx], labels[scale_idx]
                obj_mask = label[..., 0] == 1
                noobj_mask = label[..., 0] == 0
                # Count number of true objects and true non-objects
                tot_obj += torch.sum(obj_mask)
                tot_noobj += torch.sum(noobj_mask)
                # Count number of correct classified object
                correct_class += torch.sum(
                    torch.argmax(pred[..., 5:][obj_mask], dim=-1) == label[..., 5][obj_mask]
                    )
                # Count number of correct objectness & non-objectness
                obj_pred = torch.sigmoid(pred[..., 0]) > self.config['valid']['conf_threshold']
                correct_obj += torch.sum(obj_pred[obj_mask] == label[..., 0][obj_mask])
                correct_noobj += torch.sum(obj_pred[noobj_mask] == label[..., 0][noobj_mask])
        # Aggregation Result
        acc_cls = (correct_class/(tot_obj+1e-6))*100
        acc_obj = (correct_obj/(tot_obj+1e-6))*100
        acc_noobj = (correct_noobj/(tot_noobj+1e-6))*100
        accs = {
            'cls': acc_cls.item(),
            'obj': acc_obj.item(),
            'noobj': acc_noobj.item()
            }
        print("ACCs:", accs)
        return accs

    def _check_map(self):
        self.model.eval()
        loop = tqdm(self.valid_loader, leave=True, desc=f"Check mAP")
        sample_idx = 0
        all_pred_bboxes = []
        all_true_bboxes = []
        for batch_idx, (imgs, targets) in enumerate(loop):
            batch_size = imgs.size(0)
            # Move device
            imgs = imgs.to(self.device)             # (N, 3, 416, 416)
            target_s1 = targets[0].to(self.device)  # (N, 3, 13, 13, 6)
            target_s2 = targets[1].to(self.device)  # (N, 3, 26, 26, 6)
            target_s3 = targets[2].to(self.device)  # (N, 3, 52, 52, 6)
            labels = [ target_s1, target_s2, target_s3 ]
            # Model Prediction
            with torch.no_grad():
                out = self.model(imgs)
            # Extract bounding boxes for each batch sample
            true_bboxes = [ [] for _ in range(batch_size) ]
            pred_bboxes = [ [] for _ in range(batch_size) ]
            for scale_idx in range(len(out)):
                pred, label = out[scale_idx], labels[scale_idx]
                curr_scale = pred.size(2) # pred: (N, 3, S, S, 5+classes)
                curr_anchors = self.scaled_anchors[scale_idx] # curr_anchor: (3, 2)
                pbbboxes = cells_to_bboxes(preds=pred,
                                        scale=curr_scale,
                                        anchors=curr_anchors)
                tbbboxes = cells_to_bboxes(preds=label,
                                        scale=curr_scale,
                                        anchors=curr_anchors,
                                        is_preds=False)
                for idx, bboxes in enumerate(pbbboxes):
                    mask = bboxes[..., 1] > self.config['valid']['conf_threshold']
                    valid_bboxes = bboxes[mask]
                    pred_bboxes[idx] += valid_bboxes.tolist()
                for idx, bboxes in enumerate(tbbboxes):
                    mask = bboxes[..., 1] > 0.99
                    valid_bboxes = bboxes[mask]
                    true_bboxes[idx] += valid_bboxes.tolist()
            # Perform NMS
            for idx in range(batch_size):
                nms_pred_bboxes = non_max_suppression(
                    pred_bboxes[idx],
                    iou_threshold=self.config['valid']['nms_iou_threshold'],
                    prob_threshold=self.config['valid']['conf_threshold'],
                    classes=[ i for i in range(self.config['dataset']['n_classes']) ],
                    box_format='xywh'
                    )
                nms_true_bboxes = non_max_suppression(
                    true_bboxes[idx],
                    iou_threshold=0.99,
                    prob_threshold=0.99,
                    classes=[ i for i in range(self.config['dataset']['n_classes']) ],
                    box_format='xywh'
                    )
                for nms_box in nms_pred_bboxes:
                    all_pred_bboxes.append([sample_idx] + nms_box)
                for nms_box in nms_true_bboxes:
                    all_true_bboxes.append([sample_idx] + nms_box)
                sample_idx += 1
        # Compute mAP@0.5
        mapval = mean_average_precision(
                    all_pred_bboxes,
                    all_true_bboxes,
                    iou_threshold=self.config['valid']['map_iou_threshold'],
                    n_classes=self.config['dataset']['n_classes'],
                    box_format="xywh",
                    )
        print("mAP@0.5:", mapval.item())
        return mapval.item()

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'current_acc': self.current_acc,
            'current_map': self.current_map,
        }
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("Save checkpoint at '{}'".format(checkpoint_path))
