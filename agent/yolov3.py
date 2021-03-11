import os
import os.path as osp

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.dataset import YOLODataset
from data.transform import get_yolo_transform
from model.yolov3 import YOLOv3
from loss.yolo import YOLOLoss


class YOLOv3Agent:

    def __init__(self, config):
        self.config = config

        # Train on device
        target_device = config['train']['device']
        self.device = target_device if torch.cuda.is_available() else "cpu"

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
        self.current_epoch = -1
        self.current_map = 0

    def resume(self):
        pass

    def train(self):
        pass

    def finalize(self):
        pass

    def _train_one_epoch(self):
        pass

    def _validate(self):
        pass

    def _save_checkpoint(self):
        pass
