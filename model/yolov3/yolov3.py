import os
import os.path as osp
import yaml
from collections import OrderedDict

import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn


__all__ = [ "YOLOv3", "Maskv3" ]


class CNNBlock(nn.Module):
    """Basic CNN block"""
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual with repeated blocks

    Arguments:
        channels (int): number of channels in each block
        use_residual (bool): whether perform residual connection or not
        num_repeats (int): number of repeated blocks
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                    )
                ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    """Detection head in specific scale"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (5+num_classes)*3, bn_act=False, kernel_size=1)
            )
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        return (
            self.pred(x)
            .reshape(batch_size, 3, 5+self.num_classes, height, width)
            .permute(0, 1, 3, 4, 2)
            )


class MaskScalePrediction(nn.Module):
    """Detection head in specific scale with mask coefficients"""
    def __init__(self, in_channels, num_classes, num_masks):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (5+num_classes+num_masks)*3, bn_act=False, kernel_size=1)
            )
        self.num_classes = num_classes
        self.num_masks = num_masks

    def forward(self, x):
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        return (
            self.pred(x)
            .reshape(batch_size, 3, 5+self.num_classes+self.num_masks, height, width)
            .permute(0, 1, 3, 4, 2)
            )


class YOLOv3(nn.Module):
    BACKBONE_WEIGHTS = 'https://www.dropbox.com/s/cd6aufj1wwexsbp/darknet448.pth?dl=1'
    BACKBONE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'backbone.yml')
    DETECTOR_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'detector.yml')
    def __init__(self,
                in_channels,
                num_classes,
                backbone_path=None,
                detector_path=None,
                pretrain_backbone=True,
                pretrain_detector=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Generate model architecture from config file
        self.cur_channels = in_channels
        backbone_path = self.BACKBONE_PATH if backbone_path is None else backbone_path
        detector_path = self.DETECTOR_PATH if detector_path is None else detector_path
        self.backbone = self._parse_yaml(backbone_path)
        self.detector = self._parse_yaml(detector_path)
        # Initialize backbone model
        if pretrain_backbone:
            state_dict = load_state_dict_from_url(YOLOv3.BACKBONE_WEIGHTS)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if not k.startswith('feature'):
                    continue
                name = '.'.join(k.split('.')[1:])
                new_state_dict[name] = v.cpu()
            self.backbone.load_state_dict(new_state_dict)

    def forward(self, x):
        outputs, _ = self._foward_logics(x)
        return outputs

    def get_prediction_head(self, in_channels):
        return ScalePrediction(in_channels, num_classes=self.num_classes)

    def _foward_logics(self, x):
        outputs = [] # for each scale
        latent_features = [] # Feature maps for prediction heads of each scale
        route_connections = [] # Skip connection feature maps in backbone layer

        # Forward backbone feature extractor
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

        # Forward detection header
        for layer in self.detector:
            if (
                isinstance(layer, ScalePrediction)
                or isinstance(layer, MaskScalePrediction)
            ):
                latent_features.append(x)
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs, latent_features

    def _parse_yaml(self, path):
        layers = nn.ModuleList()
        in_channels = self.cur_channels # Keep track of current channels

        with open(path, 'r') as f:
            arch = yaml.full_load(f)

        for layer in arch['layers']:
            option = layer[0]
            # Basic CNN Block
            if option == 'C':
                out_channels, kernel_size, stride = layer[1:]
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                        )
                    )
                in_channels = out_channels
            # Residual Block
            elif option == 'B':
                num_repeats = layer[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            # ScalePrediction
            elif option == 'S':
                layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                    CNNBlock(in_channels, in_channels//2, kernel_size=1),
                    self.get_prediction_head(in_channels//2)
                    ]
                in_channels = in_channels // 2
            # UpSampling
            elif option == 'U':
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels * 3
                # Concat with previous layer output
                # See `route_connections` in forward method
            else:
                raise ValueError("Don't know how to parse '{}' type layer".format(option))

        self.cur_channels = in_channels
        return layers


class Maskv3(YOLOv3):

    def __init__(self, num_masks, num_features, **kwargs):
        self.num_masks = num_masks
        self.num_features = num_features
        super().__init__(**kwargs)
        # Skip Feature Layers
        # ====================================================================
        # (512, 13, 13) => (num_features, 26, 26)
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(512, num_features, **self._upscale_params(2)),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1, inplace=True),
            )
        # (256+num_features, 26, 26) => (num_features, 52, 52)
        self.upscale2 = nn.Sequential(
            nn.ConvTranspose2d(256+num_features, num_features, **self._upscale_params(2)),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1, inplace=True),
            )
        # (128+num_features, 52, 52) => (num_features, 104, 104)
        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(128+num_features, num_features, **self._upscale_params(2)),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1, inplace=True),
            )

        # Mask Generation Layer
        # ====================================================================
        self.protonet = nn.Sequential(
            # (num_features, 104, 104)
            nn.ConvTranspose2d(num_features, num_features//2, **self._upscale_params(2)),
            nn.BatchNorm2d(num_features//2),
            nn.LeakyReLU(0.1, inplace=True),
            # (num_features//2, 208, 208)
            nn.ConvTranspose2d(num_features//2, num_features//4, **self._upscale_params(2)),
            nn.BatchNorm2d(num_features//4),
            nn.LeakyReLU(0.1, inplace=True),
            # (num_features//4, 416, 416)
            nn.Conv2d(num_features//4, num_masks, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # (num_masks, 416, 416)
            )

    def forward(self, x):
        outputs, latent_features = self._foward_logics(x)

        x = self.upscale1(latent_features[0])
        x = torch.cat([latent_features[1], x], dim=1)
        x = self.upscale2(x)
        x = torch.cat([latent_features[2], x], dim=1)
        x = self.upscale3(x)
        x = self.protonet(x) # (masks)

        return outputs, x

    def get_prediction_head(self, in_channels):
        return MaskScalePrediction(in_channels,
                                num_classes=self.num_classes,
                                num_masks=self.num_masks)

    def _upscale_params(self, factor):
        if factor == 1:
            return { "stride": factor, "kernel_size": 1, "padding": 0 }
        else:
            return { "stride": factor, "kernel_size": factor*2, "padding": factor//2 }


if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    model = YOLOv3(in_channels=3, num_classes=num_classes)
    model = model.to("cuda:1")
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    x = x.to("cuda:1")

    import time
    start = time.time()
    for i in range(100):
        outs = model(x)

    print(f"Yolov3 (Elapsed Time: {(time.time()-start)/100})")
    print("Output Shape: (N, num_anchors, img_height, img_width, 5+num_class)")
    print("Scale #1:", outs[0].shape)
    print("Scale #2:", outs[1].shape)
    print("Scale #3:", outs[2].shape)
    print("")

    model = Maskv3(
                in_channels=3, num_classes=num_classes, # Object Detection Branch
                num_masks=32, num_features=256, # Mask Generation Branch
                )
    model = model.to("cuda:1")
    start = time.time()
    for i in range(100):
        outs, masks = model(x)
    print(f"Maskv3(Elapsed Time: {(time.time()-start)/100})")
    print("Output Shape: (N, num_anchors, img_height, img_width, 5+num_class+num_masks)")
    print("Scale #1:", outs[0].shape)
    print("Scale #2:", outs[1].shape)
    print("Scale #3:", outs[2].shape)
    print("Prototypes:", masks.shape)
