import os
import os.path as osp
import yaml

import torch
import torch.nn as nn


__all__ = [ "YOLOv3" ]


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
            CNNBlock(2*in_channels, (num_classes+5)*3, bn_act=False, kernel_size=1)
            )
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        return (
            self.pred(x)
            .reshape(batch_size, 3, self.num_classes+5, height, width)
            .permute(0, 1, 3, 4, 2)
            )

class YOLOv3(nn.Module):
    BACKBONE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'backbone.yml')
    DETECTOR_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'detector.yml')
    def __init__(self, in_channels, num_classes, backbone_path=None, detector_path=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Generate model architecture from config file
        self.cur_channels = in_channels
        backbone_path = self.BACKBONE_PATH if backbone_path is None else backbone_path
        detector_path = self.DETECTOR_PATH if detector_path is None else detector_path
        self.backbone = self._parse_yaml(backbone_path)
        self.detector = self._parse_yaml(detector_path)

    def forward(self, x):
        outputs = [] # for each scale
        route_connections = []

        # Forward backbone feature extractor
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

        # Forward detection header
        for layer in self.detector:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

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
                    ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                in_channels = in_channels // 2
            # UpSampling
            elif option == 'U':
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels * 3
            else:
                raise ValueError("Don't know how to parse '{}' type layer".format(option))

        self.cur_channels = in_channels
        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(in_channels=3, num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    outs = model(x)

    print("Output Shape: (N, num_anchors, img_height, img_width, num_class+5)")
    print("Scale #1:", outs[0].shape)
    print("Scale #2:", outs[1].shape)
    print("Scale #3:", outs[2].shape)
