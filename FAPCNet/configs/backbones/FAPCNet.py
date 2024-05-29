import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.module import *

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    'mish': Mish(),
    'leaky': nn.LeakyReLU(negative_slope=0.1),
    'linear': nn.Identity()
}

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='leaky'):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv(x)

class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = nn.Sequential(Conv(in_channels, out_channels, 3, stride=2),
                                             EMA(out_channels)
                                             )

        self.split_conv0 = Conv(out_channels, out_channels, 1)
        self.split_conv1 = Conv(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            Partial_conv3(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SplitAttn(out_channels)
        )

        self.concat_conv = nn.Sequential(Conv(out_channels * 2, out_channels, 1),
                                         SEAttention(out_channels))

    def forward(self, x):

        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = nn.Sequential(Conv(in_channels, out_channels, 3, stride=2),
                                             EMA(out_channels))

        self.split_conv0 = Conv(out_channels, out_channels // 2, 1)
        self.split_conv1 = Conv(out_channels, out_channels // 2, 1)

        blocks_conv = []
        for i in range(num_blocks):
            blocks_conv.append(Partial_conv3(out_channels // 2))
            blocks_conv.append(nn.BatchNorm2d(out_channels // 2))
            blocks_conv.append(nn.ReLU())
        blocks_conv.append(SplitAttn(out_channels // 2))
        self.blocks_conv = nn.Sequential(*blocks_conv)
        self.concat_conv = nn.Sequential(Conv(out_channels, out_channels, 1),
                                         SEAttention(out_channels))

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x


class FAPCNet(nn.Module):
    def __init__(self,
                 stem_channels=32,
                 feature_channels=[64, 128, 256, 512, 1024],
                 num_features=3,
                 num_classes=9):

        super(FAPCNet, self).__init__()

        self.stem_conv = Conv(3, stem_channels, 3)

        self.stages = nn.ModuleList([
            CSPFirstStage(stem_channels, feature_channels[0]),
            CSPStage(feature_channels[0], feature_channels[1], 2),
            CSPStage(feature_channels[1], feature_channels[2], 8),
            CSPStage(feature_channels[2], feature_channels[3], 12),
            CSPStage(feature_channels[3], feature_channels[4], 4)
        ])

        self.feature_channels = feature_channels
        self.num_classes = num_classes

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(1024, self.num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        features = []

        x = self.stem_conv(x)

        x = self.stages[0](x)  # //2
        x = self.stages[1](x)  # //4
        x8 = self.stages[2](x)  # //8
        features.append(x8)

        x16 = self.stages[3](x8)  # //16
        features.append(x16)

        x32 = self.stages[4](x16)  # //32
        features.append(x32)

        return features

    def forward(self, x):
        features = self.extract_features(x)
        # x = self.gap(features[-1])
        # x = self.fc(x)
        # x = x.flatten(start_dim=1)
        return features[-1]

