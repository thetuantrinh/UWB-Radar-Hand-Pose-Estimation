
"""
An experiment model for handpose estimation using radar signals
"""

import torch.nn as nn
from ..modules.residual import ResidualBlock

# from weight_init import init_weights


class RadarRes(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(RadarRes, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer7 = block(
            inplanes=32 * expansion, planes=64 * expansion, stride=2
        )
        self.layer8 = block(inplanes=64 * expansion, planes=64 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense = nn.Linear(64 * expansion, keypoints * 2)
        # self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = x.view(x.shape[0], 21, 2)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)


class SimpleNet(nn.Module):
    def __init__(
        self,
        in_channels=8,
        keypoints=21,
    ):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.dense = nn.LazyLinear(keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel1(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense = nn.Linear(32 * expansion, keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        # x = x.view(x.shape[0], 21, 2)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel2(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)
        self.layer7 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense = nn.Linear(32 * expansion, keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        # x = x.view(x.shape[0], 21, 2)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel3(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel3, self).__init__()
        self.downsample = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=1, stride=1, bias=False
        )
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(32 * expansion, keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel6(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel6, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense = nn.Linear(32 * expansion, keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        # x = x.view(x.shape[0], 21, 2)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class Model3D(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(Model3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=8,
            out_channels=16 * expansion,
            kernel_size=3,
            padding=1,
        )

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3)

        self.group1 = nn.Sequential(
            nn.Conv3d(
                16 * expansion, 32 * expansion, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(32 * expansion),
            nn.ReLU(),
        )

        self.group2 = nn.Sequential(
            nn.Conv3d(
                32 * expansion, 64 * expansion, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(64 * expansion),
            nn.ReLU(),
        )

        self.group3 = nn.Sequential(
            nn.Conv3d(
                64 * expansion, 64 * expansion, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(64 * expansion),
            nn.ReLU(),
            nn.Conv3d(
                64 * expansion, 64 * expansion, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(64 * expansion),
            nn.ReLU(),
        )

        self.avg_pooling = nn.AdaptiveAvgPool3d(output_size=(1, 1))
        self.dense = nn.Linear(64 * expansion, keypoints * 2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)

        x = self.group3(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)
        # x = x.view(x.shape[0], 21, 2)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel7(nn.Module):
    def __init__(
        self,
        expansion=2,
        in_channels=8,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel7, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.landmarks = nn.Linear(32 * expansion, keypoints * 2)
        self.hand_presence = nn.Sequential(
            nn.Linear(32 * expansion, 1),
            nn.Sigmoid(),
        )
        self.handedness = nn.Sequential(
            nn.Linear(32 * expansion, 1),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        landmarks = self.landmarks(x)
        hand_presence = self.hand_presence(x)
        handedness = self.handedness(x)

        return landmarks, hand_presence, handedness

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )


class DevModel8(nn.Module):
    def __init__(
        self,
        expansion=2,
        in_channels=8,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel8, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16 * expansion,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(16 * expansion)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = block(inplanes=16 * expansion, planes=16 * expansion)
        # self.layer2 = block(inplanes=16 * expansion, planes=16 * expansion)

        self.layer3 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        # self.layer4 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer5 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )
        # self.layer6 = block(inplanes=32 * expansion, planes=32 * expansion)

        self.layer7 = block(
            inplanes=32 * expansion, planes=32 * expansion, stride=2
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.landmarks = nn.Linear(32 * expansion, keypoints * 2)
        self.hand_presence = nn.Sequential(
            nn.Linear(32 * expansion, 1),
            nn.Sigmoid(),
        )
        self.handedness = nn.Sequential(
            nn.Linear(32 * expansion, 1),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)

        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.layer5(x)
        # x = self.layer6(x)

        x = self.layer7(x)

        x = self.avg_pooling(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        landmarks = self.landmarks(x)
        hand_presence = self.hand_presence(x)
        handedness = self.handedness(x)

        return landmarks, hand_presence, handedness

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
