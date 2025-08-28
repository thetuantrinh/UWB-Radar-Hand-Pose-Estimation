import torch.nn as nn
from ..modules.residual import ResidualBlock


class DevModel9(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel9, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
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
        self.layer2 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer3 = block(
            inplanes=32 * expansion, planes=64 * expansion, stride=2
        )
        self.layer4 = block(
            inplanes=64 * expansion, planes=64 * expansion, stride=2
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.landmarks = nn.Linear(64 * expansion, keypoints * 2)
        self.hand_presence = nn.Sequential(
            nn.Linear(64 * expansion, 1),
            nn.Sigmoid(),
        )
        self.handedness = nn.Sequential(
            nn.Linear(64 * expansion, 1),
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


class DevModel4(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel4, self).__init__()
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
            inplanes=32 * expansion, planes=64 * expansion, stride=2
        )
        self.layer6 = block(inplanes=64 * expansion, planes=64 * expansion)

        self.layer7 = block(
            inplanes=64 * expansion, planes=128 * expansion, stride=2
        )
        self.layer8 = block(inplanes=128 * expansion, planes=128 * expansion)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(128 * expansion, keypoints * 2)
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
        x = self.layer8(x)
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


class DevModel10(nn.Module):
    def __init__(
        self,
        expansion=2,
        block=ResidualBlock,
        keypoints=21,
    ):
        super(DevModel10, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
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
        self.layer2 = block(
            inplanes=16 * expansion, planes=32 * expansion, stride=2
        )
        self.layer3 = block(
            inplanes=32 * expansion, planes=64 * expansion, stride=2
        )
        self.layer4 = block(
            inplanes=64 * expansion, planes=64 * expansion, stride=2
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
 
        self.fc1 = nn.Sequential(
            nn.Linear(1152, 512), nn.ReLU(), nn.Dropout(p=0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.4)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.4)
        )
        self.landmarks = nn.Linear(64 * expansion, keypoints * 2)
        self.hand_presence = nn.Sequential(
            nn.Linear(64 * expansion, 1),
            nn.Sigmoid(),
        )
        self.handedness = nn.Sequential(
            nn.Linear(64 * expansion, 1),
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

        # x = self.avg_pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = x.view(x.size(0), -1)
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
