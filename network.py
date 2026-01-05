# STUDENT's UCO: 525265

# Description:
# This file should contain network class.
# The class should subclass the torch.nn.Module class.

import torch.nn.functional as F
from torch import Tensor, nn, flatten
from torchvision.models import (alexnet, AlexNet_Weights, resnet18,
                                vgg16, ResNet18_Weights, VGG16_Weights)


class ModelExample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class Alexnet(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self) # This is for debugging only
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x


def PretrainedAlexnet(num_classes=1000, dropout=0.5):
    model = alexnet(weights=AlexNet_Weights.DEFAULT, dropout=dropout)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model


def PretrainedResnet(num_classes=1000):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def Resnet(num_classes=1000):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def PretrainedVGG(num_classes=1000, dropout=0.5):
    model = vgg16(weights=VGG16_Weights.DEFAULT, dropout=dropout)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model
