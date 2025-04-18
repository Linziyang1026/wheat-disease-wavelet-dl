import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from config import Config


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 冻结前面的层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻后面的层（例如，最后的几个块）
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # 添加更多的 Dropout 层和 Batch Normalization
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # 替换最后的全连接层
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            self.bn1,
            nn.ReLU(),
            self.dropout1,
            nn.Linear(512, 256),
            self.bn2,
            nn.ReLU(),
            self.dropout2,
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model(num_classes=Config.num_classes, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomResNet18(num_classes=num_classes)
    model.to(device)
    return model