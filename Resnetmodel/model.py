import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 冻结前面的层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻后面的层（例如，最后的几个块）
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # 添加更多的Dropout层
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.7)

        # 替换最后的全连接层
        self.model.fc = nn.Sequential(
            self.dropout1,
            nn.Linear(self.model.fc.in_features, 256),
            self.dropout2,
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model(num_classes, device):
    model = CustomResNet50(num_classes=num_classes)
    model.to(device)
    return model