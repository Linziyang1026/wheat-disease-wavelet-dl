import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

        # 在ResNet的某些块之后添加Dropout层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)

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