import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(num_classes, device):
    model = CustomResNet50(num_classes=num_classes)
    model.to(device)
    return model