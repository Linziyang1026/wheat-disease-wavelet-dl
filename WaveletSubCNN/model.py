import torch.nn as nn
import torch.nn.functional as F

class WheatDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(WheatDiseaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x.view(-1, 128*4*4))  # 应用dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 再次应用dropout
        x = self.fc2(x)
        return x