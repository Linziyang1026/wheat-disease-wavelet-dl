import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self.load_annotations(label_file)

    def load_annotations(self, label_file):
        annotations = []
        with open(label_file, 'r') as file:
            for line in file:
                values = line.strip().split()
                category_id = int(values[0])
                points = [list(map(float, values[1:3])), list(map(float, values[3:5])),
                          list(map(float, values[5:7])), list(map(float, values[7:9]))]
                annotations.append((category_id, points))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        category_id, points = self.annotations[idx]

        if self.transform:
            transformed = self.transform(image=image, points=points, category_ids=[category_id])
            image = transformed['image']
            points = transformed['points']
            category_id = transformed['category_ids'][0]

        return image, points, category_id

# 定义数据增强流程
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OpticalDistortion(p=0.5),
    A.Perspective(p=0.5),
    A.ColorJitter(p=0.5),
    A.ToGray(p=0.5),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='xyxy', min_area=0, min_visibility=0, label_fields=['category_ids']))

# 创建数据集实例
dataset = CustomDataset(img_dir='path_to_your_images', label_file='path_to_your_labels.txt', transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 使用数据加载器
for images, points, category_ids in dataloader:
    print("Image shape:", images.shape)
    print("Points:", points)
    print("Category IDs:", category_ids)
    break  # 仅演示第一个批次