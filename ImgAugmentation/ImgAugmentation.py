import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, output_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.transform = transform
        self.img_files = self.get_img_files()
        os.makedirs(self.output_dir, exist_ok=True)

    def get_img_files(self):
        img_files = []
        for file in os.listdir(self.img_dir):
            if file.startswith("processed_") and (file.endswith(".png") or file.endswith(".jpg")):
                img_files.append(file)
        return img_files

    def load_annotations(self, label_path):
        annotations = []
        with open(label_path, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                category_id = int(values[0])
                points = np.array(values[1:]).reshape(-1, 2)  # 将剩余的值转换为二维数组，每对代表一个点的x,y坐标
                annotations.append((category_id, points))
        return annotations

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_name = img_file.replace("processed_", "").split('.')[0]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.load_annotations(label_path)
        category_id, points = annotations[0]  # 假设每个图像只有一个标注

        if self.transform:
            transformed = self.transform(image=image, keypoints=points, class_labels=[category_id])
            image = transformed['image']
            points = np.array(transformed['keypoints'])
            category_id = transformed['class_labels'][0]

        # 保存增强后的图像
        output_img_path = os.path.join(self.output_dir, f"aug_{img_file}")
        cv2.imwrite(output_img_path, cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))

        # 保存增强后的标签
        output_label_path = os.path.join(self.output_dir, f"aug_{img_name}.txt")
        points_str = ' '.join([f'{x:.6f} {y:.6f}' for x, y in points])
        with open(output_label_path, 'w') as file:
            file.write(f'{category_id} {points_str}\n')

        return image, points, category_id

# 定义数据增强流程（仅包含翻转和颜色抖动）
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.VerticalFlip(p=0.5),    # 垂直翻转
    A.ColorJitter(p=0.5),     # 颜色抖动
    ToTensorV2(),             # 转换为张量
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False), additional_targets={'class_labels': 'class_label'})

# 确保路径正确
img_dir = r'F:\wheat\output_threshold_RGB_processed_images\test\images'
label_dir = r'F:\wheat\output_threshold_RGB_processed_images\test\labels'
output_dir = r'F:\wheat\output_augmentation'

if __name__ == '__main__':
    # 创建数据集实例
    dataset = CustomDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        transform=transform
    )

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # 设置 num_workers=0

    # 使用数据加载器
    for images, points, category_ids in dataloader:
        print("Image shape:", images.shape)
        print("Points:", points)
        print("Category IDs:", category_ids)
        break  # 仅演示第一个批次

    print(f"Augmented images and labels saved to: {output_dir}")