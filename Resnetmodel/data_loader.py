from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class WheatDiseaseDataset(Dataset):



    # 应用于小波降噪后的二值图像的img&label匹配
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_labels = []
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(label_dir, filename), 'r') as f:
                    label = int(f.readline().split()[0])
                    if not (0 <= label < 2):  # 检查标签是否在有效范围内
                        raise ValueError(f"Label {label} out of range in file {filename}")

                    img_name_base = os.path.splitext(filename)[0]  # 获取不带扩展名的文件名基部
                    processed_img_name_base = f"processed_{img_name_base}"  # 构建处理后的图像文件名基部

                    # 支持多种图像格式
                    possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png']
                    found_image = False
                    for ext in possible_extensions:
                        img_path = os.path.join(img_dir, processed_img_name_base + ext)
                        if os.path.exists(img_path):
                            self.img_labels.append((Image.open(img_path).convert("RGB"), label))  # 使用RGB模式加载原始图像
                            found_image = True
                            break

                    if not found_image:
                        print(f"No matching image found for label file {filename}")

        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.img_labels[idx]
        if self.transform:
            img = self.transform(img)  # 应用转换
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_labels)


def get_data_loaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir,
                     batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = WheatDiseaseDataset(train_img_dir, train_label_dir, transform=transform_train)
    val_dataset = WheatDiseaseDataset(val_img_dir, val_label_dir, transform=transform_val_test)
    test_dataset = WheatDiseaseDataset(test_img_dir, test_label_dir, transform=transform_val_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader