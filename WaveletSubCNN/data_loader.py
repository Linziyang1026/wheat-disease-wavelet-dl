from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class WheatDiseaseDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_labels = []
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(label_dir, filename), 'r') as f:
                    label = int(f.readline().split()[0])
                    if not (0 <= label < 2):  # 检查标签是否在有效范围内
                        raise ValueError(f"Label {label} out of range in file {filename}")
                    img_name_base = filename.replace('.txt', '')
                    imgs = []
                    for suffix in ['_approx.png', '_level1_d.png', '_level1_h.png', '_level1_v.png']:
                        img_name = img_name_base + suffix
                        img_path = os.path.join(img_dir, img_name)
                        if os.path.exists(img_path):
                            imgs.append(Image.open(img_path).convert("L"))  # 使用灰度模式加载图像
                    if len(imgs) == 4:  # 确保所有子带都存在
                        self.img_labels.append((imgs, label))
        self.transform = transform

    def __getitem__(self, idx):
        imgs, label = self.img_labels[idx]
        # 将4个单通道图像合并为一个4通道图像
        transformed_imgs = [self.transform(img) for img in imgs]
        image = torch.cat(transformed_imgs, dim=0)  # 在通道维度上拼接
        return image, torch.tensor(label, dtype=torch.long)

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