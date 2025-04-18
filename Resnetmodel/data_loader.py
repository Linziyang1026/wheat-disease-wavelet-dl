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
                    if not (0 <= label < 2):
                        raise ValueError(f"Label {label} out of range in file {filename}")

                    img_name_base = os.path.splitext(filename)[0]
                    processed_img_name_base = f"processed_{img_name_base}"

                    possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png']
                    found_image = False
                    for ext in possible_extensions:
                        img_path = os.path.join(img_dir, processed_img_name_base + ext)
                        if os.path.exists(img_path):
                            img = Image.open(img_path).convert("RGB")
                            self.img_labels.append((img.copy(), label))  # 使用copy()方法复制图像
                            img.close()  # 关闭图像对象
                            found_image = True
                            break

                    if not found_image:
                        print(f"No matching image found for label file {filename}")

        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.img_labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img, label = self.img_labels[idx]
        if self.transform:
            img = self.transform(img)  # 应用转换
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_labels)


def get_data_loaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir,
                     batch_size=16):  # 减少批量大小
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = WheatDiseaseDataset(train_img_dir, train_label_dir, transform=transform_train)
    val_dataset = WheatDiseaseDataset(val_img_dir, val_label_dir, transform=transform_val_test)
    test_dataset = WheatDiseaseDataset(test_img_dir, test_label_dir, transform=transform_val_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 减少num_workers
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader