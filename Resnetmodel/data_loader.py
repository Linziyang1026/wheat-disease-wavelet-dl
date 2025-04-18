import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from config import Config

class Cutout(object):

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class GridMask(object):
    """GridMask augmentation."""
    def __init__(self, d1=96, d2=96, rotate=1, ratio=0.6, mode=0, prob=0.5):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        d = np.random.randint(self.d1, self.d2 + 1)
        m = np.random.randint(3, d // 2)
        mask = create_gridmask(mask, d, m, self.rotate, self.ratio, self.mode)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def create_gridmask(mask, d, m, rotate, ratio, mode):
    mask = mask.clone()
    h, w = mask.shape
    y = np.random.randint(0, h - d + 1)
    x = np.random.randint(0, w - d + 1)
    y1 = y
    y2 = y + d
    x1 = x
    x2 = x + d
    mask[y1:y2, x1:x2] = 0
    if mode == 1:
        mask = TF.rotate(mask, rotate)
    mask = mask.unsqueeze(0)
    return mask

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
                            self.img_labels.append((img.copy(), label))
                            img.close()
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

def get_data_loaders(train_img_dir=Config.train_img_dir, train_label_dir=Config.train_label_dir,
                     val_img_dir=Config.val_img_dir, val_label_dir=Config.val_label_dir,
                     test_img_dir=Config.test_img_dir, test_label_dir=Config.test_label_dir,
                     batch_size=Config.batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(n_holes=1, length=16),
        GridMask(d1=96, d2=96, rotate=1, ratio=0.6, mode=0, prob=0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader