import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class FeatureExtractor:
    def __init__(self):
        # 加载预训练的ResNet模型，并移除最后一层全连接层
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后一层
        self.model.eval()

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 统一调整大小到224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        """
        提取单张图像的特征向量。
        """
        img = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
        img_tensor = self.preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

        with torch.no_grad():
            feature = self.model(img_tensor)

        return feature.squeeze().numpy()


def determine_optimal_k(features, max_k=10):
    """
    使用肘部法决定最佳的k值。
    """
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        sse.append(kmeans.inertia_)

    # 绘制肘部曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, 'bx-')
    plt.xlabel('簇的数量 (k)')
    plt.ylabel('SSE')
    plt.title('肘部法显示的最佳k值')
    plt.show()


def find_representative_samples(grayscale_dir, num_clusters=5):
    """
    使用K-means聚类找到数据集中的代表性样本。
    """
    extractor = FeatureExtractor()

    all_images = [os.path.join(grayscale_dir, f) for f in os.listdir(grayscale_dir) if
                  f.endswith('.jpg') or f.endswith('.png')]
    features = [extractor.extract_features(img).flatten() for img in all_images]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    representative_samples = []
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) > 0:
            closest_idx = cluster_indices[np.argmin([np.linalg.norm(feature - kmeans.cluster_centers_[i]) for feature in
                                                     np.array(features)[cluster_indices]])]
            representative_samples.append(all_images[closest_idx])

    print(f"Selected {len(representative_samples)} representative samples.")
    return representative_samples