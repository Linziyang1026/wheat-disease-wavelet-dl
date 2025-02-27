from PIL import Image
import os
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    使用预训练的ResNet-50模型提取图像特征。
    """
    def __init__(self):
        """
        初始化FeatureExtractor类。
        移除ResNet-50模型的最后一层，使用剩余部分提取图像特征。
        """
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后一层
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),  # 将较短的一边缩放到256，保持长宽比
            transforms.CenterCrop(224),  # 从中心裁剪出224x224的区域
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])

    def extract_features(self, image_path):
        """
        从给定的图像路径中提取图像特征。

        参数:
        image_path (str): 图像文件的路径。

        返回:
        ndarray: 提取的图像特征。
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

        with torch.no_grad():
            feature = self.model(img_tensor)

        return feature.squeeze().numpy()


def determine_optimal_k(features, max_k=10):
    """
    使用SSE和轮廓系数确定最佳的KMeans聚类数。

    参数:
    features (list of ndarray): 图像特征列表。
    max_k (int): 考虑的最大聚类数，默认为10。

    返回:
    int: 最佳的聚类数。
    """
    sse = []
    silhouette_scores = []

    for k in range(2, max_k + 1):  # 从2开始是因为当k=1时无法计算轮廓系数
        kmeans = KMeans(n_clusters=k)
        preds = kmeans.fit_predict(features)

        # 计算SSE
        sse.append(kmeans.inertia_)

        # 计算轮廓系数
        score = silhouette_score(features, preds)
        silhouette_scores.append(score)

        print(f'For k={k}, SSE: {kmeans.inertia_:.2f}, Silhouette Score: {score:.3f}')

    # 找出具有最高轮廓系数的k值
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 因为k是从2开始的

    # 可视化结果
    plt.figure(figsize=(14, 7))

    # 绘制SSE图
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), sse)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')

    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores)
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    return optimal_k


def find_representative_samples(image_paths, features):
    """
    寻找给定图像路径列表中最具有代表性的样本图像。

    参数:
    image_paths (list of str): 包含图像文件路径的列表。
    features (list of ndarray): 对应于image_paths的特征列表。

    返回:
    list: 最具代表性的样本图像的路径列表。
    """
    # 自动确定最佳k值
    num_clusters = determine_optimal_k(features)
    print(f"Automatically determined optimal number of clusters: {num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    representative_samples = []
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) > 0:
            closest_idx = cluster_indices[np.argmin([np.linalg.norm(feature - kmeans.cluster_centers_[i]) for feature in
                                                     np.array(features)[cluster_indices]])]
            representative_samples.append(image_paths[closest_idx])

    print(f"Selected {len(representative_samples)} representative samples.")
    return representative_samples