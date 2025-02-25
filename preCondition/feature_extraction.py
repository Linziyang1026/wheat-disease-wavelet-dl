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
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后一层
        self.model.eval()

        # 修改后的预处理步骤，适应单通道灰度图像
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道灰度图像
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def extract_features(self, image_path):
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img)
            img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

            with torch.no_grad():
                feature = self.model(img_tensor)

            return feature.squeeze().numpy()


def determine_optimal_k(features, max_k=10):
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


def find_representative_samples(grayscale_dir, num_clusters=5):
    extractor = FeatureExtractor()
    all_images = [os.path.join(grayscale_dir, f) for f in os.listdir(grayscale_dir) if
                  f.endswith('.jpg') or f.endswith('.png')]
    features = [extractor.extract_features(img) for img in all_images]

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
            representative_samples.append(all_images[closest_idx])

    print(f"Selected {len(representative_samples)} representative samples.")
    return representative_samples