import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from scipy import linalg
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.utils.data import SubsetRandomSampler, Subset
import torch.nn.functional as F

# congan_train.pyからLMDBDatasetをインポート
from congan_train import LMDBDataset

NUM_CLASSES = 6
BATCH_SIZE = 64
DIM = 64
# Inception V3 モデルの準備
def get_inception_model():
    model = models.inception_v3(pretrained=True)
    model.eval()
    return model.to('cuda')

# 画像の前処理
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)

# 特徴抽出
def extract_features(model, dataloader):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            batch = batch.to('cuda')
            feat = model(batch)
            features.append(feat.squeeze().cpu().numpy())
    return np.concatenate(features)

# FID スコアの計算
def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# クラスごとのFID計算
def calculate_multimodal_fid(real_features, generated_features, real_labels, generated_labels, num_classes):
    fid_scores = []
    for class_id in range(num_classes):
        real_class_features = real_features[real_labels == class_id]
        generated_class_features = generated_features[generated_labels == class_id]
        
        if len(real_class_features) > 0 and len(generated_class_features) > 0:
            fid_score = calculate_fid(real_class_features, generated_class_features)
            fid_scores.append(fid_score)
    
    return np.mean(fid_scores), np.std(fid_scores)

# クラスター化を用いたFID計算
def calculate_cluster_fid(real_features, generated_features, n_clusters=5):
    # 実際の特徴量をクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    real_clusters = kmeans.fit_predict(real_features)
    
    # 生成された特徴量を同じクラスターに割り当て
    generated_clusters = kmeans.predict(generated_features)
    
    fid_scores = []
    for cluster in range(n_clusters):
        real_cluster_features = real_features[real_clusters == cluster]
        generated_cluster_features = generated_features[generated_clusters == cluster]
        
        if len(real_cluster_features) > 0 and len(generated_cluster_features) > 0:
            fid_score = calculate_fid(real_cluster_features, generated_cluster_features)
            fid_scores.append(fid_score)
    
    return np.mean(fid_scores), np.std(fid_scores)

from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# 多次元スケーリング（MDS）を用いたFID可視化
def visualize_multimodal_fid(real_features, generated_features, real_labels, generated_labels):
    mds = MDS(n_components=2, random_state=42)
    combined_features = np.vstack([real_features, generated_features])
    mds_result = mds.fit_transform(combined_features)
    
    real_mds = mds_result[:len(real_features)]
    generated_mds = mds_result[len(real_features):]
    
    plt.figure(figsize=(10, 8))
    for class_id in range(NUM_CLASSES):
        real_class = real_mds[real_labels == class_id]
        generated_class = generated_mds[generated_labels == class_id]
        
        plt.scatter(real_class[:, 0], real_class[:, 1], alpha=0.5, label=f'Real Class {class_id}')
        plt.scatter(generated_class[:, 0], generated_class[:, 1], alpha=0.5, marker='x', label=f'Generated Class {class_id}')
    
    plt.legend()
    plt.title('MDS Visualization of Real and Generated Features')
    plt.show()

# 生成画像の取得
def get_generated_images(generator, num_images):
    generator.eval()
    images = []
    with torch.no_grad():
        for _ in tqdm(range(0, num_images, BATCH_SIZE), desc="Generating images"):
            noise = torch.randn(BATCH_SIZE, 128).to('cuda')
            fake_images = generator(noise)
            images.append(fake_images.cpu())
    return torch.cat(images)[:num_images]

# 生成画像とラベルを取得する関数
def get_generated_images_and_labels(generator, num_images):
    generator.eval()
    images = []
    labels = []
    total_generated = 0
    with torch.no_grad():
        for _ in tqdm(range(0, num_images, BATCH_SIZE), desc="Generating images"):
            current_batch_size = min(BATCH_SIZE, num_images - total_generated)
            fake_labels = np.random.choice([0,1], size=current_batch_size)
            noise = gen_rand_noise_with_label(fake_labels)
            fake_images = generator(noise)
            images.append(fake_images.cpu())
            labels.extend(fake_labels)
            total_generated += current_batch_size
            if total_generated >= num_images:
                break
    images = torch.cat(images)[:num_images]
    labels = np.array(labels)[:num_images]
    
    # 画像の形状を確認し、必要に応じて調整
    if images.dim() == 2:
        images = images.view(-1, 3, 64, 64)  # 仮定: 64x64の画像
    elif images.dim() == 3:
        images = images.unsqueeze(1)  # チャンネル次元を追加
    
    return images, labels

# 特徴量とラベルを同時に抽出する関数
def extract_features_and_labels(model, dataloader, num_images):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, total=num_images // BATCH_SIZE, desc="Extracting real features"):
            if len(features) >= num_images:
                break
            images = images.to('cuda')
            feat = model(images)
            features.append(feat.cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.concatenate(features)[:num_images], np.array(labels)[:num_images]

def gen_rand_noise_with_label(label=None):
    if label is None:
        label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    else:
        # ラベルが与えられた場合、そのサイズを使用
        batch_size = len(label)
    
    # 実際のバッチサイズを使用
    actual_batch_size = len(label)
    
    #attach label into noise
    noise = np.random.normal(0, 1, (actual_batch_size, 128))
    prefix = np.zeros((actual_batch_size, NUM_CLASSES))
    prefix[np.arange(actual_batch_size), label] = 1
    noise[np.arange(actual_batch_size), :NUM_CLASSES] = prefix[np.arange(actual_batch_size)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to('cuda')

    return noise

def get_filtered_indices(dataset, target_classes):
    indices = []
    for i in tqdm(range(len(dataset)), desc="Filtering indices", unit="image"):
        _, label = dataset[i]
        if label in target_classes:
            indices.append(i)
        if len(indices) > 20000:
            break
    return indices

# メイン処理
def main():
    # データセットとモデルのパスを設定
    real_data_path = '/content/drive/MyDrive/living_annotation_train_data_lmdb'
    model_path = '/content/drive/MyDrive/output_wgan/generator.pt'

    # Inception V3 モデルの準備
    inception_model = get_inception_model()
    print("completed load inception")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    num_images = 20000  # FID計算に使用する画像の数

    real_dataset = LMDBDataset(real_data_path, transform=transform)
    print("completed data load")

    # クラスラベルが0のインデックスを取得
    class_0_indices = get_filtered_indices(real_dataset, target_classes=[0,1])
    print("completed filter")

    # num_imagesを超えないように、必要な数のインデックスをランダムに選択
    if len(class_0_indices) > num_images:
        selected_indices = np.random.choice(class_0_indices, num_images, replace=False)
    else:
        selected_indices = class_0_indices

    # 選択されたインデックスでSubsetを作成
    subset = Subset(real_dataset, selected_indices)
    print("completed create subset")

    # DataLoaderの作成
    real_dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 生成器の準備
    generator = torch.load(model_path)

    # 実際の画像と生成画像の特徴抽出
    real_features, real_labels = extract_features_and_labels(inception_model, real_dataloader, num_images)
    generated_images, generated_labels = get_generated_images_and_labels(generator, num_images)
    print("Generated images shape:", generated_images.shape)
    generated_features = extract_features(inception_model, DataLoader(generated_images, batch_size=BATCH_SIZE))

    # FIDスコアの計算
    fid_score = calculate_fid(real_features, generated_features)
    print(f"FID Score: {fid_score}")

    # 多峰性FIDの計算
    mean_fid, std_fid = calculate_multimodal_fid(real_features, generated_features, real_labels, generated_labels, NUM_CLASSES)
    print(f"Multimodal FID - Mean: {mean_fid}, Std: {std_fid}")

    # クラスター化を用いたFIDの計算
    mean_cluster_fid, std_cluster_fid = calculate_cluster_fid(real_features, generated_features, n_clusters=NUM_CLASSES)
    print(f"Cluster FID - Mean: {mean_cluster_fid}, Std: {std_cluster_fid}")

    # 多峰性FIDの可視化
    visualize_multimodal_fid(real_features, generated_features, real_labels, generated_labels)

if __name__ == "__main__":
    main()