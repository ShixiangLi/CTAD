import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shutil  # 用于文件操作（复制文件）

def load_images_from_folder(folder_path, img_size=(64, 64)):
    """Load images from a folder and preprocess (resize and flatten into vectors)."""
    images = []
    filenames = []  # 存储图像文件名，以便后续保存时使用
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, img_size)  # Resize to fixed size
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB mode
                images.append(img)
                filenames.append(filename)  # 保存文件名
            except Exception as e:
                print(f"Skipping file {img_path}, due to error: {e}")
    return images, filenames

def extract_features(images):
    """Flatten images into feature vectors."""
    features = []
    for img in images:
        features.append(img.flatten())
    return np.array(features)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os

def find_optimal_clusters(data, dataset, max_k=10):
    """使用肘部法则（Elbow Method）找到最佳聚类数"""
    from sklearn.cluster import KMeans

    wcss = []
    for n in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # 画出 WCSS 曲线
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='#007acc', linewidth=2)
    plt.xticks(range(1, max_k + 1))
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('WCSS', fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)

    elbow_plot_path = os.path.join("./cls_results/", dataset, "optimal_k_elbow_plot.png")
    plt.savefig(elbow_plot_path, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    # 自动检测"拐点"
    reductions = np.diff(wcss)
    changes = np.diff(reductions)
    optimal_k = np.argmax(changes) + 2  # 由于求二阶差分，需要加2

    print(f"Optimal number of clusters determined by Elbow Method: {optimal_k}")
    return optimal_k if optimal_k > 2 else 3

def visualize_with_lda_and_centers(features, labels, cluster_centers, original_images, n_clusters, save_path):
    """使用 LDA 可视化聚类结果，并改进图像的展示效果"""
    
    lda = LDA(n_components=2)
    reduced_features = lda.fit_transform(features, labels)
    cluster_centers_reduced = lda.transform(cluster_centers)

    # 颜色映射，确保不同簇颜色区分明显
    colors = plt.cm.get_cmap('tab10', n_clusters)

    plt.figure(figsize=(6, 4), dpi=150)
    
    # 画出不同类别的点
    for cluster_idx in range(n_clusters):
        cluster_points = reduced_features[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'Cluster {cluster_idx}', alpha=0.7, edgecolors='k', 
                    color=colors(cluster_idx), s=50)

    # 画出聚类中心
    plt.scatter(cluster_centers_reduced[:, 0], cluster_centers_reduced[:, 1], 
                color='black', marker='X', label='Cluster Centers', s=150, edgecolors='white')

    # plt.title("LDA Visualization of Clusters", fontsize=14, fontweight='bold')
    plt.xlabel("LD1", fontsize=12)
    plt.ylabel("LD2", fontsize=12)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)

    lda_plot_path = os.path.join(save_path, "lda_cluster_plot.png")
    plt.savefig(lda_plot_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"LDA cluster plot saved to: {lda_plot_path}")

    # 展示聚类中心的原始图像
    fig, axes = plt.subplots(1, n_clusters, figsize=(12, 6), dpi=150)
    for i, ax in enumerate(axes):
        ax.imshow(original_images[i])
        ax.axis('off')
        ax.set_title(f'Cluster {i}', fontsize=10, fontweight='bold')
    # plt.suptitle("Cluster Centers' Original Images", fontsize=14, fontweight='bold')

    cluster_centers_img_path = os.path.join(save_path, "cluster_centers_images.png")
    plt.savefig(cluster_centers_img_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"Cluster centers' images saved to: {cluster_centers_img_path}")


def save_images_by_cluster(filenames, labels, original_image_paths, save_path):
    """Copy images to subdirectories based on their cluster labels."""
    for cluster_idx in np.unique(labels):
        cluster_dir = os.path.join(save_path, f"cluster_{cluster_idx}")
        os.makedirs(cluster_dir, exist_ok=True)  # Create the cluster folder if it doesn't exist

        # Find the indices of the images belonging to the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        
        # Copy each image to the corresponding cluster directory
        for idx in cluster_indices:
            original_image_path = original_image_paths[idx]
            filename = filenames[idx]
            destination_path = os.path.join(cluster_dir, filename)

            # Copy image to the cluster directory
            shutil.copy(original_image_path, destination_path)

        print(f"Copied {len(cluster_indices)} images to {cluster_dir}")

def save_cluster_labels_to_txt(filenames, labels, original_image_folder):
    """Save the cluster labels to a txt file, ensuring the order corresponds to sorted filenames."""
    
    # Zip filenames with labels and sort by filenames
    sorted_filenames_labels = sorted(zip(filenames, labels), key=lambda x: x[0].lower())
    
    # Filepath to save the labels
    labels_path = os.path.join(original_image_folder, 'cluster_labels.txt')
    
    try:
        with open(labels_path, 'w') as f:
            for filename, label in sorted_filenames_labels:
                f.write(f"{label}\n")  # Write each filename and label pair
        print(f"Cluster labels saved to: {labels_path}")
    except Exception as e:
        print(f"Error saving cluster labels: {e}")


def main(dataset='smd'):
    # Set the folder path and image size
    folder_path = os.path.join("datasets/", dataset, "time_series", "train", "good")
    img_size = (64, 64)

    # Specify a path to save the results
    save_path = os.path.join("cls_results/", dataset)
    os.makedirs(save_path, exist_ok=True)

    # Load images from folder
    images, filenames = load_images_from_folder(folder_path, img_size)
    print(f"Loaded {len(images)} images.")

    # Get the full paths of the original images for later use in copying
    original_image_paths = [os.path.join(folder_path, filename) for filename in filenames]

    # Extract features
    features = extract_features(images)
    print(f"Image feature shape: {features.shape}")

    # Find the optimal number of clusters using the elbow method
    optimal_k = find_optimal_clusters(features, dataset=dataset, max_k=10)

    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(features)

    # Get cluster labels and cluster centers
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Find the closest image to each cluster center
    original_cluster_center_images = []
    for center_idx, cluster_center in enumerate(cluster_centers):
        distances = np.linalg.norm(features - cluster_center, axis=1)  # Compute L2 norm
        closest_image_idx = np.argmin(distances)  # Get the index of closest image
        original_cluster_center_images.append(images[closest_image_idx])  # Append the closest image

    # Visualize LDA results and cluster centers
    visualize_with_lda_and_centers(features, labels, cluster_centers, 
                                   original_cluster_center_images, optimal_k, save_path)

    # Copy images by their cluster labels
    save_images_by_cluster(filenames, labels, original_image_paths, save_path)

    # Save the cluster labels as a text file
    # save_cluster_labels_to_txt(filenames, labels, folder_path)

if __name__ == "__main__":
    main(dataset='swat')
