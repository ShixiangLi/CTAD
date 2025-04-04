import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 数据集配置
dataset = 'SWAT'
phase = 'test'
window_size = 100

# 加载数据
features = np.load(f"dataset/{dataset}/{dataset}_{phase}.npy")
num_windows = len(features) // window_size

# 计算全局最大最小值用于归一化
global_min = features.min()
global_max = features.max()

# 创建保存目录
def make_dirs(*dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

# 可视化函数
def save_image(data, output_path, cmap='hot', dpi=60):
    plt.figure(figsize=(5, 5))
    plt.imshow(data, aspect='auto', cmap=cmap, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_mask(data, output_path, cmap='hot', dpi=60):
    plt.figure(figsize=(5, 5))
    plt.imshow(data, aspect='auto', cmap=cmap, origin='lower', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

# 测试阶段带标签处理
if phase == 'test':
    labels = np.load(f"dataset/{dataset}/{dataset}_{phase}_label.npy")

    normal_output_dir = f'dataset/output_images/{dataset}/time_series/{phase}/good'
    anomaly_output_dir = f'dataset/output_images/{dataset}/time_series/{phase}/anomaly'
    res_dir = f'dataset/output_images/{dataset}/time_series/ground_truth/anomaly'
    make_dirs(normal_output_dir, anomaly_output_dir, res_dir)

    for i in tqdm(range(num_windows)):
        # 窗口切片
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        feature_window = features[start_idx:end_idx]
        label_window = labels[start_idx:end_idx]

        # 全局归一化
        feature_window_norm = (feature_window - global_min) / (global_max - global_min)

        # 保存特征图
        output_dir = normal_output_dir if label_window.max() == 0 else anomaly_output_dir
        save_image(feature_window_norm.T, os.path.join(output_dir, f'window_{i + 1}.png'))

        # 生成并保存掩码图
        label_fig = np.expand_dims(label_window, axis=0).repeat(feature_window.shape[1], axis=0)
        if label_window.max() == 1:
            save_mask(label_fig, os.path.join(res_dir, f'window_{i + 1}_mask.png'), cmap='gray')

# 训练阶段无标签，仅保存特征图
else:
    normal_output_dir = f'dataset/output_images/{dataset}/time_series/{phase}/good'
    make_dirs(normal_output_dir)

    for i in tqdm(range(num_windows)):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        feature_window = features[start_idx:end_idx]
        feature_window_norm = (feature_window - global_min) / (global_max - global_min)

        save_image(feature_window_norm.T, os.path.join(normal_output_dir, f'window_{i + 1}.png'))

print("可视化完成 ✅")
