import torch
import numpy as np
from sklearn.cluster import KMeans
import os

class ClusterAwareLoss:
    """
    聚类感知损失函数，用于处理正常样本分布差异大的情况。
    
    自动识别特征空间中的不同簇，并基于样本所属簇应用不同权重的MSE和余弦相似度损失。
    """
    
    def __init__(self, n_clusters=3, device='cuda', feature_levels=1, save_path=None):
        """
        初始化聚类感知损失函数。
        
        参数:
            n_clusters (int): 正常数据中要识别的簇数量
            device (str): 用于计算的设备('cuda'或'cpu')
            feature_levels (int): 要处理的特征层级数量
            save_path (str, optional): 保存/加载聚类模型的路径
        """
        self.n_clusters = n_clusters
        self.device = device
        self.feature_levels = feature_levels
        self.save_path = save_path
        
        # 初始化聚类相关组件
        self.cluster_models = {}  # 存储K-means模型
        self.feature_memory = {}  # 存储用于聚类的特征
        self.is_initialized = False
        
        # 初始化每个簇的可学习权重
        self.mse_weights = torch.ones(n_clusters).to(device)
        self.cos_weights = torch.ones(n_clusters).to(device)
        self.mse_weights.requires_grad = True
        self.cos_weights.requires_grad = True
        
        # 簇权重的优化器
        self.cluster_optimizer = torch.optim.Adam([self.mse_weights, self.cos_weights], lr=0.01)
        
        # 基础损失函数
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.cos_criterion = torch.nn.CosineSimilarity(dim=-1)
        
        # 统计信息
        self.cluster_counts = [0] * n_clusters
        
    def collect_features(self, encoder, train_loader, no_avg=False):
        """
        从训练集收集特征用于聚类。
        
        参数:
            encoder: 特征提取模型
            train_loader: 训练数据的DataLoader
            no_avg (bool): 是否跳过平均池化
        """
        print("======================特征收集阶段======================")
        
        # 初始化特征内存
        for fl in range(self.feature_levels):
            self.feature_memory[fl] = []
        
        # 收集特征
        with torch.no_grad():
            encoder.eval()
            for i, (images, _, _, _, _) in enumerate(train_loader):
                if i % 10 == 0:  # 采样以减少计算量
                    images = images.float().to(self.device)
                    features = encoder(images)
                    
                    for fl in range(self.feature_levels):
                        if no_avg:
                            input_feat = features[fl]
                        else:
                            m = torch.nn.AvgPool2d(3, 1, 1)
                            input_feat = m(features[fl])
                        
                        N, D, _, _ = input_feat.shape
                        input_flat = input_feat.permute(0, 2, 3, 1).reshape(N, -1, D)
                        
                        # 计算每个样本的特征统计量
                        for n in range(N):
                            feat_stats = torch.cat([
                                torch.mean(input_flat[n], dim=0),
                                torch.std(input_flat[n], dim=0)
                            ]).cpu().numpy().astype(np.float64)  # 确保使用double类型
                            self.feature_memory[fl].append(feat_stats)
        
        print(f"从{len(train_loader)}个批次收集了特征")
    
    def initialize_clusters(self):
        """
        基于收集的特征初始化聚类模型。
        """
        # 对每个特征层级执行聚类
        for fl in range(self.feature_levels):
            feature_array = np.array(self.feature_memory[fl], dtype=np.float64)  # 确保使用double类型
            
            # 标准化特征
            feature_mean = np.mean(feature_array, axis=0)
            feature_std = np.std(feature_array, axis=0) + 1e-8
            feature_array = (feature_array - feature_mean) / feature_std
            
            # 训练K-means
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
            kmeans.fit(feature_array)
            
            # 保存聚类模型和标准化参数
            self.cluster_models[fl] = {
                'kmeans': kmeans,
                'mean': feature_mean,
                'std': feature_std
            }
        
        # 标记为已初始化
        self.is_initialized = True
        print(f"聚类完成，已识别{self.n_clusters}个簇")
        
        # 如果提供了路径，保存聚类模型
        if self.save_path:
            self.save_cluster_models()
    
    def get_cluster_id(self, features, level):
        """
        确定批次中每个样本的簇ID。
        
        参数:
            features: 输入特征 (N, L, D)
            level: 特征层级索引
        
        返回:
            list: 每个样本的簇ID
        """
        N = features.size(0)
        cluster_ids = []
        
        for n in range(N):
            feat_stats = torch.cat([
                torch.mean(features[n], dim=0),
                torch.std(features[n], dim=0)
            ]).cpu().numpy()
            
            # 标准化特征统计量，并确保使用double类型
            norm_feat = ((feat_stats - self.cluster_models[level]['mean']) / 
                         self.cluster_models[level]['std']).astype(np.float64)
            
            # 预测簇
            cluster_id = self.cluster_models[level]['kmeans'].predict([norm_feat])[0]
            cluster_ids.append(cluster_id)
            self.cluster_counts[cluster_id] += 1
        
        return cluster_ids
    
    def calculate_loss(self, output, target, cluster_ids):
        """
        计算聚类感知加权损失。
        
        参数:
            output: 重建的特征 (N, L, D)
            target: 原始特征 (N, L, D)
            cluster_ids: 每个样本的簇ID列表
        
        返回:
            torch.Tensor: 加权损失
        """
        batch_size = output.size(0)
        weighted_mse = 0
        weighted_cos = 0
        
        for n in range(batch_size):
            c_id = cluster_ids[n]
            
            # 计算此样本的MSE损失
            sample_mse = torch.mean(self.mse_criterion(output[n:n+1], target[n:n+1]))
            
            # 计算此样本的余弦相似度损失
            sample_cos = torch.mean(1 - self.cos_criterion(output[n:n+1], target[n:n+1]))
            
            # 应用簇特定权重
            weighted_mse += self.mse_weights[c_id] * sample_mse
            weighted_cos += self.cos_weights[c_id] * sample_cos
        
        # 按批次大小归一化
        weighted_mse /= batch_size
        weighted_cos /= batch_size
        
        # 合并损失
        total_loss = weighted_mse + weighted_cos
        return total_loss
    
    def update_weights(self):
        """
        更新簇特定权重。
        """
        self.cluster_optimizer.step()
        self.cluster_optimizer.zero_grad()
        
        # 确保权重为正值
        with torch.no_grad():
            self.mse_weights.data = torch.relu(self.mse_weights.data)
            self.cos_weights.data = torch.relu(self.cos_weights.data)
            
            # 防止所有权重为零
            if torch.sum(self.mse_weights.data) < 1e-6:
                self.mse_weights.data = torch.ones_like(self.mse_weights.data)
            if torch.sum(self.cos_weights.data) < 1e-6:
                self.cos_weights.data = torch.ones_like(self.cos_weights.data)
    
    def get_weights_info(self):
        """
        获取有关簇权重和计数的信息。
        
        返回:
            dict: 关于簇的信息
        """
        return {
            'cluster_counts': self.cluster_counts,
            'mse_weights': self.mse_weights.detach().cpu().numpy(),
            'cos_weights': self.cos_weights.detach().cpu().numpy()
        }
    
    def reset_counts(self):
        """
        为新的epoch重置簇计数。
        """
        self.cluster_counts = [0] * self.n_clusters
    
    def save_cluster_models(self):
        """
        保存聚类模型和权重。
        """
        if not self.save_path:
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        save_data = {
            'cluster_models': self.cluster_models,
            'mse_weights': self.mse_weights.detach().cpu().numpy(),
            'cos_weights': self.cos_weights.detach().cpu().numpy()
        }
        torch.save(save_data, os.path.join(self.save_path, 'cluster_models.pth'))
    
    def load_cluster_models(self, path=None):
        """
        加载保存的聚类模型和权重。
        
        参数:
            path (str, optional): 加载路径。如果为None，使用self.save_path
        
        返回:
            bool: 加载是否成功
        """
        load_path = path or self.save_path
        if not load_path:
            return False
        
        model_path = os.path.join(load_path, 'cluster_models.pth')
        if not os.path.exists(model_path):
            return False
        
        saved_data = torch.load(model_path)
        self.cluster_models = saved_data['cluster_models']
        self.mse_weights.data = torch.tensor(saved_data['mse_weights']).to(self.device)
        self.cos_weights.data = torch.tensor(saved_data['cos_weights']).to(self.device)
        self.is_initialized = True
        
        return True
    
    def state_dict(self):
        """
        获取用于保存的状态字典。
        
        返回:
            dict: 状态字典
        """
        return {
            'cluster_models': self.cluster_models,
            'mse_weights': self.mse_weights.detach().cpu().numpy(),
            'cos_weights': self.cos_weights.detach().cpu().numpy()
        }
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载。
        
        参数:
            state_dict (dict): 要加载的状态字典
        """
        self.cluster_models = state_dict['cluster_models']
        self.mse_weights.data = torch.tensor(state_dict['mse_weights']).to(self.device)
        self.cos_weights.data = torch.tensor(state_dict['cos_weights']).to(self.device)
        self.is_initialized = True
