# 加载特定文件
"""
概述
该文件实现了对抗解耦图神经网络（CADGNN），用于从因果图和相关图中分离偏差和因果表示。
通过对抗训练学习解耦表示，为下游任务提供高质量的因果特征。

类说明

1. GradientReverseLayer: 梯度反转层
   功能: 在前向传播中保持输入不变，在反向传播中反转梯度方向
   方法: forward(ctx, x, alpha), backward(ctx, grad_output)

2. SignedGCNLayer: 符号图卷积层
   功能: 处理带正负权重的图结构，分别聚合正向和负向邻居信息
   方法: __init__(in_features, out_features), forward(x, edge_index, edge_weight)

3. BiasEncoder: 偏差编码器
   功能: 使用标准GCN从相关图中提取偏差表示
   方法: __init__(input_dim, hidden_dim), forward(x, edge_index)

4. CausalEncoder: 因果编码器
   功能: 使用Signed GCN从因果图中提取因果表示
   方法: __init__(input_dim, hidden_dim, num_layers=2), forward(x, edge_index, edge_weight)

5. BiasDecoder: 偏差解码器
   功能: 通过内积重建相关图的邻接矩阵
   方法: __init__(), forward(z_b)

6. CausalDecoder: 因果解码器
   功能: 预测因果图中的正边概率、负边概率和边权重
   方法: __init__(hidden_dim), forward(z_c)

7. Discriminator: 对抗判别器
   功能: 区分偏差表示和因果表示，通过对抗训练促进解耦
   方法: __init__(hidden_dim), forward(z)

8. CADGNN: 对抗解耦GNN模型
   功能: 整合所有组件，实现完整的对抗解耦训练
   方法:
     - __init__(input_dim, hidden_dim, num_layers=2, lambda_adv=1.0, lambda_weight=0.1)
     - update_epoch(epoch)
     - get_adv_weight()
     - forward(x_corr, x_causal, edge_index_corr, edge_index_causal, edge_weight_causal)
     - compute_loss(outputs, adj_corr, adj_causal, edge_index_causal)
     - get_causal_representation(x_causal, edge_index_causal, edge_weight_causal)

辅助函数：
1. load_data(causal_adj_path, corr_adj_path, features_path, k_features=500)
   功能: 加载数据并进行特征选择
2. plot_losses(loss_history, model_config, timestamp, save_dir="CADGNN_results", train_config=None)
   功能: 绘制并保存训练损失曲线
3. set_random_seed(seed=42)
   功能: 设置随机种子以确保实验可重现性
4. train_CADGNN(model, data, num_epochs=200, lr=0.0001, weight_decay=1e-5, device='cuda', model_config=None)
   功能: 训练CADGNN模型
5. evaluate_discriminator(model, data, device='cuda')
   功能: 评估判别器性能（应接近50%准确率表示解耦成功）
6. save_model_and_representations(model, data, causal_representations, model_config, timestamp, save_dir="CADGNN_results")
   功能: 保存训练好的模型和因果表示

主要特点：
- 对抗解耦学习：分离偏差和因果表示
- 符号图卷积：处理带正负权重的因果图
- 渐进式对抗训练：逐步增加对抗损失权重
- 数值稳定性控制：防止训练过程中的梯度爆炸
- 全面的可视化支持：损失曲线和模型评估

应用场景：
- 因果表示学习
- 图结构数据的特征提取
- 偏差与因果效应的分离
- 下游任务的特征增强
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.feature_expansion import expand_features_for_causal_graph, verify_temporal_alignment
from utils.get_data_file import get_project_root,find_causal_adj_file,find_corr_adj_file,find_features_file
from utils.Causal_dataset_process import expand_scmi_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import random


class GradientReverseLayer(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SignedGCNLayer(nn.Module):
    """
    符号图卷积层 (Signed GCN Layer) with enhanced stability
    """

    def __init__(self, in_features, out_features):
        super(SignedGCNLayer, self).__init__()
        # 自连接权重
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 正向邻居权重
        self.W_plus = nn.Linear(in_features, out_features, bias=False)
        # 负向邻居权重
        self.W_minus = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features)  # 添加层归一化
        self.activation = nn.ReLU()  # 添加激活函数

    def forward(self, x, edge_index, edge_weight):
        # 添加输入归一化
        x = torch.clamp(x, min=-10, max=10)  # 限制输入范围

        # 分离正负边权重
        edge_weight_plus = torch.clamp(edge_weight, min=0, max=10)
        edge_weight_minus = torch.abs(torch.clamp(edge_weight, max=0))

        # 计算节点度数（正负分开）
        degree_plus = torch.zeros(x.size(0), device=x.device)
        degree_minus = torch.zeros(x.size(0), device=x.device)

        degree_plus.scatter_add_(0, edge_index[0], edge_weight_plus)
        degree_minus.scatter_add_(0, edge_index[0], edge_weight_minus)

        # 避免除零错误
        degree_plus = torch.clamp(degree_plus, min=1e-8)
        degree_minus = torch.clamp(degree_minus, min=1e-8)

        # 计算归一化系数
        norm_plus = torch.pow(degree_plus[edge_index[0]] * degree_plus[edge_index[1]], -0.5)
        norm_minus = torch.pow(degree_minus[edge_index[0]] * degree_minus[edge_index[1]], -0.5)

        # 限制归一化系数范围
        norm_plus = torch.clamp(norm_plus, min=0, max=10)
        norm_minus = torch.clamp(norm_minus, min=0, max=10)

        # 应用归一化
        norm_edge_weight_plus = edge_weight_plus * norm_plus
        norm_edge_weight_minus = edge_weight_minus * norm_minus

        # 初始化输出
        h = self.W(x)  # 自连接部分

        # 正向邻居聚合
        plus_messages = self.W_plus(x)[edge_index[1]] * norm_edge_weight_plus.unsqueeze(1)
        plus_aggregated = torch.zeros_like(h)
        plus_aggregated.scatter_add_(0, edge_index[0].unsqueeze(1).expand_as(plus_messages), plus_messages)
        h += plus_aggregated

        # 负向邻居聚合
        minus_messages = self.W_minus(x)[edge_index[1]] * norm_edge_weight_minus.unsqueeze(1)
        minus_aggregated = torch.zeros_like(h)
        minus_aggregated.scatter_add_(0, edge_index[0].unsqueeze(1).expand_as(minus_messages), minus_messages)
        h += minus_aggregated

        # 应用归一化和激活
        h = self.layer_norm(h)
        h = self.activation(h)

        return h


class BiasEncoder(nn.Module):
    """
    偏差编码器 g_b (标准 GCN)
    """

    def __init__(self, input_dim, hidden_dim):
        super(BiasEncoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        h = self.relu(self.gcn1(x, edge_index))
        z_b = self.gcn2(h, edge_index)
        return z_b


class CausalEncoder(nn.Module):
    """
    因果编码器 g_c (Signed GCN)
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(CausalEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(SignedGCNLayer(input_dim, hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(SignedGCNLayer(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
            if i < self.num_layers - 1:  # 除了最后一层都加激活函数
                h = self.relu(h)
        return h


class BiasDecoder(nn.Module):
    """
    偏差解码器 DEC_b
    """

    def __init__(self):
        super(BiasDecoder, self).__init__()

    def forward(self, z_b):
        # 通过内积重建邻接矩阵
        adj_recon = torch.sigmoid(torch.matmul(z_b, z_b.t()))
        return adj_recon


class CausalDecoder(nn.Module):
    """
    因果解码器 DEC_c with enhanced stability
    """

    def __init__(self, hidden_dim):
        super(CausalDecoder, self).__init__()
        # 正边概率预测参数
        self.theta_plus = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 负边概率预测参数
        self.theta_minus = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 边权重预测参数
        self.theta_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 添加缩放因子
        self.scale = 0.1

    def forward(self, z_c):
        # 限制输入范围
        z_c = torch.clamp(z_c, min=-10, max=10)

        # 预测正边概率
        p_plus = torch.sigmoid(torch.matmul(self.theta_plus(z_c), z_c.t()) * self.scale)
        # 预测负边概率
        p_minus = torch.sigmoid(torch.matmul(self.theta_minus(z_c), z_c.t()) * self.scale)
        # 预测边权重
        w_pred = torch.matmul(self.theta_weight(z_c), z_c.t()) * self.scale

        return p_plus, p_minus, w_pred


# class Discriminator(nn.Module):
#     """
#     对抗判别器 D
#     """
#
#     def __init__(self, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.linear = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, z):
#         return self.sigmoid(self.linear(z))
#
class Discriminator(nn.Module):
    """
    增强型对抗判别器
    """

    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.layers(z)


class CADGNN(nn.Module):
    """
    对抗解耦GNN模型
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, lambda_adv=1.0, lambda_weight=0.1):
        super(CADGNN, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_weight = lambda_weight
        self.current_epoch = 0

        # 编码器
        self.bias_encoder = BiasEncoder(input_dim, hidden_dim)
        self.causal_encoder = CausalEncoder(input_dim, hidden_dim, num_layers)

        # 解码器
        self.bias_decoder = BiasDecoder()
        self.causal_decoder = CausalDecoder(hidden_dim)

        # 判别器
        self.discriminator = Discriminator(hidden_dim)

        # 梯度反转层参数
        self.grl_alpha = 2.0

        # 渐进式对抗训练参数
        self.adv_start_epoch = 10  # 前10个epoch不进行对抗训练
        self.adv_warmup_epochs = 50  # 渐进式增加对抗损失权重

    def update_epoch(self, epoch):
        """更新当前epoch，用于渐进式训练"""
        self.current_epoch = epoch

    def get_adv_weight(self):
        """计算当前对抗损失权重"""
        if self.current_epoch < self.adv_start_epoch:
            return 0.0
        elif self.current_epoch < self.adv_start_epoch + self.adv_warmup_epochs:
            # 渐进式增加对抗权重
            progress = (self.current_epoch - self.adv_start_epoch) / self.adv_warmup_epochs
            return self.lambda_adv * progress
        else:
            return self.lambda_adv

    def forward(self, x_corr, x_causal, edge_index_corr, edge_index_causal, edge_weight_causal):
        """
        前向传播

        Args:
            x_corr: 相关图节点特征 (N_corr, F)
            x_causal: 因果图节点特征 (N_causal, F)
            edge_index_corr: 相关图边索引
            edge_index_causal: 因果图边索引
            edge_weight_causal: 因果图边权重
        """
        # 编码阶段
        z_b = self.bias_encoder(x_corr, edge_index_corr)
        z_c = self.causal_encoder(x_causal, edge_index_causal, edge_weight_causal)

        # 解码阶段
        adj_recon_corr = self.bias_decoder(z_b)
        p_plus, p_minus, w_pred = self.causal_decoder(z_c)

        # 对抗判别阶段（根据当前epoch决定是否启用对抗训练）
        if self.current_epoch >= self.adv_start_epoch:
            z_b_grl = GradientReverseLayer.apply(z_b, self.grl_alpha)
            z_c_grl = GradientReverseLayer.apply(z_c, self.grl_alpha)
            d_z_b = self.discriminator(z_b_grl)
            d_z_c = self.discriminator(z_c_grl)
        else:
            # 在预热期，返回固定的判别结果
            d_z_b = torch.zeros(z_b.size(0), 1, device=z_b.device)
            d_z_c = torch.ones(z_c.size(0), 1, device=z_c.device)

        return {
            'z_b': z_b,
            'z_c': z_c,
            'adj_recon_corr': adj_recon_corr,
            'p_plus': p_plus,
            'p_minus': p_minus,
            'w_pred': w_pred,
            'd_z_b': d_z_b,
            'd_z_c': d_z_c
        }

    def compute_loss(self, outputs, adj_corr, adj_causal, edge_index_causal):
        """
        计算总损失，使用更严格的损失平衡和数值稳定性控制
        """
        # 偏差分支重建损失 (MSE) - 添加数值稳定性检查
        adj_recon_corr = torch.clamp(outputs['adj_recon_corr'], min=0, max=1)
        adj_corr = torch.clamp(adj_corr, min=0, max=1)
        loss_recon_b = F.mse_loss(adj_recon_corr, adj_corr) * 0.01  # 大幅降低权重

        # 因果分支重建损失
        p_plus, p_minus, w_pred = outputs['p_plus'], outputs['p_minus'], outputs['w_pred']

        # 数值稳定性处理
        p_plus = torch.clamp(p_plus, min=1e-7, max=1 - 1e-7)
        p_minus = torch.clamp(p_minus, min=1e-7, max=1 - 1e-7)
        w_pred = torch.clamp(w_pred, min=-10, max=10)  # 限制权重预测范围

        # 提取边索引对应的预测值
        pred_w = w_pred[edge_index_causal[0], edge_index_causal[1]]
        true_w = adj_causal[edge_index_causal[0], edge_index_causal[1]]

        # 限制真实权重范围以增加稳定性
        true_w = torch.clamp(true_w, min=-10, max=10)

        # 符号分类损失
        true_plus = (true_w > 0).float()
        true_minus = (true_w < 0).float()

        pred_p_plus = p_plus[edge_index_causal[0], edge_index_causal[1]]
        pred_p_minus = p_minus[edge_index_causal[0], edge_index_causal[1]]

        # 添加数值稳定性处理
        pred_p_plus = torch.clamp(pred_p_plus, min=1e-7, max=1 - 1e-7)
        pred_p_minus = torch.clamp(pred_p_minus, min=1e-7, max=1 - 1e-7)

        loss_sign = F.binary_cross_entropy(pred_p_plus, true_plus) + \
                    F.binary_cross_entropy(pred_p_minus, true_minus)

        # 权重回归损失 - 使用Huber损失增加鲁棒性
        loss_weight = F.smooth_l1_loss(pred_w, true_w) * 0.0001  # 大幅降低权重并使用更鲁棒的损失函数

        # 重新平衡各项损失
        loss_sign = loss_sign * 0.01  # 适度降低符号损失权重

        loss_recon_c = loss_sign + self.lambda_weight * loss_weight

        # 对抗损失 - 添加更多数值稳定性处理
        if 'd_z_b' in outputs and 'd_z_c' in outputs and self.current_epoch >= self.adv_start_epoch:
            d_z_b = torch.clamp(outputs['d_z_b'], min=1e-7, max=1 - 1e-7)
            d_z_c = torch.clamp(outputs['d_z_c'], min=1e-7, max=1 - 1e-7)

            # 使用更稳定的对抗损失计算方式
            loss_adv_b = F.binary_cross_entropy(d_z_b, torch.zeros_like(d_z_b))  # 偏差表示应该被判别为0
            loss_adv_c = F.binary_cross_entropy(d_z_c, torch.ones_like(d_z_c))  # 因果表示应该被判别为1
            loss_adv = (loss_adv_b + loss_adv_c) * self.get_adv_weight()

            # 调试信息
            if self.current_epoch % 50 == 0:
                print(f"  Epoch {self.current_epoch}: adv_weight={self.get_adv_weight():.4f}, "
                      f"d_z_b_mean={d_z_b.mean().item():.4f}, d_z_c_mean={d_z_c.mean().item():.4f}")
        else:
            loss_adv = torch.tensor(0.0, device=loss_recon_b.device)

        # # 总损失 - 进一步调整权重
        # loss_total = loss_recon_b + loss_recon_c * 0.1 + loss_adv
        # 或者使用更精细的平衡
        loss_total = (
                loss_recon_b * 1 +  # 偏差重建损失权重降低
                loss_recon_c * 1 +  # 因果重建损失权重降低
                loss_adv * 0.01  # 对抗损失权重大幅降低
        )
        # 检查是否有NaN或inf
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"NaN/Inf detected in loss components:")
            print(f"  recon_b: {loss_recon_b}")
            print(f"  recon_c: {loss_recon_c}")
            print(f"  sign: {loss_sign}")
            print(f"  weight: {loss_weight}")
            print(f"  adv: {loss_adv}")
            # 返回一个大的常数损失以避免训练崩溃
            return {
                'loss_total': torch.tensor(100.0, requires_grad=True, device=loss_total.device),
                'loss_recon_b': torch.tensor(0.1, device=loss_total.device),
                'loss_recon_c': torch.tensor(10.0, device=loss_total.device),
                'loss_sign': torch.tensor(1.0, device=loss_total.device),
                'loss_weight': torch.tensor(1.0, device=loss_total.device),
                'loss_adv': torch.tensor(1.0, device=loss_total.device)
            }

        return {
            'loss_total': loss_total,
            'loss_recon_b': loss_recon_b,
            'loss_recon_c': loss_recon_c,
            'loss_sign': loss_sign,
            'loss_weight': loss_weight,
            'loss_adv': loss_adv
        }


    def get_causal_representation(self, x_causal, edge_index_causal, edge_weight_causal):
        """
        获取因果表示用于下游任务
        """
        with torch.no_grad():
            z_c = self.causal_encoder(x_causal, edge_index_causal, edge_weight_causal)
        return z_c



def load_data(causal_adj_path, corr_adj_path, features_path, k_features=500):
    """
    加载数据并进行特征选择
    """
    # 加载数据
    causal_adj = np.load(causal_adj_path)
    corr_adj = np.load(corr_adj_path)
    print(f"Original corr_adj shape: {corr_adj.shape}")  # 应为 (100, 100)

    # 使用expand_scmi_matrix扩展相关图
    corr_adj_expanded = expand_scmi_matrix(corr_adj, n_nodes=100, tau_max=3)
    print(f"Expanded corr_adj shape: {corr_adj_expanded.shape}")  # 应为 (400, 400)
    # 替换原始相关图
    corr_adj_tensor = torch.FloatTensor(corr_adj_expanded)  # (400, 400)

    # 加载特征数据 (CSV格式)
    features_df = pd.read_csv(features_path, header=0, low_memory=False)

    # 提取数值数据
    features_values = features_df.iloc[:, :].values.T

    # 处理特征形状，确保是 (N, T) 格式，其中N是节点数，T是时间步数
    try:
        features = features_values.astype(np.float32)
    except ValueError:
        features = np.zeros(features_values.shape, dtype=np.float32)
        for i in range(features_values.shape[0]):
            for j in range(features_values.shape[1]):
                try:
                    features[i, j] = float(features_values[i, j])
                except ValueError:
                    features[i, j] = 0.0

    print(f"Original features shape: {features.shape}")

    # 统一特征扩展（适用于因果图和相关图）
    features_expanded = expand_features_for_causal_graph(features, causal_nodes=400, original_nodes=100, tau_max=3)
    scaler = StandardScaler()
    features_expanded = scaler.fit_transform(features_expanded.T).T
    print(f"Features shape after expansion: {features_expanded.shape}")  # 应该是 (400, 9997)

    # 验证扩展后的特征维度
    print(f"原始特征维度: {features.shape}")  # (100, 10000)
    print(f"扩展特征维度: {features_expanded.shape}")  # (400, 9997)
    print(
        f"时间步缩减: {features.shape[1]} → {features_expanded.shape[1]} (减少{features.shape[1] - features_expanded.shape[1]}步)")


    # 特征选择 - 选择时间步（特征维度）
    if k_features < features_expanded.shape[1]:  # 按时间步选择
        # 简单的特征选择：选择前k_features个时间步
        features_selected = features_expanded[:, :k_features]
        selected_indices = np.arange(k_features)
    else:
        features_selected = features_expanded
        selected_indices = np.arange(features_expanded.shape[1])

    print(f"Selected {features_selected.shape[1]} time steps from {features_expanded.shape[1]} original time steps")

    # 为相关图和因果图准备特征（统一使用扩展后的特征）
    features_corr = features_selected  # (400, k_features)
    features_causal = features_selected  # (400, k_features)

    # 转换为torch张量
    features_corr_tensor = torch.FloatTensor(features_corr)
    features_causal_tensor = torch.FloatTensor(features_causal)
    causal_adj_tensor = torch.FloatTensor(causal_adj)
    corr_adj_tensor = torch.FloatTensor(corr_adj_expanded)

    # 构建边索引 (COO格式)
    # 相关图边索引 (400 nodes)
    corr_edges = np.where(corr_adj_expanded != 0)
    edge_index_corr = torch.LongTensor(np.array([corr_edges[0], corr_edges[1]]))
    edge_weight_corr = torch.FloatTensor(corr_adj_expanded[corr_edges])

    # 因果图边索引 (400 nodes)
    causal_edges = np.where(causal_adj != 0)
    edge_index_causal = torch.LongTensor(np.array([causal_edges[0], causal_edges[1]]))
    edge_weight_causal = torch.FloatTensor(causal_adj[causal_edges])

    return {
        'features_corr': features_corr_tensor,
        'features_causal': features_causal_tensor,
        'causal_adj': causal_adj_tensor,
        'corr_adj': corr_adj_tensor,
        'edge_index_corr': edge_index_corr,
        'edge_weight_corr': edge_weight_corr,
        'edge_index_causal': edge_index_causal,
        'edge_weight_causal': edge_weight_causal,
        'selected_feature_indices': selected_indices
    }


def plot_losses(loss_history, model_config, timestamp, save_dir="CADGNN_results", train_config=None):
    """
    绘制并保存训练损失曲线

    Args:
        loss_history: 损失历史记录字典
        model_config: 模型配置信息
        timestamp: 时间戳
        save_dir: 保存目录
    """
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # 打印损失统计信息
    print("\n=== 损失统计信息 ===")
    for loss_name, loss_values in loss_history.items():
        if len(loss_values) > 0:
            start_val = loss_values[0]
            max_val = max(loss_values)
            final_val = loss_values[-1]
            print(f"{loss_name}:")
            print(f"  起始值: {start_val:.6f}")
            print(f"  最大值: {max_val:.6f}")
            print(f"  最终值: {final_val:.6f}")
            print(f"  收敛变化: {start_val - final_val:.6f}")
            print()

    # 绘制损失曲线
    plt.figure(figsize=(16, 10))

    # 定义损失项和对应的子图位置
    loss_items = [
        ('total_loss', 'Total Loss', 1),
        ('recon_b_loss', 'Bias Reconstruction Loss', 3),
        ('recon_c_loss', 'Causal Reconstruction Loss', 6),
        ('sign_loss', 'Sign Classification Loss', 4),
        ('weight_loss', 'Weight Regression Loss', 5),
        ('adv_loss', 'Adversarial Loss', 2)
    ]

    for loss_key, title, subplot_idx in loss_items:
        if loss_key in loss_history and len(loss_history[loss_key]) > 0:
            plt.subplot(2, 3, subplot_idx)
            loss_values = loss_history[loss_key]
            plt.plot(loss_values, linewidth=1.5)

            # 添加统计信息文本
            start_val = loss_values[0]
            max_val = max(loss_values)
            final_val = loss_values[-1]

            # 在图上添加关键数值标注
            plt.text(0.02, 0.98, f'Start: {start_val:.8f}',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.90, f'Max: {max_val:.8f}',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(0.02, 0.82, f'Final: {final_val:.8f}',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.title(title, fontsize=12, pad=15)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

            # 如果有明显的尖刺，可以标记出来
            if max_val > start_val * 2 and len(loss_values) > 10:
                max_idx = loss_values.index(max_val)
                plt.annotate(f'Peak\n{max_val:.4f}',
                             xy=(max_idx, max_val),
                             xytext=(max_idx, max_val * 1.1),
                             arrowprops=dict(arrowstyle='->', color='red'),
                             fontsize=8, color='red')

    plt.tight_layout()

    config_str = f"N{model_config['N']}_T{model_config['T']}_hid{model_config['hidden_dim']}_lay{model_config['num_layers']}_ladv{model_config['lambda_adv']}_lw{model_config['lambda_weight']}"

    # 添加训练配置到文件名
    if train_config:
        train_str = f"ep{train_config.get('num_epochs', 'N/A')}_lr{train_config.get('lr', 'N/A')}_wd{train_config.get('weight_decay', 'N/A')}_grl{train_config.get('grl_alpha', 'N/A')}"
        filename = f"{config_str}_{train_str}_{timestamp}.png"
    else:
        filename = f"{config_str}_{timestamp}.png"

    filepath = save_path / filename

    # 保存图像
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to: {filepath}")

    return filepath


def set_random_seed(seed=42):
    """
    设置随机种子以确保实验可重现性
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_CADGNN(model, data, num_epochs=200, lr=0.0001, weight_decay=1e-5, device='cuda', model_config=None):
    """
    训练CADGNN模型（简化版）
    """
    # 移动数据到设备
    features_corr = data['features_corr'].to(device)
    features_causal = data['features_causal'].to(device)
    causal_adj = data['causal_adj'].to(device)
    corr_adj = data['corr_adj'].to(device)
    edge_index_corr = data['edge_index_corr'].to(device)
    edge_weight_causal = data['edge_weight_causal'].to(device)
    edge_index_causal = data['edge_index_causal'].to(device)

    # 设置模型到设备
    model = model.to(device)

    # 为不同组件使用不同的学习率
    optimizer = optim.Adam([
        {'params': model.bias_encoder.parameters(), 'lr': lr},
        {'params': model.causal_encoder.parameters(), 'lr': lr},
        {'params': model.bias_decoder.parameters(), 'lr': lr},
        {'params': model.causal_decoder.parameters(), 'lr': lr},
        {'params': model.discriminator.parameters(), 'lr': lr * 10}  # 判别器使用更高的学习率
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # 记录损失历史
    loss_history = {
        'total_loss': [],
        'recon_b_loss': [],
        'recon_c_loss': [],
        'sign_loss': [],
        'weight_loss': [],
        'adv_loss': []
    }

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        # 更新模型的当前epoch
        model.update_epoch(epoch)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(
            features_corr,
            features_causal,
            edge_index_corr,
            edge_index_causal,
            edge_weight_causal
        )

        # 计算损失
        loss_dict = model.compute_loss(
            outputs,
            corr_adj,
            causal_adj,
            edge_index_causal
        )

        # 检查损失是否为NaN或inf
        if torch.isnan(loss_dict['loss_total']) or torch.isinf(loss_dict['loss_total']):
            print(f"Epoch {epoch}: NaN or Inf loss detected, skipping...")
            continue

        # 总损失反向传播
        loss_dict['loss_total'].backward()
        # 在训练循环中添加梯度裁剪来防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_dict['loss_total'])

        # 记录损失
        loss_history['total_loss'].append(loss_dict['loss_total'].item())
        loss_history['recon_b_loss'].append(loss_dict['loss_recon_b'].item())
        loss_history['recon_c_loss'].append(loss_dict['loss_recon_c'].item())
        loss_history['sign_loss'].append(loss_dict['loss_sign'].item())
        loss_history['weight_loss'].append(loss_dict['loss_weight'].item())
        # 确保记录的是数值而不是张量
        adv_loss_val = loss_dict['loss_adv'].item() if isinstance(loss_dict['loss_adv'], torch.Tensor) else loss_dict[
            'loss_adv']
        loss_history['adv_loss'].append(adv_loss_val)

        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Total Loss: {loss_dict['loss_total'].item():.4f}, "
                  f"Recon B: {loss_dict['loss_recon_b'].item():.4f}, "
                  f"Recon C: {loss_dict['loss_recon_c'].item():.4f}, "
                  f"Sign: {loss_dict['loss_sign'].item():.4f}, "
                  f"Weight: {loss_dict['loss_weight'].item():.4f}, "
                  f"Adv: {adv_loss_val:.4f}")

    # 保存损失曲线
    if model_config is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 准备训练配置信息
        train_config = {
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'device': device,
            'grl_alpha': model.grl_alpha if hasattr(model, 'grl_alpha') else 'N/A'
        }
        plot_losses(loss_history, model_config, timestamp, train_config=train_config)

    return model, loss_history


# def evaluate_discriminator(model, data, device='cuda'):
#     """
#     评估判别器性能（应接近50%准确率表示解耦成功）
#     """
#     model.eval()
#     with torch.no_grad():
#         features_corr = data['features_corr'].to(device)
#         features_causal = data['features_causal'].to(device)
#         edge_index_corr = data['edge_index_corr'].to(device)
#         edge_index_causal = data['edge_index_causal'].to(device)
#         edge_weight_causal = data['edge_weight_causal'].to(device)
#
#         # 前向传播
#         outputs = model(
#             features_corr,
#             features_causal,
#             edge_index_corr,
#             edge_index_causal,
#             edge_weight_causal
#         )
#
#         # 计算判别器准确率
#         d_z_b = outputs['d_z_b']
#         d_z_c = outputs['d_z_c']
#
#         # 判别器对偏差表示预测为0，对因果表示预测为1
#         acc_b = (d_z_b < 0.5).float().mean()
#         acc_c = (d_z_c > 0.5).float().mean()
#         overall_acc = (acc_b + acc_c) / 2
#
#         print(f"Discriminator Accuracy - Bias: {acc_b.item():.4f}, "
#               f"Causal: {acc_c.item():.4f}, Overall: {overall_acc.item():.4f}")
#
#     return overall_acc.item()
def evaluate_discriminator(model, data, device='cuda'):
    """
    评估判别器性能（应接近50%准确率表示解耦成功）
    """
    model.eval()
    with torch.no_grad():
        features_corr = data['features_corr'].to(device)
        features_causal = data['features_causal'].to(device)
        edge_index_corr = data['edge_index_corr'].to(device)
        edge_index_causal = data['edge_index_causal'].to(device)
        edge_weight_causal = data['edge_weight_causal'].to(device)

        # 前向传播
        outputs = model(
            features_corr,
            features_causal,
            edge_index_corr,
            edge_index_causal,
            edge_weight_causal
        )

        # 计算判别器准确率
        d_z_b = outputs['d_z_b']
        d_z_c = outputs['d_z_c']

        # 判别器对偏差表示预测为0，对因果表示预测为1
        acc_b = (d_z_b < 0.5).float().mean()
        acc_c = (d_z_c > 0.5).float().mean()
        overall_acc = (acc_b + acc_c) / 2

        # 计算更详细的指标
        pred_b_mean = d_z_b.mean()
        pred_c_mean = d_z_c.mean()
        pred_diff = torch.abs(pred_b_mean - pred_c_mean)

        print(f"Discriminator Performance:")
        print(f"  Bias Accuracy (should be <0.5): {acc_b.item():.4f} (mean pred: {pred_b_mean.item():.4f})")
        print(f"  Causal Accuracy (should be >0.5): {acc_c.item():.4f} (mean pred: {pred_c_mean.item():.4f})")
        print(f"  Overall Accuracy: {overall_acc.item():.4f}")
        print(f"  Prediction Difference: {pred_diff.item():.4f}")

        # 分析判别器性能
        if torch.abs(overall_acc - 0.5) < 0.1 and pred_diff > 0.3:
            print("  Status: GOOD - Well decoupled representations")
        elif acc_b < 0.7 and acc_c > 0.3 and pred_diff > 0.2:
            print("  Status: FAIR - Reasonable decoupling")
        elif acc_b > 0.9 and acc_c > 0.9:
            print("  Status: POOR - Both representations classified as causal")
        elif acc_b < 0.1 and acc_c < 0.1:
            print("  Status: POOR - Both representations classified as bias")
        else:
            print("  Status: UNCLEAR - Further analysis needed")

    model.train()  # 恢复训练模式
    return overall_acc.item()


def save_model_and_representations(model, data, causal_representations, model_config, timestamp,
                                   save_dir="CADGNN_results"):
    """
    保存训练好的模型和因果表示

    Args:
        model: 训练好的CADGNN模型
        data: 训练数据
        causal_representations: 因果表示
        model_config: 模型配置
        timestamp: 时间戳
        save_dir: 保存目录
    """
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # 构建文件名前缀
    config_str = f"N{model_config['N']}_T{model_config['T']}_hid{model_config['hidden_dim']}_lay{model_config['num_layers']}_ladv{model_config['lambda_adv']}_lw{model_config['lambda_weight']}"
    filename_prefix = f"{config_str}_{timestamp}"

    # 1. 保存模型权重
    model_save_path = save_path / f"{filename_prefix}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'timestamp': timestamp
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    # 2. 保存因果表示
    causal_repr_save_path = save_path / f"{filename_prefix}_causal_representations.npy"
    np.save(causal_repr_save_path, causal_representations.cpu().numpy())
    print(f"Causal representations saved to: {causal_repr_save_path}")

    # 3. 保存训练数据信息（可选）
    data_info_save_path = save_path / f"{filename_prefix}_data_info.pt"
    torch.save({
        'edge_index_causal': data['edge_index_causal'],
        'edge_weight_causal': data['edge_weight_causal'],
        'edge_index_corr': data['edge_index_corr'],
        'selected_feature_indices': data['selected_feature_indices']
    }, data_info_save_path)
    print(f"Data info saved to: {data_info_save_path}")

    return {
        'model_path': model_save_path,
        'causal_repr_path': causal_repr_save_path,
        'data_info_path': data_info_save_path
    }


# 使用示例
if __name__ == "__main__":
    set_random_seed(42)

    # 直接定义文件路径
    project_root = Path(__file__).parent.parent  # Module的父目录即为项目根目录
    print(f"Project root: {project_root}")

    # 直接指定文件路径
    causal_adj_path = project_root / "Combined_Results" / "combined_causal_matrix_N100_T10000_t3_pcmci_alpha0p0001_scmi_delta20.0_k10_bins5_threshold1e-05_pcmci_w0.5_scmi_w0.5.npy"
    corr_adj_path = project_root / "Correlation_Graph" / "corr_adjacency_matrix.npy"
    features_path = project_root / "dataset" / "simulation_hubs_core" / "N100_T10000_Hubs15_Viz_Final3" / "Env_A_timeseries.csv"

    # 配置参数
    params = {
        'N': 100,
        'T': 10000,
        't': 3
    }


    # 建议的 hidden_dim 设置规则
    def suggest_hidden_dim(input_dim):
        # 通常 hidden_dim 应该是 input_dim 的 1/4 到 1/2
        if input_dim <= 1000:
            return min(512, input_dim)
        elif input_dim <= 5000:
            return min(256, input_dim // 4)
        else:
            return min(128, input_dim // 8)


    # 建议的 num_layers 设置规则
    def suggest_num_layers(input_dim):
        # 根据输入维度调整层数
        if input_dim <= 1000:
            return 3
        elif input_dim <= 5000:
            return 4
        else:
            return 5

    input_dim = 8000  # 特征维度（经过特征选择后）
    hidden_dim = suggest_hidden_dim(input_dim)
    num_layers = suggest_num_layers(input_dim)
    lambda_adv = 0.2
    # 对抗损失（Adversarial Loss）在总损失函数中的权重系数
    # ↑更强的解耦效果，但可能导致训练不稳定
    # ↓训练稳定但解耦效果可能不足
    lambda_weight = 0.02
    # 因果分支中权重回归损失在因果重建损失中的权重
    # ↑ 更精确的边权重预测，但可能忽略符号分类
    # ↓ 更关注正负边判断，权重预测精度下降
    k_features = input_dim  # 选择的特征数量

    try:
        # 检查文件是否存在
        print(f"Checking causal adjacency matrix file: {causal_adj_path}")
        print(f"File exists: {causal_adj_path.exists()}")

        print(f"Checking correlation adjacency matrix file: {corr_adj_path}")
        print(f"File exists: {corr_adj_path.exists()}")

        print(f"Checking features file: {features_path}")
        print(f"File exists: {features_path.exists()}")

        # 检查CUDA可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 加载和预处理数据
        print("Loading data...")
        data = load_data(
            str(causal_adj_path),
            str(corr_adj_path),
            str(features_path),
            k_features=k_features
        )

        # 创建模型
        print("Creating model...")
        model = CADGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lambda_adv=lambda_adv,
            lambda_weight=lambda_weight
        )

        # 保存模型配置用于文件命名
        model_config = {
            'N': params['N'],
            'T': params['T'],
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lambda_adv': lambda_adv,
            'lambda_weight': lambda_weight,
            'input_dim': input_dim,
            'k_features': k_features
        }

        # 训练模型
        print("Training model...")
        trained_model, loss_history = train_CADGNN(
            model,
            data,
            num_epochs=1500,
            lr=0.0001,
            device=device,
            model_config=model_config
        )

        # 评估判别器
        print("Evaluating discriminator...")
        disc_acc = evaluate_discriminator(trained_model, data, device)

        # 获取因果表示
        print("Getting causal representations...")
        features_causal = data['features_causal'].to(device)
        edge_index_causal = data['edge_index_causal'].to(device)
        edge_weight_causal = data['edge_weight_causal'].to(device)

        causal_representations = trained_model.get_causal_representation(
            features_causal,
            edge_index_causal,
            edge_weight_causal
        )

        print(f"Causal representations shape: {causal_representations.shape}")
        print("Training completed!")

        # 保存模型和结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = save_model_and_representations(
            trained_model,
            data,
            causal_representations,
            model_config,
            timestamp
        )

        print("Training completed and results saved!")
        print("Saved files:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")


    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the data files exist in the expected directories.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
