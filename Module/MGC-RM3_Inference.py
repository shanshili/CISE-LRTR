"""
MGC-RM3_Inference.py - MGC-RM3模型推理与可视化 (最终修正版)
功能：
1. 加载训练好的 MGC-RM3 模型 (支持 PyTorch 2.6+ 安全加载)。
2. 自动清洗数据中的孤立节点，彻底解决 NaN/Divide-by-zero 错误。
3. 计算模型预测的相似度分数。
4. 基于梯度 (Saliency Map) 计算节点贡献度，找出关键节点。
5. 生成可视化图表并保存。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import networkx as nx
from datetime import datetime
import math
import argparse

# ==================== 1. 路径与环境设置 ====================
# 添加项目根目录以导入 Module
sys.path.append(str(Path(__file__).parent.parent))

# 尝试导入可视化模块
try:
    from Module.PageRank import create_causal_graph_from_matrices, data_color_graph

    HAS_PAGERANK_MODULE = True
except ImportError:
    print("Warning: Could not import visualization modules from Module.PageRank. Advanced plotting skipped.")
    HAS_PAGERANK_MODULE = False

# 绘图字体配置
config = {
    "font.family": 'Times New Roman',
    "font.size": 10.5,
    'axes.unicode_minus': False
}
rcParams.update(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 2. 模型类定义 (必须保留以支持反序列化) ====================

class MFC_RMF(nn.Module):
    def __init__(self, cadgnn_model, args):
        super(MFC_RMF, self).__init__()
        torch.manual_seed(1234)
        self.args = args
        self.cadgnn = cadgnn_model
        self.readout_rep = readout(args.cadgnn_hidden_dim, num_heads=4)
        self.readout_match = readout(args.match_size, num_heads=4)
        self.match = matching(args.cadgnn_hidden_dim, args.match_size)
        self.mlp = MLP(args.match_size)

    def get_causal_representation(self, x, edge_index, edge_weight):
        return self.cadgnn.causal_encoder(x, edge_index, edge_weight)

    def mutual_representation(self, h_1, h_2):
        cos_sim = torch.cosine_similarity(h_1.unsqueeze(1), h_2.unsqueeze(0), dim=-1)
        cos_ceo = cos_sim / torch.sum(cos_sim, dim=0)
        return torch.mm(cos_ceo, h_2)

    def forward(self, graph_pair):
        feature_1 = graph_pair.x_s
        feature_2 = graph_pair.x_t

        # 标准化
        feature_1 = (feature_1 - feature_1.min()) / (feature_1.max() - feature_1.min() + 1e-8)
        feature_2 = (feature_2 - feature_2.min()) / (feature_2.max() - feature_2.min() + 1e-8)

        # 获取表示
        h_1_self = self.get_causal_representation(feature_1, graph_pair.edge_index_s, graph_pair.edge_weight_s)
        h_2_self = self.get_causal_representation(feature_2, graph_pair.edge_index_t, graph_pair.edge_weight_t)

        h_1_self_global = self.readout_rep(h_1_self)
        h_2_self_global = self.readout_rep(h_2_self)

        h_1_mutual = self.mutual_representation(h_1_self, h_2_self)
        h_2_mutual = self.mutual_representation(h_2_self, h_1_self)

        h_1_mutual_global = self.readout_rep(h_1_mutual)
        h_2_mutual_global = self.readout_rep(h_2_mutual)

        # 匹配层
        miu_1 = self.match(h_1_self, h_1_mutual)
        miu_2 = self.match(h_2_self, h_2_mutual)
        miu_1_global = self.readout_rep(miu_1)
        miu_2_global = self.readout_rep(miu_2)

        phi_1 = self.match(h_1_self, h_1_mutual_global)
        phi_2 = self.match(h_2_self, h_2_mutual_global)
        phi_1_global = self.readout_rep(phi_1)
        phi_2_global = self.readout_rep(phi_2)

        psi_1 = self.match(h_1_self_global, h_1_mutual)
        psi_2 = self.match(h_2_self_global, h_2_mutual)
        psi_1_global = self.readout_rep(psi_1)
        psi_2_global = self.readout_rep(psi_2)

        omega_1 = self.match(h_1_self_global, h_1_mutual_global)
        omega_2 = self.match(h_2_self_global, h_2_mutual_global)

        # 融合与预测
        z_1 = torch.cat([miu_1_global, phi_1_global, psi_1_global, omega_1], dim=1).t()
        z_2 = torch.cat([miu_2_global, phi_2_global, psi_2_global, omega_2], dim=1).t()
        z = torch.cat([z_1, z_2], dim=0)
        z = self.mlp(z.t())

        return z, None, None


class readout(nn.Module):
    def __init__(self, output_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_size // num_heads
        self.W_q = nn.Parameter(torch.Tensor(output_size, output_size))
        self.W_k = nn.Parameter(torch.Tensor(output_size, output_size))
        self.W_v = nn.Parameter(torch.Tensor(output_size, output_size))
        self.W_o = nn.Parameter(torch.Tensor(output_size, output_size))
        self.weight = nn.Parameter(torch.Tensor(output_size, output_size))
        self.norm = nn.LayerNorm(output_size)
        for w in [self.W_q, self.W_k, self.W_v, self.W_o, self.weight]:
            if w.dim() > 1: nn.init.xavier_uniform_(w)

    def forward(self, h):
        Q = torch.matmul(h, self.W_q)
        K = torch.matmul(h, self.W_k)
        V = torch.matmul(h, self.W_v)
        Q = Q.view(h.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(h.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(h.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        h_attn = torch.matmul(attn_weights, V).transpose(0, 1).contiguous().view(h.shape[0], -1)
        h_attn = torch.matmul(h_attn, self.W_o)
        scores = torch.mm(h, self.weight.t()).sum(dim=1)
        sigmoid_scores = torch.sigmoid(scores)
        h_global = sigmoid_scores.unsqueeze(-1) * h
        h_combined = h_attn + h_global + h
        h_combined = self.norm(h_combined)
        return h_combined.sum(dim=0).unsqueeze(0)


class matching(nn.Module):
    def __init__(self, output_size, match_size):
        super(matching, self).__init__()
        self.output_size = output_size
        self.match_size = match_size
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.match_size))
        nn.init.xavier_uniform_(self.weight.data)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_1, h_2):
        h_1 = h_1.unsqueeze(1)
        h_2 = h_2.unsqueeze(1)
        weight = self.weight.unsqueeze(0)
        h_1_w = torch.mul(h_1, weight)
        h_2_w = torch.mul(h_2, weight)
        return torch.cosine_similarity(h_1_w, h_2_w, dim=-1) * F.softplus(self.temperature)


class MLP(nn.Module):
    def __init__(self, match_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(match_size * 8, match_size * 2)
        self.linear2 = nn.Linear(match_size * 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(match_size * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear1(z)
        z = self.layer_norm1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return self.sigmoid(z)


class GraphPair:
    def __init__(self, edge_index_s, edge_index_t, x_s, x_t, edge_weight_s, edge_weight_t, label=None):
        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.x_s = x_s
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.label = label


# ==================== 3. 核心辅助函数 ====================

def clean_weight_matrix(matrix):
    """
    【关键修复】清洗权重矩阵，防止全零行导致 Divide by zero (NaN)。
    对孤立节点（全零行）添加微小的自环。
    """
    row_sums = matrix.sum(axis=1)
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        print(f"  [Data Cleaning] Found {len(zero_rows)} isolated nodes. Adding epsilon self-loops.")
        for idx in zero_rows:
            matrix[idx, idx] = 1e-6
    return matrix


def select_features(features, target_dim=8000):
    if features.shape[1] > target_dim:
        return features[:, :target_dim]
    else:
        return features


def load_trained_model(model_path):
    """加载模型对象 (处理 PyTorch 2.6+ 安全限制)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        # 设置 weights_only=False 以允许加载自定义类
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        if "weights_only" in str(e):
            print("Please downgrade PyTorch or enable weights_only=False explicitly.")
        sys.exit(1)


def compute_node_importance(model, graph_pair):
    """计算节点梯度重要性 (Saliency Map)"""
    model.eval()

    # 开启特征梯度
    if not graph_pair.x_s.requires_grad:
        graph_pair.x_s.requires_grad = True

    # 前向传播
    z, _, _ = model(graph_pair)

    # 反向传播计算对输入的梯度
    model.zero_grad()
    z.backward()

    # 获取梯度 (绝对值 -> 平均特征维度)
    node_gradients = graph_pair.x_s.grad.data.abs().mean(dim=1)

    # 归一化
    grad_sum = node_gradients.sum()
    if grad_sum == 0:
        node_weights = torch.ones_like(node_gradients) / len(node_gradients)
    else:
        node_weights = node_gradients / grad_sum

    return z.item(), node_weights.cpu().numpy()


def aggregate_weights(contribution_weights, original_node_count=100):
    """聚合时序扩展节点权重"""
    aggregated = np.zeros(original_node_count)
    multiplier = len(contribution_weights) // original_node_count

    for i in range(original_node_count):
        indices = [i + j * original_node_count for j in range(multiplier)]
        aggregated[i] = np.sum(contribution_weights[indices])

    agg_sum = np.sum(aggregated)
    if agg_sum > 0:
        aggregated = aggregated / agg_sum
    else:
        aggregated = np.ones(original_node_count) / original_node_count

    return aggregated


# ==================== 4. 主程序 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpn", type=int, default=62, help="Graph Pair Index")
    parser.add_argument("--model_path", type=str,
                        default="./MGC-RM_result/model_save/model_GP_62_n_400_d_9997_e_150_l_1e-05_20250907_153259.pth",
                        help="Path to trained model")
    args_cmd = parser.parse_args()

    Gpn = args_cmd.gpn
    node_num = 400
    original_node = 100
    data_num = 9997
    select_node_count = 15
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== Starting Inference for Graph Pair {Gpn} ===")
    print(f"Using device: {device}")

    # 1. 加载数据
    print("1. Loading Data...")
    try:
        w_ori_path = f'./perturbation_data/origin_data_weight_ori_node_{node_num}_data_{data_num}.npz'
        f_ori_path = f'./perturbation_data/origin_data_fea_ori_node_{node_num}_data_{data_num}.npz'
        weight_ori = np.load(w_ori_path, allow_pickle=True)['arr_0']
        fea_ori = np.load(f_ori_path, allow_pickle=True)['arr_0']

        w_pert_path = f'./perturbation_data/perturbed_graph_weight_node_{node_num}_data_{data_num}.npz'
        f_pert_path = f'./perturbation_data/perturbed_fea_node_{node_num}_data_{data_num}.npz'
        perturbed_w = np.load(w_pert_path)['arr_0']
        perturbed_fe = np.load(f_pert_path)['arr_0']

        pos_path = os.path.join(str(Path(__file__).parent), '..',
                                'dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final/node_positions.csv')
        if os.path.exists(pos_path):
            positions_df = pd.read_csv(pos_path)
            positions = positions_df[['x', 'y']].values
            lon, lat = positions[:, 0], positions[:, 1]
        else:
            print("  Warning: Position file not found. Using random positions.")
            positions = np.random.rand(original_node, 2)
            lon, lat = positions[:, 0], positions[:, 1]

    except FileNotFoundError as e:
        print(f"  Error loading data: {e}")
        return

    # 2. 清洗数据 (彻底解决 NaN 问题)
    print("2. Cleaning Data (fixing isolated nodes)...")
    weight_ori = clean_weight_matrix(weight_ori)
    perturbed_graph_w = clean_weight_matrix(perturbed_w[Gpn])

    # 3. 构建图对象
    print("3. Constructing GraphPair...")
    G_o = nx.from_numpy_array(weight_ori, create_using=nx.Graph())
    edge_index_s = np.array(list(G_o.edges)).T
    edge_weight_s = torch.tensor([G_o[u][v]['weight'] for u, v in G_o.edges], dtype=torch.float32).to(device)

    G_p = nx.from_numpy_array(perturbed_graph_w, create_using=nx.Graph())
    edge_index_t = np.array(list(G_p.edges)).T
    edge_weight_t = torch.tensor([G_p[u][v]['weight'] for u, v in G_p.edges], dtype=torch.float32).to(device)

    x_s_tensor = torch.tensor(fea_ori, dtype=torch.float32).to(device)
    x_t_tensor = torch.tensor(perturbed_fe[Gpn], dtype=torch.float32).to(device)
    x_s_tensor = select_features(x_s_tensor, 8000)
    x_t_tensor = select_features(x_t_tensor, 8000)

    edge_index_s_tensor = torch.tensor(edge_index_s, dtype=torch.int64).to(device)
    edge_index_t_tensor = torch.tensor(edge_index_t, dtype=torch.int64).to(device)

    gp = GraphPair(edge_index_s_tensor, edge_index_t_tensor, x_s_tensor, x_t_tensor,
                   edge_weight_s, edge_weight_t)

    # 4. 加载模型
    print(f"4. Loading Model from {args_cmd.model_path}...")
    model = load_trained_model(args_cmd.model_path)

    # 5. 推理与梯度分析
    print("5. Running Inference & Gradient Saliency Analysis...")
    similarity_score, node_weights = compute_node_importance(model, gp)
    print(f"  >>> Predicted Similarity Score: {similarity_score:.6f}")

    # 6. 结果处理
    aggregated_weights = aggregate_weights(node_weights, original_node)

    # 打印 Top 5
    node_rank_idx = np.argsort(-aggregated_weights)
    print(f"  >>> Top 5 Significant Nodes: {node_rank_idx[:5]}")
    print(f"  >>> Top 5 Weights: {aggregated_weights[node_rank_idx[:5]]}")

    # 7. 保存与可视化
    output_dir = './MGC-RM_inference_result'
    os.makedirs(output_dir, exist_ok=True)

    # 保存排名
    save_path = os.path.join(output_dir, f'Node_Rank_GP{Gpn}_{timestamp}.txt')
    np.savetxt(save_path, node_rank_idx, fmt='%d')
    print(f"6. Results saved to {output_dir}")

    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, original_node + 1), aggregated_weights, '-o', markersize=3, linewidth=1, color='#1f77b4')
    plt.xlabel('Sensor Node Index')
    plt.ylabel('Contribution Weight')
    plt.title(f'Node Contributions (GP {Gpn}, Score: {similarity_score:.3f})')

    max_idx = np.argmax(aggregated_weights)
    plt.annotate(f'Max: {aggregated_weights[max_idx]:.4f}', xy=(max_idx + 1, aggregated_weights[max_idx]),
                 xytext=(max_idx + 1, aggregated_weights[max_idx] * 1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'GP{Gpn}_Weights_{timestamp}.svg'), format='svg')
    plt.close()

    # 高级可视化 (如果模块存在)
    if HAS_PAGERANK_MODULE and os.path.exists(pos_path):
        try:
            print("  Generating advanced Causal Graph visualization...")
            # 尝试查找 PCMCI/SCMI 文件构建背景图
            base_path = str(Path(__file__).parent)
            pcmci_path = os.path.join(base_path,
                                      'PCMCI_/N100_T10000_Hubs15_Viz_Final/pcmci_results_tau3_alpha0p0001.pkl')
            scmi_path = os.path.join(base_path,
                                     'SCMI_N100_T10000_Hubs15_delta20.0_k10_bins30/scmi_causal_graph_delta20.0_k10_bins30_threshold1e-07.npy')

            if os.path.exists(pcmci_path) and os.path.exists(scmi_path):
                G_causal = create_causal_graph_from_matrices(pcmci_path, scmi_path, positions)
                data_color_graph(aggregated_weights, G_causal, positions, 'plasma_r', False)
                plt.savefig(os.path.join(output_dir, f'GP{Gpn}_Causal_Network_{timestamp}.svg'), format='svg')
                plt.close()
            else:
                print("  Note: PCMCI/SCMI files not found, skipping advanced network plot.")
        except Exception as e:
            print(f"  Advanced visualization skipped: {e}")

    print("=== Inference Completed Successfully ===")


if __name__ == "__main__":
    main()
