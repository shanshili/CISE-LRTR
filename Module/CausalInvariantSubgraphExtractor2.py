"""
核心特点: 理论修正版本
流程: 修正了LRTR Loss计算逻辑，防止平凡解
功能: 融合SCMI和PCMCI结果
改进: 使用归一化LRTR Loss，数值稳定性更好
掩码处理: 基于时间滞后模板的掩码结构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from pathlib import Path
import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# 假设 Module 在父目录下
sys.path.append(str(Path(__file__).parent.parent))
from Module.CADGNN import CADGNN

# ==========================================
# 0. 辅助类与工具函数
# ==========================================



def enhance_mask_contrast(mask_np):
    return 1 / (1 + np.exp(-10 * (mask_np - 0.5)))


def analyze_mask_distribution(mask_matrix, title="Mask Distribution Analysis"):
    mask_np = mask_matrix.detach().cpu().numpy()
    print(f"\n[{title}]")
    print(f"  Shape: {mask_np.shape}")
    print(f"  Mean: {mask_np.mean():.4f}, Std: {mask_np.std():.4f}")
    print(f"  Min: {mask_np.min():.4f}, Max: {mask_np.max():.4f}")
    return mask_np


def expand_features_for_causal_graph(features, causal_nodes, original_nodes, tau_max):
    if features.shape[0] != original_nodes: features = features.T
    T = features.shape[1]
    expanded_features = np.zeros((causal_nodes, T - tau_max), dtype=features.dtype)
    for t in range(tau_max + 1):
        start_idx, end_idx = t * original_nodes, (t + 1) * original_nodes
        start_offset, end_offset = tau_max - t, t
        sl = features[:original_nodes, start_offset:] if end_offset == 0 else features[:original_nodes,
                                                                              start_offset:-end_offset]
        expanded_features[start_idx:end_idx, :] = sl
    return expanded_features


def pad_and_expand_matrix(matrix, node_idx, n_nodes_target=100, tau_max=3, verbose=False, tag=""):
    if torch.is_tensor(matrix): matrix = matrix.detach().cpu().numpy()
    if matrix.ndim == 3: matrix = np.max(matrix, axis=2)

    current_nodes = matrix.shape[0]
    matrix_full = None

    if current_nodes == n_nodes_target - 1:
        matrix_full = np.zeros((n_nodes_target, n_nodes_target))
        matrix_full[:node_idx, :node_idx] = matrix[:node_idx, :node_idx]
        matrix_full[:node_idx, node_idx + 1:] = matrix[:node_idx, node_idx:]
        matrix_full[node_idx + 1:, :node_idx] = matrix[node_idx:, :node_idx]
        matrix_full[node_idx + 1:, node_idx + 1:] = matrix[node_idx:, node_idx:]
        if verbose: print(f"    [{tag}] Padding: Inserted zeros at {node_idx}. {current_nodes} -> {n_nodes_target}")
    elif current_nodes == n_nodes_target:
        matrix_full = matrix
        if verbose: print(f"    [{tag}] Padding: Matched target. No padding.")
    else:
        return np.zeros((n_nodes_target * (tau_max + 1), n_nodes_target * (tau_max + 1)))

    total_dim = n_nodes_target * (tau_max + 1)
    expanded = np.zeros((total_dim, total_dim))
    for t in range(tau_max + 1):
        s, e = t * n_nodes_target, (t + 1) * n_nodes_target
        expanded[s:e, s:e] = matrix_full.T

    if verbose: print(f"    [{tag}] Expansion: -> {total_dim}x{total_dim}")
    return expanded


# ==========================================
# 1. 数据加载逻辑
# ==========================================

def load_dual_perturbed_graphs(scmi_dir, pcmci_dir, feature_csv_path, n_nodes_target=100, tau_max=3):
    scmi_dir = Path(scmi_dir)
    pcmci_dir = Path(pcmci_dir)

    print(f"\n=== Step 1: Loading Features ===")
    df = pd.read_csv(feature_csv_path).fillna(0)
    raw_feat = df.values
    if raw_feat.shape[0] != n_nodes_target: raw_feat = raw_feat.T

    total_dim = n_nodes_target * (tau_max + 1)
    expanded_feat_np = expand_features_for_causal_graph(raw_feat, total_dim, n_nodes_target, tau_max)
    global_features = torch.FloatTensor(expanded_feat_np)

    cadgnn_dim = 8000
    if global_features.shape[1] > cadgnn_dim:
        global_features = global_features[:, :cadgnn_dim]
    elif global_features.shape[1] < cadgnn_dim:
        pad = torch.zeros(total_dim, cadgnn_dim - global_features.shape[1])
        global_features = torch.cat([global_features, pad], dim=1)

    print(f"Features: {global_features.shape}")
    s_list, p_list = [], []

    print(f"=== Step 2: Loading Perturbed Graphs ===")
    for i in range(n_nodes_target):
        verbose = (i < 2)
        f_s, f_p = None, None

        s_subdir = scmi_dir / f"scmi_results_node_{i}"
        if s_subdir.exists():
            files = list(s_subdir.glob("scmi_causal_graph_*.npy"))
            if files: f_s = sorted(files)[0]

        p_subdir = pcmci_dir / f"pcmci_results_node_{i}"
        if p_subdir.exists():
            files = list(p_subdir.glob("*.pkl"))
            if files: f_p = files[0]

        if f_s and f_p:
            try:
                mat_s = np.load(f_s)
                with open(f_p, 'rb') as f:
                    data_p = pickle.load(f)

                mat_p = None
                if isinstance(data_p, dict):
                    for k in ['val_matrix', 'graph', 'adjacency', 'causal_matrix']:
                        if k in data_p: mat_p = data_p[k]; break
                    if mat_p is None and 'selected_links' in data_p: mat_p = np.zeros((99, 99))
                if mat_p is None: mat_p = list(data_p.values())[0] if isinstance(data_p, dict) else data_p

                adj_s = pad_and_expand_matrix(mat_s, i, n_nodes_target, tau_max, verbose, "SCMI")
                adj_p = pad_and_expand_matrix(mat_p, i, n_nodes_target, tau_max, verbose, "PCMCI")

                def to_data(adj):
                    t = torch.FloatTensor(adj)
                    mask = torch.abs(t) > 1e-6
                    edges = torch.nonzero(mask).t().contiguous()
                    weights = t[edges[0], edges[1]]
                    if edges.size(1) == 0:
                        edges = torch.arange(total_dim).unsqueeze(0).repeat(2, 1)
                        weights = torch.zeros(total_dim)
                    return global_features, edges, weights

                s_list.append(to_data(adj_s))
                p_list.append(to_data(adj_p))
            except Exception as e:
                print(f"Error Node {i}: {e}")

    print(f"Loaded {len(s_list)} pairs.")
    return s_list, p_list


# ==========================================
# 2. 模型定义 (CGBP + Mask)
# ==========================================

class CausalGatedBilinearProjection(nn.Module):
    """
    对应理论：Phi_i = A^(i) * (H W H^T)
    """

    def __init__(self, hidden_dim):
        super(CausalGatedBilinearProjection, self).__init__()
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, H, A):
        H_proj = self.W_p(H)
        Phi = torch.matmul(H_proj, H.t())
        # A 是因果门控，Phi 是语义交互
        return A * Phi


class CausalInvariantSubgraphExtractor(nn.Module):
    def __init__(self, cadgnn_model, input_dim, hidden_dim, target_nodes=400, tau_max=3):
        super(CausalInvariantSubgraphExtractor, self).__init__()
        self.cadgnn = cadgnn_model

        # 核心：软掩码 M = Sigmoid(Theta)
        self.causal_mask_theta = nn.Parameter(torch.randn(target_nodes, target_nodes) * 0.5)

        self.w_scmi = nn.Parameter(torch.tensor(0.0))
        self.w_pcmci = nn.Parameter(torch.tensor(0.0))

        self.causal_bilinear = CausalGatedBilinearProjection(hidden_dim)

        mask_template = torch.zeros(target_nodes, target_nodes)
        n = target_nodes // (tau_max + 1)
        for i in range(tau_max + 1):
            for j in range(tau_max + 1):
                if j >= i:
                    rs, re = i * n, (i + 1) * n
                    cs, ce = j * n, (j + 1) * n
                    mask_template[rs:re, cs:ce] = 1.0
        self.register_buffer('mask_template', mask_template)
        self.temperature = 1.0

    def set_temperature(self, temp):
        self.temperature = temp

    def get_causal_mask(self):
        return torch.sigmoid(self.causal_mask_theta / self.temperature) * self.mask_template

    def get_causal_mask_reduced(self, original_nodes=100, tau_max=3):
        full_mask = self.get_causal_mask()
        reshaped = full_mask.view(tau_max + 1, original_nodes, tau_max + 1, original_nodes)
        return reshaped.mean(dim=0).mean(dim=1)

    def forward(self, scmi_data_list, pcmci_data_list):
        mask = self.get_causal_mask()

        def process_list(data_list):
            features_list = []
            if not data_list: return None
            for feat, edges, weights in data_list:
                z = self.cadgnn.causal_encoder(feat, edges, weights)

                adj = torch.zeros(z.size(0), z.size(0), device=z.device)
                adj[edges[0], edges[1]] = weights

                # 理论核心：扰动子图 = 原始邻接 * 软掩码
                adj_masked = adj * mask

                phi = self.causal_bilinear(z, adj_masked)
                features_list.append(phi)
            return torch.stack(features_list, dim=2)

        return {
            'scmi_phi': process_list(scmi_data_list),
            'pcmci_phi': process_list(pcmci_data_list),
            'mask': mask
        }


# ==========================================
# 3. 训练与Loss (修正版)
# ==========================================

def compute_lrtr_loss_normalized(Phi_tensor):
    """
    【修正版】：归一化 LRTR Loss
    改动：返回 Sum 而不是 Mean，或者乘以放大系数，防止梯度消失。
    """
    if Phi_tensor is None or Phi_tensor.size(2) == 0:
        return torch.tensor(0.0, device=Phi_tensor.device)

    # Phi_tensor: [Nodes, Nodes, Batch]
    N = Phi_tensor.size(0)

    # 1. 归一化 (保持方向一致性)
    norms = torch.norm(Phi_tensor, p=2, dim=(0, 1), keepdim=True) + 1e-6
    Phi_norm = Phi_tensor / norms

    # 2. 计算均值图 (Centroid)
    mean_slice = torch.mean(Phi_norm, dim=2, keepdim=True)

    # 3. 计算残差平方 (Difference Squared)
    diff_sq = (Phi_norm - mean_slice) ** 2

    # 【关键修改】
    # 原来是 torch.mean(diff_sq)，数值在 1e-7 级别。
    # 现在改为乘以 N*N (即求 Sum)，或者直接放大。
    # 这里我们乘以 1e5 作为基础放大倍率，使其对齐到 0.1 ~ 1.0 的范围
    loss = torch.mean(diff_sq) * (N * N)

    return loss


def train_causal_invariant_subgraph_extractor(model, s_data, p_data, num_epochs=100, lr=0.005, device='cuda'):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print("\n=== Step 3: Training Start (Rescaled Loss) ===")
    s_gpu = [(f.to(device), e.to(device), w.to(device)) for f, e, w in s_data]
    p_gpu = [(f.to(device), e.to(device), w.to(device)) for f, e, w in p_data]

    loss_history = {'total_loss': [], 'causal_lrtr_loss': []}

    for epoch in range(num_epochs):
        temp = 0.5 + 0.5 * (0.98 ** epoch)
        model.set_temperature(temp)

        model.train()
        optimizer.zero_grad()

        out = model(s_gpu, p_gpu)
        mask = out['mask']

        # 1. 计算 Loss (现在数值应该在 0.01 ~ 10.0 之间)
        loss_s = compute_lrtr_loss_normalized(out['scmi_phi'])
        loss_p = compute_lrtr_loss_normalized(out['pcmci_phi'])

        # 2. 动态加权
        w_s, w_p = torch.sigmoid(model.w_scmi), torch.sigmoid(model.w_pcmci)
        causal_loss = (w_s * loss_s + w_p * loss_p) / (w_s + w_p + 1e-6)

        # 3. 正则化 (保持小权重)
        # Binary Reg: 0.01 足够
        binary_reg = 0.01 * torch.mean(mask * (1 - mask))

        # Sparsity: 必须非常小，否则 mask 会全 0
        sparsity = 1e-5 * torch.norm(mask, p=1)

        # 4. 防塌缩惩罚 (如果 Mask 均值 < 0.05，施加重罚)
        mask_mean = torch.mean(mask)
        collapse_penalty = 0.0
        if mask_mean < 0.05:
            # 指数级惩罚，迫使 Mask 长出来
            collapse_penalty = 10.0 * torch.exp(5.0 * (0.05 - mask_mean))

        total_loss = causal_loss + binary_reg + sparsity + collapse_penalty

        if torch.isnan(total_loss):
            print("NaN Loss detected!")
            break

        total_loss.backward()

        # 梯度裁剪 (防止因为 Loss 放大导致梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        loss_history['total_loss'].append(total_loss.item())
        loss_history['causal_lrtr_loss'].append(causal_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Total: {total_loss.item():.4f} | "
                  f"Causal: {causal_loss.item():.4f} | "
                  f"Mask Mean: {mask_mean.item():.4f} | "
                  f"Collapse Pen: {collapse_penalty:.4f if isinstance(collapse_penalty, float) else collapse_penalty.item():.4f}")

    return model, loss_history


def save_results(model, loss_history, save_dir, original_nodes=100, tau_max=3):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"run_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pth")
    mask_full = model.get_causal_mask().detach().cpu().numpy()
    mask_reduced = model.get_causal_mask_reduced(original_nodes, tau_max).detach().cpu().numpy()

    np.save(path / "mask_full.npy", mask_full)
    np.save(path / "mask_reduced.npy", mask_reduced)
    visualize_mask_heatmap(mask_full, path / "heatmap_full.png", "Full Mask", 400, tau_max)
    visualize_mask_heatmap(mask_reduced, path / "heatmap_reduced.png", "Reduced Mask", 100, tau_max)

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history['total_loss'])
    plt.title("Total Loss")
    plt.savefig(path / "loss.png")
    plt.close()
    print(f"Saved to {path}")


def visualize_mask_heatmap(mask_np, save_path, title, nodes, tau):
    enhanced = enhance_mask_contrast(mask_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(enhanced, cmap='Reds', center=0.5, vmin=0, vmax=1)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# ==========================================
# 4. 主函数
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    root = Path(__file__).parent.parent

    # 路径配置
    base_data = root / "dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final"
    feature_csv = base_data / "Env_A_timeseries.csv"
    scmi_dir = root / "dataset/Perturbed_SCMI_Causal_Graphs_N100_T10000_Hubs15/scmi_perturbations"
    pcmci_dir = root / "dataset/Perturbed_Causal_Graphs_N100_T10000_Hubs15"
    cadgnn_path = "./CADGNN_results/N100_T10000_hid128_lay5_ladv0.1_lw0.02_20250902_203304_model.pth"

    s_list, p_list = load_dual_perturbed_graphs(scmi_dir, pcmci_dir, feature_csv, 100, 3)
    if not s_list: return

    feat_dim = s_list[0][0].shape[1]

    if os.path.exists(cadgnn_path):
        ckpt = torch.load(cadgnn_path, map_location=device)
        cfg = ckpt.get('model_config', {'hidden_dim': 128})
        cadgnn = CADGNN(feat_dim, cfg['hidden_dim'], 5, 0.1, 0.02).to(device)
        cadgnn.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("CADGNN Loaded.")
    else:
        cadgnn = CADGNN(feat_dim, 128, 5, 0.1, 0.02).to(device)

    extractor = CausalInvariantSubgraphExtractor(cadgnn, feat_dim, 128, target_nodes=400, tau_max=3)

    trained_model, hist = train_causal_invariant_subgraph_extractor(
        extractor, s_list, p_list, num_epochs=150, lr=0.005, device=device
    )

    analyze_mask_distribution(trained_model.get_causal_mask_reduced(), "Final Mask")
    save_results(trained_model, hist, "causal_invariant_results", 100, 3)


if __name__ == "__main__":
    main()
