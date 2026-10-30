import os
import glob
import time
import math
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv


# =========================================================
# GAT backbone
# =========================================================
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.tanh(x)
        return x


# =========================================================
# GraphPair (aligned mapping)
# =========================================================
class GraphPair:
    def __init__(self, edge_index_s, edge_index_t, x_s, x_t, keep_idx, label=None):
        """
        keep_idx: LongTensor shape (N_t,)
          - mapping from perturbed nodes -> original nodes
          - original N=100, perturbed N=99, keep_idx length=99
        """
        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.x_s = x_s
        self.x_t = x_t
        self.keep_idx = keep_idx
        self.label = label


# =========================================================
# readout / matching / MLP
# =========================================================
class readout(nn.Module):
    def __init__(self, output_size):
        super(readout, self).__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h):
        global_h = torch.matmul(torch.mean(h, dim=0), self.weight)
        transformed_global = torch.tanh(global_h)
        sigmoid_scores = torch.sigmoid(torch.mm(h, transformed_global.view(-1, 1)).sum(dim=1))
        h_global = sigmoid_scores.unsqueeze(-1) * h
        return h_global.sum(dim=0).unsqueeze(0)


class Matching(nn.Module):
    def __init__(self, input_dim, match_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, match_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, a, b):
        # a: (Na, D), b: (Nb, D)
        a = F.normalize(self.proj(a), dim=-1)
        b = F.normalize(self.proj(b), dim=-1)
        return a @ b.T   # (Na, Nb) pairwise cosine


class MLP(nn.Module):
    def __init__(self, match_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(match_size * 8, match_size * 2)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)

        self.linear2 = nn.Linear(match_size * 2, match_size)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.zeros_(self.linear2.bias.data)

        self.linear3 = nn.Linear(match_size, match_size // 2)
        nn.init.xavier_uniform_(self.linear3.weight.data)
        nn.init.zeros_(self.linear3.bias.data)

        self.linear4 = nn.Linear(match_size // 2, 1)
        nn.init.xavier_uniform_(self.linear4.weight.data)
        nn.init.zeros_(self.linear4.bias.data)

        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.linear4(z)   # raw
        return z


# =========================================================
# MFC_RMF (Aligned version for stable node importance)
# =========================================================
class MFC_RMF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super().__init__()
        torch.manual_seed(1234)
        self.args = args

        self.gat = GAT(input_size, hidden_size, output_size)

        # node readout (for D=output_size)
        self.readout_node = readout(output_size)

        # pairwise matcher -> (Na, Nb)
        self.match = Matching(output_size, args.match_size)

        # 把 (Na,Nb) 的相似度矩阵压成固定长度向量（关键！）
        # 这里用 4 个统计量：mean/max over rows + mean/max over cols -> 得到 4 个标量图级特征
        # 再投影到 match_size
        self.pair2vec = nn.Sequential(
            nn.Linear(4, args.match_size),
            nn.ReLU(),
        )

        # 现在 z 会是：8 个 (1, match_size) 拼起来 -> (1, match_size*8)
        self.mlp = MLP(args.match_size)

    def self_representation(self, x, edge_index):
        return self.gat(x, edge_index)

    def mutual_representation_aligned(self, h_1_keep, h_2):
        cos_sim = torch.cosine_similarity(
            h_1_keep.unsqueeze(1), h_2.unsqueeze(0), dim=-1
        )  # (N2,N2)
        attn = torch.softmax(cos_sim, dim=1)  # ✅ 行归一化
        return attn @ h_2  # (N2, D)

    def pairwise_to_vec(self, S, k=10):
        """
        S: (Na, Nb) 相似度矩阵
        return: (1, match_size)

        Top-k 统计版本：对每行/每列取 top-k，再做 mean/max，
        比全局 mean/max 对结构差异更敏感，更容易拉开 sim_pred 的动态范围。
        """
        Na, Nb = S.shape

        # --- row top-k: each row selects top-k across columns ---
        k_row = min(k, Nb)
        row_topk = torch.topk(S, k_row, dim=1).values          # (Na, k_row)
        row_mean = row_topk.mean()                              # scalar
        row_max  = row_topk.max()                               # scalar

        # --- col top-k: each col selects top-k across rows ---
        k_col = min(k, Na)
        col_topk = torch.topk(S, k_col, dim=0).values          # (k_col, Nb)
        col_mean = col_topk.mean()                              # scalar
        col_max  = col_topk.max()                               # scalar

        stats = torch.stack([row_mean, row_max, col_mean, col_max], dim=0).unsqueeze(0)  # (1,4)
        return self.pair2vec(stats)  # (1, match_size)

    def forward(self, graph_pair):
        edge_index_1 = graph_pair.edge_index_s
        edge_index_2 = graph_pair.edge_index_t
        keep_idx = graph_pair.keep_idx

        x1 = graph_pair.x_s
        x2 = graph_pair.x_t

        # 建议：按特征维标准化更稳（但不是必须）
        x1 = (x1 - x1.mean(dim=0, keepdim=True)) / (x1.std(dim=0, keepdim=True) + 1e-9)
        x2 = (x2 - x2.mean(dim=0, keepdim=True)) / (x2.std(dim=0, keepdim=True) + 1e-9)

        h1_self = self.self_representation(x1, edge_index_1)  # (N1,D)
        h2_self = self.self_representation(x2, edge_index_2)  # (N2,D)

        h1_keep = h1_self.index_select(dim=0, index=keep_idx)  # (N2,D)

        # global node readout
        h1_global = self.readout_node(h1_self)  # (1,D)
        h2_global = self.readout_node(h2_self)  # (1,D)

        # mutual
        h1_mutual_keep = self.mutual_representation_aligned(h1_keep, h2_self)  # (N2,D)
        h2_mutual      = self.mutual_representation_aligned(h2_self, h1_keep)  # (N2,D)

        h1_mutual_global = self.readout_node(h1_mutual_keep)  # (1,D)
        h2_mutual_global = self.readout_node(h2_mutual)       # (1,D)

        # 计算 8 个匹配分支：每个分支输出 (1, match_size)
        miu_1 = self.pairwise_to_vec(self.match(h1_keep, h1_mutual_keep))
        miu_2 = self.pairwise_to_vec(self.match(h2_self, h2_mutual))

        phi_1 = self.pairwise_to_vec(self.match(h1_keep, h1_mutual_global))  # (N2,1) -> 也能算
        phi_2 = self.pairwise_to_vec(self.match(h2_self, h2_mutual_global))

        psi_1 = self.pairwise_to_vec(self.match(h1_global, h1_mutual_keep))  # (1,N2)
        psi_2 = self.pairwise_to_vec(self.match(h2_global, h2_mutual))

        omega_1 = self.pairwise_to_vec(self.match(h1_global, h1_mutual_global))  # (1,1)
        omega_2 = self.pairwise_to_vec(self.match(h2_global, h2_mutual_global))

        z_1 = torch.cat([miu_1, phi_1, psi_1, omega_1], dim=1)  # (1, match*4)
        z_2 = torch.cat([miu_2, phi_2, psi_2, omega_2], dim=1)  # (1, match*4)
        z = torch.cat([z_1, z_2], dim=1)                        # (1, match*8)

        score = torch.sigmoid(self.mlp(z))
        return score


# =========================================================
# Utilities
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    lat/lon in degrees
    return distance in km
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def geo_dist_ok(u, v, pos_latlon, threshold_km):
    lat1, lon1 = pos_latlon[u]
    lat2, lon2 = pos_latlon[v]
    return haversine_km(lat1, lon1, lat2, lon2) <= threshold_km



def rankdata(a):
    a = np.asarray(a)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense_rank = np.cumsum(obs) - 1

    counts = np.bincount(dense_rank)
    cumsum = np.cumsum(counts)
    start = cumsum - counts

    ranks = np.zeros_like(a_sorted, dtype=np.float64)
    for r in range(len(counts)):
        if counts[r] > 0:
            s = start[r]
            e = cumsum[r]
            ranks[s:e] = (s + e - 1) / 2.0 + 1

    return ranks[inv]


def spearman_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    assert len(a) == len(b)

    ra = rankdata(a)
    rb = rankdata(b)

    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float(np.mean(ra * rb))


def corr_to_edge_index(corr_matrix, threshold=0.5, use_abs=True):
    mask = (np.abs(corr_matrix) > threshold) if use_abs else (corr_matrix > threshold)
    np.fill_diagonal(mask, False)
    row, col = np.where(mask)
    return np.vstack([row, col]).astype(np.int64)



def build_keep_idx(remove_node, N=100):
    keep = [i for i in range(N) if i != remove_node]
    return keep


def adjacency_similarity_L1(A, B):
    diff = np.abs(A - B).sum()
    denom = np.abs(A).sum() + 1e-9
    sim = 1.0 - diff / denom
    sim = max(0.0, min(1.0, float(sim)))
    return sim


def build_graph_pair_for_node(
    i,
    N,
    node_feat,
    x_s_tensor,
    edge_index_s_tensor,
    pert_files,
    corr_threshold,
    device,
    pert_corr_override=None
):
    if pert_corr_override is None:
        pdata = np.load(pert_files[i], allow_pickle=True).item()
        pert_corr = pdata["graph"]
    else:
        pert_corr = pert_corr_override

    edge_index_t_np = corr_to_edge_index(pert_corr, threshold=corr_threshold, use_abs=True)
    edge_index_t_tensor = torch.tensor(edge_index_t_np, dtype=torch.long, device=device)

    node_feat_pert = np.delete(node_feat, i, axis=0)
    x_t_tensor = torch.tensor(node_feat_pert, dtype=torch.float32, device=device)

    keep_idx = torch.tensor(build_keep_idx(i, N), dtype=torch.long, device=device)

    gp = GraphPair(edge_index_s_tensor, edge_index_t_tensor, x_s_tensor, x_t_tensor, keep_idx)
    return gp, pert_corr


def proxy_similarity_on_common_nodes(corr_adj, pert_corr, remove_node, N=100):
    keep = build_keep_idx(remove_node, N)
    A_99 = corr_adj[np.ix_(keep, keep)]
    return adjacency_similarity_L1(A_99, pert_corr)


def compute_corr_from_timeseries(X):
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def augment_pert_corr(pert_corr, noise_std=0.02, drop_prob=0.05, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    B = pert_corr.copy()

    noise = rng.normal(0.0, noise_std, size=B.shape).astype(np.float32)
    noise = (noise + noise.T) / 2.0
    B = B + noise

    if drop_prob > 0:
        mask = rng.random(B.shape) < drop_prob
        np.fill_diagonal(mask, False)
        mask = np.triu(mask, 1)
        mask = mask | mask.T
        B[mask] = 0.0

    B = np.clip(B, -1.0, 1.0)
    np.fill_diagonal(B, 1.0)

    return B


def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def select_top_nodes_by_importance(importance: np.ndarray, top_k: int = 30):
    """
    importance: shape [N]
    return:
      top_nodes: np.ndarray shape [top_k]  (按重要性从高到低)
    """
    importance = np.asarray(importance, dtype=np.float64)
    N = importance.shape[0]
    k = min(int(top_k), N)
    top_nodes = np.argsort(-importance)[:k]
    return top_nodes

def corr_knn_edges_on_top_nodes(
    corr_adj: np.ndarray,
    top_nodes: np.ndarray,
    knn_k: int = 10,
    use_abs_for_rank: bool = True,
    remove_self: bool = True,
    symmetrize: str = "union",   # "union" / "mutual" / "none"
):
    """
    return:
      edges_undirected: [(u, v, abs_corr), ...]  (u < v)  用于可视化
      edges_directed:   [(u, v), ...]            用于保存 PCMCI-like（有向）
    """
    corr_adj = np.asarray(corr_adj, dtype=float)
    top_nodes = np.asarray(top_nodes, dtype=int)
    S = top_nodes.tolist()
    m = len(S)
    if m < 2:
        return [], []

    C = corr_adj[np.ix_(S, S)]  # (m,m)
    score = np.abs(C) if use_abs_for_rank else C
    score = score.copy()

    if remove_self:
        np.fill_diagonal(score, -np.inf)

    k_eff = min(int(knn_k), m - 1) if remove_self else min(int(knn_k), m)
    A_dir = np.zeros((m, m), dtype=bool)

    for ii in range(m):
        idx = np.argpartition(-score[ii], k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-score[ii, idx])]
        A_dir[ii, idx] = True

    if symmetrize == "none":
        A = A_dir
    elif symmetrize == "union":
        A = A_dir | A_dir.T
    elif symmetrize == "mutual":
        A = A_dir & A_dir.T
    else:
        raise ValueError(f"Unknown symmetrize: {symmetrize}")

    if remove_self:
        np.fill_diagonal(A, False)

    # directed edges
    edges_directed = []
    rows, cols = np.where(A)
    for ii, jj in zip(rows.tolist(), cols.tolist()):
        u = S[ii]
        v = S[jj]
        if u != v:
            edges_directed.append((u, v))

    # undirected unique edges
    edges_undirected = []
    seen = set()
    for (u, v) in edges_directed:
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges_undirected.append((a, b, float(abs(corr_adj[a, b]))))

    return edges_undirected, edges_directed


# =========================================================
# ✅ train helper (random i + random aug_k + grad clip)
# =========================================================
def train_all_nodes_random(
    model,
    optimizer,
    mse_loss,
    corr_adj,
    pert_files,
    node_feat,
    x_s_tensor,
    edge_index_s_tensor,
    N,
    args,
    device,
    steps=8000,
    noise_std=0.02,
    drop_prob=0.05,
    print_every=200,
    grad_clip=5.0,  # ✅ NEW: gradient clipping
):
    model.train()
    loss_hist = []

    for step in range(steps):
        i_train = np.random.randint(0, N)

        _, pert_corr_base = build_graph_pair_for_node(
            i=i_train, N=N, node_feat=node_feat,
            x_s_tensor=x_s_tensor, edge_index_s_tensor=edge_index_s_tensor,
            pert_files=pert_files, corr_threshold=args["corr_threshold"],
            device=device
        )

        # ✅ 每步随机 2~4 个 augment
        aug_k = np.random.randint(2, 5)  # 2,3,4

        for k in range(aug_k):
            pert_corr_aug = augment_pert_corr(
                pert_corr_base,
                noise_std=noise_std,
                drop_prob=drop_prob,
                seed=step * 100 + k
            )

            gp, _ = build_graph_pair_for_node(
                i=i_train, N=N, node_feat=node_feat,
                x_s_tensor=x_s_tensor, edge_index_s_tensor=edge_index_s_tensor,
                pert_files=pert_files, corr_threshold=args["corr_threshold"],
                device=device,
                pert_corr_override=pert_corr_aug
            )

            if args["label_mode"] == "proxy":
                sim = proxy_similarity_on_common_nodes(corr_adj, pert_corr_aug, remove_node=i_train, N=N)
            else:
                raise NotImplementedError("GED mode not integrated in this training helper yet.")

            target = torch.tensor([[sim]], dtype=torch.float32, device=device)

            pred = model(gp)
            loss = mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()

            # ✅ 梯度裁剪：就在这里（loss.backward() 之后，optimizer.step() 之前）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            loss_hist.append(loss.item())

        if step % print_every == 0:
            print(f"[TRAIN] step={step:05d}/{steps} node={i_train} aug={aug_k} loss={np.mean(loss_hist[-aug_k:]):.6f}")

    return loss_hist


# =========================================================
# ✅ inference helper
# =========================================================
def inference_all_nodes(
    model,
    corr_adj,
    pert_files,
    node_feat,
    x_s_tensor,
    edge_index_s_tensor,
    N,
    args,
    device
):
    model.eval()
    sim_pred = []
    sim_gt = []
    mse_list = []

    with torch.no_grad():
        for i in range(N):
            pdata = np.load(pert_files[i], allow_pickle=True).item()
            pert_corr = pdata["graph"]

            gp, _ = build_graph_pair_for_node(
                i=i, N=N, node_feat=node_feat,
                x_s_tensor=x_s_tensor, edge_index_s_tensor=edge_index_s_tensor,
                pert_files=pert_files, corr_threshold=args["corr_threshold"],
                device=device,
                pert_corr_override=pert_corr
            )

            pred = float(model(gp).item())
            sim_pred.append(pred)

            if args["label_mode"] == "proxy":
                gt = proxy_similarity_on_common_nodes(corr_adj, pert_corr, remove_node=i, N=N)
            else:
                raise NotImplementedError("GED mode not integrated in this inference helper yet.")

            sim_gt.append(gt)
            mse_list.append((pred - gt) ** 2)

    sim_pred = np.array(sim_pred, dtype=np.float64)
    sim_gt = np.array(sim_gt, dtype=np.float64)
    mse_list = np.array(mse_list, dtype=np.float64)

    importance_pred = 1.0 - sim_pred
    importance_gt = 1.0 - sim_gt

    sp_sim = spearman_corr(sim_pred, sim_gt)
    sp_imp = spearman_corr(importance_pred, importance_gt)

    print(f"[EVAL] Spearman(sim_pred, sim_gt) = {sp_sim:.4f}")
    print(f"[EVAL] Spearman(importance_pred, importance_gt) = {sp_imp:.4f}")
    print(f"[EVAL] MSE mean = {mse_list.mean():.6f}")

    return sim_pred, sim_gt, importance_pred, importance_gt, mse_list, sp_sim, sp_imp


# =========================================================
# Visualization
# =========================================================
def plot_training_loss(loss_history, save_path, title):
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_node_importance_with_index(
    node_positions,
    importance,
    save_path,
    title="Node Importance",
    edges_base=None,
    draw_edges=True,
    node_size=220,
    font_size=8,
    edge_alpha=1,
    edge_width=1,
    dpi=600,
    spatial_distance_threshold=None,
    knn_connections=None,
    corr_adj=None,
    top_k_nodes=30,
    top_nodes_override=None,
    corr_knn_k=None,
    corr_knn_symmetrize="union",
    use_abs_for_rank=True,

):
    """
    模式无关版绘图：
    1) 先确定“哪些节点属于高重要性”
    2) 根据模式(KNN 或 edges_base)统一生成 edge_list = [(u,v,corr_or_None), ...]
    3) 统一绘制 edge_list（如果 corr_adj 存在则用 Reds 映射）
    4) 若存在 corr 值 -> 统一生成 cbar_corr
    5) 绘制节点 importance 的 colorbar（左侧）
    """

    import numpy as np
    import matplotlib.pyplot as plt

    pos = node_positions[:, :2]
    x = pos[:, 0]
    y = pos[:, 1]
    N = pos.shape[0]

    # ---------- 1) 高重要性节点集合（前30%） ----------
    # importance_threshold = np.percentile(importance, 70)
    # high_importance_nodes = set(np.where(importance >= importance_threshold)[0])
    if top_nodes_override is None:
        top_nodes = select_top_nodes_by_importance(importance, top_k=top_k_nodes)
    else:
        top_nodes = np.asarray(top_nodes_override, dtype=int)

    high_importance_nodes = set(top_nodes.tolist())

    # ---------- 2) 统一收集边 edge_list ----------
    # edge_list: list of (u, v, corr_val_or_None)
    edge_list = []

    # 假设 pos[:,0]=lon, pos[:,1]=lat
    pos_lonlat = np.stack([pos[:, 0], pos[:, 1]], axis=1)  # (N,2)

    def _dist_ok(u, v):
        # 注意这里 pos_lonlat 是 [lon,lat]，haversine 需要 [lat,lon]
        lon1, lat1 = pos_lonlat[u]
        lon2, lat2 = pos_lonlat[v]
        return haversine_km(lat1, lon1, lat2, lon2) <= 10

    if draw_edges:
        # ✅ 优先：相关性KNN（保证与保存子图一致）
        if corr_knn_k is not None and corr_adj is not None:
            edges_undirected, _ = corr_knn_edges_on_top_nodes(
                corr_adj=corr_adj,
                top_nodes=top_nodes,
                knn_k=int(corr_knn_k),
                use_abs_for_rank=use_abs_for_rank,
                remove_self=True,
                symmetrize=corr_knn_symmetrize,
            )
            # edges_undirected: (u,v,abs_corr)
            for (a, b, corr_val) in edges_undirected:
                if not _dist_ok(a, b):
                    continue
                edge_list.append((a, b, corr_val))
    # ---------- 3) 开始绘图 ----------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect("equal", adjustable="box")

    # ---------- 4) 统一绘制边 & 统一准备 corr colorbar ----------
    corr_vals = [c for (_, _, c) in edge_list if c is not None]

    # 若有 corr 值，则使用 Reds 映射；否则灰色
    if corr_adj is not None and len(corr_vals) > 0:
        vmin, vmax = min(corr_vals), max(corr_vals)
        # 防止 vmin==vmax 导致 Normalize 奇怪
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Reds
    else:
        norm = None
        cmap = None

    # 画边
    for (u, v, c) in edge_list:
        if cmap is not None and c is not None:
            color = cmap(norm(c))
            line_width = edge_width * (c * 3)  # 你原来逻辑：相关性越大线越粗
        else:
            color = "gray"
            line_width = edge_width

        ax.plot(
            [x[u], x[v]],
            [y[u], y[v]],
            linewidth=line_width,
            alpha=edge_alpha,
            color=color,
            zorder=1
        )

    # ---------- 5) 画节点 ----------
    sc = ax.scatter(
        x, y,
        c=importance,
        s=node_size,
        cmap="YlGn",
        edgecolors="none",
        linewidths=0.0,
        zorder=2
    )

    # 节点编号（高重要性黑字，其它浅灰）
    for i in range(N):
        if i in high_importance_nodes:
            # 高重要性节点使用黑色文字
            ax.text(
                x[i], y[i], str(i),
                fontsize=font_size,
                color="black",
                ha="center", va="center",
                zorder=3
            )

    # ---------- 6) cbar_corr（右侧） ----------
    if cmap is not None and len(corr_vals) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_corr = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.030,
            pad=0.02,
            location="right"
        )
        cbar_corr.set_label("Correlation Strength", rotation=270, labelpad=12)

    # ---------- 7) importance colorbar（左侧） ----------
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.05, location="left")
    cb.set_label("Node Importance Score", rotation=90, labelpad=7)

    # ---------- 8) 画布美化 ----------
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.margins(0.05)

    fig.tight_layout(pad=3.0)
    fig.savefig(save_path, format='svg', bbox_inches="tight")
    plt.close(fig)


import pickle
import numpy as np
import os


def build_pcmci_like_from_corr(
    corr_adj: np.ndarray,
    top_nodes: np.ndarray,
    knn_k: int = 10,
    tau_max: int = 5,
    lag_fixed: int = 1,
    use_abs_for_rank: bool = True,
    keep_signed_weight: bool = True,
    remove_self: bool = True,
    symmetrize: str = "union",

    # ✅ NEW: 与可视化一致的空间过滤
    node_positions: np.ndarray = None,             # shape (N,>=2)
    spatial_distance_threshold: float = None,      # e.g. 20
):
    """
    在 top_nodes 子集上做相关性 KNN 构图，再写回全图 PCMCI-like。
    ✅ 新增：如果给了 node_positions + spatial_distance_threshold，
           则只保留距离 <= threshold 的边（与 plot 中 _dist_ok 一致）
    """
    corr_adj = np.asarray(corr_adj, dtype=float)
    N = corr_adj.shape[0]
    assert corr_adj.shape[0] == corr_adj.shape[1]

    top_nodes = np.asarray(top_nodes, dtype=int)
    S = top_nodes.tolist()
    m = len(S)
    if m < 2:
        raise ValueError("top_nodes size must be >= 2 to build KNN subgraph.")

    # ✅ 准备坐标（与 plot: node_positions[:, :2] 一致）
    if spatial_distance_threshold is not None:
        if node_positions is None:
            raise ValueError("node_positions must be provided when spatial_distance_threshold is not None.")
        pos2 = np.asarray(node_positions, dtype=float)[:, :2]
        if pos2.shape[0] != N:
            raise ValueError(f"node_positions first dim must be N={N}, got {pos2.shape[0]}")
    else:
        pos2 = None

    pos2 = np.asarray(node_positions, dtype=float)[:, :2]

    # 假设 node_positions[:,0]=lon, [:,1]=lat
    pos_lonlat = np.stack([pos2[:, 0], pos2[:, 1]], axis=1)

    def _dist_ok(u: int, v: int) -> bool:
        lon1, lat1 = pos_lonlat[u]
        lon2, lat2 = pos_lonlat[v]
        return haversine_km(lat1, lon1, lat2, lon2) <= 10

    L = tau_max + 1
    if lag_fixed < 0 or lag_fixed >= L:
        raise ValueError(f"lag_fixed={lag_fixed} must be in [0, {L-1}] for tau_max={tau_max}")

    graph = np.full((N, N, L), "", dtype=object)
    val = np.zeros((N, N, L), dtype=float)

    C = corr_adj[np.ix_(S, S)]  # (m,m)
    score = np.abs(C) if use_abs_for_rank else C
    score = score.copy()

    if remove_self:
        np.fill_diagonal(score, -np.inf)

    k_eff = min(int(knn_k), m - 1) if remove_self else min(int(knn_k), m)
    A_dir = np.zeros((m, m), dtype=bool)

    for ii in range(m):
        idx = np.argpartition(-score[ii], k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-score[ii, idx])]
        A_dir[ii, idx] = True

    if symmetrize == "none":
        A = A_dir
    elif symmetrize == "union":
        A = A_dir | A_dir.T
    elif symmetrize == "mutual":
        A = A_dir & A_dir.T
    else:
        raise ValueError(f"Unknown symmetrize: {symmetrize}")

    if remove_self:
        np.fill_diagonal(A, False)

    # ✅ 写边时加入空间过滤（关键改动）
    rows, cols = np.where(A)
    for ii, jj in zip(rows.tolist(), cols.tolist()):
        i = S[ii]
        j = S[jj]
        if i == j:
            continue

        # ✅ 空间过滤：与 plot 完全一致
        if not _dist_ok(i, j):
            continue

        graph[i, j, lag_fixed] = "-->"
        w = corr_adj[i, j] if keep_signed_weight else abs(corr_adj[i, j])
        val[i, j, lag_fixed] = float(w)

    return graph, val



def save_subgraph_pkl(pkl_path: str, graph: np.ndarray, val_matrix: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    payload = {"graph": graph, "val_matrix": val_matrix}
    payload.update(meta)
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[SAVE] subgraph pkl -> {pkl_path}")




# =========================================================
# Config
# ========================================================= # simulation_hubs_core/N100_T10000_Hubs15_Viz_Final"
CONFIG = {
    "dataset_dir": "../dataset/pems_spatial_kmeans_topK" ,
    "perturbed_dir": "../dataset/Perturbed_corr_Graphs_PEMS_N100_1h_standardize_id",
    "save_dir": "./outputs_importance_PEMS",

    "lr": 1e-4,
    "weight_decay": 1e-4,

    "hidden_dim": 512,
    "output_dim": 256,
    "match_size": 256,

    "corr_threshold": 0.5,
    "label_mode": "proxy",

    "draw_edges": True,

    # training random
    "train_steps": 2000,
    "noise_std": 0.02,
    "drop_prob": 0.05,
    "print_every": 200,
    "grad_clip": 5.0,  # ✅ NEW
}


def enhance_contrast(x, q_low=5, q_high=95, gamma=0.5):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [q_low, q_high])
    x = np.clip(x, lo, hi)
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    x = x ** gamma   # gamma<1：增强对比
    return x



def main():
    args = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    os.makedirs(args["save_dir"], exist_ok=True)
    os.makedirs(os.path.join(args["save_dir"], "models"), exist_ok=True)
    os.makedirs(os.path.join(args["save_dir"], "figures"), exist_ok=True)
    os.makedirs(os.path.join(args["save_dir"], "scores"), exist_ok=True)

    # Load time series
    feature_path = os.path.join(args["dataset_dir"], "id_standardize_Env_A_timeseries.csv")
    X = pd.read_csv(feature_path).values
    T, N = X.shape
    print(f"[INFO] timeseries shape: {X.shape} (T={T}, N={N})")

    feat_mean = np.mean(X, axis=0)
    feat_std = np.std(X, axis=0)
    feat_absmean = np.mean(np.abs(X), axis=0)

    # ✅ 基于 corr_adj 的结构特征（建议放在 compute_corr_from_timeseries(X) 之后更合理）
    # 这里先计算 corr_adj，再用它做特征
    corr_adj = compute_corr_from_timeseries(X)

    abs_corr = np.abs(corr_adj).copy()
    np.fill_diagonal(abs_corr, 0.0)

    feat_corr_mean = abs_corr.mean(axis=0)  # (N,)
    feat_corr_max = abs_corr.max(axis=0)  # (N,)
    feat_degree = (abs_corr > args["corr_threshold"]).sum(axis=0).astype(np.float32)  # (N,)

    # ✅ 拼成更强的 node feature：3（时序统计）+ 3（相关结构）
    node_feat = np.stack(
        [feat_mean, feat_std, feat_absmean, feat_corr_mean, feat_corr_max, feat_degree],
        axis=1
    ).astype(np.float32)

    input_dim = node_feat.shape[1]
    print(f"[INFO] node_feat shape: {node_feat.shape}, input_dim={input_dim}")

    x_s_tensor = torch.tensor(node_feat, dtype=torch.float32, device=device)


    # Corr adjacency
    edge_index_s_np = corr_to_edge_index(corr_adj, threshold=args["corr_threshold"], use_abs=True)
    edge_index_s_tensor = torch.tensor(edge_index_s_np, dtype=torch.long, device=device)

    # Load node positions
    pos_path = os.path.join(args["dataset_dir"], "id_node_positions.csv")
    pos_df = pd.read_csv(pos_path)
    # 常见情况：列名就是 x,y
    if {"x", "y"}.issubset(pos_df.columns):
        node_positions = pos_df[["x", "y"]].to_numpy(dtype=np.float32)
    # node_positions = pd.read_csv(pos_path).values.astype(np.float32)

    # Load perturbed files
    pert_files = sorted(glob.glob(os.path.join(args["perturbed_dir"], "perturbed_corr_graph_node_*.npy")))
    assert len(pert_files) == N, f"Need {N} perturbed files, got {len(pert_files)}"

    # Init model
    class SimpleArgs:
        pass

    model_args = SimpleArgs()
    model_args.match_size = args["match_size"]

    model = MFC_RMF(input_dim, args["hidden_dim"], args["output_dim"], model_args).to(device)
    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    mse_loss = nn.SmoothL1Loss(beta=0.05)

    # Train
    print("[INFO] start training on ALL nodes (random sampling)...")
    train_loss_history = train_all_nodes_random(
        model=model,
        optimizer=optimizer,
        mse_loss=mse_loss,
        corr_adj=corr_adj,
        pert_files=pert_files,
        node_feat=node_feat,
        x_s_tensor=x_s_tensor,
        edge_index_s_tensor=edge_index_s_tensor,
        N=N,
        args=args,
        device=device,
        steps=args["train_steps"],
        noise_std=args["noise_std"],
        drop_prob=args["drop_prob"],
        print_every=args["print_every"],
        grad_clip=args["grad_clip"],
    )

    # Save model + loss
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_path = os.path.join(args["save_dir"], "models", f"mfc_rmf_aligned_{args['label_mode']}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print("[SAVE] model ->", model_path)

    loss_txt_path = os.path.join(args["save_dir"], "scores", f"train_loss_{args['label_mode']}_{timestamp}.txt")
    np.savetxt(loss_txt_path, np.array(train_loss_history))
    print("[SAVE] train loss ->", loss_txt_path)

    loss_fig_path = os.path.join(args["save_dir"], "figures", f"train_loss_{args['label_mode']}_{timestamp}.png")
    plot_training_loss(
        train_loss_history,
        loss_fig_path,
        title=f"Training Loss (random all nodes) steps={args['train_steps']} lr={args['lr']}"
    )
    print("[SAVE] train loss fig ->", loss_fig_path)

    # Inference
    print("[INFO] start inference...")
    sim_pred, sim_gt, importance_pred, importance_gt, mse_list, sp_sim, sp_imp = inference_all_nodes(
        model=model,
        corr_adj=corr_adj,
        pert_files=pert_files,
        node_feat=node_feat,
        x_s_tensor=x_s_tensor,
        edge_index_s_tensor=edge_index_s_tensor,
        N=N,
        args=args,
        device=device
    )
    print("sim_pred min/mean/max:",
          sim_pred.min(), sim_pred.mean(), sim_pred.max())
    print("sim_gt   min/mean/max:",
          sim_gt.min(), sim_gt.mean(), sim_gt.max())

    # Save importance csv
    score_csv = os.path.join(args["save_dir"], "scores", f"node_importance_{args['label_mode']}_{timestamp}.csv")
    pd.DataFrame({
        "node": np.arange(N),
        "sim_pred": sim_pred,
        "sim_gt": sim_gt,
        "importance_pred": importance_pred,
        "importance_gt": importance_gt,
        "mse": mse_list
    }).to_csv(score_csv, index=False)
    print("[SAVE] importance csv ->", score_csv)

    # Plot importance maps
    importance_pred_norm = enhance_contrast(importance_pred, q_low=5, q_high=95, gamma=0.5)
    importance_gt_norm = enhance_contrast(importance_gt, q_low=5, q_high=95, gamma=0.5)

    imp_fig_pred = os.path.join(args["save_dir"], "figures", f"node_importance_pred_{args['label_mode']}_{timestamp}.svg")
    imp_fig_gt = os.path.join(args["save_dir"], "figures", f"node_importance_gt_{args['label_mode']}_{timestamp}.svg")

    # plot_node_importance_with_index(
    #     node_positions=node_positions,
    #     importance=importance_pred_norm,
    #     edges_base=edge_index_s_np,
    #     draw_edges=args["draw_edges"],
    #     save_path=imp_fig_pred,
    #     title=f"Importance PRED (1-sim_pred) Spearman={sp_imp:.3f}",
    #     node_size=220,
    #     font_size=8
    # )

    TOPK_NODES = 30
    top_nodes_raw = select_top_nodes_by_importance(importance_pred, top_k=TOPK_NODES)

    # 使用K近邻连接策略（推荐）
    plot_node_importance_with_index(
        node_positions=node_positions,
        importance=importance_pred_norm,
        edges_base=edge_index_s_np,
        draw_edges=args["draw_edges"],
        save_path=imp_fig_pred,
        title=f"Nodes Importance PRED",
        node_size=220,
        font_size=8,
        spatial_distance_threshold=10,  # 你想保留距离过滤也可以
        corr_adj=corr_adj,
        top_k_nodes=30,
        top_nodes_override=top_nodes_raw,
        # ✅ 关键：改用“相关性KNN边”，与保存一致
        corr_knn_k=10,
        corr_knn_symmetrize="union",
        use_abs_for_rank=True,
    )

    print("[SAVE] importance pred fig ->", imp_fig_pred)
    print("[SAVE] importance gt fig ->", imp_fig_gt)
    print("[DONE] All results saved to:", args["save_dir"])
    # =========================
    # Build & save corr-KNN subgraph on TOP-30 important nodes
    # =========================
    TAU_MAX_FOR_EVAL = 5

    top_nodes = top_nodes_raw
    sub_out_dir = os.path.join(args["save_dir"], "subgraphs_corr", f"corr_knn_top{TOPK_NODES}_{timestamp}")

    # 你可以扫不同的 knn_k / lag_fixed 做对比
    KNN_LIST = [5, 10, 15]
    LAG_LIST = [1, 0]  # 你说要试 1，也可以试 0
    SYMM = "union"  # 双向

    for lag_fixed in LAG_LIST:
        for knn_k in KNN_LIST:
            g_pred, v_pred = build_pcmci_like_from_corr(
                corr_adj=corr_adj,
                top_nodes=top_nodes,
                knn_k=knn_k,
                tau_max=TAU_MAX_FOR_EVAL,
                lag_fixed=lag_fixed,
                use_abs_for_rank=True,
                keep_signed_weight=True,
                remove_self=True,
                symmetrize=SYMM,

                # ✅ NEW: 与可视化一致的空间过滤
                node_positions=node_positions,
                spatial_distance_threshold=20,  # 这里要与你 plot 里保持一致
            )

            tag = f"top{TOPK_NODES}_knn{knn_k}_{SYMM}_lag{lag_fixed}_tau{TAU_MAX_FOR_EVAL}"
            out_pkl = os.path.join(sub_out_dir, tag, f"subgraph_{tag}.pkl")

            meta = {
                "build_mode": "corr_knn_on_top_nodes",
                "top_k_nodes": int(TOPK_NODES),
                "top_nodes": top_nodes.tolist(),
                "knn_k": int(knn_k),
                "symmetrize": SYMM,
                "tau_max": int(TAU_MAX_FOR_EVAL),
                "lag_fixed": int(lag_fixed),
                "edges_count": int(np.sum(g_pred == "-->")),
                "N": int(g_pred.shape[0]),
                "L": int(g_pred.shape[2]),
                "spatial_distance_threshold": 20,

            }
            save_subgraph_pkl(out_pkl, g_pred, v_pred, meta)


if __name__ == "__main__":
    main()
