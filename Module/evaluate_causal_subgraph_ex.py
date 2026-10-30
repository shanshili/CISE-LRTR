# Module/evaluate_causal_subgraph.py
"""
Evaluate predicted causal subgraph vs groundtruth core subgraph.

Supports:
- Hard match: edge/node Precision/Recall/F1 (direct (src,dst,lag) match)
- Weighted metrics: weighted edge recall (wER), weighted IoU (wIoU) using continuous weights
- Soft match (near-miss): node/edge soft recall based on Euclidean distance (node_positions.csv)
- Optional Top-K evaluation to reduce threshold sensitivity

Assumptions:
- Graph format is PCMCI-like:
  graph: ndarray [N, N, L] with '-->' marking directed edges
  val_matrix: ndarray [N, N, L] with signed weights (continuous allowed)
- Positions: CSV with columns ['x','y'] and optionally node_id

Author: (generated)
"""

from __future__ import annotations

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Set


# =========================
# I/O helpers
# =========================

def load_pcmci_pkl(pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a PCMCI pkl that stores {'graph':..., 'val_matrix':...}."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "graph" in data and "val_matrix" in data:
        graph = data["graph"]
        val = data["val_matrix"]
    else:
        raise ValueError(f"Unrecognized PCMCI pkl format: {pkl_path}")
    graph = np.array(graph)
    val = np.array(val, dtype=float)
    return graph, val


def load_subgraph_pkl(pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predicted subgraph pkl saved by visualize_mask_subgraph.py:
    {'graph': subgraph, 'val_matrix': subval_matrix, ...}
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "graph" not in data or "val_matrix" not in data:
        raise ValueError(f"Unrecognized subgraph pkl format: {pkl_path}")
    graph = np.array(data["graph"])
    val = np.array(data["val_matrix"], dtype=float)
    return graph, val


def load_positions_csv(csv_path: str, N: int) -> np.ndarray:
    """Load node positions. Expected columns: x,y (node_id optional)."""
    df = pd.read_csv(csv_path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"positions csv must contain x,y columns: {csv_path}")

    # If node_id exists, sort by node_id
    if "node_id" in df.columns:
        df = df.sort_values("node_id")
    pos = df[["x", "y"]].values.astype(float)

    if pos.shape[0] != N:
        raise ValueError(f"Positions row count {pos.shape[0]} != N={N}")
    return pos


# =========================
# Core extraction
# =========================

def edges_from_graph(graph: np.ndarray, val: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, float]]:
    """
    Extract edges list from PCMCI graph.
    Returns list of (src, dst, lag, weight).
    """
    N, N2, L = graph.shape
    assert N == N2
    edges = []
    for src in range(N):
        for dst in range(N):
            for lag in range(L):
                if graph[src, dst, lag] == "-->":
                    w = float(val[src, dst, lag]) if val is not None else 1.0
                    edges.append((src, dst, lag, w))
    return edges


def node_set_from_edges(edges: List[Tuple[int, int, int, float]]) -> Set[int]:
    s: Set[int] = set()
    for u, v, lag, w in edges:
        s.add(u); s.add(v)
    return s


# =========================
# Hard metrics
# =========================

def prf_from_edge_sets(E_gt: List[Tuple[int, int, int, float]],
                       E_pred: List[Tuple[int, int, int, float]]) -> Dict[str, float]:
    gt = set((u, v, lag) for u, v, lag, w in E_gt)
    pr = set((u, v, lag) for u, v, lag, w in E_pred)
    inter = gt & pr
    prec = len(inter) / (len(pr) + 1e-12)
    rec = len(inter) / (len(gt) + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(len(inter)), "gt": float(len(gt)), "pred": float(len(pr))}


def prf_from_node_sets(V_gt: Set[int], V_pred: Set[int]) -> Dict[str, float]:
    inter = V_gt & V_pred
    prec = len(inter) / (len(V_pred) + 1e-12)
    rec = len(inter) / (len(V_gt) + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(len(inter)), "gt": float(len(V_gt)), "pred": float(len(V_pred))}


# =========================
# Weighted metrics
# =========================

def weighted_edge_recall(E_gt: List[Tuple[int, int, int, float]],
                         E_pred: List[Tuple[int, int, int, float]]) -> float:
    """wER = sum_{hit} |w_gt| / sum_{gt} |w_gt|"""
    gt_map = {(u, v, lag): abs(w) for u, v, lag, w in E_gt}
    pr_set = set((u, v, lag) for u, v, lag, w in E_pred)
    num = sum(w for k, w in gt_map.items() if k in pr_set)
    den = sum(gt_map.values()) + 1e-12
    return float(num / den)


def weighted_iou(E_gt: List[Tuple[int, int, int, float]],
                 E_pred: List[Tuple[int, int, int, float]]) -> float:
    """
    wIoU over |w| on union keys:
    sum min(|w_gt|, |w_pred|) / sum max(|w_gt|, |w_pred|)
    """
    gt = {(u, v, lag): abs(w) for u, v, lag, w in E_gt}
    pr = {(u, v, lag): abs(w) for u, v, lag, w in E_pred}
    keys = set(gt.keys()) | set(pr.keys())
    num = sum(min(gt.get(k, 0.0), pr.get(k, 0.0)) for k in keys)
    den = sum(max(gt.get(k, 0.0), pr.get(k, 0.0)) for k in keys) + 1e-12
    return float(num / den)


# =========================
# Soft metrics (Euclidean near-miss)
# =========================

def pairwise_dist_matrix(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def soft_node_recall(V_core: Set[int], V_pred: Set[int],
                     dist_mat: np.ndarray, tau: float) -> float:
    """
    For each core node v, score = exp(-min_{u in V_pred} d(v,u) / tau)
    Average over core nodes.
    """
    if len(V_core) == 0:
        return 1.0
    if len(V_pred) == 0:
        return 0.0
    V_pred_list = list(V_pred)
    scores = []
    for v in V_core:
        d = float(dist_mat[v, V_pred_list].min())
        scores.append(np.exp(-d / tau))
    return float(np.mean(scores))


def soft_edge_recall(E_gt: List[Tuple[int, int, int, float]],
                     E_pred: List[Tuple[int, int, int, float]],
                     dist_mat: np.ndarray,
                     tau_e: float, tau_l: float) -> float:
    """
    For each gt edge (u->v,lag), best sim over predicted edges:
      sim = exp(-(d(u,uh)+d(v,vh))/tau_e) * exp(-|lag-lagh|/tau_l)
    Average over gt edges.
    """
    if len(E_gt) == 0:
        return 1.0
    if len(E_pred) == 0:
        return 0.0

    pred_triples = [(u, v, lag) for u, v, lag, w in E_pred]
    scores = []
    for u, v, lag, w in E_gt:
        best = 0.0
        for uh, vh, lagh in pred_triples:
            du = dist_mat[u, uh]
            dv = dist_mat[v, vh]
            s = np.exp(-(du + dv) / tau_e) * np.exp(-abs(lag - lagh) / tau_l)
            if s > best:
                best = float(s)
        scores.append(best)
    return float(np.mean(scores))


# =========================
# Optional: Top-K evaluation
# =========================

def topk_edges_from_scores(score_tensor: np.ndarray, k: int) -> List[Tuple[int, int, int, float]]:
    """
    score_tensor: [N,N,L] (e.g., mask or |mask*val|)
    return top-k edges as (u,v,lag,score) excluding self if you want (here keep self).
    """
    N, _, L = score_tensor.shape
    flat = score_tensor.reshape(-1)
    if k <= 0:
        return []
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k-1)[:k]
    # sort for reproducibility
    idx = idx[np.argsort(-flat[idx])]
    edges = []
    for id_ in idx:
        u = id_ // (N * L)
        rem = id_ % (N * L)
        v = rem // L
        lag = rem % L
        edges.append((int(u), int(v), int(lag), float(score_tensor[u, v, lag])))
    return edges


# =========================
# Main evaluation
# =========================

def evaluate(gt_graph: np.ndarray, gt_val: np.ndarray,
             pred_graph: np.ndarray, pred_val: np.ndarray,
             positions: np.ndarray,
             core_nodes: Optional[List[int]] = None,
             tau_node: float = 20.0,
             tau_edge: float = 20.0,
             tau_lag: float = 1.0,
             do_topk: bool = True) -> Dict[str, float]:

    E_gt = edges_from_graph(gt_graph, gt_val)
    E_pr = edges_from_graph(pred_graph, pred_val)

    V_gt = node_set_from_edges(E_gt)
    V_pr = node_set_from_edges(E_pr)

    edge_prf = prf_from_edge_sets(E_gt, E_pr)
    node_prf = prf_from_node_sets(V_gt, V_pr)

    wer = weighted_edge_recall(E_gt, E_pr)
    wiou = weighted_iou(E_gt, E_pr)

    D = pairwise_dist_matrix(positions)

    if core_nodes is None:
        V_core = V_gt  # fallback
    else:
        V_core = set(core_nodes)

    sNR = soft_node_recall(V_core, V_pr, D, tau=tau_node)
    sER = soft_edge_recall(E_gt, E_pr, D, tau_e=tau_edge, tau_l=tau_lag)

    out = {}
    # Hard
    out.update({f"edge_{k}": v for k, v in edge_prf.items()})
    out.update({f"node_{k}": v for k, v in node_prf.items()})
    # Weighted
    out["wER"] = wer
    out["wIoU"] = wiou
    # Soft
    out["sNodeRecall"] = sNR
    out["sEdgeRecall"] = sER

    # Top-K optional: compare under same budget as GT edges
    if do_topk:
        K = int(edge_prf["gt"])
        # Use score = abs(pred_val) as ranking source (you can replace by mask itself if you have it)
        score = np.abs(pred_val)
        topk = topk_edges_from_scores(score, K)
        # Convert to edge set format, but need a "graph" notion: treat as predicted edges
        # We'll compare (u,v,lag) keys against GT keys
        topk_prf = prf_from_edge_sets(E_gt, topk)
        out.update({f"topK_edge_{k}": v for k, v in topk_prf.items()})
        out["topK_K"] = float(K)

    return out


def print_report(metrics: Dict[str, float]) -> None:
    def pct(x): return f"{100*x:.2f}%"

    print("\n================ Evaluation Report ================")
    print(f"Edges: GT={int(metrics['edge_gt'])} | Pred={int(metrics['edge_pred'])} | TP={int(metrics['edge_tp'])}")
    print(f"  Hard Edge Precision: {pct(metrics['edge_precision'])}")
    print(f"  Hard Edge Recall:    {pct(metrics['edge_recall'])}")
    print(f"  Hard Edge F1:        {pct(metrics['edge_f1'])}")

    print(f"\nNodes: GT={int(metrics['node_gt'])} | Pred={int(metrics['node_pred'])} | TP={int(metrics['node_tp'])}")
    print(f"  Hard Node Precision: {pct(metrics['node_precision'])}")
    print(f"  Hard Node Recall:    {pct(metrics['node_recall'])}")
    print(f"  Hard Node F1:        {pct(metrics['node_f1'])}")

    print("\nWeighted:")
    print(f"  wER   (weighted edge recall): {pct(metrics['wER'])}")
    print(f"  wIoU  (weighted IoU):         {pct(metrics['wIoU'])}")

    print("\nSoft (near-miss with Euclidean distance):")
    print(f"  sNodeRecall (tau_node): {pct(metrics['sNodeRecall'])}")
    print(f"  sEdgeRecall (tau_edge,tau_lag): {pct(metrics['sEdgeRecall'])}")

    if "topK_edge_precision" in metrics:
        print("\nTop-K (budget = |E_gt|, ranking by |pred_val|):")
        print(f"  K = {int(metrics['topK_K'])}")
        print(f"  Edge Precision@K: {pct(metrics['topK_edge_precision'])}")
        print(f"  Edge Recall@K:    {pct(metrics['topK_edge_recall'])}")
        print(f"  Edge F1@K:        {pct(metrics['topK_edge_f1'])}")

    print("===================================================\n")


def main():
    # ===============================
    # ======= 固定配置区域 ==========
    # ===============================

    GT_PKL = "../dataset/simulation_hubs_core/N500_T10000_Hubs40/Env_A_gt_core_pcmci_tau5.pkl"

    # PRED_PKL = (
    #     "../mask_subgraph_visualization/"
    #     "N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3/"
    #     "threshold_0.8/subgraph_threshold_0.8.pkl"
    # )
    PRED_PKL = (
        #"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"
        #"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results2_508_4\pkl\B2_GCN_LRTR.pkl"
        # "C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results2_508_4\pkl\B3_CORR_INV.pkl"
        # "C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results2_508_4\pkl\B4_NOCGBP.pkl"
         "../mask_subgraph_visualization_ex/N500/threshold_0.1323/subgraph_threshold_0.1323.pkl"
         # "../mask_subgraph_visualization_ex/N1000/threshold_0.39/subgraph_threshold_0.39.pkl"
    )

    POS_CSV = (
        "../dataset/simulation_hubs_core/"
        "N500_T10000_Hubs40/"
        "node_positions.csv"
    )

    # 核心节点（hub nodes）
    # ⚠️ 请把 ... 替换成真实的节点编号
    CORE_NODES = [
        # 96, 11, 54, 51, 41, 15, 38, 62, 12, 52, 75, 89, 10, 8, 61

        42, 295, 479, 54, 333, 349, 34, 385, 188, 377, 341, 231, 421, 450, 161, 307, 99, 47, 414, 206, 181, 82, 447,
        270, 9, 131, 68, 389, 345, 362, 379, 417, 491, 280, 70, 183, 449, 35, 18, 444

        # 717, 105, 690, 683, 521, 890, 230, 990, 605, 330, 922, 859, 255, 165, 756, 908, 354, 499, 912, 486, 199, 701,
        # 79, 931, 243, 784, 321, 810, 470, 302, 474, 14, 880, 854,
        # 196, 510, 344, 538, 958, 846, 325, 150, 457, 832, 228, 906, 681, 556, 149, 641, 783, 957, 883, 262, 573, 989,
        # 427, 982, 136, 904, 498, 853, 5, 715, 229, 862, 35, 501, 256, 818, 845, 685, 825, 544, 615, 315, 838, 750, 1,
        # 179
    ]

    # Soft-metric 超参数
    TAU_NODE = 10.0   # 核心节点“附近”的空间尺度
    TAU_EDGE = 10.0   # 边端点空间尺度
    TAU_LAG  = 1.0    # 滞后差异尺度

    DO_TOPK = True    # 是否启用 Top-K 评估

    # ===============================
    # ========== 加载数据 ===========
    # ===============================

    gt_graph, gt_val = load_pcmci_pkl(GT_PKL)
    pred_graph, pred_val = load_subgraph_pkl(PRED_PKL)

    N = gt_graph.shape[0]
    positions = load_positions_csv(POS_CSV, N=N)

    # ===============================
    # ========== 评估 ===========
    # ===============================

    metrics = evaluate(
        gt_graph=gt_graph,
        gt_val=gt_val,
        pred_graph=pred_graph,
        pred_val=pred_val,
        positions=positions,
        core_nodes=CORE_NODES if len(CORE_NODES) > 0 else None,
        tau_node=TAU_NODE,
        tau_edge=TAU_EDGE,
        tau_lag=TAU_LAG,
        do_topk=DO_TOPK
    )

    print_report(metrics)


if __name__ == "__main__":
    main()
