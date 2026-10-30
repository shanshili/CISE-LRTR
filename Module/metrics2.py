# -*- coding: utf-8 -*-
r"""
metrics.py —— 评估指标（整合版）

设计口径
------------------------------------------------------------------
本模块以 subgraph.Subgraph 为唯一输入契约：
    V   : List[int]        原始节点 ID（0..N-1），即 top_nodes，|V|=B
    E   : List[(i,j,l,w)]  已 induced 的核心边，lag 独立保留
    directed / weighted    corr 无向、pcmci 有向
与 GT 比较全程用原始 0..N-1 下标，不做任何 remap（save_pkl 存全尺寸全阵）。

数值量级对齐前序研究
------------------------------------------------------------------
- Ent      : 沿用 evaluate_full_v5 的完整高斯微分熵 + 逐节点归一，cov 用标准化后协方差
- RMSE     : 沿用 v5 的空间 KNN 插值（位置为特征、标准化信号为目标）
- EC       : 沿用 v5 的平方距离（物理布线，去重物理边只算一次）
- P/R/F1   : 沿用 causal_subgraph 的 +1e-12 稳定化
- sER/sNR  : 沿用 causal_subgraph 的高斯核，tau_e=tau_node=20, tau_l=1
- wER      : 沿用 causal_subgraph 的 weighted_edge_recall（GT 权重加权召回）

坍缩 vs 展开（冲突 5）
------------------------------------------------------------------
坍缩图 G_col：i->j <=> 存在任一 lag 的边（v5 口径）。
展开图 G_exp：节点实例 φ(v,t)=tN+v，边 (i,t)->(j,t+l)，纯 lag>=1 时为 DAG。
D / ρ / H / V结构 / AR / Specificity 同时给 _col 与 _exp 两套。
source / sink / |T| / EC / 准确性 为单套（与坍缩无关）。
网络效率维持率只在展开图上算（移除物理节点=删其全部时间副本）。

lag 截断（用户要求：截断与不截断都记录）
------------------------------------------------------------------
GT 到 tau5(L=6)，预测到 tau_max。边级 P/R/F1/sER/wER 同时给：
    _notrunc : 用全部 GT 边（含 lag>tau_max，永不可命中，recall 偏低）
    _trunc   : 只用 lag<=tau_max 的 GT 边（结构上可命中）

nan 的含义
------------------------------------------------------------------
nan = 该指标在该方法上无定义（不是 0）。正文写 "—" 并脚注。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

__version__ = "2.0"

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz
_EPS = 1e-12

# ==== 软指标超参（取自 causal_subgraph / v5）====
TAU_NODE = 20.0
TAU_EDGE = 20.0
TAU_LAG = 1.0
# ==== Ent 正则（取自 v5）====
ENT_ALPHA = 1e-4


# ==========================================================
# 0. 基础：从 GT 图 / 子图边表构造标准结构
# ==========================================================

def edges_from_graph(graph, val=None) -> List[Tuple[int, int, int, float]]:
    """PCMCI 图 [N,N,L] -> [(src,dst,lag,w)]，w 取 val（缺省 1.0）。"""
    g = np.asarray(graph)
    out = []
    for (i, j, l) in np.argwhere(g == "-->"):
        w = float(val[i, j, l]) if val is not None else 1.0
        out.append((int(i), int(j), int(l), w))
    return out


def truncate_edges(E, tau_max) -> List[Tuple[int, int, int, float]]:
    """只保留 lag<=tau_max 的边。"""
    return [e for e in E if int(e[2]) <= int(tau_max)]


def node_set(E) -> Set[int]:
    s: Set[int] = set()
    for (i, j, *_x) in E:
        s.add(int(i)); s.add(int(j))
    return s


def pairwise_dist(pos) -> np.ndarray:
    p = np.asarray(pos, float)[:, :2]
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


# ==========================================================
# 1. 边级 / 节点级 硬命中（保留 lag 三元组）
# ==========================================================

def _prf(gt_keys: Set, pr_keys: Set) -> Dict[str, float]:
    tp = len(gt_keys & pr_keys)
    p = tp / (len(pr_keys) + _EPS)
    r = tp / (len(gt_keys) + _EPS)
    f1 = 2 * p * r / (p + r + _EPS)
    return {"TP": float(tp), "P": float(p), "R": float(r), "F1": float(f1),
            "n_gt": float(len(gt_keys)), "n_pred": float(len(pr_keys))}


def edge_prf(E_gt, E_pred) -> Dict[str, float]:
    """边级 P/R/F1，键为 (src,dst,lag)（保留方向与滞后）。"""
    gt = {(int(i), int(j), int(l)) for (i, j, l, *_w) in E_gt}
    pr = {(int(i), int(j), int(l)) for (i, j, l, *_w) in E_pred}
    return _prf(gt, pr)


def node_prf(V_core, V_pred) -> Dict[str, float]:
    """节点级 P/R/F1。V_core=CORE_NODES，V_pred=top_nodes。"""
    gt = {int(v) for v in V_core}
    pr = {int(v) for v in V_pred}
    return _prf(gt, pr)


# ==========================================================
# 2. 软命中（近邻高斯核）
# ==========================================================

def soft_node_recall(V_core, V_pred, D, tau=TAU_NODE) -> float:
    """sNR = mean_v exp(-min_u d(v,u)/tau)，参照集 V_core=CORE_NODES。"""
    V_core, V_pred = list(V_core), list(V_pred)
    if not V_core:
        return 1.0
    if not V_pred:
        return 0.0
    return float(np.mean([np.exp(-float(D[v, V_pred].min()) / tau) for v in V_core]))


def soft_edge_recall(E_gt, E_pred, D, tau_e=TAU_EDGE, tau_l=TAU_LAG) -> float:
    """sER = mean_gt max_pred exp(-(d(u,u')+d(v,v'))/tau_e - |l-l'|/tau_l)。"""
    E_gt = list(E_gt)
    P = [(int(i), int(j), int(l)) for (i, j, l, *_w) in E_pred]
    if not E_gt:
        return 1.0
    if not P:
        return 0.0
    Pa = np.array(P)
    return float(np.mean([
        float(np.exp(-(D[u, Pa[:, 0]] + D[v, Pa[:, 1]]) / tau_e
                     - np.abs(l - Pa[:, 2]) / tau_l).max())
        for (u, v, l, *_w) in E_gt]))


def weighted_edge_recall(E_gt, E_pred) -> float:
    """wER = sum_{命中}|w_gt| / sum_{gt}|w_gt|（GT 权重加权召回）。"""
    gt_map = {(int(i), int(j), int(l)): abs(float(w)) for (i, j, l, w) in E_gt}
    pr = {(int(i), int(j), int(l)) for (i, j, l, *_w) in E_pred}
    num = sum(w for k, w in gt_map.items() if k in pr)
    den = sum(gt_map.values()) + _EPS
    return float(num / den)


# ==========================================================
# 3. 边级指标：截断 / 不截断 两套
# ==========================================================

def edge_metrics_dual(E_gt_full, E_pred, D, tau_max) -> Dict[str, float]:
    """边级 P/R/F1/sER/wER，给 _notrunc（全 GT）与 _trunc（lag<=tau_max）两套。"""
    out: Dict[str, float] = {}
    for tag, E_gt in (("notrunc", E_gt_full),
                      ("trunc", truncate_edges(E_gt_full, tau_max))):
        prf = edge_prf(E_gt, E_pred)
        out[f"edge_n_gt_{tag}"] = prf["n_gt"]
        out[f"edge_n_pred_{tag}"] = prf["n_pred"]
        out[f"edge_TP_{tag}"] = prf["TP"]
        out[f"edge_P_{tag}"] = prf["P"]
        out[f"edge_R_{tag}"] = prf["R"]
        out[f"edge_F1_{tag}"] = prf["F1"]
        out[f"sER_{tag}"] = soft_edge_recall(E_gt, E_pred, D)
        out[f"wER_{tag}"] = weighted_edge_recall(E_gt, E_pred)
    return out


def node_metrics(V_core, V_pred, D) -> Dict[str, float]:
    """节点级 P/R/F1 + sNR（单套；节点与 lag 无关）。"""
    prf = node_prf(V_core, V_pred)
    return {"node_n_gt": prf["n_gt"], "node_n_pred": prf["n_pred"],
            "node_TP": prf["TP"], "node_P": prf["P"], "node_R": prf["R"],
            "node_F1": prf["F1"], "sNR": soft_node_recall(V_core, V_pred, D)}


# ==========================================================
# 4. Task：RMSE（v5 空间 KNN 插值）+ Ent（v5 高斯微分熵）
# ==========================================================

def compute_rmse(V_pred, positions, std_data) -> float:
    """空间 KNN 插值 RMSE（v5 口径）。
    选点 S=V_pred 为训练，其余节点为测试；位置为特征、标准化信号为目标。"""
    from sklearn.neighbors import KNeighborsRegressor
    N = std_data.shape[1]
    S = sorted({int(v) for v in V_pred if int(v) < N})
    if not S:
        return 999.0
    Q = np.setdiff1d(np.arange(N), S)
    if len(Q) == 0:
        return 0.0
    Xtr = np.array([positions.get(i, (0.0, 0.0)) for i in S])
    Xte = np.array([positions.get(i, (0.0, 0.0)) for i in Q])
    Ytr = std_data[:, S].T
    Yte = std_data[:, Q].T
    try:
        knn = KNeighborsRegressor(n_neighbors=min(5, len(S)), weights="distance")
        knn.fit(Xtr, Ytr)
        Yp = knn.predict(Xte)
        return float(np.sqrt(np.mean((Yte - Yp) ** 2)))
    except Exception:
        return 999.0


def compute_entropy(V_pred, cov_std, normalize=True, alpha=ENT_ALPHA) -> float:
    """完整高斯微分熵 + 逐节点归一（v5 口径）。cov_std 必须是标准化后协方差。
        H = 0.5*ln|Σ_S+αI| + 0.5*k*(1+ln 2π)，返回 H/k（normalize=True）。"""
    idx = sorted({int(v) for v in V_pred})
    k = len(idx)
    if k < 1:
        return 0.0
    sub = cov_std[np.ix_(idx, idx)] + np.eye(k) * alpha
    sign, logdet = np.linalg.slogdet(sub)
    if sign <= 0:
        return -999.0
    const = 0.5 * k * (1 + np.log(2 * np.pi))
    ent = 0.5 * logdet + const
    return float(ent / k) if normalize else float(ent)


# ==========================================================
# 5. Cost：EC（v5 平方距离，去重物理边）+ Density（双套）
# ==========================================================

def physical_edges(E) -> Set[Tuple[int, int]]:
    """去重物理边（无向对，忽略 lag/方向）：{(min,max)}。"""
    return {(min(int(i), int(j)), max(int(i), int(j))) for (i, j, *_x) in E}


def energy_cost(E, positions) -> float:
    """EC = sum_{物理边} ||p_i - p_j||^2（v5 平方距离，每条物理连线只算一次）。"""
    ec = 0.0
    for (i, j) in physical_edges(E):
        p1 = np.asarray(positions.get(int(i), (0.0, 0.0)), float)
        p2 = np.asarray(positions.get(int(j), (0.0, 0.0)), float)
        ec += float(np.dot(p1 - p2, p1 - p2))
    return ec


def density_collapsed(G_col) -> float:
    return float(nx.density(G_col))


def density_expanded(n_edges_exp, B, L) -> float:
    M = B * L
    return float(n_edges_exp / (M * (M - 1))) if M > 1 else 0.0


# ==========================================================
# 6. 图构造：坍缩图 / 展开图
# ==========================================================

def build_collapsed(V, E, directed) -> nx.Graph:
    """坍缩图 G_col：i->j <=> 存在任一 lag 的边（v5 口径）。"""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from([int(v) for v in V])
    for (i, j, l, w) in E:
        G.add_edge(int(i), int(j), weight=float(w), lag=int(l))
    return G


def build_expanded(V, E, N, L, directed):
    """展开图 G_exp：节点 φ(v,t)=tN+v（v∈V, t=0..L-1），边 (i,t)->(j,t+l)。
    返回 (G_exp, node_groups)；node_groups[v] = [tN+v for t]，攻击按组删。"""
    G = nx.DiGraph() if directed else nx.Graph()
    Vs = [int(v) for v in V]
    for v in Vs:
        for t in range(L):
            G.add_node(t * N + v)
    for (i, j, l, w) in E:
        for t in range(L - int(l)):
            a = t * N + int(i)
            b = (t + int(l)) * N + int(j)
            G.add_edge(a, b, weight=float(w))
    node_groups = {v: [t * N + v for t in range(L)] for v in Vs}
    return G, node_groups


# ==========================================================
# 7. 因果结构：坍缩 / 展开 双套
# ==========================================================

def _reciprocity(G: nx.DiGraph) -> float:
    if not isinstance(G, nx.DiGraph) or G.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.reciprocity(G) or 0.0)
    except Exception:
        return 0.0


def _flow_hierarchy(G: nx.DiGraph) -> float:
    if not isinstance(G, nx.DiGraph) or G.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.flow_hierarchy(G))
    except Exception:
        return 0.0


def _v_structures(G: nx.DiGraph) -> int:
    """严格对撞 v-structure 计数（v5 口径）：
    对每个 n，纯父 pa（无 n->pa 反向边）中两两既无边者计一次。"""
    if not isinstance(G, nx.DiGraph):
        return 0
    vd = 0
    for n in G.nodes():
        pa = [p for p in G.predecessors(n) if not G.has_edge(n, p)]
        for a in range(len(pa)):
            for b in range(a + 1, len(pa)):
                if not G.has_edge(pa[a], pa[b]) and not G.has_edge(pa[b], pa[a]):
                    vd += 1
    return vd


def source_sink_from_edges(V, E) -> Dict[str, float]:
    """source/sink 由"任一 lag 是否有入/出边"决定（坍缩不改变 -> 单套）。"""
    Vs = {int(v) for v in V}
    din = {v: 0 for v in Vs}
    dout = {v: 0 for v in Vs}
    for (i, j, l, w) in E:
        i, j = int(i), int(j)
        if i in dout:
            dout[i] += 1
        if j in din:
            din[j] += 1
    src = sum(1 for v in Vs if din[v] == 0 and dout[v] > 0)
    snk = sum(1 for v in Vs if din[v] > 0 and dout[v] == 0)
    return {"Sources": float(src), "Sinks": float(snk)}


def _v_structures_expanded(V, E, N, L) -> int:
    """展开图 v-structure（去重成物理三元组 {i,k,j}）：
    碰撞点 k 的物理父 i,j 若在物理骨架上无边（任一 lag 均无 i-j），计一对。"""
    Vs = {int(v) for v in V}
    # 物理骨架（无向，忽略 lag）
    skel = set()
    parents = {v: set() for v in Vs}
    for (i, j, l, w) in E:
        i, j = int(i), int(j)
        skel.add((min(i, j), max(i, j)))
        if j in parents and i in Vs:
            parents[j].add(i)
    vd = 0
    for k in Vs:
        pa = sorted(parents[k])
        for a in range(len(pa)):
            for b in range(a + 1, len(pa)):
                i, j = pa[a], pa[b]
                if (min(i, j), max(i, j)) not in skel:
                    vd += 1
    return vd


def causal_structure_dual(V, E, N, L, directed) -> Dict[str, float]:
    """ρ/H/V结构 双套 + source/sink 单套。"""
    out: Dict[str, float] = {}
    # 单套：source / sink
    out.update(source_sink_from_edges(V, E))
    if not directed:
        # 无向图（corr）：有向语义指标无定义
        for tag in ("col", "exp"):
            out[f"Reciprocity_{tag}"] = float("nan")
            out[f"FlowHierarchy_{tag}"] = float("nan")
            out[f"V_struct_{tag}"] = float("nan")
        return out
    # 坍缩
    Gc = build_collapsed(V, E, directed=True)
    out["Reciprocity_col"] = _reciprocity(Gc)
    out["FlowHierarchy_col"] = _flow_hierarchy(Gc)
    out["V_struct_col"] = float(_v_structures(Gc))
    # 展开
    Ge, _ = build_expanded(V, E, N, L, directed=True)
    out["Reciprocity_exp"] = _reciprocity(Ge)
    out["FlowHierarchy_exp"] = _flow_hierarchy(Ge)
    out["V_struct_exp"] = float(_v_structures_expanded(V, E, N, L))
    return out


# ==========================================================
# 8. 干预分析：坍缩 / 展开 双套
# ==========================================================

def _targets(G_col: nx.DiGraph, V) -> List[int]:
    """目标集：source 节点；无 source 时取出度 top-5（v5 逻辑）。"""
    if isinstance(G_col, nx.DiGraph) and G_col.number_of_nodes() > 0:
        src = [n for n in G_col if G_col.in_degree(n) == 0 and G_col.out_degree(n) > 0]
        if src:
            return src
        outd = dict(G_col.out_degree())
        return sorted(outd, key=outd.get, reverse=True)[:5]
    return list(V)[:5]


def intervention_dual(V, E, N, L, directed) -> Dict[str, float]:
    """|T|（单套）+ AR/Specificity 双套。无向图返回 nan。"""
    B = len(V)
    if not directed:
        return {"Num_Targets": float("nan"),
                "AR_col": float("nan"), "Specificity_col": float("nan"),
                "AR_exp": float("nan"), "Specificity_exp": float("nan")}
    Gc = build_collapsed(V, E, directed=True)
    T = _targets(Gc, V)
    # 坍缩 reach
    reach_col = [len(nx.descendants(Gc, s)) for s in T if s in Gc]
    AR_col = float(np.mean(reach_col)) if reach_col else 0.0
    # 展开 reach（映回物理节点去重）
    Ge, _ = build_expanded(V, E, N, L, directed=True)
    reach_exp = []
    for s in T:
        src_inst = 0 * N + int(s)  # 最早层 t=0
        if src_inst not in Ge:
            # s 在 t=0 无实例（不太可能，V 全部建了 t=0..L-1），退回任意存在实例
            cand = [t * N + int(s) for t in range(L) if (t * N + int(s)) in Ge]
            if not cand:
                continue
            src_inst = cand[0]
        desc = nx.descendants(Ge, src_inst)
        phys = {d % N for d in desc}
        reach_exp.append(len(phys))
    AR_exp = float(np.mean(reach_exp)) if reach_exp else 0.0
    return {"Num_Targets": float(len(T)),
            "AR_col": AR_col, "Specificity_col": float(1.0 - AR_col / max(1, B)),
            "AR_exp": AR_exp, "Specificity_exp": float(1.0 - AR_exp / max(1, B))}


# ==========================================================
# 9. 网络效率维持率（展开图，v5 归一化）
# ==========================================================

def global_efficiency_v5(G) -> float:
    """E_glob = (1/(n(n-1))) sum_{i!=j} 1/d_hop（v5 口径，当前节点数 n 归一）。"""
    n = len(G)
    if n < 2:
        return 0.0
    s = 0.0
    max_possible = n * (n - 1) / 2  # 完全图的最大可能效率
    for src, lens in nx.all_pairs_shortest_path_length(G):
        for tgt, d in lens.items():
            if src != tgt and d > 0:
                s += 1.0 / d
    return float(np.clip(s / max_possible, 0.0, 1.0))
    #return float(s / (n * (n - 1)))


def resilience_expanded(V, E, N, L, directed, steps=10, repeats=20, seed=0):
    """网络效率维持率（展开图）。移除一个物理节点=删其全部时间副本。
    返回 (gamma, Q_mean)。20 次随机序列取均值（v5 口径）。"""
    Ge, groups = build_expanded(V, E, N, L, directed)
    phys_nodes = list(groups.keys())
    B = len(phys_nodes)
    gamma = np.array([i / steps for i in range(steps + 1)])
    if B < 2:
        # return gamma, np.ones(steps + 1)
        return gamma, np.ones((repeats, steps + 1))  # 返回完整的矩阵
    E0 = global_efficiency_v5(Ge)
    if E0 <= 0:
        #return gamma, np.zeros(steps + 1)
        return gamma, np.zeros((repeats, steps + 1))
    rng = np.random.default_rng(seed)
    all_q = []
    for _ in range(repeats):
        order = phys_nodes.copy()
        rng.shuffle(order)
        Gc = Ge.copy()
        removed = 0                       # 已删除的物理节点数
        q = np.zeros(steps + 1)
        q[0] = 1.0
        for s in range(1, steps + 1):
            target = int(round(gamma[s] * B))     # 到第 s 步应累计删除的物理节点数
            while removed < target and order:
                v = order.pop(0)
                for inst in groups[v]:            # 删该物理节点的全部时间副本
                    if inst in Gc:
                        Gc.remove_node(inst)
                removed += 1
            current_eff = global_efficiency_v5(Gc)
            q[s] = current_eff / E0 if E0 > 0 else 0.0
            # q[s] = global_efficiency_v5(Gc) / E0
        all_q.append(q)
    # return gamma, np.mean(all_q, axis=0)
    return gamma, np.array(all_q)  # 返回完整的 [repeats, steps+1] 矩阵


def resilience_auc(gamma, q) -> float:
    """计算弹性曲线 AUC（确保返回标量）"""
    if isinstance(q, np.ndarray) and q.ndim == 1:
        return float(_TRAPZ(q, gamma))
    elif isinstance(q, np.ndarray) and q.ndim == 2:
        return float(_TRAPZ(q.mean(axis=0), gamma))
    else:
        return 0.0
    # 确保 AUC 值在合理范围内
    return float(np.clip(auc_val, 0.0, 1.0))


# ==========================================================
# 10. 顶层：一次性评估一个子图
# ==========================================================

def evaluate_subgraph(V, E, *, E_gt_full, gt_val_shape_L,
                      positions, std_data, cov_std, core_nodes,
                      N, tau_max, directed,
                      resil_steps=10, resil_repeats=20) -> Dict[str, float]:
    """
    评估一个核心子图，返回扁平 dict。

    参数
    ----
    V, E          : Subgraph.V / Subgraph.E
    E_gt_full     : GT 全部边 [(i,j,l,w)]（不截断）
    positions     : {node_id: (x,y)}
    std_data      : [T,N] 标准化后信号（供 RMSE）
    cov_std       : [N,N] 标准化后协方差（供 Ent）
    core_nodes    : CORE_NODES 硬编码核心节点集（供节点 P/R/F1 与 sNR）
    N, tau_max    : 节点总数、最大滞后
    directed      : 该子图是否有向
    """
    L = int(tau_max) + 1
    D = pairwise_dist([positions[i] if i in positions else (0.0, 0.0)
                       for i in range(N)])
    out: Dict[str, float] = {}

    # ---- 边级（截断/不截断双套）----
    out.update(edge_metrics_dual(E_gt_full, E, D, tau_max))
    # ---- 节点级（单套，参照 CORE_NODES）----
    out.update(node_metrics(core_nodes, V, D))
    # ---- Task ----
    out["RMSE"] = compute_rmse(V, positions, std_data)
    out["Ent"] = compute_entropy(V, cov_std, normalize=True)
    # ---- Cost ----
    out["EC"] = energy_cost(E, positions)
    Gc = build_collapsed(V, E, directed)
    out["Density_col"] = density_collapsed(Gc)
    Ge, _ = build_expanded(V, E, N, L, directed)
    out["Density_exp"] = density_expanded(Ge.number_of_edges(), len(V), L)
    # ---- Causal 结构（双套 + source/sink 单套）----
    out.update(causal_structure_dual(V, E, N, L, directed))
    # ---- Intervention（双套 + |T| 单套）----
    out.update(intervention_dual(V, E, N, L, directed))
    # ---- 网络效率维持率（展开图）----
    gamma, q = resilience_expanded(V, E, N, L, directed,
                                   steps=resil_steps, repeats=resil_repeats)
    out["Resil_AUC_exp"] = resilience_auc(gamma, q)
    out["_resil_gamma"] = gamma
    out["_resil_q"] = q
    return out
