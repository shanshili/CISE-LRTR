# -*- coding: utf-8 -*-
r"""
metrics.py —— 评估指标

指标分组
------------------------------------------------------------------
  因果准确性  skel/dir 的 P·R·F1（硬命中）、sNR / sER（软命中）、Cover@r、AUPRC
  结构        Density、Reciprocity、FlowHierarchy、V_Delta、Sources/Sinks
  成本收益    EC、EC_per_edge、Eff_per_EC、E_glob（辅助）、Ent_per_node
  弹性传播    attack_curve / auc、Focus、DescJaccard

网络效率口径
------------------------------------------------------------------
"边多 => 能耗高"。主表用 EC（能耗总量）、EC_per_edge（平均边长）、
Eff_per_EC = E_glob/EC（单位能耗换来的效率）。
E_glob 仅作辅助列：|V| 固定时它近乎单调于 |E|，稠密全双向图必然最高，
不能用它论证精简方法更好。

nan 的含义
------------------------------------------------------------------
nan = 该方法在该指标上没有定义（不是 0）。正文写 "—" 并脚注原因。
典型：无向图的 dir/*、Focus、DescJaccard；无边排序能力方法的 AUPRC。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

__version__ = "1.0"

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ==========================================================
# GT
# ==========================================================

def edges_from_graph(graph, val=None):
    E, W = set(), {}
    for (i, j, t) in np.argwhere(np.asarray(graph) == "-->"):
        k = (int(i), int(j), int(t))
        E.add(k)
        W[k] = float(val[i, j, t]) if val is not None else 1.0
    return E, W


def truncate_gt(E_core, tau_max_pred):
    """GT 到 lag 5，预测只到 lag tau_max_pred —— 超出的 GT 边结构上不可命中，截断并提示。
    正文报截断后的 |E_core|。"""
    lags = np.array([t for (_, _, t) in E_core])
    print("[GT] lag 分布:", {int(l): int((lags == l).sum()) for l in np.unique(lags)})
    kept = [e for e in E_core if e[2] <= tau_max_pred]
    if len(kept) < len(E_core):
        print(f"[GT] {len(E_core)-len(kept)}/{len(E_core)} 条 lag>{tau_max_pred}，已截断。"
              f"正文报 |E_core|={len(kept)}（原 {len(E_core)}）")
    return kept


def skeleton(E) -> Set:
    return {(min(int(i), int(j)), max(int(i), int(j))) for (i, j, *_x) in E}


def directed(E) -> Set:
    return {(int(i), int(j), int(t)) for (i, j, t, *_x) in E}


def prf(gt: Set, pr: Set) -> Dict[str, float]:
    """TP = |E_core ∩ E_pred|，P = TP/|E_pred|，R = TP/|E_core|，F1 = 2PR/(P+R)"""
    tp = len(gt & pr)
    p = tp / len(pr) if pr else 0.0
    r = tp / len(gt) if gt else 0.0
    return {"TP": float(tp), "P": p, "R": r,
            "F1": 2 * p * r / (p + r) if (p + r) > 0 else 0.0}


def pairwise_dist(pos):
    p = np.asarray(pos, float)[:, :2]
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


# ==========================================================
# 1. 因果准确性
# ==========================================================

def soft_node_recall(V_core, V_pred, D, mu=20.0) -> float:
    """sNR = (1/|V_core|) sum_i exp(-min_j dist(v_i, v_j) / mu)
    V_pred 取模型选出的核心节点集（若取"被边碰到的节点"会饱和到 1.0）。"""
    V_core, V_pred = list(V_core), list(V_pred)
    if not V_core: return 1.0
    if not V_pred: return 0.0
    return float(np.mean([np.exp(-float(D[v, V_pred].min()) / mu) for v in V_core]))


def soft_edge_recall(E_core, E_pred, D, mu_e=20.0, mu_t=1.0) -> float:
    """sER = (1/|E_core|) sum_i max_j exp(-(dist(u_i,u_j)+dist(v_i,v_j))/mu_e
                                          - |tau_i - tau_j|/mu_t)"""
    E_core = list(E_core); E_pred = [(i, j, t) for (i, j, t, *_x) in E_pred]
    if not E_core: return 1.0
    if not E_pred: return 0.0
    P = np.array(E_pred)
    return float(np.mean([
        float(np.exp(-(D[u, P[:, 0]] + D[v, P[:, 1]]) / mu_e
                     - np.abs(t - P[:, 2]) / mu_t).max())
        for (u, v, t) in E_core]))


def coverage(V_core, V_pred, D, radii=(0, 5, 10, 20)) -> Dict[str, float]:
    """C(r) = 命中半径 r 内的 GT 核心节点比例。C(0) 即硬命中率；
    C(r) 随 r 增长是"未命中也在附近"的直接量化，比单一 sNR 可读。"""
    V_core, V_pred = list(V_core), list(V_pred)
    if not V_core or not V_pred:
        return {f"Cover@{r}": float("nan") for r in radii}
    dmin = np.array([D[v, V_pred].min() for v in V_core])
    return {f"Cover@{r}": float((dmin <= r).mean()) for r in radii}


def auprc(edge_score, candidate, gt_skel: Set) -> float:
    """无阈值排序质量，在候选集内排序（|E| 自由时 P/R/F1 被边数混淆，这一列不受影响）。
    候选集必须给：否则 N^2·L 个负例会把 AUPRC 压到 ~0，且各方法候选集不同、不可比。
    分数全并列（无权图）或没有边排序能力（B4）-> nan。"""
    if edge_score is None or candidate is None:
        return float("nan")
    S, C = np.asarray(edge_score, float), np.asarray(candidate, float)
    if S.shape[:2] != C.shape[:2]:
        return float("nan")
    d: Dict = {}
    for i, j, l in zip(*np.where(C > 0)):
        if i == j: continue
        l2 = min(int(l), S.shape[2] - 1)
        k = (min(int(i), int(j)), max(int(i), int(j)))
        d[k] = max(d.get(k, -np.inf), float(S[i, j, l2]))
    if not d: return float("nan")
    keys = list(d)
    y = np.array([1 if k in gt_skel else 0 for k in keys])
    s = np.array([d[k] for k in keys])
    if y.sum() == 0 or float(np.std(s)) < 1e-12:
        return float("nan")
    yt = y[np.argsort(-s)]
    tp = np.cumsum(yt)
    return float(_TRAPZ(tp / np.arange(1, len(yt) + 1), tp / yt.sum()))


# ==========================================================
# 2. 结构
# ==========================================================

def structure(G) -> Dict[str, float]:
    """Reciprocity = 双向边占比；FlowHierarchy = 1 - 环上边占比；
    V_Delta = v_i -> v_j <- v_k 且 v_i, v_k 非邻接的 v-structure 计数。"""
    if not isinstance(G, nx.DiGraph) or G.number_of_edges() == 0:
        return {k: float("nan") for k in
                ("Reciprocity", "FlowHierarchy", "V_Delta", "Sources", "Sinks")}
    try: rho = nx.reciprocity(G) or 0.0
    except Exception: rho = 0.0
    try: H = nx.flow_hierarchy(G)
    except Exception: H = 0.0
    vd = 0
    for n in G:
        pr = [p for p in G.predecessors(n) if not G.has_edge(n, p)]
        for a in range(len(pr)):
            for b in range(a + 1, len(pr)):
                if not G.has_edge(pr[a], pr[b]) and not G.has_edge(pr[b], pr[a]):
                    vd += 1
    ind, outd = dict(G.in_degree()), dict(G.out_degree())
    return {"Reciprocity": float(rho), "FlowHierarchy": float(H), "V_Delta": float(vd),
            "Sources": float(sum(1 for n in G if ind[n] == 0 and outd[n] > 0)),
            "Sinks": float(sum(1 for n in G if ind[n] > 0 and outd[n] == 0))}


def density(G) -> float:
    """D = |E_pred| / (|V_pred|(|V_pred|-1))"""
    n = G.number_of_nodes()
    return float(G.number_of_edges() / (n * (n - 1))) if n > 1 else 0.0


# ==========================================================
# 3. 成本 / 收益
# ==========================================================

def global_efficiency(G, n_norm=None) -> float:
    """E_glob = (1/(N(N-1))) sum_{i!=j} 1/d_ij，d_ij 为跳数，对所有节点对求和。
    n_norm：攻击仿真必须传初始 N_0，否则 Q(gamma) 会超过 1。
    辅助列：|V| 固定时近乎单调于 |E|。"""
    n = n_norm if n_norm is not None else len(G)
    if n < 2 or len(G) < 2: return 0.0
    s = 0.0
    for src, lens in nx.all_pairs_shortest_path_length(G):
        for tgt, d in lens.items():
            if src != tgt and d > 0:
                s += 1.0 / d
    return float(s / (n * (n - 1)))


def energy_cost(G, pos) -> Dict[str, float]:
    """EC = sum_{(i,j) in E_pred} dist(v_i, v_j)（线性距离）；EC_per_edge = EC/|E|。
    边多 => 能耗高，这是网络效率口径的主指标。"""
    p = np.asarray(pos, float)
    ec = float(sum(np.linalg.norm(p[u][:2] - p[v][:2]) for u, v in G.edges()))
    m = G.number_of_edges()
    return {"EC": ec, "EC_per_edge": float(ec / m) if m > 0 else float("nan")}


def logdet_entropy(cov, nodes, ridge=1e-4, per_node=True) -> float:
    """Ent = logdet(Cov(X_v))。logdet 随节点数线性增长，默认返回 Ent/k 便于横比。"""
    idx = sorted(set(int(i) for i in nodes))
    if len(idx) < 2: return float("nan")
    sub = cov[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
    sign, ld = np.linalg.slogdet(sub)
    if sign <= 0: return float("nan")
    return float(ld / len(idx)) if per_node else float(ld)


# ==========================================================
# 4. 弹性 / 传播
# ==========================================================

def attack_curve(G, steps=20, seed=0):
    """单次随机移除序列的 Q(gamma) = E_glob(G_rem) / E_glob(G_init)，
    全程用初始 N_0 归一化 -> Q ∈ [0,1]（用剩余节点数归一化会让移除外围节点时 Q>1）。
    返回 (gamma, Q)。run.py 用 20 个不同 seed 出 20 张图。"""
    N0 = G.number_of_nodes()
    gamma = np.linspace(0, 1, steps + 1)
    if N0 < 2: return gamma, np.ones(steps + 1)
    E0 = global_efficiency(G, n_norm=N0)
    if E0 <= 0: return gamma, np.zeros(steps + 1)
    rng = np.random.default_rng(seed)
    order = list(G.nodes()); rng.shuffle(order)
    Gc = G.copy()
    q = np.zeros(steps + 1); q[0] = 1.0
    for s in range(1, steps + 1):
        while Gc.number_of_nodes() > N0 - int(round(gamma[s] * N0)) and order:
            Gc.remove_node(order.pop())
        q[s] = global_efficiency(Gc, n_norm=N0) / E0
    return gamma, q


def auc(gamma, q) -> float:
    return float(_TRAPZ(q, gamma))


def direction_focus(G) -> Dict[str, float]:
    """Focus = (1/|S|) sum_s (1 - |后代(s)| / |同连通分量(s)|)，S 为源节点集。
    量化"传播范围是否可控"：对全双向图（相关 KNN union）后代集 = 连通分量，
    Focus 解析恒为 0。"""
    if not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
        return {"Focus": float("nan"), "Reach_dir": float("nan")}
    U = G.to_undirected()
    src = [n for n in G if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    if not src:
        outd = dict(G.out_degree())
        src = sorted(outd, key=outd.get, reverse=True)[:5]
    fo, rd = [], []
    for s in src:
        d = len(nx.descendants(G, s))
        u = len(nx.node_connected_component(U, s)) - 1
        rd.append(d)
        if u > 0:
            fo.append(1.0 - d / u)
    return {"Focus": float(np.mean(fo)) if fo else float("nan"),
            "Reach_dir": float(np.mean(rd)) if rd else float("nan")}


def descendant_jaccard(G_pred, G_gt) -> Dict[str, float]:
    """J = 后代集与 GT 后代集的 Jaccard 均值，量化"波及得对不对"（而不是波及了多少）。"""
    if not isinstance(G_pred, nx.DiGraph) or G_gt is None:
        return {"DescJaccard": float("nan")}
    tg = [n for n in G_gt if G_gt.in_degree(n) == 0 and G_gt.out_degree(n) > 0]
    if not tg:
        outd = dict(G_gt.out_degree())
        tg = sorted(outd, key=outd.get, reverse=True)[:5]
    js = []
    for s in tg:
        if s not in G_pred: continue
        bp, bg = nx.descendants(G_pred, s), nx.descendants(G_gt, s)
        if bp or bg:
            js.append(len(bp & bg) / max(1, len(bp | bg)))
    return {"DescJaccard": float(np.mean(js)) if js else float("nan")}
