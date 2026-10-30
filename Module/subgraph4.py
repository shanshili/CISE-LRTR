# -*- coding: utf-8 -*-
r"""
subgraph3.py —— 掩码驱动的核心节点与邻接边筛选

本模块采用 visualize_mask_subgraph_clear.py 的筛选口径，同时保持原有
Subgraph / build / save_pkl / from_pkl 等接口不变：

1. 上游先用掩码阈值得到强掩码区域，并形成边级分数 s_E。
2. 节点分数 s_V 为强掩码区域内效应的入边和 + 出边和。
3. 所有方法统一按 s_V 选择 Top-B 核心节点，不再按边排序累积端点。
4. 默认 edge_keep_mode='or'：边的任意一端属于 Top-B 即通过节点门控；
   设为 'and' 时，两端都必须属于 Top-B。
5. 最终边仍须属于对应底图；掩码型方法还须满足 s_E > 0。
   PCMCI 边还会删除 |val| <= 1e-6 的近零效应。

因此 Subgraph.V 仍表示 Top-B 核心节点，Subgraph.E 在默认 OR 模式下可包含
核心节点的一阶非核心邻居。这与目标可视化脚本保存 top_nodes、但在完整 N×N×L
矩阵中保留核心节点邻接边的语义一致。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

__version__ = "1.1"

# ======== 节点截止阈值：唯一来源 ========
NODE_BUDGET = 30


@dataclass
class Subgraph:
    name: str
    V: List[int]
    E: List[Tuple[int, int, int, float]]      # (i, j, tau, w)
    directed: bool
    weighted: bool
    node_score: Optional[np.ndarray] = None
    edge_score: Optional[np.ndarray] = None   # [N,N,L]；无边排序能力的方法为 None
    candidate: Optional[np.ndarray] = None    # [N,N,L] AUPRC 候选集
    branch: str = "edge"
    induce: str = ""
    meta: Dict = field(default_factory=dict)

    def to_nx(self):
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from([int(v) for v in self.V])
        for (i, j, t, w) in self.E:
            G.add_edge(int(i), int(j), weight=float(w), lag=int(t))
        return G

    def skeleton(self) -> Set:
        return {(min(int(i), int(j)), max(int(i), int(j))) for (i, j, *_x) in self.E}

    def directed_keys(self) -> Set:
        return {(int(i), int(j), int(t)) for (i, j, t, *_x) in self.E}

    def line(self) -> str:
        return (f"{self.name:<16} |V|={len(self.V):<3} |E|={len(self.E):<4} "
                f"{'有向' if self.directed else '无向'} "
                f"{'有权' if self.weighted else '无权'}  "
                f"branch={self.branch:<5} induce={self.induce}")


# ==========================================================
# 第一段：选节点
# ==========================================================

def node_scores_from_masked_effect(sE, mode="inout") -> np.ndarray:
    """由已过阈值的掩码效应 sE 计算节点分数。"""
    S = np.abs(np.asarray(sE, float))
    if S.ndim != 3:
        raise ValueError(f"sE 必须是 [N,N,L]，实际为 {S.shape}")
    out_score = S.sum(axis=(1, 2))
    in_score = S.sum(axis=(0, 2))
    if mode == "out":
        return out_score
    if mode == "in":
        return in_score
    if mode == "inout":
        return out_score + in_score
    raise ValueError(f"Unknown node_score_mode: {mode}")


def nodes_by_node_score(sV, B=None) -> List[int]:
    B = NODE_BUDGET if B is None else B
    s = np.asarray(sV, float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"node_score 必须是一维非空数组，实际为 {s.shape}")
    B = min(max(1, int(B)), int(s.size))
    if float(np.std(s)) < 1e-12:
        print("  [WARN] node_score 全部并列，Top-B 将按原节点顺序确定")
    return [int(v) for v in np.argsort(-s, kind="stable")[:B]]


def nodes_by_edge_score(sE, B=None, fallback=None) -> List[int]:
    """兼容旧调用名；新协议不再逐边累积端点，而是按入+出聚合后选点。"""
    scores = (np.asarray(fallback, float) if fallback is not None
              else node_scores_from_masked_effect(sE, mode="inout"))
    return nodes_by_node_score(scores, B)


# ==========================================================
# 第二段：诱导边
# ==========================================================

def _dist(pos):
    p = np.asarray(pos, float)[:, :2]
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


def _keep_by_node_gate(i, j, top_set, edge_keep_mode):
    if edge_keep_mode == "or":
        return i in top_set or j in top_set
    if edge_keep_mode == "and":
        return i in top_set and j in top_set
    raise ValueError(f"Unknown edge_keep_mode: {edge_keep_mode}")


def induce_corr_knn(V, corr_adj, pos, knn_k=15, radius=10.0, skel=None,
                    edge_keep_mode="or"):
    r"""相关骨架中通过 Top 节点门控的边 -> 无向无权。

    skel 给定（--corr-edges-npy 的权威骨架）：直接用它，不再重算 KNN。
        重算出的骨架与 MGC-RM 产物当初用的差一条边，两组结构指标就不可比。
    skel=None：在完整节点集上现算 KNN，再应用 OR/AND 节点门控。

    权重一律置 1.0：|rho| 来自相关性模块，不是重要性模型的输出，不能当作它预测的强度。
    （绘图可用 |rho| 做深浅/粗细，但指标里恒为无权，见 viz.py 的图注要求。）
    """
    top_set = set(int(v) for v in V)
    N = int(np.asarray(corr_adj).shape[0])
    S = list(range(N)) if edge_keep_mode == "or" else sorted(top_set)
    m = len(S)
    if m < 2:
        return []
    if skel is not None:
        A = np.asarray(skel, float)[np.ix_(S, S)] > 0
        A = A | A.T
        np.fill_diagonal(A, False)
    else:
        C = np.abs(np.asarray(corr_adj, float)[np.ix_(S, S)])
        np.fill_diagonal(C, -np.inf)
        D = _dist(pos)
        if radius is not None:
            for a in range(m):
                for b in range(m):
                    if a != b and D[S[a], S[b]] > radius:
                        C[a, b] = -np.inf
        A = np.zeros((m, m), bool)
        for ii in range(m):
            v = np.isfinite(C[ii])
            if not v.any():
                continue
            kk = min(int(knn_k), int(v.sum()))
            idx = np.argpartition(-C[ii], kk - 1)[:kk]
            A[ii, idx[np.isfinite(C[ii, idx])]] = True
        A = A | A.T
        np.fill_diagonal(A, False)
    out, seen = [], set()
    for ii, jj in zip(*np.where(A)):
        u, v = S[ii], S[jj]
        a, b = (u, v) if u < v else (v, u)
        if not _keep_by_node_gate(a, b, top_set, edge_keep_mode):
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b, 0, 1.0))
    return out


def induce_corr_knn_mask(V, sE, corr_adj, pos, knn_k=15, radius=10.0, skel=None,
                         edge_keep_mode="or"):
    """B3_CORR_IRM：KNN 骨架 ∩ IRM 学得掩码 -> 无向无权；lag 取掩码最强项。"""
    base = set((a, b) for (a, b, _t, _w) in
               induce_corr_knn(V, corr_adj, pos, knn_k, radius, skel,
                               edge_keep_mode=edge_keep_mode))
    S = np.asarray(sE, float)
    out = []
    for (a, b) in base:
        pair = np.maximum(S[a, b, :], S[b, a, :])
        if float(pair.max()) > 0:
            out.append((a, b, int(np.argmax(pair)), 1.0))
    return out


def induce_pcmci(V, base_graph, base_val, sE=None, edge_keep_mode="or"):
    r"""有向有权，底图为原始 PCMCI 图（--pcmci-full-pkl）。
    sE=None    -> 通过节点门控的全部 PCMCI 边（B4，冗余）
    sE 给定    -> 节点门控 ∩ 掩码过阈值的 PCMCI 边（B2 / FULL，精简）"""
    top_set = set(int(v) for v in V)
    g = np.asarray(base_graph)
    val = np.asarray(base_val, float)
    N, _, L = g.shape
    out = []
    for l in range(L):
        for i in range(N):
            for j in range(N):
                if not _keep_by_node_gate(i, j, top_set, edge_keep_mode):
                    continue
                if str(g[i, j, l]) != "-->":
                    continue
                if sE is not None and sE[i, j, l] <= 0:
                    continue
                if abs(float(val[i, j, l])) <= 1e-6:
                    continue
                out.append((int(i), int(j), int(l), float(val[i, j, l])))
    return out


# ==========================================================
# 组装
# ==========================================================

def build(name, branch, induce, *, sE=None, sV=None, B=None,
          base_graph=None, base_val=None, corr_adj=None, pos=None,
          knn_k=15, radius=10.0, corr_skel=None, rank_edges=True,
          edge_keep_mode="or", node_score_mode="inout") -> Subgraph:
    B = NODE_BUDGET if B is None else B
    # visualize_mask_subgraph_clear 口径：所有分支都按掩码效应的节点聚合分数选 Top-B。
    node_scores = np.asarray(sV, float) if sV is not None else np.asarray([])
    if node_scores.size == 0:
        if sE is None:
            raise ValueError("缺少 sV，且无法从空 sE 派生节点分数")
        node_scores = node_scores_from_masked_effect(sE, mode=node_score_mode)
    V = nodes_by_node_score(node_scores, B)

    if induce in ("corr_knn", "corr_knn_mask"):
        E = (induce_corr_knn(
                V, corr_adj, pos, knn_k, radius, corr_skel,
                edge_keep_mode=edge_keep_mode
             ) if induce == "corr_knn" else induce_corr_knn_mask(
                V, sE, corr_adj, pos, knn_k, radius, corr_skel,
                edge_keep_mode=edge_keep_mode
             ))
        directed, weighted = False, False
        N = int(np.asarray(corr_adj).shape[0])
        L = int(np.asarray(sE).shape[2]) if sE is not None else 1

        # AUPRC 的候选集应是掩码前的相关 KNN 骨架，而不是最终已选出的边。
        if corr_skel is not None:
            base_cand = (np.asarray(corr_skel, float) > 0).astype(float)
        else:
            base_cand = np.zeros((N, N), float)
            for (a, b, _t, _w) in induce_corr_knn(
                    list(range(N)), corr_adj, pos, knn_k, radius, None,
                    edge_keep_mode="or"):
                base_cand[a, b] = base_cand[b, a] = 1.0
        cand = np.repeat(base_cand[:, :, None], L, axis=2)
    elif induce in ("pcmci_mask", "pcmci_all"):
        E = induce_pcmci(
            V, base_graph, base_val,
            sE if induce == "pcmci_mask" else None,
            edge_keep_mode=edge_keep_mode,
        )
        directed, weighted = True, True
        cand = (np.asarray(base_graph) == "-->").astype(float)
    else:
        raise ValueError(induce)

    # rank_edges=False（MGC-RM / B4）：该方法没有对边排序的能力 -> AUPRC 如实报 "—"
    es = sE if rank_edges else None
    return Subgraph(name=name, V=V, E=E, directed=directed, weighted=weighted,
                    node_score=node_scores, edge_score=es, candidate=cand,
                    branch=branch, induce=induce,
                    meta=dict(B=B, knn_k=knn_k, radius=radius,
                              node_score_mode=node_score_mode,
                              edge_keep_mode=edge_keep_mode))


def save_pkl(path, sub: Subgraph, tau_max=3, N=100):
    """导出为 PCMCI-like 结构（graph / val_matrix / meta），兼容现有绘图脚本。"""
    L = tau_max + 1
    g = np.full((N, N, L), "", dtype=object)
    val = np.zeros((N, N, L))
    for (i, j, t, w) in sub.E:
        t = min(int(t), L - 1)
        g[i, j, t] = "-->"; val[i, j, t] = w
        if not sub.directed:
            g[j, i, t] = "-->"; val[j, i, t] = w
    meta = dict(sub.meta)
    structural_nodes = sorted(
        set(int(v) for v in sub.V)
        | {int(i) for i, _j, _t, _w in sub.E}
        | {int(j) for _i, j, _t, _w in sub.E}
    )
    meta.update(top_nodes=[int(v) for v in sub.V], top_nodes_count=len(sub.V),
                structural_nodes=structural_nodes,
                structural_nodes_count=len(structural_nodes),
                directed=sub.directed, weighted=sub.weighted,
                branch=sub.branch, induce=sub.induce,
                edges_count=int((g == "-->").sum()), N=N, L=L, tau_max=tau_max)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"graph": g, "val_matrix": val, "meta": meta}
    payload.update(meta)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[SAVE] {sub.name} -> {path}")
    return path


def from_pkl(path, name, branch, induce, **kw) -> Subgraph:
    """复用已有产物（MGC-RM / CISE-LRTR），再走同一套选点+诱导，保证口径一致。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    m = d.get("meta", {}) or {}
    sV = np.asarray(m.get("node_scores", d.get("node_scores", [])), float)
    g = np.asarray(d["graph"])
    val = np.asarray(d["val_matrix"], float)
    if sV.size == 0:
        sV = np.abs(val).sum((1, 2)) + np.abs(val).sum((0, 2))
        print(f"  [{name}] pkl 无 node_scores，节点分数由 |val| 入+出派生（derived，"
              f"与原脚本排序不保证一致）")
    sE = np.abs(val) * (g == "-->")
    print(f"  [{name}] 复用 {path}  graph{g.shape}  |E|={int((g=='-->').sum())}")
    return build(name, branch, induce, sE=sE, sV=sV, base_graph=g, base_val=val, **kw)


def from_pkl_asis(path, name, directed=True, weighted=True, candidate=None) -> Subgraph:
    """按产物**原样**导入：V 与 E 都取 pkl 里的，不重新选点、不重新连边。
    用于把旧协议的核心子图与统一协议的结果并排比较。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    m = d.get("meta", {}) or {}
    g = np.asarray(d["graph"])
    val = np.asarray(d["val_matrix"], float)
    E = [(int(i), int(j), int(t), float(val[i, j, t]))
         for (i, j, t) in np.argwhere(g == "-->")]
    tn = m.get("top_nodes", d.get("top_nodes"))
    V = ([int(v) for v in tn] if tn is not None
         else sorted({i for (i, j, _t, _w) in E} | {j for (i, j, _t, _w) in E}))
    sV = np.asarray(m.get("node_scores", d.get("node_scores", [])), float)
    if sV.size == 0:
        sV = np.abs(val).sum((1, 2)) + np.abs(val).sum((0, 2))
    sE = np.abs(val) * (g == "-->")
    print(f"  [{name}] 原样导入 {path}  |V|={len(V)}  |E|={len(E)}（旧协议，不重新选点/连边）")
    return Subgraph(name=name, V=V, E=E, directed=directed, weighted=weighted,
                    node_score=sV, edge_score=sE, candidate=candidate,
                    branch="asis", induce="asis",
                    meta=dict(source=str(path), protocol="legacy"))


def report(subs: Sequence[Subgraph]) -> str:
    L = ["", "=" * 96,
         f"掩码子图（Top-B={NODE_BUDGET}；默认 OR 门控：边任意一端在 Top-B 即保留）", "=" * 96]
    L += ["  " + s.line() for s in subs]
    L += ["=" * 96,
          "|V| 表示 Top-B 核心节点数；OR 模式下 E 可连接一阶非核心邻居。",
          "|E| 不同 -> P/R/F1 被边数混淆，必须并排报无阈值 AUPRC。",
          "网络效率口径：边多 => 能耗高 => 主表用 EC / EC_per_edge / Eff_per_EC(=E_glob/EC)；",
          "E_glob 在 |V| 固定时近乎单调于 |E|，仅作辅助列。"]
    return "\n".join(L)
