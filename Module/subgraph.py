# -*- coding: utf-8 -*-
r"""
subgraph.py —— 统一子图协议：从分数到核心子图

节点截止阈值（唯一来源）
------------------------------------------------------------------
    NODE_BUDGET = 27
    单次实验临时改：python run.py --node-budget 30
    与 keep_frac / target_density(掩码边密度目标) 无关，不要跟着改。

协议：两类方法，同一个节点预算 B
------------------------------------------------------------------
所有方法 |V| = B，边的连接方式由各自的原生规则决定。

  第一类 · 边级方法（B2 / B3 / FULL）        branch='edge'
      按 s_E 降序遍历、累积端点、凑够 B 停 -> V，再保留 V 内所有过掩码阈值的边。
      重要的边直接决定重要的节点，不经过求和聚合这一步有损压缩
      （求和会让"连了很多弱边"的节点压过"连了一条强边"的节点）。
      边的候选列表按各自图源给：
        B2 / FULL  induce='pcmci_mask'     PCMCI 边 ∩ 掩码过阈值     有向有权
        B3         induce='corr_knn_mask'  KNN 构图的边 ∩ 掩码过阈值 无向无权

  第二类 · 只有节点重要性（MGC-RM / B4）      branch='node'
      V = top-B by s_V，再按各自构图方式连边：
        MGC-RM     induce='corr_knn'       V 上重新 KNN 构图         无向无权
        B4         induce='pcmci_all'      V 节点的 PCMCI 出度边     有向有权

边一律严格 induced（两端都在 V 内）。B4 的"出度边"若允许落点在 V 外，节点数会溢出 B，
"所有子图节点数一致"当场破功、|E| 也不可比 —— 因此只保留落点同在 V 内的出度边。

留边不用"恰好遍历到的那几条"：那会把 |E| 钉死在 ~B，B2 与 FULL 的结构对比会塌掉。

B4 为什么用 'pcmci_all'
------------------------------------------------------------------
CGBP 把节点特征抬到边级 Phi = A~ ⊙ (H W_p H^T) ∈ R^{N×N}；消融后
Phi = Linear(GConv(Z)) ∈ R^{N×1} 只剩节点因果分数，同一节点的所有入边梯度同号同向，
边之间不可分辨 —— 没有对边排序的能力，只能保留 top 节点上的全部 PCMCI 边。
底图仍是 PCMCI，故有向有权、相对准确，但冗余、噪声多。

|E| 自由带来的代价
------------------------------------------------------------------
|V| 固定 -> RMSE / Ent / EC / Density 是方法性质，可直接比。
|E| 自由 -> 承载"冗余 vs 精简"，但 P/R/F1 会被边数混淆 -> 必须并排报 AUPRC（不依赖边数）。
网络效率口径：边多 => 能耗高 => 用 EC / EC_per_edge / Eff_per_EC；
E_glob 在 |V| 固定时近乎单调于 |E|（稠密全双向图必然最高），仅作辅助列。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

__version__ = "1.0"

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

def nodes_by_node_score(sV, B=None) -> List[int]:
    B = NODE_BUDGET if B is None else B
    s = np.asarray(sV, float)
    if float(np.std(s)) < 1e-12:
        raise ValueError("node_score 全部并列 -> top_nodes 退化为 [0..B-1]，结果不可用")
    return [int(v) for v in np.argsort(-s)[:B]]


def nodes_by_edge_score(sE, B=None, fallback=None) -> List[int]:
    B = NODE_BUDGET if B is None else B
    S = np.asarray(sE, float).copy()
    N, _, L = S.shape
    for l in range(L):
        np.fill_diagonal(S[:, :, l], -np.inf)
    flat = S.ravel()
    ok = np.isfinite(flat) & (flat > 0)
    if not ok.any():
        if fallback is None:
            raise ValueError("edge_score 全零且无 node_score 兜底")
        return nodes_by_node_score(fallback, B)
    V, seen = [], set()
    for t in np.argsort(-np.where(ok, flat, -np.inf)):
        if not ok[t]:
            break
        i, rem = divmod(int(t), N * L)
        j, _ = divmod(rem, L)
        for v in (i, j):
            if v not in seen:
                seen.add(v); V.append(v)
                if len(V) >= B:
                    return V
    if len(V) < B and fallback is not None:      # 边分数覆盖不到 B 个节点 -> 节点分数补齐
        print(f"  [WARN] 边分数仅覆盖 {len(V)} 个节点 (<B={B})，用 node_score 补齐")
        for v in nodes_by_node_score(fallback, N):
            if v not in seen:
                V.append(v); seen.add(v)
                if len(V) >= B:
                    break
    return V


# ==========================================================
# 第二段：诱导边
# ==========================================================

def _dist(pos):
    p = np.asarray(pos, float)[:, :2]
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


def induce_corr_knn(V, corr_adj, pos, knn_k=15, radius=10.0, skel=None):
    r"""相关骨架内、两端都在 V 的边 -> 无向无权。

    skel 给定（--corr-edges-npy 的权威骨架）：直接用它，不再重算 KNN。
        重算出的骨架与 MGC-RM 产物当初用的差一条边，两组结构指标就不可比。
    skel=None：现算 KNN —— 每个节点在半径 r 内取 k 个最强 |rho| 邻居，union。

    权重一律置 1.0：|rho| 来自相关性模块，不是重要性模型的输出，不能当作它预测的强度。
    （绘图可用 |rho| 做深浅/粗细，但指标里恒为无权，见 viz.py 的图注要求。）
    """
    S = [int(v) for v in V]
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
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b, 0, 1.0))
    return out


def induce_corr_knn_mask(V, sE, corr_adj, pos, knn_k=15, radius=10.0, skel=None):
    """B3：KNN 构图的边 ∩ 掩码过阈值 -> 无向无权。lag 取该点对上掩码最强的那个。"""
    base = set((a, b) for (a, b, _t, _w) in
               induce_corr_knn(V, corr_adj, pos, knn_k, radius, skel))
    S = np.asarray(sE, float)
    out = []
    for (a, b) in base:
        pair = np.maximum(S[a, b, :], S[b, a, :])
        if float(pair.max()) > 0:
            out.append((a, b, int(np.argmax(pair)), 1.0))
    return out


def induce_pcmci(V, base_graph, base_val, sE=None):
    r"""有向有权，底图为原始 PCMCI 图（--pcmci-full-pkl）。
    sE=None    -> V 上的全部 PCMCI 边（B4，冗余）
    sE 给定    -> 仅掩码过阈值的边（B2 / FULL，精简）"""
    vs = set(int(v) for v in V)
    g = np.asarray(base_graph)
    val = np.asarray(base_val, float)
    _, _, L = g.shape
    out = []
    for l in range(L):
        for i in vs:
            for j in vs:
                if i == j or str(g[i, j, l]) != "-->":
                    continue
                if sE is not None and sE[i, j, l] <= 0:
                    continue
                out.append((int(i), int(j), int(l), float(val[i, j, l])))
    return out


# ==========================================================
# 组装
# ==========================================================

def build(name, branch, induce, *, sE=None, sV=None, B=None,
          base_graph=None, base_val=None, corr_adj=None, pos=None,
          knn_k=15, radius=10.0, corr_skel=None, rank_edges=True) -> Subgraph:
    B = NODE_BUDGET if B is None else B
    V = (nodes_by_edge_score(sE, B, fallback=sV) if branch == "edge"
         else nodes_by_node_score(sV, B))

    if induce in ("corr_knn", "corr_knn_mask"):
        E = (induce_corr_knn(V, corr_adj, pos, knn_k, radius, corr_skel) if induce == "corr_knn"
             else induce_corr_knn_mask(V, sE, corr_adj, pos, knn_k, radius, corr_skel))
        directed, weighted = False, False
        N = int(np.asarray(corr_adj).shape[0])
        cand = np.zeros((N, N, 1))
        for (a, b, _t, _w) in E:
            cand[a, b, 0] = cand[b, a, 0] = 1.0
    elif induce in ("pcmci_mask", "pcmci_all"):
        E = induce_pcmci(V, base_graph, base_val, sE if induce == "pcmci_mask" else None)
        directed, weighted = True, True
        cand = (np.asarray(base_graph) == "-->").astype(float)
    else:
        raise ValueError(induce)

    # rank_edges=False（MGC-RM / B4）：该方法没有对边排序的能力 -> AUPRC 如实报 "—"
    es = sE if rank_edges else None
    return Subgraph(name=name, V=V, E=E, directed=directed, weighted=weighted,
                    node_score=sV, edge_score=es, candidate=cand,
                    branch=branch, induce=induce,
                    meta=dict(B=B, knn_k=knn_k, radius=radius))


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
    meta.update(top_nodes=[int(v) for v in sub.V], top_nodes_count=len(sub.V),
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
         f"统一子图（节点预算 B={NODE_BUDGET}；|E| 为方法性质，不设边预算）", "=" * 96]
    L += ["  " + s.line() for s in subs]
    L += ["=" * 96,
          "|E| 不同 -> P/R/F1 被边数混淆，必须并排报无阈值 AUPRC。",
          "网络效率口径：边多 => 能耗高 => 主表用 EC / EC_per_edge / Eff_per_EC(=E_glob/EC)；",
          "E_glob 在 |V| 固定时近乎单调于 |E|，仅作辅助列。"]
    return "\n".join(L)
