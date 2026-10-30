# -*- coding: utf-8 -*-
r"""
viz.py —— 核心子图绘图 + 扰动干扰曲线

核心子图规格
------------------------------------------------------------------
共同：核心节点红色高亮 / 非核心节点灰色小点 / 非核心边一律不画 /
      无黑边框 / Times New Roman / 仅核心节点标号

  因果核心子图（有向有权：B2 / B4 / FULL）  plot_directed_subgraph_clear
      节点大小  ∝ 有向加权出度 sum_{j,tau} |w_ij,tau|
      边颜色    红白蓝色条 RdBu_r + TwoSlopeNorm(vcenter=0)：负蓝 / 0 白 / 正红
      边粗细    ∝ |因果效应|

  相关核心子图（无向无权：MGC-RM / B3）      plot_undirected_subgraph
      节点大小  ∝ 子图内节点度
      边颜色    红白色条 Reds：深浅 ∝ |rho|
      边粗细    ∝ |rho|

      |rho| 只用于绘图可读性，指标里这些边恒为无权。图注必须写明，否则会被质疑
      "图上带强度、表里当无权"：
        Edge shading/width reflects |rho| for readability only;
        all metrics treat these edges as unweighted.

两个实现细节
------------------------------------------------------------------
  加权出度用 sum|w| 而不是 sum w：PCMCI 的 val 有正有负，直接求和会让 hub 被抵消成小点。
  大小/线宽映射设下限：sink 的加权出度为 0，不设下限会画没；_scale 另外处理了
  "全相等""全零"两种退化（除零）。

扰动干扰曲线
------------------------------------------------------------------
出 20 张独立图 + 1 张 mean ± s.e.m. 汇总。移除顺序是随机的，单条曲线只是一次实现，
只给平均图会掩盖方差结构。第 r 张图里所有方法用同一个 seed=r 打乱各自的核心节点：
run 间差异 = 移除顺序，方法间差异 = 方法本身，属配对比较。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm

try:
    from metrics import attack_curve, auc
except ImportError:
    from .metrics import attack_curve, auc

__version__ = "1.0"

NODE_SIZE_RANGE = (60.0, 460.0)     # 核心节点大小映射区间
NONCORE_SIZE = 55.0  # 非核心节点大小
EDGE_WIDTH_RANGE = (1.5, 4.0)
UNDIRECTED_EDGE_WIDTH_RANGE = (2.5, 6.0)


# 主方法：红、粗、大点、置顶；其余细线弱色陪衬
STYLE = {
    "MGC-RM":       dict(c="#7f7f7f", m="s", ls="--", lw=1.6, ms=5, z=2, a=0.85),
    "B2_GCN_LRTR":  dict(c="#1f77b4", m="^", ls="-.", lw=1.6, ms=5, z=2, a=0.85),
    "B3_CORR_INV":  dict(c="#2ca02c", m="v", ls=":",  lw=1.6, ms=5, z=2, a=0.85),
    "B4_NOCGBP":    dict(c="#9467bd", m="P", ls="-.", lw=1.6, ms=5, z=2, a=0.85),
    "FULL_CISE":    dict(c="#d62728", m="o", ls="-",  lw=3.2, ms=7, z=10, a=1.0),
    "FULL_LEGACY":  dict(c="#ff9896", m="D", ls="--", lw=2.0, ms=5, z=9, a=0.95),
    "PRIOR_ONLY":   dict(c="#17becf", m="*", ls="--", lw=1.4, ms=6, z=3, a=0.8),
}
LABEL = {
    "MGC-RM": "MGC-RM",
    "B2_GCN_LRTR": "GCN-LRTR",
    "B3_CORR_INV": "CORR-INV",
    "B4_NOCGBP": "CISE-LRTR (w/o CGBP)",
    "FULL_CISE": "CISE-LRTR",
    "FULL_LEGACY": "CISE-LRTR (ours, legacy protocol)",
    "PRIOR_ONLY": "Prior mask only (no training)",
}


def _st(n): return STYLE.get(n, dict(c=None, m=".", ls="-", lw=1.5, ms=4, z=2, a=0.8))
def _lb(n): return LABEL.get(n, n)


def _scale(vals, lo, hi):
    v = np.asarray(vals, float)
    if v.size == 0:
        return v
    vmax = float(np.max(v))
    if vmax <= 1e-12:                       # 全零 -> 统一取下限
        return np.full_like(v, lo)
    vmin = float(np.min(v))
    if abs(vmax - vmin) < 1e-12:            # 全相等 -> 取中值
        return np.full_like(v, 0.5 * (lo + hi))
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


def _save(fig, save_path, dpi, formats):
    base = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for f in formats:
        fig.savefig(base + f, format=f[1:], bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return base


def _frame(ax):
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.margins(0.05)


# ==========================================================
# 1. 因果核心子图（有向有权）
# ==========================================================

def plot_directed_subgraph_clear(sub, positions, save_path, font_size=12, dpi=600,
                                 formats=(".png", ".pdf", ".svg")):
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 12
    p = np.asarray(positions, float)[:, :2]
    x, y = p[:, 0], p[:, 1]
    N = p.shape[0]
    core = [int(v) for v in sub.V]
    non = sorted(set(range(N)) - set(core))

    outw = {v: 0.0 for v in core}
    for (i, j, _t, w) in sub.E: # 有向有权
        if int(i) in outw:
            outw[int(i)] += abs(float(w))
    sizes = _scale([outw[v] for v in core], *NODE_SIZE_RANGE) # 节点大小

    ws = [float(w) for (_i, _j, _t, w) in sub.E] # 边权重
    m = max([abs(v) for v in ws], default=1.0) or 1.0 # 边作为颜色条（colorbar）的对称边界
    norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    cmap = plt.cm.RdBu_r                     # 负=蓝 / 0=白 / 正=红
    lws = _scale([abs(v) for v in ws], *EDGE_WIDTH_RANGE) # 边粗细

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect("equal", adjustable="box")

    for k, (i, j, _t, w) in enumerate(sub.E):
        ax.add_patch(FancyArrowPatch(
            (x[i], y[i]), (x[j], y[j]), arrowstyle="-|>", mutation_scale=11,
            connectionstyle="arc3,rad=0.08", color=cmap(norm(float(w))),
            lw=float(lws[k]), alpha=0.9, shrinkA=7, shrinkB=7, zorder=1))

    ax.scatter([x[i] for i in non], [y[i] for i in non],
               s=NONCORE_SIZE, c="gray", alpha=0.7, linewidths=0, zorder=2)
    ax.scatter([x[i] for i in core], [y[i] for i in core],
               s=sizes, c="red", alpha=0.9, linewidths=0, zorder=3)
    for i in core:
        ax.text(x[i], y[i], str(i), fontsize=font_size, ha="center", va="center",
                zorder=4, color="black", fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=1, pad=0.02, location="right")
    cb.set_label("Causal Effect (signed)", rotation=270, labelpad=14)

    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=11,
               label="Core Causal Nodes"), #  \n(size $\\propto$ weighted out-degree)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7,
               label="Non-Core Nodes"),
        # Line2D([0], [0], color=cmap(0.95), lw=3.0,
        #        label="Core Causal Edges\n(width $\\propto$ |effect|)"),
    ], fontsize=10, loc="upper right", labelspacing=1.1)
    _frame(ax)
    fig.tight_layout(pad=3.0)
    base = _save(fig, save_path, dpi, formats)
    print(f"[PLOT] {sub.name} 因果核心子图（|V|={len(core)}，|E|={len(sub.E)}） -> {base}.png")
    return base


# ==========================================================
# 2. 相关核心子图（无向无权）
# ==========================================================

def plot_undirected_subgraph(sub, positions, save_path, corr_adj=None,
                             font_size=10, dpi=600, formats=(".png", ".pdf", ".svg")):
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 12
    p = np.asarray(positions, float)[:, :2]
    x, y = p[:, 0], p[:, 1]
    N = p.shape[0]
    core = [int(v) for v in sub.V]
    non = sorted(set(range(N)) - set(core))

    pairs = sorted({(min(int(i), int(j)), max(int(i), int(j))) for (i, j, *_r) in sub.E})

    deg = {v: 0 for v in core}
    for (a, b) in pairs:
        for v in (a, b):
            if v in deg:
                deg[v] += 1
    sizes = _scale([deg[v] for v in core], *NODE_SIZE_RANGE)

    if corr_adj is not None:
        vals = [float(abs(np.asarray(corr_adj, float)[a, b])) for (a, b) in pairs]
    else:
        vals = [1.0] * len(pairs)
    if vals:
        vmin, vmax = min(vals), max(vals)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
        norm, cmap = plt.Normalize(vmin, 0.2), plt.cm.Reds    # 0.72, 0.95 # 白 -> 红
        lws = _scale(vals, *UNDIRECTED_EDGE_WIDTH_RANGE)
    else:
        norm = cmap = None; lws = []

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect("equal", adjustable="box")

    for k, (a, b) in enumerate(pairs):
        ax.plot([x[a], x[b]], [y[a], y[b]], lw=float(lws[k]), alpha=1,
                color=cmap(norm(vals[k])), zorder=1, solid_capstyle="round")

    ax.scatter([x[i] for i in non], [y[i] for i in non],
               s=NONCORE_SIZE, c="gray", alpha=0.7, linewidths=0, zorder=2)
    ax.scatter([x[i] for i in core], [y[i] for i in core],
               s=sizes, c="red", alpha=0.9, linewidths=0, zorder=3)
    for i in core:
        ax.text(x[i], y[i], str(i), fontsize=font_size, ha="center", va="center",
                zorder=4, color="black", fontweight="bold")

    if cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=1, pad=0.02, location="right")
        cb.set_label("Correlation Strength  ", rotation=270, labelpad=14)

    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=11,
               label="Core Nodes"),#\n(size $\\propto$ degree)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7,
               label="Non-Core Nodes"),
        # Line2D([0], [0], color=plt.cm.Reds(0.85), lw=3.0,
        #        label="Core Edges (unweighted;\nshading/width $\\propto|\\rho|$ for display only)"),
    ], fontsize=10, loc="upper right", labelspacing=1.1)
    _frame(ax)
    fig.tight_layout(pad=3.0)
    base = _save(fig, save_path, dpi, formats)
    print(f"[PLOT] {sub.name} 相关核心子图（|V|={len(core)}，|E|={len(pairs)}） -> {base}.png")
    return base


# ==========================================================
# 3. 有向图：复用外部绘图（--directed-plot reuse）
# ==========================================================

def plot_directed_subgraph(sub, positions, out_dir, VIS, tau_max=3, N=100):
    """复用 pcmci_draw_MultiDiGraph2.build_multi_di_graph + plot_causal_graph2_clear2。
    该函数的节点大小来自它自己算的 nsc、配色也由它决定，与本文件的规格不一致；
    只在需要保持外部图口径时使用（--directed-plot reuse）。"""
    L = tau_max + 1
    g = np.full((N, N, L), "", dtype=object)
    val = np.zeros((N, N, L))
    for (i, j, t, w) in sub.E:
        t = min(int(t), L - 1)
        g[i, j, t] = "-->"; val[i, j, t] = w
    try:
        try:
            from Module.pcmci_draw_MultiDiGraph2 import build_multi_di_graph
        except ImportError:
            from pcmci_draw_MultiDiGraph2 import build_multi_di_graph
        os.makedirs(out_dir, exist_ok=True)
        G, ec, nsc = build_multi_di_graph(g, val, N, tau_max, positions)
        VIS.plot_causal_graph2_clear2(G, dict(enumerate(positions)), ec, nsc,
                                      str(out_dir), f"tau{tau_max}", "mask",
                                      title=f"Core Causal Invariant Subgraph — {sub.name}")
        VIS.plot_causal_matrix(ec, N, str(out_dir), f"tau{tau_max}", "mask")
        print(f"[PLOT] {sub.name} 有向有权子图（reuse） -> {out_dir}")
        return out_dir
    except Exception as e:
        print(f"[WARN] {sub.name} 有向绘图失败 ({e})；指标不受影响")
        return None


# ==========================================================
# 4. 扰动干扰曲线
# ==========================================================

def _draw(ax, x, y, name, band=None):

    """
    在给定的坐标轴上绘制数据曲线和可选的置信带

    参数:
        ax: matplotlib的坐标轴对象，用于绘图
        x: x轴数据，通常是数组或列表
        y: y轴数据，通常是数组或列表
        name: 数据系列名称，用于确定绘图样式
        band: 可选参数，表示置信带的宽度，如果提供则绘制填充区域

    返回:
        无返回值，直接在给定的坐标轴上绘图
    """
    s = _st(name)  # 获取与名称对应的样式配置
    # 判断是否为主曲线（FULL_CISE）
    main = name == "FULL_CISE"
    # 绘制主曲线，设置各种样式参数
    ax.plot(x, y, label=_lb(name), color=s["c"], marker=s["m"], linestyle=s["ls"],
            linewidth=s["lw"], markersize=s["ms"], zorder=s["z"], alpha=s["a"],
            markeredgecolor="white" if main else None,  # 主曲线添加白色边框
            markeredgewidth=1.0 if main else 0.0)      # 主曲线设置边框宽度
    # 如果提供了置信带参数，则绘制填充区域
    if band is not None:
        ax.fill_between(x, np.maximum(0, y - band), y + band, color=s["c"],
                        alpha=0.18 if main else 0.07, linewidth=0, zorder=s["z"] - 1)


def _legend_main_first(ax):
    """
    为图表的图例进行排序，确保带有"FULL_CISE"标签的图例项显示在第一位
    参数:
        ax: matplotlib的Axes对象，包含图例信息
    返回:
        无直接返回值，但会修改传入的ax对象的图例
    """
    # 获取当前图例的句柄和标签
    h, l = ax.get_legend_handles_labels()
    # 创建一个排序索引列表，根据标签是否为"FULL_CISE"进行排序
    # "FULL_CISE"标签的项会被排在最前面，其他项按原始顺序排列
    o = sorted(range(len(l)), key=lambda i: l[i] != _lb("FULL_CISE"))
    # 使用排序后的索引重新排列图例的句柄和标签，并设置字体大小为8
    ax.legend([h[i] for i in o], [l[i] for i in o], fontsize=8)


def plot_attack_runs(subs, save_dir, n_runs=20, steps=20, ts=""):
    """第 r 张图：所有方法在 seed=r 的随机移除序列下的 Q(gamma)（配对）。
    返回 {method: [AUC_run0, ...]}，供表里报 mean ± std（跨随机序列，不是跨 seed）。"""
    os.makedirs(save_dir, exist_ok=True)
    curves: Dict[str, List[np.ndarray]] = {s.name: [] for s in subs}
    aucs: Dict[str, List[float]] = {s.name: [] for s in subs}
    gamma = None

    for r in range(n_runs):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        order = sorted(subs, key=lambda s: s.name == "FULL_CISE")   # 主方法最后画 -> 置顶
        for s in order:
            if s.name == "FULL_LEGACY":  # 跳过不需要的条目
                continue
            G = s.to_nx()
            if G.number_of_nodes() < 2:
                continue
            g, q = attack_curve(G, steps=steps, seed=r)
            gamma = g
            curves[s.name].append(q)
            aucs[s.name].append(auc(g, q))
            _draw(ax, g, q, s.name)
        ax.set_title(f"Resilience: Network Efficiency under Random Attack")# — run {r+1}/{n_runs} (seed={r})")
        ax.set_xlabel("Fraction of Nodes Removed  $\\gamma$")
        ax.set_ylabel("Efficiency Retention  $Q(\\gamma)$  (Higher is Better)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        _legend_main_first(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"Resilience_run{r:02d}_{ts}.png"), dpi=300)
        plt.close(fig)
    print(f"[PLOT] {n_runs} 张独立扰动图 -> {save_dir}/Resilience_run00..{n_runs-1:02d}_{ts}.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in sorted(subs, key=lambda s: s.name == "FULL_CISE"):
        cs = curves[s.name]
        if not cs:
            continue
        A = np.stack(cs, 0)
        _draw(ax, gamma, A.mean(0), s.name, band=A.std(0) / np.sqrt(len(cs)))
    ax.set_title(f"Resilience: Efficiency under Random Attack\n"
                 f"(mean ± s.e.m. over {n_runs} random removal orders)")
    ax.set_xlabel("Fraction of Nodes Removed  $\\gamma$")
    ax.set_ylabel("Efficiency Retention  $Q(\\gamma)$  (Higher is Better)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    _legend_main_first(ax)
    fig.tight_layout()
    p = os.path.join(save_dir, f"Resilience_summary_{ts}.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"[PLOT] 汇总 -> {p}")
    return aucs


def plot_attack_runs2(subs, resil_curves, save_dir, ts=""):
    """使用预先计算好的展开图弹性曲线绘制扰动干扰曲线（20次独立 + mean±s.e.m.汇总）。
    resil_curves: {method_name: (gamma, q_matrix)}，其中 q_matrix 形状为 [n_runs, steps]
                  或 {method_name: (gamma, q_array)} 如果只有单条曲线。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 统一提取 gamma（所有方法共享同一个 gamma）
    gamma = None
    for name, (g, q) in resil_curves.items():
        gamma = g
        break

    if gamma is None:
        print("[WARN] resil_curves 为空，跳过绘图")
        return

    # 1. 绘制每次独立运行的图
    n_runs = 0
    # 探测运行次数（以 FULL_CISE 或首个方法为基准）
    for s in subs:
        if s.name in resil_curves:
            _, q_data = resil_curves[s.name]
            if q_data.ndim == 2:
                n_runs = max(n_runs, q_data.shape[0])
            elif q_data.ndim == 1:
                n_runs = max(n_runs, 1)

    for r in range(n_runs):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        order = sorted(subs, key=lambda s: s.name == "FULL_CISE")  # 主方法最后画 -> 置顶
        for s in order:
            if s.name == "FULL_LEGACY":  # 跳过不需要的条目
                continue
            if s.name not in resil_curves:
                continue
            g, q_data = resil_curves[s.name]
            if q_data.ndim != 2 or r >= q_data.shape[0]:
                continue
            q = q_data[r]
            _draw(ax, g, q, s.name)

        ax.set_title(f"Resilience: Network Efficiency under Random Attack")
        ax.set_xlabel("Fraction of Nodes Removed  $\\gamma$")
        ax.set_ylabel("Efficiency Retention  $Q(\\gamma)$  (Higher is Better)")
        ax.set_ylim(-0.02, 1.1)
        ax.grid(True, alpha=0.3)
        _legend_main_first(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"Resilience_run{r:02d}_{ts}.png"), dpi=300)
        plt.close(fig)


    print(f"[PLOT] {n_runs} 张独立扰动图 -> {save_dir}/Resilience_run00..{n_runs - 1:02d}_{ts}.png")

    # 2. 绘制 mean ± s.e.m. 汇总图
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in sorted(subs, key=lambda s: s.name == "FULL_CISE"):
        if s.name not in resil_curves:
            continue
        g, q_data = resil_curves[s.name]
        if q_data.ndim == 2:
            A = q_data
            _draw(ax, g, A.mean(0), s.name, band=A.std(0) / np.sqrt(len(A)))
        else:
            _draw(ax, g, q_data, s.name)

    ax.set_title(f"Resilience: Efficiency under Random Attack\n"
                 f"(mean ± s.e.m. over {n_runs} random removal orders)")
    ax.set_xlabel("Fraction of Nodes Removed  $\\gamma$")
    ax.set_ylabel("Efficiency Retention  $Q(\\gamma)$  (Higher is Better)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    _legend_main_first(ax)
    fig.tight_layout()
    p = os.path.join(save_dir, f"Resilience_summary_{ts}.png")
    fig.savefig(p, dpi=300);
    plt.close(fig)
    print(f"[PLOT] 汇总 -> {p}")


def plot_deployment(deploy: Dict[str, Tuple], save_dir, ts=""):
    """部署仿真（RMSE vs 传感器数）。曲线由 evaluate_full_v3-1.SubGraphEvaluator 提供。"""
    if not deploy:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sorted(deploy, key=lambda n: n == "FULL_CISE"):
        x, y = deploy[name]
        _draw(ax, np.asarray(x), np.asarray(y), name)
    ax.set_title("Optimization: Sensor Deployment Efficiency")
    ax.set_xlabel("Number of Sensors Deployed")
    ax.set_ylabel("Network Reconstruction RMSE (Lower is Better)")
    ax.grid(True, alpha=0.3)
    _legend_main_first(ax)
    fig.tight_layout()
    p = os.path.join(save_dir, f"Deployment_{ts}.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"[PLOT] {p}")
    return p
