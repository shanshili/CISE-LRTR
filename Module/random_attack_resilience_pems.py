# -*- coding: utf-8 -*-
r"""
只读取已有子图 PKL，执行展开图上的随机节点攻击并绘制网络效率维持率曲线。

依赖
----
把本脚本放在项目 Module 目录（与 metrics2.py、subgraph.py、viz.py 同级）。
脚本不会训练模型，也不会读取时间序列、坐标或 ground truth。

默认输入
--------
D:\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems\pkl

维持率修正
----------
旧实现每次攻击后都用“剩余节点数”重新归一化全局效率。删除低贡献节点会缩小
分母，因此 Q(gamma) 可能上升并明显超过 1。本脚本固定使用攻击前的节点对集合：
被删除节点涉及的节点对贡献按 0 计。等价地，直接用攻击后的 1/d 总和除以攻击前
的 1/d 总和。节点删除不会创造更短路径，所以 Q 理论上单调不增且位于 [0, 1]。

示例
----
python random_attack_resilience_pems.py
python random_attack_resilience_pems.py --runs 200 --steps 10
python random_attack_resilience_pems.py --runs 200 --band std
python random_attack_resilience_pems.py --runs 200 --band sem
python random_attack_resilience_pems.py --pkl-dir "D:\\other\\pkl" --out "D:\\other\\figures"
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

import metrics2 as M
import subgraph as SG


DEFAULT_PKL_DIR = Path(
    r".\baseline_results_pems3\pkl"
)

METHOD_ORDER = [
    "MGC-RM",
    "B2_GCN_LRTR",
    "B3_CORR_INV",
    "B4_NOCGBP",
    "FULL_LEGACY",
]
UNDIRECTED_METHODS = {"MGC-RM", "B3_CORR_INV"}
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def canonical_method_name(raw_name: str) -> str:
    """把常见文件名统一成 viz.py 已定义样式的方法名。"""
    compact = re.sub(r"[^a-z0-9]+", "", raw_name.lower())
    if "mgcrm" in compact:
        return "MGC-RM"
    if "b2" in compact and ("gcn" in compact or "lrtr" in compact):
        return "B2_GCN_LRTR"
    if "b3" in compact or "corrinv" in compact:
        return "B3_CORR_INV"
    if "b4" in compact or "nocgbp" in compact:
        return "B4_NOCGBP"
    # 这 5 个对比子图里主方法可能按方法名、显示名或旧协议名保存；绘图时统一成
    # viz.py 的主方法键 FULL_LEGACY，保证使用 CISE-LRTR 的红色高亮样式。
    if (
        "fullcise" in compact
        or "fulllegacy" in compact
        or "ciselrtr" in compact
        or compact == "cise"
    ):
        return "FULL_LEGACY"
    return raw_name


def _read_identity(path: Path) -> Tuple[str, bool, bool]:
    """只读取名称/有向性元数据；图本体仍交给 subgraph.from_pkl_asis 导入。"""
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} 的顶层对象不是 dict")

    meta = payload.get("meta", {}) or {}
    raw_name = str(
        meta.get("method")
        or meta.get("name")
        or payload.get("method")
        or payload.get("name")
        or path.stem
    )
    name = canonical_method_name(raw_name)

    directed_value = meta.get("directed", payload.get("directed"))
    weighted_value = meta.get("weighted", payload.get("weighted"))
    directed = (
        bool(directed_value)
        if directed_value is not None
        else name not in UNDIRECTED_METHODS
    )
    weighted = (
        bool(weighted_value)
        if weighted_value is not None
        else directed
    )
    return name, directed, weighted


def load_five_subgraphs(pkl_dir: Path, expected_count: int) -> List[SG.Subgraph]:
    """复用 subgraph.py 的原样导入逻辑，载入目录中的全部 PKL。"""
    paths = sorted(pkl_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"目录中没有 .pkl 文件：{pkl_dir}")
    if expected_count > 0 and len(paths) != expected_count:
        names = "\n  ".join(p.name for p in paths)
        raise ValueError(
            f"期望 {expected_count} 个 PKL，实际找到 {len(paths)} 个：\n  {names}\n"
            "如确实要处理当前全部文件，可传 --expected-count 0。"
        )

    subs: List[SG.Subgraph] = []
    seen = set()
    for path in paths:
        name, directed, weighted = _read_identity(path)
        if name in seen:
            raise ValueError(f"方法名重复：{name}（请检查 PKL 文件名或 meta.name）")
        sub = SG.from_pkl_asis(
            str(path), name=name, directed=directed, weighted=weighted
        )
        if not directed:
            # save_pkl 会把无向边同时写入 g[i,j,l] 和 g[j,i,l]；原样读回后先还原成
            # 一条无向边，避免 build_expanded 在相反时间方向上重复构边。
            unique_edges = {}
            for i, j, lag, weight in sub.E:
                key = (min(int(i), int(j)), max(int(i), int(j)), int(lag))
                old = unique_edges.get(key)
                if old is None or abs(float(weight)) > abs(float(old[3])):
                    unique_edges[key] = (*key, float(weight))
            sub.E = list(unique_edges.values())
        if sub.edge_score is None or np.asarray(sub.edge_score).ndim != 3:
            raise ValueError(f"{path.name} 中 graph/val_matrix 不是 [N,N,L] 结构")
        if not sub.V:
            raise ValueError(f"{path.name} 未解析出任何核心节点")
        subs.append(sub)
        seen.add(name)

    rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    return sorted(subs, key=lambda s: (rank.get(s.name, 999), s.name))


def efficiency_mass(G: nx.Graph) -> float:
    """返回所有可达有序节点对的 sum(1 / d_hop)，不按当前节点数重新归一化。"""
    total = 0.0
    for source, lengths in nx.all_pairs_shortest_path_length(G):
        for target, distance in lengths.items():
            if source != target and distance > 0:
                total += 1.0 / float(distance)
    return float(total)


def stable_resilience_expanded(
    sub: SG.Subgraph,
    *,
    steps: int,
    runs: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """复用 metrics2.build_expanded，按固定初始节点对口径计算随机攻击曲线。"""
    score_shape = np.asarray(sub.edge_score).shape
    N, N2, L = map(int, score_shape)
    if N != N2:
        raise ValueError(f"{sub.name}: graph 前两维不相等：{score_shape}")

    expanded, groups = M.build_expanded(
        list(sub.V), list(sub.E), N, L, bool(sub.directed)
    )
    physical_nodes = sorted(groups)
    B = len(physical_nodes)
    if B < 2:
        raise ValueError(f"{sub.name}: 核心物理节点数 {B} < 2，无法计算网络效率")

    baseline_mass = efficiency_mass(expanded)
    if baseline_mass <= 0.0:
        raise ValueError(f"{sub.name}: 展开图无可达节点对，网络效率维持率无定义")

    gamma = np.linspace(0.0, 1.0, steps + 1)
    targets = np.rint(gamma * B).astype(int)
    all_q = np.zeros((runs, steps + 1), dtype=float)

    for run in range(runs):
        # 每个 run 使用 seed=base_seed+run；各方法节点数相同时共享同一随机排列索引。
        rng = np.random.default_rng(base_seed + run)
        order = physical_nodes.copy()
        rng.shuffle(order)

        current = expanded.copy()
        removed = 0
        q = np.zeros(steps + 1, dtype=float)
        q[0] = 1.0

        for step in range(1, steps + 1):
            target = int(targets[step])
            while removed < target:
                physical_node = order[removed]
                current.remove_nodes_from(groups[physical_node])
                removed += 1

            ratio = efficiency_mass(current) / baseline_mass
            q[step] = min(1.0, max(0.0, float(ratio)))

        # 理论上已经单调且不超过 1；这里只消除浮点舍入量级的越界。
        q = np.minimum.accumulate(q)
        all_q[run] = np.clip(q, 0.0, 1.0)

    return gamma, all_q


def _draw_builtin(
    ax,
    gamma: np.ndarray,
    q: np.ndarray,
    name: str,
    band: np.ndarray | None = None,
    band_alpha: float = 0.22,
) -> None:
    """viz.py 不可导入时的轻量回退绘图。"""
    colors = {
        "MGC-RM": "#767171",
        "B2_GCN_LRTR": "#1f77b4",
        "B3_CORR_INV": "#2ca02c",
        "B4_NOCGBP": "#FFC000",
        "FULL_LEGACY": "#d62728",
    }
    markers = {
        "MGC-RM": "s",
        "B2_GCN_LRTR": "^",
        "B3_CORR_INV": "v",
        "B4_NOCGBP": "P",
        "FULL_LEGACY": "o",
    }
    labels = {
        "MGC-RM": "MGC-RM",
        "B2_GCN_LRTR": "STCG-GCN",
        "B3_CORR_INV": "Corr-IRM",
        "B4_NOCGBP": "CINE",
        "FULL_LEGACY": "CISE-LRTR",
    }
    is_main = name == "FULL_LEGACY"
    color = colors.get(name)
    ax.plot(
        gamma,
        np.clip(q, 0.0, 1.0),
        label=labels.get(name, name),
        color=color,
        marker=markers.get(name, "o"),
        linewidth=3.4 if is_main else 1.6,
        markersize=7 if is_main else 5,
        zorder=10 if is_main else 2,
        markeredgecolor="white" if is_main else None,
        markeredgewidth=0.9 if is_main else 0.0,
    )
    if band is not None:
        lower = np.clip(q - band, 0.0, 1.0)
        upper = np.clip(q + band, 0.0, 1.0)
        ax.fill_between(
            gamma,
            lower,
            upper,
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )


def plot_with_builtin(
    subs: Sequence[SG.Subgraph],
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: Path,
    ts: str,
    band_mode: str,
    band_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    # 设置全局字体为 Times New Roman
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"  # 使数学字体也与 Times 风格一致

    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    runs = max(q.shape[0] for _, q in curves.values())

    for run in range(runs):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        # 主方法最后绘制，并用更高 zorder，避免与其他方法重合时被覆盖。
        for sub in sorted(subs, key=lambda item: item.name == "FULL_LEGACY"):
            gamma, q = curves[sub.name]
            if run < q.shape[0]:
                _draw_builtin(ax, gamma, q[run], sub.name)
        ax.set_title("Resilience: Network Efficiency under Random Attack")
        ax.set_xlabel(r"Fraction of Physical Nodes Removed  $\gamma$")
        ax.set_ylabel(r"Efficiency Retention  $Q(\gamma)$  (Higher is Better)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / f"Resilience_run{run:03d}_{ts}.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for sub in sorted(subs, key=lambda item: item.name == "FULL_LEGACY"):
        gamma, q = curves[sub.name]
        std = q.std(axis=0, ddof=1) if q.shape[0] > 1 else np.zeros_like(q[0])
        if band_mode == "std":
            band = std
        elif band_mode == "sem":
            band = std / np.sqrt(q.shape[0])
        else:
            band = None
        _draw_builtin(
            ax,
            gamma,
            q.mean(axis=0),
            sub.name,
            band=band,
            band_alpha=band_alpha,
        )
    band_title = {
        "std": "mean ± std",
        "sem": "mean ± s.e.m.",
        "none": "mean",
    }[band_mode]
    ax.set_title(
        f"Resilience: Efficiency under Random Attack\n"
        f"({band_title} over 500 random removal orders)"
    )
    ax.set_xlabel(r"Fraction of Physical Nodes Removed  $\gamma$")
    ax.set_ylabel(r"Efficiency Retention  $Q(\gamma)$  (Higher is Better)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_dir / f"Resilience_summary_{ts}.png", dpi=300)
    plt.close(fig)


def plot_curves(
    subs: Sequence[SG.Subgraph],
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: Path,
    ts: str,
    band_mode: str,
    band_alpha: float,
) -> None:
    """绘图时再次限制到 [0,1]，并保证 CISE-LRTR 红线最后绘制。"""
    plot_with_builtin(subs, curves, save_dir, ts, band_mode, band_alpha)


def write_auc_tables(
    subs: Sequence[SG.Subgraph],
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: Path,
    base_seed: int,
) -> None:
    """保存逐次 AUC 与汇总，便于核对多次攻击的波动。"""
    run_path = save_dir / "resilience_auc_by_run.csv"
    summary_path = save_dir / "resilience_auc_summary.csv"

    summary_rows = []
    with run_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "run", "seed", "auc"])
        for sub in subs:
            gamma, q = curves[sub.name]
            aucs = _TRAPZ(q, gamma, axis=1)
            for run, auc_value in enumerate(aucs):
                writer.writerow([sub.name, run, base_seed + run, f"{auc_value:.10f}"])
            summary_rows.append(
                [
                    sub.name,
                    len(sub.V),
                    len(sub.E),
                    len(aucs),
                    float(np.mean(aucs)),
                    float(np.std(aucs, ddof=0)),
                    float(np.std(aucs, ddof=0) / np.sqrt(len(aucs))),
                    float(np.min(aucs)),
                    float(np.max(aucs)),
                ]
            )

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "physical_nodes",
                "stored_edges",
                "runs",
                "auc_mean",
                "auc_std",
                "auc_sem",
                "auc_min",
                "auc_max",
            ]
        )
        for row in summary_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 5 个已有子图 PKL，仅运行随机攻击网络维持率实验。"
    )
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录；默认写入 pkl 目录同级的 random_attack_results_时间戳。",
    )
    parser.add_argument("--runs", type=int, default=100, help="随机移除序列数量。")
    parser.add_argument("--steps", type=int, default=10, help="每条曲线的攻击步数。")
    parser.add_argument("--seed", type=int, default=4, help="第 0 次实验的随机种子。")
    parser.add_argument(
        "--band",
        choices=["std", "sem", "none"],
        default="std",
        help=(
            "汇总图阴影：std=随机攻击结果的实际离散程度（默认，推荐）；"
            "sem=均值估计误差；none=不绘制阴影。"
        ),
    )
    parser.add_argument(
        "--band-alpha",
        type=float,
        default=0.22,
        help="阴影透明度，范围 [0,1]；默认 0.22。",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=5,
        help="期望 PKL 数；传 0 可处理目录中的任意数量。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")
    if args.steps < 1:
        raise ValueError("--steps 必须 >= 1")
    if not 0.0 <= args.band_alpha <= 1.0:
        raise ValueError("--band-alpha 必须在 [0,1] 内")

    pkl_dir = args.pkl_dir.expanduser().resolve()
    if not pkl_dir.is_dir():
        raise NotADirectoryError(f"PKL 目录不存在：{pkl_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.out.expanduser().resolve()
        if args.out is not None
        else pkl_dir.parent / f"random_attack_results_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = load_five_subgraphs(pkl_dir, args.expected_count)
    print("\n已载入子图：")
    for sub in subs:
        shape = tuple(np.asarray(sub.edge_score).shape)
        print(
            f"  {sub.name:<16} |V|={len(sub.V):<3} |E|={len(sub.E):<5} "
            f"directed={sub.directed!s:<5} graph={shape}"
        )

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sub in subs:
        print(f"[ATTACK] {sub.name}: runs={args.runs}, steps={args.steps}")
        curves[sub.name] = stable_resilience_expanded(
            sub,
            steps=args.steps,
            runs=args.runs,
            base_seed=args.seed,
        )

    plot_curves(subs, curves, out_dir, ts, args.band, args.band_alpha)
    write_auc_tables(subs, curves, out_dir, args.seed)

    max_q = max(float(q.max()) for _, q in curves.values())
    monotone = all(
        bool(np.all(np.diff(q, axis=1) <= 1e-12)) for _, q in curves.values()
    )
    print(f"\n[DONE] 输出目录：{out_dir}")
    print(f"[CHECK] max Q={max_q:.12f}; 全部曲线单调不增={monotone}")
    print("[CHECK] 使用固定初始节点对口径；clip 仅处理浮点误差，不用于掩盖超 1。")
    print(f"[PLOT] 汇总图阴影={args.band}; alpha={args.band_alpha:.2f}")


if __name__ == "__main__":
    main()
