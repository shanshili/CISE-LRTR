# -*- coding: utf-8 -*-
r"""
run2.py —— 主入口（整合新版 metrics v2.0）：复用/训练 -> 统一子图协议 -> 单表评估 -> 出图

与旧 run.py 的差异
------------------------------------------------------------------
1. 评估段改调 metrics.evaluate_subgraph(...)，一次性产出：
   - 边级 P/R/F1/sER/wER：截断(_trunc) 与 不截断(_notrunc) 两套
   - 节点级 P/R/F1/sNR：参照硬编码 CORE_NODES（--core-nodes）
   - Task：RMSE（v5 空间 KNN 插值）、Ent（v5 高斯微分熵，标准化协方差）
   - Cost：EC（v5 平方距离，去重物理边）、Density（坍缩/展开两套）
   - Causal：ρ / H / V结构（坍缩/展开两套）、Sources / Sinks（单套）
   - Intervention：|T| / AR / Specificity（AR、Spec 坍缩/展开两套）
   - 网络效率维持率 Resil_AUC_exp（仅展开图；移除物理节点=删其全部时间副本）
2. 数据准备新增 std_data（StandardScaler 后）与 cov_std（标准化协方差），供 RMSE / Ent。
3. 新增 --core-nodes 参数（GT hub 节点，务必按真实数据填写）。
4. 网络效率维持率改由展开图绘制（plot_resilience_expanded）。

关键输入的角色
------------------------------------------------------------------
  --pcmci-dir / --scmi-dir   100 个 leave-one-out 扰动视图 = 训练输入
  --pcmci-full-pkl           原始时空因果图 = 掩码作用的底图，导出子图的边来自它
  --corr-adj-npy             相关邻接矩阵（与 MGC-RM 产物同源）
  --corr-edges-npy           权威相关骨架；给了就不再重算 KNN
  --full-mask-dir            已有不变子图结果目录 -> 取掩码 -> 统一协议 -> FULL_CISE
  --full-legacy-pkl          旧协议核心子图 pkl -> 原样导入 -> FULL_LEGACY
  --gt-pkl                   评估用 ground truth 核心图
  --core-nodes               GT 核心 hub 节点集（节点 P/R/F1 与 sNR 的参照）

参数位置速查
------------------------------------------------------------------
  节点截止阈值   subgraph.NODE_BUDGET（默认见 subgraph.py）；临时改用 --node-budget N
  扰动图张数     --attack-figs（默认 20，第 r 张用 seed=r，方法间配对）
  掩码边密度     --keep-frac（默认 0.0265）
  KNN 参数       --knn-k / --radius
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import pipeline as P
import subgraph as SG
import metrics2 as M
import viz as V


# ==========================================================
# I/O helpers
# ==========================================================

def load_pkl_graph(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["graph"]), np.asarray(d["val_matrix"], float)


def edges_to_skel(arr, N):
    """--corr-edges-npy -> [N,N] 对称二值骨架。兼容 [E,2] / [E,3] / object 数组。"""
    a = np.asarray(arr)
    if a.dtype == object or a.ndim == 1:
        a = np.array([list(e)[:2] for e in a])
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"edges_list 形状无法解析: {a.shape}")
    S = np.zeros((N, N), float)
    for i, j in a[:, :2].astype(int):
        if 0 <= i < N and 0 <= j < N and i != j:
            S[i, j] = S[j, i] = 1.0
    print(f"[CORR] 权威骨架 edges_list: {a.shape[0]} 条记录 -> |E|={int(S.sum()/2)}（去重对称化后）")
    return S


def corr_from_ts(X):
    c = np.corrcoef(X, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c.astype(float)


# ==========================================================
# 评估输入准备（新版 metrics 需要的额外原料）
# ==========================================================

def prepare_eval_inputs(X, pos, num_nodes):
    """
    X   : [T,N] 原始信号（若为 [N,T] 上游已转置）
    pos : [N,2] 坐标
    返回 std_data[T,N]、cov_std[N,N]、positions{id:(x,y)}
    """
    clean = np.nan_to_num(X)
    scaler = StandardScaler()
    std_data = scaler.fit_transform(clean)              # [T,N]（v5 口径：先标准化）
    cov_std = np.cov(std_data, rowvar=False)            # [N,N] 标准化协方差（供 Ent）
    positions = {int(i): (float(pos[i][0]), float(pos[i][1]))
                 for i in range(num_nodes)}             # 缺失补 0（一般不缺）
    return std_data, cov_std, positions


def plot_resilience_expanded(resil_curves, save_path):
    """网络效率维持率（展开图）：移除物理节点=删其全部时间副本，20 次平均。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, (gamma, q) in resil_curves.items():
        plt.plot(gamma, q, marker="o", label=name)
    plt.title("Network Efficiency Retention under Random Attack\n"
              "(Expanded graph; remove physical node = drop all time copies; 20-run avg)")
    plt.xlabel("Fraction of physical nodes removed  γ")
    plt.ylabel("Efficiency Retention  Q(γ)")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")


# ==========================================================
# 训练一个方法 -> Subgraph
# ==========================================================

def run_one(name, seed, cfg, base_g, base_v, corr, pos, dev, corr_skel=None):
    spec = P.METHODS[name]
    print(f"\n{'='*72}\n[RUN] {name} | seed={seed} | {spec['desc']}\n{'='*72}")
    P.set_seed(seed)
    C = P.cise(cfg.get("cise_path"))
    data = P.causal_views(cfg) if spec["views"] == "causal" else P.corr_views(cfg)
    prior = C.get_stationary_prior_mask(data, original_nodes=cfg["num_nodes"],
                                        tau_max=cfg["tau_max"])
    model = P.Extractor(P.build_encoder(spec["encoder"], cfg, dev), cfg["hidden_dim"],
                        cfg["num_nodes"], cfg["tau_max"], prior,
                        cgbp=spec["cgbp"], freeze=spec["freeze"])
    if spec.get("skip_train"):
        print("  [PRIOR_ONLY] 跳过训练")
        model = model.to(dev); hist = {"mask_std": [float("nan")]}
    else:
        model, hist = P.train(model, data, device=dev, **cfg)

    with torch.no_grad():
        fm = model.full_mask().cpu().numpy()

    if spec["views"] == "corr":
        # B3：掩码打在相关骨架上，不碰 PCMCI 的 graph/val
        skel = (corr_skel if corr_skel is not None
                else P.corr_knn_skeleton(corr, pos, cfg["knn_k"], cfg["radius"]))
        sE, sV, thr = P.mask_to_scores_corr(fm, skel, cfg["num_nodes"], cfg["tau_max"],
                                            keep_frac=cfg["keep_frac"])
        src = "edges_list.npy 权威骨架" if corr_skel is not None else \
              f"现算 KNN (k={cfg['knn_k']}, r={cfg['radius']})"
        print(f"  [B3] 相关骨架 |E|={int(skel.sum()/2)}  <- {src}")
    else:
        sE, sV, thr = P.mask_to_scores(fm, base_g, base_v, cfg["num_nodes"], cfg["tau_max"],
                                       keep_frac=cfg["keep_frac"],
                                       transposed=cfg["mask_transposed"])

    sub = SG.build(name, spec["branch"], spec["induce"], sE=sE, sV=sV,
                   B=cfg["node_budget"], base_graph=base_g, base_val=base_v,
                   corr_adj=corr, pos=pos, knn_k=cfg["knn_k"], radius=cfg["radius"],
                   corr_skel=corr_skel, rank_edges=spec.get("rank_edges", True))
    sub.meta.update(seed=seed, thr=thr, mask_std=float(hist["mask_std"][-1]))

    # B4 一致性检查：head 的节点分数 vs 掩码派生的节点分数（读出仍以掩码为准）
    if not spec["cgbp"] and not spec.get("skip_train"):
        try:
            from scipy.stats import spearmanr
            hs = P.node_head_scores(model, data, device=dev)
            if hs is not None:
                rho = float(spearmanr(hs, sV).correlation)
                sub.meta["head_vs_mask_spearman"] = rho
                print(f"  [B4] head 节点分数 vs 掩码派生节点分数 Spearman = {rho:.3f}"
                      + ("  [WARN] 相关性低 -> 掩码与 head 学到了两回事，查训练"
                         if abs(rho) < 0.3 else ""))
        except Exception as e:
            print(f"  [B4] 一致性检查跳过 ({e})")
    return sub


# ==========================================================
# main
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+",
                    default=["B2_GCN_LRTR", "B3_CORR_INV", "B4_NOCGBP"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--feature-csv", default="Env_A_timeseries.csv")
    ap.add_argument("--pos-csv", default="node_positions.csv")
    ap.add_argument("--gt-pkl", default=None)
    ap.add_argument("--pcmci-dir", required=True, help="扰动视图（训练输入），不是底图")
    ap.add_argument("--scmi-dir", required=True, help="扰动视图（训练输入），不是底图")
    ap.add_argument("--pcmci-full-pkl", required=True,
                    help="原始时空因果图 = 掩码作用的底图")
    ap.add_argument("--cadgnn-path", required=True)
    ap.add_argument("--cise-path", default=None)
    ap.add_argument("--vis-path", default=None)
    ap.add_argument("--mgcrm-pkl", default=None, help="MGC-RM 已有产物（复用，不训练）")
    ap.add_argument("--full-mask-dir", default=None,
                    help="已有不变子图结果目录（如 ./causal_invariant_results/results_20260108_180245）"
                         "，从中取掩码，按统一协议重新选点连边 -> FULL_CISE")
    ap.add_argument("--full-legacy-pkl", default=None,
                    help="旧协议核心子图 pkl（如 .../threshold_0.8/subgraph_threshold_0.8.pkl），"
                         "原样导入、不重新选点 -> FULL_LEGACY")
    ap.add_argument("--mask-temperature", type=float, default=0.6,
                    help="--full-mask-dir 里若存的是 causal_mask_kernel(logits)，"
                         "按 sigmoid(kernel/T) 还原，T 取训练末期温度")
    ap.add_argument("--corr-adj-npy", default=None,
                    help="corr_adjacency_matrix.npy；不给则由时间序列现算"
                         "（可能与 MGC-RM 产物不同源）")
    ap.add_argument("--corr-edges-npy", default=None,
                    help="edges_list.npy 权威相关骨架；给了就不再重算 KNN")
    ap.add_argument("--node-budget", type=int, default=SG.NODE_BUDGET,
                    help=f"节点截止阈值（默认 {SG.NODE_BUDGET}，来源 subgraph.NODE_BUDGET）")
    ap.add_argument("--core-nodes", type=int, nargs="+", default=None,
                    help="GT 核心 hub 节点集（节点 P/R/F1 与 sNR 的参照）。"
                         "不给则回退为 GT 边覆盖到的节点集。务必按真实数据填写。")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--keep-frac", type=float, default=0.0265)
    ap.add_argument("--attack-figs", type=int, default=200)
    ap.add_argument("--directed-plot", choices=["builtin", "reuse"], default="builtin",
                    help="builtin=viz 规格（红白蓝色条 + 加权出度大小 + 灰非核心）；"
                         "reuse=复用 plot_causal_graph2_clear2")
    ap.add_argument("--mask-not-transposed", action="store_true",
                    help="expand_and_merge 未做转置时加此项（先跑 README 的三行验证）")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    cfg = dict(P.DEFAULT_CFG)
    cfg.update(feature_file=os.path.join(args.dataset_dir, args.feature_csv),
               position_csv=os.path.join(args.dataset_dir, args.pos_csv),
               pcmci_dir=args.pcmci_dir, scmi_dir=args.scmi_dir,
               pcmci_full_pkl=args.pcmci_full_pkl, cadgnn_path=args.cadgnn_path,
               cise_path=args.cise_path, vis_path=args.vis_path,
               num_epochs=args.epochs, node_budget=args.node_budget,
               knn_k=args.knn_k, radius=args.radius, keep_frac=args.keep_frac,
               mask_transposed=not args.mask_not_transposed, out_dir=args.out)

    # ---- 数据 ----
    X = pd.read_csv(cfg["feature_file"]).select_dtypes(include=[np.number]).values
    if X.shape[0] == cfg["num_nodes"] and X.shape[1] != cfg["num_nodes"]:
        X = X.T
    pos_df = pd.read_csv(cfg["position_csv"])
    if "node_id" in pos_df.columns:               # 坐标按 node_id 升序，防止行序错位
        pos_df = pos_df.sort_values("node_id")
    pos = pos_df[["x", "y"]].values

    if args.corr_adj_npy:
        corr = np.asarray(np.load(args.corr_adj_npy, allow_pickle=True), float)
        assert corr.shape == (cfg["num_nodes"], cfg["num_nodes"]), f"corr 形状 {corr.shape}"
        print(f"[CORR] 复用 {args.corr_adj_npy}  shape={corr.shape}")
    else:
        corr = corr_from_ts(X)
        print("[CORR] 未给 --corr-adj-npy，由时间序列现算 —— 与 MGC-RM 产物不保证同源")
    corr_skel = (edges_to_skel(np.load(args.corr_edges_npy, allow_pickle=True), cfg["num_nodes"])
                 if args.corr_edges_npy else None)
    if corr_skel is None:
        print(f"[CORR] 未给 --corr-edges-npy，相关骨架由 KNN 现算 "
              f"(k={args.knn_k}, r={args.radius})")

    # 评估输入：标准化信号 / 标准化协方差 / positions 字典（供 RMSE / Ent / 各指标）
    std_data, cov_std, positions = prepare_eval_inputs(X, pos, cfg["num_nodes"])
    base_g, base_v = load_pkl_graph(args.pcmci_full_pkl)

    # ---- GT ----
    gt_pkl = args.gt_pkl or os.path.join(args.dataset_dir, "Env_A_gt_core_pcmci_tau5.pkl")
    gt_g, gt_v = load_pkl_graph(gt_pkl)
    E_gt_full = M.edges_from_graph(gt_g, gt_v)                 # 全部 GT 边（不截断），[(i,j,l,w)]
    E_gt_trunc = M.truncate_edges(E_gt_full, cfg["tau_max"])
    V_gt_edges = sorted(M.node_set(E_gt_full))                # GT 边覆盖到的节点（回退用）
    CORE_NODES = args.core_nodes if args.core_nodes else V_gt_edges
    if not args.core_nodes:
        print("[GT] 未给 --core-nodes，节点 P/R/F1 与 sNR 回退为 GT 边覆盖节点集。"
              "建议显式传入真实 hub 节点。")
    print(f"[GT] |E_gt|={len(E_gt_full)}（不截断）| {len(E_gt_trunc)}（lag<=tau_max={cfg['tau_max']}）"
          f" | |CORE_NODES|={len(CORE_NODES)}")
    print(f"[CFG] 节点截止阈值 B={cfg['node_budget']}（subgraph.NODE_BUDGET / --node-budget）")

    # ---- 子图：复用 ----
    subs = []
    if args.mgcrm_pkl:
        subs.append(SG.from_pkl(args.mgcrm_pkl, "MGC-RM", "node", "corr_knn",
                                corr_adj=corr, pos=pos, B=args.node_budget,
                                knn_k=args.knn_k, radius=args.radius, corr_skel=corr_skel,
                                rank_edges=False))
    if args.full_mask_dir:
        fm = P.load_mask_dir(args.full_mask_dir, cfg["num_nodes"], cfg["tau_max"],
                             temperature=args.mask_temperature)
        sE, sV, thr = P.mask_to_scores(fm, base_g, base_v, cfg["num_nodes"], cfg["tau_max"],
                                       keep_frac=cfg["keep_frac"],
                                       transposed=cfg["mask_transposed"])
        sub = SG.build("FULL_CISE", "edge", "pcmci_mask", sE=sE, sV=sV, B=args.node_budget,
                       base_graph=base_g, base_val=base_v, rank_edges=True)
        sub.meta.update(thr=thr, protocol="unified", source=args.full_mask_dir)
        subs.append(sub)
    if args.full_legacy_pkl:
        subs.append(SG.from_pkl_asis(args.full_legacy_pkl, "FULL_LEGACY",
                                     directed=True, weighted=True,
                                     candidate=(base_g == "-->").astype(float)))

    # ---- 子图：训练 ----
    trained = [m for m in args.methods if m not in {s.name for s in subs}]
    for seed in args.seeds:
        for m in trained:
            try:
                subs.append(run_one(m, seed, cfg, base_g, base_v, corr, pos, dev, corr_skel))
            except Exception as e:
                import traceback; print(f"[ERROR] {m} seed={seed}: {e}"); traceback.print_exc()

    print(SG.report(subs))

    # ==========================================================
    # 评估（新版 metrics v2.0）
    # ==========================================================
    rows = []
    resil_curves = {}
    for s in subs:
        try:
            res = M.evaluate_subgraph(
                V=list(s.V), E=list(s.E),
                E_gt_full=E_gt_full, gt_val_shape_L=cfg["tau_max"] + 1,
                positions=positions, std_data=std_data, cov_std=cov_std,
                core_nodes=CORE_NODES, N=cfg["num_nodes"], tau_max=cfg["tau_max"],
                directed=s.directed, resil_steps=10, resil_repeats=args.attack_figs)
        except Exception as e:
            import traceback
            print(f"[ERROR] evaluate {s.name}: {e}"); traceback.print_exc()
            continue

        # 弹性曲线单独存（展开图绘图用），其余进指标表
        gamma = res.pop("_resil_gamma"); q = res.pop("_resil_q")
        # resil_curves[s.name] = (gamma, q) # 弹性曲线
        # 将 q 扩充为 [1, steps] 以便兼容 viz 绘图矩阵要求
        # resil_curves[s.name] = (gamma, q.reshape(1, -1) if q.ndim == 1 else q)
        resil_curves[s.name] = (gamma, q)
        row = {"method": s.name, "|V|": len(s.V), "|E|": len(s.E),
               "结构": "有向有权" if s.directed else "无向无权",
               "协议": s.meta.get("protocol", "unified"),
               "mask_std": s.meta.get("mask_std", float("nan"))}
        row.update({k: v for k, v in res.items() if not k.startswith("_")})
        rows.append(row)

    # ==========================================================
    # 出图：子图可视化 + 保存 pkl + 网络效率维持率（展开）
    # ==========================================================
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    VIS = None
    for s in subs:
        d = os.path.join(fig_dir, s.name)
        if s.directed:
            if args.directed_plot == "builtin":
                V.plot_directed_subgraph_clear(s, pos, os.path.join(d, f"{s.name}_subgraph"))
            else:
                if VIS is None:
                    try:
                        p = args.vis_path or P.search_upward("visualize_mask_subgraph_clear.py")
                        VIS = P.load_module(p, "vis_mod",
                                            [str(Path(p).parent), str(Path(p).parent.parent)])
                    except Exception as e:
                        print(f"[WARN] 无法加载外部绘图模块 ({e})，回退 builtin")
                if VIS is not None:
                    V.plot_directed_subgraph(s, pos, d, VIS, cfg["tau_max"], cfg["num_nodes"])
                else:
                    V.plot_directed_subgraph_clear(s, pos, os.path.join(d, f"{s.name}_subgraph"))
        else:
            V.plot_undirected_subgraph(s, pos, os.path.join(d, f"{s.name}_subgraph"),
                                       corr_adj=corr)
        SG.save_pkl(os.path.join(args.out, "pkl", f"{s.name}.pkl"), s,
                    cfg["tau_max"], cfg["num_nodes"])

    # 网络效率维持率：展开图版本（唯一口径）
    # plot_resilience_expanded(resil_curves, os.path.join(fig_dir, f"resilience_expanded_{ts}.png"))
    # 网络效率维持率：调用 viz 的 plot_attack_runs2 绘制（展开图版本，唯一口径）
    V.plot_attack_runs2(subs, resil_curves, fig_dir, ts=ts)

    # ==========================================================
    # 写表
    # ==========================================================
    df = pd.DataFrame(rows)
    order = ["MGC-RM", "B2_GCN_LRTR", "B3_CORR_INV", "B4_NOCGBP",
             "FULL_CISE", "FULL_LEGACY", "PRIOR_ONLY"]
    df["_o"] = df["method"].apply(lambda m: order.index(m) if m in order else 99)
    df = df.sort_values("_o").drop(columns="_o")

    p_csv = os.path.join(args.out, f"metrics_{ts}.csv")
    df.to_csv(p_csv, index=False)
    print("\n" + df.to_string(index=False))
    try:
        p_excel = os.path.join(args.out, f"metrics_{ts}.xlsx")
        df.to_excel(p_excel, index=False, engine="openpyxl")
        print(f"[Saved] {p_excel}")
    except Exception as e:
        print(f"[WARN] 写 xlsx 失败（{e}），已存 csv")
    print(f"[Saved] {p_csv}")

    # ==========================================================
    # 口径说明
    # ==========================================================
    print("\n" + "=" * 72)
    print("nan = 该方法在该指标上没有定义（不是 0）。正文写 '—' 并脚注原因。")
    print("边级 P/R/F1/sER/wER 有 _notrunc（全 GT）与 _trunc（lag<=tau_max）两套；正文建议用 _trunc。")
    print("结构/干预/密度有 _col（坍缩，v5 口径）与 _exp（展开）两套：")
    print("  - Reciprocity/AR 越小越好、FlowHierarchy/Specificity 越大越好；")
    print("  - 坍缩会把不同 lag 的前向边误记为互惠、伪造有向环 -> _col 低估有向+lag 的优势；")
    print("  - 主张因果优势用 _exp，_col 作对照。")
    print("EC = v5 平方距离，只在去重物理边上算一次（不随时间副本复制），越小越好。")
    print("网络效率维持率 Resil_AUC_exp 仅展开图；移除一个物理节点=删其全部时间副本，20 次平均。")
    print("Sources/Sinks/|T|/EC/节点级/RMSE/Ent 为单套（与坍缩无关）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
