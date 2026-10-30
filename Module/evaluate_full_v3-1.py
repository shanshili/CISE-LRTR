'''
版本3-1 (evaluate_full_v3-1.py) - 严格子图解析版

流程介绍：
重新设计图解析逻辑，确保严格按top_nodes构建诱导子图
添加了更精确的节点映射机制
修复了不同数据源间节点数量不一致的问题
优化了攻击模拟算法
区别：重点解决图结构解析中的节点数量不匹配问题，确保因果图和相关性图比较时基于相同数量的节点，避免因节点数量差异导致的不公平比较。

函数介绍：
parse_graph_structure: 严格子图解析，仅保留top_nodes指定的节点
simulation_deployment: 优化的部署仿真，考虑节点ID映射
simulate_attack: 改进的攻击模拟，处理边界情况
load_positions_csv: 简化的加载函数

实现功能：
确保公平的图结构比较（相同节点集合）
处理不同格式的邻接矩阵（字符串vs数值）
准确映射矩阵索引到实际节点ID
稳定的网络分析结果输出
'''
import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import re
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Set, Any, List


# ==========================================
# 1. 基础工具
# ==========================================

def load_positions_csv(pos_csv: str) -> Dict[int, Tuple[float, float]]:
    if not os.path.exists(pos_csv): raise FileNotFoundError(f"{pos_csv}")
    df = pd.read_csv(pos_csv)
    candidates = [("node_id", "x", "y"), ("id", "x", "y"), ("node_id", "lon", "lat")]
    for a, b, c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            id_col, lon_col, lat_col = a, b, c
            break
    else:
        raise ValueError(f"Columns?")
    pos = {}
    for _, row in df.iterrows():
        raw = row[id_col]
        nid = int(re.search(r"\d+", raw).group()) if isinstance(raw, str) else int(raw)
        pos[nid] = (float(row[lon_col]), float(row[lat_col]))
    return pos


def load_pkl_dict(p):
    with open(p, "rb") as f: return pickle.load(f)


def parse_graph_structure(data, positions, label="Graph"):
    """
    [严格版解析]
    1. 解析图结构。
    2. 强制只返回 meta['top_nodes'] 定义的诱导子图 (Induced Subgraph)。
    3. 剔除不在 top_nodes 中的多余节点。
    """
    adj = np.array(data["graph"])
    num_matrix_nodes = adj.shape[0]
    G = nx.DiGraph()

    # --- 1. 读取并标准化 Top Nodes ---
    top_nodes_list = []
    if "meta" in data and "top_nodes" in data["meta"]:
        raw_tn = data["meta"]["top_nodes"]
        if isinstance(raw_tn, np.ndarray):
            top_nodes_list = raw_tn.tolist()
        elif isinstance(raw_tn, list):
            top_nodes_list = raw_tn

        # 转 int
        if top_nodes_list:
            top_nodes_list = [int(n) for n in top_nodes_list]

    print(f"[{label}] Raw Matrix: {adj.shape}, Target Top Nodes: {len(top_nodes_list)}")

    # --- 2. 建立映射 (Index -> ID) ---
    # 如果矩阵很大（全图），但 top_nodes 很小，我们假设矩阵索引对应全量 ID（0..N）
    # 如果矩阵和 top_nodes 一样大，我们假设是一一对应
    node_map = {}
    if top_nodes_list and len(top_nodes_list) == num_matrix_nodes:
        # 完美匹配
        node_map = {i: n for i, n in enumerate(top_nodes_list)}
    else:
        # 不匹配 (通常是 Causal 图：矩阵是 58x58，Top Nodes 是 30)
        # 此时假设矩阵索引 i 就是节点 ID i
        node_map = {i: i for i in range(num_matrix_nodes)}

    # --- 3. 解析连边 ---
    is_string_graph = False
    if adj.dtype.kind in {'U', 'S'}:
        is_string_graph = True
    elif adj.dtype == object:
        if "-->" in adj: is_string_graph = True

    if is_string_graph:
        if adj.ndim == 3:
            mask = (adj == "-->").any(axis=2)
        else:
            mask = (adj == "-->")
    else:
        try:
            adj_numeric = adj.astype(float)
            if adj_numeric.ndim == 3:
                mask = (np.abs(adj_numeric) > 1e-6).any(axis=2)
            else:
                mask = (np.abs(adj_numeric) > 1e-6)
        except:
            mask = np.zeros(adj.shape[:2], dtype=bool)

    srcs, dsts = np.where(mask)

    # --- 4. 构建初步图 ---
    for s_idx, d_idx in zip(srcs, dsts):
        u = node_map.get(s_idx, s_idx)
        v = node_map.get(d_idx, d_idx)
        if u not in G: G.add_node(u, pos=positions.get(u, (0, 0)))
        if v not in G: G.add_node(v, pos=positions.get(v, (0, 0)))
        G.add_edge(u, v)

    # --- 5. [关键修改] 严格子图过滤 (Strict Subgraph) ---
    if top_nodes_list:
        # A. 确保所有 top_nodes 都在图里 (即使是孤立的)
        for nid in top_nodes_list:
            if nid not in G:
                G.add_node(nid, pos=positions.get(nid, (0, 0)))

        # B. 剔除所有不在 top_nodes 里的节点
        # 这是你之前遇到 "Causal 节点太多" 的根本原因
        nodes_to_keep = [n for n in top_nodes_list if n in G]
        G = G.subgraph(nodes_to_keep).copy()

    return G, list(G.nodes())


# ==========================================
# 2. 评估器
# ==========================================

class SubGraphEvaluator:
    def __init__(self, full_data, positions):
        self.raw_data = full_data
        self.positions = positions
        self.std_data = StandardScaler().fit_transform(np.nan_to_num(full_data))
        self.num_nodes = full_data.shape[1]

    def _eff(self, G):
        n = len(G)
        if n < 2: return 0.0
        try:
            lengths = dict(nx.all_pairs_shortest_path_length(G))
        except:
            return 0.0
        s = sum(1.0 / d for src in lengths for tgt, d in lengths[src].items() if src != tgt)
        return s / (n * (n - 1))

    def compute_rmse(self, nodes):
        if not nodes: return 999.0
        sel = sorted(list(set(nodes)))
        sel = [n for n in sel if n < self.num_nodes]
        if not sel: return 999.0

        tgt = np.setdiff1d(np.arange(self.num_nodes), sel)
        if len(tgt) == 0: return 0.0

        X_tr = np.array([self.positions.get(i, (0, 0)) for i in sel])
        X_te = np.array([self.positions.get(i, (0, 0)) for i in tgt])

        if X_tr.shape[0] != len(sel) or X_te.shape[0] != len(tgt): return 999.0

        Y_tr = self.std_data[:, sel].T
        Y_te = self.std_data[:, tgt].T

        try:
            knn = KNeighborsRegressor(min(5, len(sel)), weights='distance')
            knn.fit(X_tr, Y_tr)
            return np.sqrt(np.mean((Y_te - knn.predict(X_te)) ** 2))
        except:
            return 999.0

    def compute_topo_metrics(self, G):
        return {
            "Eff": self._eff(G),
            "Density": nx.density(G),
            "Energy (EC)": sum(
                np.linalg.norm(np.array(self.positions.get(u, (0, 0))) - np.array(self.positions.get(v, (0, 0)))) ** 2
                for u, v in G.edges()
            )
        }

    def compute_causal_metrics(self, G):
        if len(G) == 0: return {}
        ind = dict(G.in_degree());
        outd = dict(G.out_degree())
        v_structs = 0
        nodes = list(G.nodes())
        for n in nodes:
            preds = list(G.predecessors(n))
            strict_preds = [p for p in preds if not G.has_edge(n, p)]
            if len(strict_preds) < 2: continue
            for i in range(len(strict_preds)):
                for j in range(i + 1, len(strict_preds)):
                    p1, p2 = strict_preds[i], strict_preds[j]
                    if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                        v_structs += 1
        return {
            "Reciprocity": nx.reciprocity(G),
            "Flow Hierarchy": nx.flow_hierarchy(G) if len(G) > 0 else 0,
            "Sources": sum(1 for n in G if ind[n] == 0 and outd[n] > 0),
            "V-Structures (Strict)": v_structs
        }

    def simulation_deployment(self, G, strategy='causal'):
        nodes = list(G.nodes())
        if not nodes: return [], []

        if strategy == 'causal':
            ind = dict(G.in_degree());
            outd = dict(G.out_degree())
            score = {n: (10000 if ind[n] == 0 else 0) + outd[n] for n in nodes}
            ordered = sorted(nodes, key=lambda n: score[n], reverse=True)
        else:
            deg = dict(G.degree())
            ordered = sorted(nodes, key=lambda n: deg[n], reverse=True)

        x, y = [], []
        step = max(1, len(nodes) // 10)
        for k in range(1, len(nodes) + 1, step):
            sub_nodes = ordered[:k]
            rmse = self.compute_rmse(sub_nodes)
            x.append(k)
            y.append(rmse)
        return x, y

    # def simulate_attack(self, G, steps=10):
    #     G_curr = G.copy()
    #     initial_eff = self._eff(G_curr)
    #     if initial_eff == 0: initial_eff = 1.0
    #
    #     nodes_to_remove = list(G_curr.nodes())
    #     np.random.shuffle(nodes_to_remove)
    #
    #     x = [0]
    #     y = [1.0]
    #     total_nodes = len(nodes_to_remove)
    #     if total_nodes == 0: return [0], [0]
    #
    #     batch_size = max(1, int(np.ceil(total_nodes / steps)))
    #     removed_count = 0
    #
    #     while removed_count < total_nodes:
    #         current_batch_size = min(batch_size, total_nodes - removed_count)
    #         for _ in range(current_batch_size):
    #             if nodes_to_remove:
    #                 n = nodes_to_remove.pop()
    #                 if n in G_curr: G_curr.remove_node(n)
    #
    #         removed_count += current_batch_size
    #         curr_eff = self._eff(G_curr)
    #         x.append(removed_count)
    #         y.append(curr_eff / initial_eff)
    #
    #     return x, y

    def simulate_attack(self, G, steps=10, num_runs=5):
        """
        模拟攻击并多次运行，返回平均效率和标准差。

        :param num_runs: 进行多少次模拟
        :return: x, y 坐标列表
        """
        G_curr = G.copy()
        initial_eff = self._eff(G_curr)
        if initial_eff == 0: initial_eff = 1.0

        nodes_to_remove = list(G_curr.nodes())
        np.random.shuffle(nodes_to_remove)

        all_x = []
        all_y = []
        total_nodes = len(nodes_to_remove)
        if total_nodes == 0: return [0], [0], [0]  # 返回3个值

        batch_size = max(1, int(np.ceil(total_nodes / steps)))

        # 多次模拟
        for _ in range(num_runs):
            G_copy = G.copy()  # 复制一份图
            nodes_to_remove_copy = nodes_to_remove.copy()
            np.random.shuffle(nodes_to_remove_copy)

            x = [0]
            y = [1.0]
            removed_count = 0

            while removed_count < total_nodes:
                current_batch_size = min(batch_size, total_nodes - removed_count)
                for _ in range(current_batch_size):
                    if nodes_to_remove_copy:
                        n = nodes_to_remove_copy.pop()
                        if n in G_copy: G_copy.remove_node(n)

                removed_count += current_batch_size
                curr_eff = self._eff(G_copy)
                x.append(removed_count)
                y.append(curr_eff / initial_eff)

            all_x.append(x)
            all_y.append(y)

        # 计算均值和标准差
        mean_y = np.mean(all_y, axis=0)
        std_y = np.std(all_y, axis=0)

        return all_x[0], mean_y, std_y  # 返回3个值


# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    SAVE_DIR = "./evaluate_under_Attack"
    os.makedirs(SAVE_DIR, exist_ok=True)
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 路径
    d_dir = "../dataset/pems_spatial_kmeans_topK"
    f_path = os.path.join(d_dir, "id_standardize_Env_A_timeseries.csv")
    pos_path = os.path.join(d_dir, "id_node_positions.csv")
    # pkl_cse = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    pkl_cse = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3_2/threshold_0.8/subgraph_threshold_0.8.pkl"
    pkl_corr = r"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B4_NOCGBP.pkl"# B2_GCN_LRTR.pkl"
    # PKL_CSE = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    #PKL_CSE = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3_2/threshold_0.8/subgraph_threshold_0.8.pkl"
    # PKL_CSE = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
    # PKL_CORR = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B3_CORR_INV.pkl"

    # pkl_cse = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
    # pkl_corr = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

    print(f"[{TIMESTAMP}] Loading Data...")
    df = pd.read_csv(f_path)
    X = df.select_dtypes(include=[np.number]).values
    pos = load_positions_csv(pos_path)
    ev = SubGraphEvaluator(X, pos)

    d_cse = load_pkl_dict(pkl_cse)
    d_corr = load_pkl_dict(pkl_corr)

    results_attack = {}
    results_deploy = {}

    print("\nParsing Graphs...")
    G_cse, _ = parse_graph_structure(d_cse, pos, label="Causal")
    G_corr, _ = parse_graph_structure(d_corr, pos, label="Correlation")

    # [验证] 现在这里的数字应该完全对齐 (比如都是 30)
    print(f"\n[Validation] Causal Nodes: {G_cse.number_of_nodes()} | Correlation Nodes: {G_corr.number_of_nodes()}")

    for name, G in [("Causal", G_cse), ("Correlation", G_corr)]:
        print(f"\n=== {name} ===")
        print(f"Stats: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")

        topo = ev.compute_topo_metrics(G)
        print(f"Topo : Eff {topo['Eff']:.4f}, Density {topo['Density']:.4f}")

        causal = ev.compute_causal_metrics(G)
        print(f"Causal: Recip {causal['Reciprocity']:.2f}, V-Struct {causal['V-Structures (Strict)']}")

        try:
            dx, dy = ev.simulation_deployment(G, strategy='causal' if name == 'Causal' else 'degree')
            results_deploy[name] = (dx, dy)
        except Exception as e:
            print(f"Error Deploy {name}: {e}")
            results_deploy[name] = ([0], [0])

        try:
            # ax, ay = ev.simulate_attack(G, steps=10)
            # results_attack[name] = (ax, ay)
            ax, mean_ay, std_ay = ev.simulate_attack(G, steps=15, num_runs=10)
            results_attack[name] = (ax, mean_ay, std_ay)

        except Exception as e:
            print(f"Error Attack {name}: {e}")
            results_attack[name] = ([0], [0])

    # Plot 1: Deployment
    # try:
    #     plt.clf()
    #     plt.figure(figsize=(8, 4))
    #     if 'Causal' in results_deploy:
    #         plt.plot(results_deploy['Causal'][0], results_deploy['Causal'][1], 'r-o', label='Causal (Source First)')
    #     if 'Correlation' in results_deploy:
    #         plt.plot(results_deploy['Correlation'][0], results_deploy['Correlation'][1], 'b--s',
    #                  label='Correlation (Hub First)')
    #     plt.title(f"Optimization: Sensor Deployment Efficiency")
    #     plt.xlabel("Number of Sensors Deployed")
    #     plt.ylabel("Network Reconstruction RMSE (Lower is Better)")
    #     plt.legend()
    #     plt.grid(True)
    #     path_deploy = os.path.join(SAVE_DIR, f"Deployment_Optimization_{TIMESTAMP}.png")
    #     plt.savefig(path_deploy, dpi=300)
    #     print(f"\n[Saved] {path_deploy}")
    #     plt.close('all')
    # except Exception as e:
    #     print(e)

    # Plot 2: Attack
    # try:
    #     plt.clf()
    #     plt.figure(figsize=(8, 4))
    #     if 'Causal' in results_attack:
    #         plt.plot(results_attack['Causal'][0], results_attack['Causal'][1], 'r-o', label='Causal Topology')
    #     if 'Correlation' in results_attack:
    #         plt.plot(results_attack['Correlation'][0], results_attack['Correlation'][1], 'b--s',
    #                  label='Correlation Topology')
    #     plt.title(f"Resilience: Network Efficiency under Random Attack")
    #     plt.xlabel("Number of Nodes Removed")
    #     plt.ylabel("Global Efficiency Retention (Higher is Better)")
    #     plt.legend()
    #     plt.grid(True)
    #     path_attack = os.path.join(SAVE_DIR, f"Resilience_Attack_{TIMESTAMP}.png")
    #     plt.savefig(path_attack, dpi=300)
    #     print(f"[Saved] {path_attack}")
    #     plt.close('all')
    # except Exception as e:
    #     print(e)
    # Plot 2: Attack (绘制均值和标准差)
    try:
        plt.clf()
        plt.figure(figsize=(8, 4))

        if 'Causal' in results_attack:
            ax, mean_ay, std_ay = results_attack['Causal']
            plt.plot(ax, mean_ay, 'r-o', label='Causal Topology')
            # plt.fill_between(ax, mean_ay - std_ay, mean_ay + std_ay, color='red', alpha=0.2)

        if 'Correlation' in results_attack:
            ax, mean_by, std_by = results_attack['Correlation']
            plt.plot(ax, mean_by, 'b--s', label='Correlation Topology')
            # plt.fill_between(ax, mean_by - std_by, mean_by + std_by, color='blue', alpha=0.2)

        plt.title(f"Resilience: Network Efficiency under Random Attack")
        plt.xlabel("Number of Nodes Removed")
        plt.ylabel("Global Efficiency Retention (Higher is Better)")
        plt.legend()
        plt.grid(True)

        path_attack = os.path.join(SAVE_DIR, f"Resilience_Attack_{TIMESTAMP}.png")
        plt.savefig(path_attack, dpi=300)
        print(f"[Saved] {path_attack}")
        plt.close('all')
    except Exception as e:
        print(e)

    print("\nDone.")
