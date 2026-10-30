'''
版本5 (evaluate_full_v5.py) - 终极整合版

更新日志：
1. [修复] 熵计算：启用高斯微分熵公式 + 维度归一化，解决维度偏差问题。
2. [修复] 子图对齐：在 parse_graph_structure 中增加 Strict Subgraph 逻辑，确保节点数完全对齐。
3. [增强] 鲁棒性评估：simulate_attack 现在运行 20 次并取平均值，消除随机抖动。
4. [保留] 能耗分析：保留了 V4 的物理距离能耗计算。
5. [保留] 综合指标：保留了因果结构、干预分析等所有指标。
'''

import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Set, Any, List


# ==========================================
# 1. 基础数据加载
# ==========================================

def load_positions_csv(pos_csv: str) -> Dict[int, Tuple[float, float]]:
    if not os.path.exists(pos_csv):
        raise FileNotFoundError(f"Position file not found: {pos_csv}")
    df = pd.read_csv(pos_csv)
    candidates = [("node_id", "x", "y"), ("id", "x", "y"), ("node_id", "lon", "lat")]
    for a, b, c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            id_col, lon_col, lat_col = a, b, c
            break
    else:
        raise ValueError(f"Unrecognized columns in {pos_csv}")

    pos = {}
    for _, row in df.iterrows():
        raw_id = row[id_col]
        if isinstance(raw_id, str):
            m = re.search(r"\d+", raw_id)
            nid = int(m.group()) if m else int(raw_id)
        else:
            nid = int(raw_id)
        pos[nid] = (float(row[lon_col]), float(row[lat_col]))
    return pos


def load_pkl_dict(pkl_path: str) -> Dict[str, Any]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def parse_graph_structure(data: Dict[str, Any], positions: Dict[int, Tuple], label: str = "Graph") -> nx.DiGraph:
    """
    [V5 升级版解析]
    1. 解析邻接矩阵。
    2. 强制补全 top_nodes。
    3. 强制剔除多余节点 (Strict Induced Subgraph)，确保公平对比。
    """
    adj = np.array(data["graph"])
    G = nx.DiGraph()

    # 1. 获取 Top Nodes 列表
    top_nodes_list = []
    if "meta" in data and "top_nodes" in data["meta"]:
        raw_tn = data["meta"]["top_nodes"]
        if isinstance(raw_tn, np.ndarray):
            top_nodes_list = raw_tn.tolist()
        elif isinstance(raw_tn, list):
            top_nodes_list = raw_tn
        if top_nodes_list: top_nodes_list = [int(n) for n in top_nodes_list]

    num_matrix_nodes = adj.shape[0]

    # 2. 建立 ID 映射
    node_map = {}
    if top_nodes_list and len(top_nodes_list) == num_matrix_nodes:
        node_map = {i: n for i, n in enumerate(top_nodes_list)}
    else:
        # 如果矩阵大小与top_nodes不一致(通常是GT)，假设矩阵索引=节点ID
        node_map = {i: i for i in range(num_matrix_nodes)}

    # 3. 解析连边
    is_string = (adj.dtype.kind in {'U', 'S'}) or (adj.dtype == object and "-->" in adj)
    if is_string:
        mask = (adj == "-->").any(axis=2) if adj.ndim == 3 else (adj == "-->")
    else:
        try:
            adj_num = adj.astype(float)
            mask = (np.abs(adj_num) > 1e-6).any(axis=2) if adj_num.ndim == 3 else (np.abs(adj_num) > 1e-6)
        except:
            mask = np.zeros(adj.shape[:2], dtype=bool)

    srcs, dsts = np.where(mask)
    for s_idx, d_idx in zip(srcs, dsts):
        u, v = node_map.get(s_idx, s_idx), node_map.get(d_idx, d_idx)
        if u not in G: G.add_node(u, pos=positions.get(u, (0, 0)))
        if v not in G: G.add_node(v, pos=positions.get(v, (0, 0)))
        G.add_edge(u, v)

    # 4. [关键步骤] 严格子图对齐
    if top_nodes_list:
        # 补全孤立点
        for nid in top_nodes_list:
            if nid not in G: G.add_node(nid, pos=positions.get(nid, (0, 0)))

        # 只保留 top_nodes 中的点，剔除全图中的其他点
        nodes_to_keep = [n for n in top_nodes_list if n in G]
        G = G.subgraph(nodes_to_keep).copy()

    return G


# ==========================================
# 2. 评估器
# ==========================================

class SubGraphEvaluator:
    def __init__(self, full_data: np.ndarray, positions: Dict[int, Tuple]):
        self.raw_data = full_data
        self.positions = positions
        self.num_nodes = full_data.shape[1]

        # 数据标准化
        clean_data = np.nan_to_num(full_data)
        scaler = StandardScaler()
        self.std_data = scaler.fit_transform(clean_data)
        self.cov_matrix = np.cov(self.std_data, rowvar=False)

    def _custom_global_efficiency(self, G: nx.Graph) -> float:
        n = len(G)
        if n < 2: return 0.0
        try:
            # 这里的距离是指跳数(Hops)，不是物理距离
            lengths = dict(nx.all_pairs_shortest_path_length(G))
        except:
            return 0.0
        sum_inv_dist = 0.0
        for source in lengths:
            for target, dist in lengths[source].items():
                if source != target and dist > 0:
                    sum_inv_dist += 1.0 / dist
        return sum_inv_dist / (n * (n - 1))

    # [V4 Feature] 网络维护能耗
    def compute_maintenance_energy(self, G: nx.Graph) -> Dict[str, float]:
        total_energy = 0.0
        edge_count = G.number_of_edges()
        if edge_count == 0:
            return {"Total Maintenance Energy": 0.0, "Avg Energy per Link": 0.0}

        for u, v in G.edges():
            p1 = np.array(self.positions.get(u, (0, 0)))
            p2 = np.array(self.positions.get(v, (0, 0)))
            dist = np.linalg.norm(p1 - p2)
            # 能耗模型: proportional to distance squared
            total_energy += dist ** 2

        return {
            "Total Maintenance Energy": total_energy,
            "Avg Energy per Link": total_energy / edge_count
        }

    def compute_rmse(self, selected_nodes: List[int]) -> float:
        if not selected_nodes: return 999.0
        sel_indices = sorted(list(set(selected_nodes)))
        sel_indices = [n for n in sel_indices if n < self.num_nodes]  # 安全检查
        target_indices = np.setdiff1d(np.arange(self.num_nodes), sel_indices)

        if not sel_indices: return 999.0
        if len(target_indices) == 0: return 0.0

        X_train = np.array([self.positions.get(i, (0, 0)) for i in sel_indices])
        X_test = np.array([self.positions.get(i, (0, 0)) for i in target_indices])

        # 维度对齐检查
        if X_train.shape[0] != len(sel_indices) or X_test.shape[0] != len(target_indices):
            return 999.0

        Y_train = self.std_data[:, sel_indices].T
        Y_test_true = self.std_data[:, target_indices].T

        try:
            knn = KNeighborsRegressor(n_neighbors=min(5, len(sel_indices)), weights='distance')
            knn.fit(X_train, Y_train)
            Y_pred = knn.predict(X_test)
            return np.sqrt(np.mean((Y_test_true - Y_pred) ** 2))
        except:
            return 999.0

    # [V5 Fix] 修正后的高斯熵计算
    def compute_entropy(self, selected_nodes: List[int], normalize: bool = True) -> float:
        if not selected_nodes: return 0.0
        idx = list(set(selected_nodes))
        k = len(idx)

        # 提取协方差子矩阵
        sub_cov = self.cov_matrix[np.ix_(idx, idx)]

        # 防止奇异矩阵
        alpha = 1e-4
        sub_cov_reg = sub_cov + np.eye(k) * alpha

        # 1. 计算 Log Determinant
        sign, logdet = np.linalg.slogdet(sub_cov_reg)
        if sign <= 0: return -999.0  # 数值错误兜底

        # 2. 完整的高斯微分熵公式
        # H(X) = 0.5 * ln(|Sigma|) + 0.5 * k * (1 + ln(2*pi))
        const_term = 0.5 * k * (1 + np.log(2 * np.pi))
        entropy_total = 0.5 * logdet + const_term

        # 3. 维度归一化 (Per-Node Entropy)
        if normalize:
            return entropy_total / k
        else:
            return entropy_total

    def compute_causal_structure_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        if G.number_of_nodes() == 0: return {}

        # 基础指标
        reciprocity = nx.reciprocity(G)
        try:
            hierarchy = nx.flow_hierarchy(G)
        except:
            hierarchy = 0.0

        in_d = dict(G.in_degree())
        out_d = dict(G.out_degree())

        # 修正的 V-Structure 计算 (严格对撞)
        v_structures = 0
        for n in G.nodes():
            preds = list(G.predecessors(n))
            # 必须是纯父节点，不能有双向边
            strict_preds = [p for p in preds if not G.has_edge(n, p)]
            if len(strict_preds) < 2: continue

            # 检查父节点之间是否无边
            for i in range(len(strict_preds)):
                for j in range(i + 1, len(strict_preds)):
                    p1, p2 = strict_preds[i], strict_preds[j]
                    if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                        v_structures += 1

        return {
            "Reciprocity": reciprocity,
            "Flow Hierarchy": hierarchy,
            "Sources": sum(1 for n in G if in_d[n] == 0 and out_d[n] > 0),
            "Sinks": sum(1 for n in G if in_d[n] > 0 and out_d[n] == 0),
            "V-Structures": v_structures
        }

    def perform_intervention_analysis(self, G: nx.DiGraph) -> Dict[str, float]:
        if len(G) == 0: return {}
        # 优先选择 Source 节点作为干预对象
        targets = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        # 如果没有 source，选择出度最大的 Top 5
        if not targets:
            out_degrees = dict(G.out_degree())
            targets = sorted(out_degrees, key=out_degrees.get, reverse=True)[:5]

        impacts = []
        for t in targets:
            descendants = nx.descendants(G, t)
            impacts.append(len(descendants))

        avg_impact = np.mean(impacts) if impacts else 0.0
        # Specificity: 影响越精准(范围越小)，特异性越高。 1 - (avg_impact / total_nodes)
        total = max(1, G.number_of_nodes())
        specificity = 1.0 - (avg_impact / total)

        return {
            "Num Targets": len(targets),
            "Avg Impact Reach": avg_impact,
            "Impact Specificity": specificity
        }

    def compute_network_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        eff = self._custom_global_efficiency(G)
        G_undir = G.to_undirected()
        if len(G) > 0:
            largest_cc = max(nx.connected_components(G_undir), key=len)
            gc_ratio = len(largest_cc) / len(G)
        else:
            gc_ratio = 0
        return {
            "Global Efficiency": eff,
            "Giant Component Ratio": gc_ratio,
            "Edge Density": nx.density(G)
        }

    # [V5 升级] 多次模拟取平均，消除抖动
    def simulate_attack_averaged(self, G: nx.Graph, steps=10, repeats=20) -> Tuple[List[float], List[float]]:
        all_y = []
        x_axis = [i / steps for i in range(steps + 1)]

        for _ in range(repeats):
            G_curr = G.copy()
            initial_eff = self._custom_global_efficiency(G_curr)
            if initial_eff == 0: initial_eff = 1.0

            nodes = list(G_curr.nodes())
            np.random.shuffle(nodes)  # 随机移除顺序

            y_run = []
            batch = max(1, int(np.ceil(len(nodes) / steps)))
            removed = 0

            # Step 0
            y_run.append(1.0)

            for _ in range(steps):
                # Remove batch
                for _ in range(batch):
                    if nodes:
                        n = nodes.pop(0)
                        if n in G_curr: G_curr.remove_node(n)

                curr_eff = self._custom_global_efficiency(G_curr)
                y_run.append(curr_eff / initial_eff)

            # 截断或填充以确保长度一致 (防止除零导致的长度不一)
            if len(y_run) > steps + 1: y_run = y_run[:steps + 1]
            while len(y_run) < steps + 1: y_run.append(y_run[-1])

            all_y.append(y_run)

        # 计算平均值
        avg_y = np.mean(all_y, axis=0)
        return x_axis, avg_y


# ==========================================
# 3. 主流程
# ==========================================

if __name__ == "__main__":
    dataset_dir = "../dataset/pems_spatial_kmeans_topK"
    feature_path = os.path.join(dataset_dir, "id_standardize_Env_A_timeseries.csv")
    pos_csv = os.path.join(dataset_dir, "id_node_positions.csv")

    # 合成数据集 B2 B3 B4
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B2_GCN_LRTR.pkl"
    # PKL_CORR = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B3_CORR_INV.pkl"
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B4_NOCGBP.pkl"\

    # 合成数据集
    # PKL_CSE = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    # 合成数据集 MGC-RM  CISE-LRTR
    # PKL_CSE = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3_2/threshold_0.8/subgraph_threshold_0.8.pkl"
    # PKL_CORR = r"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"


    # PEMS MGC-RM  CISE-LRTR
    # PKL_CSE = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
    # PKL_CORR = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

    # PEMS B2 B3 B4
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems\pkl\B2_GCN_LRTR.pkl"
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems3\pkl\B4_NOCGBP.pkl"
    PKL_CORR= r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems3\pkl\B3_CORR_IRM.pkl"


    # PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results3_0000001\pkl\B2_GCN_LRTR.pkl"
    # PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results3_0000001\pkl\B4_NOCGBP.pkl"
    # PKL_CORR=r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_resultS3_0000001\pkl\B3_CORR_IRM.pkl"
    # PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_resultS3_0000001\pkl\FULL_CISE.pkl"

    print("Loading data...")
    df = pd.read_csv(feature_path)
    X = df.select_dtypes(include=[np.number]).values
    positions = load_positions_csv(pos_csv)
    evaluator = SubGraphEvaluator(X, positions)

    data_cse = load_pkl_dict(PKL_CSE)
    data_corr = load_pkl_dict(PKL_CORR)

    results_attack = {}

    # 进行对比评估
    for name, data in [("Causal", data_cse), ("Correlation", data_corr)]:
        print(f"\n{'=' * 20} Evaluating {name} {'=' * 20}")

        # 1. 严格解析图结构 (Strict Subgraph)
        G = parse_graph_structure(data, positions, label=name)
        nodes_list = list(G.nodes())

        # 2. 基础统计
        print(f"[Statistics]")
        print(f"  - Nodes: {G.number_of_nodes()} (Aligned)")
        print(f"  - Edges: {G.number_of_edges()}")

        # 3. 任务指标 (RMSE & Corrected Entropy)
        rmse = evaluator.compute_rmse(nodes_list)
        ent = evaluator.compute_entropy(nodes_list, normalize=True)
        print(f"[Task Performance]")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - Avg Entropy (Gaussian): {ent:.4f} (Higher=More Information)")

        # 4. 维护能耗 (Physical Cost)
        energy_metrics = evaluator.compute_maintenance_energy(G)
        print(f"[Physical Maintenance Cost]")
        print(f"  - Total Energy:   {energy_metrics['Total Maintenance Energy']:.2f}")
        print(f"  - Avg Energy/Link:{energy_metrics['Avg Energy per Link']:.2f}")

        # 5. 因果结构
        causal_metrics = evaluator.compute_causal_structure_metrics(G)
        print(f"[Causal Structure]")
        for k, v in causal_metrics.items(): print(f"  - {k:<20}: {v:.4f}")

        # 6. 干预分析
        interv_metrics = evaluator.perform_intervention_analysis(G)
        print(f"[Intervention Analysis]")
        for k, v in interv_metrics.items(): print(f"  - {k:<20}: {v:.4f}")

        # 7. 弹性模拟 (多轮平均)
        print(f"[Resilience Sim] Running 20 iterations...")
        ax, ay = evaluator.simulate_attack_averaged(G, steps=10, repeats=20)
        results_attack[name] = (ax, ay)

    # 绘图
    plt.figure(figsize=(8, 6))
    for name, (x, y) in results_attack.items():
        plt.plot(x, y, marker='o', label=f"{name} Topology")
    plt.title("Network Resilience: Efficiency Retention under Random Attack\n(Averaged over 20 runs)")
    plt.xlabel("Fraction of Nodes Removed")
    plt.ylabel("Global Efficiency (Relative)")
    plt.legend()
    plt.grid(True)

    save_path = f"./resilience_comparison_v5.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()
