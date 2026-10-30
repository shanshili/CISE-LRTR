
'''
版本4 (evaluate_full_v4.py) - 能耗分析增强版

流程介绍：
1.整合了前几个版本的所有改进
2.新增网络维护能耗计算功能
3.保留了标准化数据处理
4.维持完整的评估指标体系
区别：在保持前版本所有功能的基础上，新增了物理成本评估维度，通过计算网络维护能耗来评估图结构的物理实现成本，特别适合评估远距离连接的成本。

函数介绍：
compute_maintenance_energy: 计算网络维护能耗，基于节点间物理距离
compute_causal_structure_metrics: 改进的因果指标，标注更清晰
perform_intervention_analysis: 完整的干预分析功能
compute_network_metrics: 标准网络拓扑指标

实现功能：
评估图结构的物理实现成本
对比因果图和相关性图的能耗差异
分析远距离连接对网络成本的影响
提供综合性的网络结构评估框架
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


def parse_graph_structure(data: Dict[str, Any], positions: Dict[int, Tuple], is_causal: bool = True) -> nx.DiGraph:
    """直接从 PKL 的 'graph' 矩阵中解析拓扑结构，强制为有向图 DiGraph"""
    if "graph" not in data:
        raise ValueError("PKL data missing 'graph' key.")

    adj_matrix = np.array(data["graph"])
    G = nx.DiGraph()

    if adj_matrix.ndim == 3:
        edge_mask = (adj_matrix == "-->").any(axis=2)
    elif adj_matrix.ndim == 2:
        edge_mask = (adj_matrix == "-->")
    else:
        raise ValueError(f"Unknown graph shape: {adj_matrix.shape}")

    srcs, dsts = np.where(edge_mask)

    for s, d in zip(srcs, dsts):
        s, d = int(s), int(d)
        if s not in G: G.add_node(s, pos=positions.get(s, (0, 0)))
        if d not in G: G.add_node(d, pos=positions.get(d, (0, 0)))
        G.add_edge(s, d)

    if "meta" in data and "top_nodes" in data["meta"]:
        top_nodes = data["meta"]["top_nodes"]
        if top_nodes is not None:
            for n in top_nodes:
                n = int(n)
                if n not in G:
                    G.add_node(n, pos=positions.get(n, (0, 0)))
    return G


# ==========================================
# 2. 增强版评估器
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
            lengths = dict(nx.all_pairs_shortest_path_length(G))
        except:
            return 0.0
        sum_inv_dist = 0.0
        for source in lengths:
            for target, dist in lengths[source].items():
                if source != target and dist > 0:
                    sum_inv_dist += 1.0 / dist
        return sum_inv_dist / (n * (n - 1))

    # [新增] 计算网络维护能耗/通信代价
    def compute_maintenance_energy(self, G: nx.Graph) -> Dict[str, float]:
        """
        计算网络的维护能耗指标。
        假设通信/连接维护能耗与物理距离的平方成正比 (Path Loss Model)。
        这能凸显相关性图因为连接了大量远距离节点而导致的巨大开销。
        """
        total_energy = 0.0
        edge_count = G.number_of_edges()
        if edge_count == 0:
            return {"Total Energy": 0.0, "Avg Energy per Link": 0.0}

        for u, v in G.edges():
            p1 = np.array(self.positions.get(u, (0, 0)))
            p2 = np.array(self.positions.get(v, (0, 0)))
            # 欧几里得距离
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
        target_indices = np.setdiff1d(np.arange(self.num_nodes), sel_indices)
        if len(target_indices) == 0: return 0.0

        X_train = np.array([self.positions.get(i, (0, 0)) for i in sel_indices])
        X_test = np.array([self.positions.get(i, (0, 0)) for i in target_indices])
        Y_train = self.std_data[:, sel_indices].T
        Y_test_true = self.std_data[:, target_indices].T

        knn = KNeighborsRegressor(n_neighbors=min(5, len(sel_indices)), weights='distance')
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        return np.sqrt(np.mean((Y_test_true - Y_pred) ** 2))

    # def compute_entropy(self, selected_nodes: List[int]) -> float:
    #     if not selected_nodes: return 0.0
    #     idx = list(set(selected_nodes))
    #     sub_cov = self.cov_matrix[np.ix_(idx, idx)]
    #     alpha = 1e-4
    #     sign, logdet = np.linalg.slogdet(sub_cov + np.eye(len(idx)) * alpha)
    #     if sign <= 0 or logdet < -1000: return -100.0
    #     return 0.5 * logdet
    # def compute_entropy(self, selected_nodes: List[int], normalize: bool = True) -> float:
    #     if not selected_nodes: return 0.0
    #
    #     idx = list(set(selected_nodes))
    #     k = len(idx)
    #
    #     # 提取子矩阵
    #     sub_cov = self.cov_matrix[np.ix_(idx, idx)]
    #
    #     # 正则化，防止奇异矩阵（保持 alpha，但要知道它代表只要冗余就会很低）
    #     alpha = 1e-4
    #     sub_cov_reg = sub_cov + np.eye(k) * alpha
    #
    #     # 计算 Log Determinant
    #     sign, logdet = np.linalg.slogdet(sub_cov_reg)
    #
    #     # 错误处理：如果行列式非正（理论上加了 alpha 不会，但为了保险）
    #     if sign <= 0:
    #         return -999.0
    #
    #     # --- 修正核心 ---
    #     # 1. 完整熵公式 (Differential Entropy)
    #     # H = 0.5 * log(|Sigma|) + 0.5 * k * (1 + log(2*pi))
    #     const_term = 0.5 * k * (1 + np.log(2 * np.pi))
    #     entropy_total = 0.5 * logdet + const_term
    #
    #     # 2. 维度归一化 (Average Entropy per Node)
    #     if normalize:
    #         return entropy_total / k
    #     else:
    #         return entropy_total


    def compute_entropy(self, selected_nodes: List[int]) -> float:
        if not selected_nodes:
            return 0.0

        idx = list(set(selected_nodes))  # 去除重复节点
        sub_cov = self.cov_matrix[np.ix_(idx, idx)]  # 提取子协方差矩阵

        # 设置微小扰动值（alpha），防止协方差矩阵为奇异矩阵
        alpha = 1e-4  # 可以根据实验调整
        n = len(idx)

        # 对协方差矩阵加上微小扰动
        sub_cov_regularized = sub_cov + np.eye(n) * alpha

        # 计算协方差矩阵的对数行列式
        sign, logdet = np.linalg.slogdet(sub_cov_regularized)

        # 检查协方差矩阵的特征值是否存在负值
        eigvals = np.linalg.eigvals(sub_cov_regularized)
        if np.any(eigvals < 0):
            print(f"警告：协方差矩阵有负特征值，特征值：{eigvals}")

        # 如果行列式小于某个阈值或为负，则返回默认熵值
        if sign <= 0 or logdet < -500:  # 调整这个阈值
            return -100.0  # 可以根据需要调整这个默认值

        # 返回对数行列式熵，0.5 是比例常数
        return 0.5 * logdet

    def compute_causal_structure_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        if G.number_of_nodes() == 0: return {}

        reciprocity = nx.reciprocity(G)
        try:
            hierarchy = nx.flow_hierarchy(G)
        except:
            hierarchy = 0.0

        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        num_sources = sum(1 for n in G.nodes() if in_degrees[n] == 0 and out_degrees[n] > 0)
        num_sinks = sum(1 for n in G.nodes() if in_degrees[n] > 0 and out_degrees[n] == 0)

        # [修正] 严格的 V-Structure 检测 (排除双向边)
        v_structures = 0
        nodes = list(G.nodes())
        for n in nodes:
            preds = list(G.predecessors(n))
            # 关键：只保留单向连接的父节点
            strict_preds = [p for p in preds if not G.has_edge(n, p)]
            if len(strict_preds) < 2: continue

            for i in range(len(strict_preds)):
                for j in range(i + 1, len(strict_preds)):
                    p1, p2 = strict_preds[i], strict_preds[j]
                    if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                        v_structures += 1

        return {
            "Reciprocity (Low is Causal)": reciprocity,
            "Flow Hierarchy (High is Causal)": hierarchy,
            "Source Nodes (Root Causes)": num_sources,
            "Sink Nodes (Effects)": num_sinks,
            "V-Structures (Valid Colliders)": v_structures
        }

    def perform_intervention_analysis(self, G: nx.DiGraph) -> Dict[str, float]:
        if len(G) == 0: return {}
        sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        if not sources:
            degrees = dict(G.degree())
            sources = sorted(degrees, key=degrees.get, reverse=True)[:5]

        impact_sizes = []
        for src in sources:
            descendants = nx.descendants(G, src)
            impact_sizes.append(len(descendants))

        avg_impact = np.mean(impact_sizes) if impact_sizes else 0.0
        total_nodes = max(1, G.number_of_nodes())
        specificity = 1.0 - (avg_impact / total_nodes)

        return {
            "Num Intervention Targets": len(sources),
            "Avg Impact Reach (Nodes)": avg_impact,
            "Impact Specificity [0-1]": specificity
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

    def simulate_attack(self, G: nx.Graph, steps=10) -> Tuple[List[float], List[float]]:
        G_curr = G.copy()
        initial_eff = self._custom_global_efficiency(G_curr)
        if initial_eff == 0: initial_eff = 1.0
        nodes = list(G_curr.nodes())
        np.random.shuffle(nodes)
        x, y = [], []
        batch = max(1, len(nodes) // steps)
        for i in range(steps + 1):
            curr_metric = self._custom_global_efficiency(G_curr)
            x.append(i / steps)
            y.append(curr_metric / initial_eff)
            if i < steps:
                for _ in range(batch):
                    if nodes:
                        n = nodes.pop(0)
                        if n in G_curr: G_curr.remove_node(n)
        return x, y


# ==========================================
# 3. 主流程
# ==========================================

if __name__ == "__main__":
    dataset_dir = "../dataset/pems_spatial_kmeans_topK"
    feature_path = os.path.join(dataset_dir, "id_standardize_Env_A_timeseries.csv")
    pos_csv = os.path.join(dataset_dir, "id_node_positions.csv")

    # 合成数据集 B2 B3 B4
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B4_NOCGBP.pkl"# B2_GCN_LRTR.pkl"
    # PKL_CORR = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results2_508_4\pkl\B3_CORR_INV.pkl"

    # 合成数据集
    # PKL_CSE = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    # 合成数据集 MGC-RM  CISE-LRTR
    # PKL_CSE = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3_2/threshold_0.8/subgraph_threshold_0.8.pkl"
    # PKL_CORR = r"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"


    # PEMS MGC-RM  CISE-LRTR
    # PKL_CSE = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
    # PKL_CORR = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

    # PEMS B2 B3 B4
    PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems3\pkl\B2_GCN_LRTR.pkl"
    # PKL_CSE = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems3\pkl\B4_NOCGBP.pkl"
    PKL_CORR= r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\baseline_results_pems3\pkl\B3_CORR_IRM.pkl"


    #PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results3_0000001\pkl\B2_GCN_LRTR.pkl"
    # PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_results3_0000001\pkl\B4_NOCGBP.pkl"
    #PKL_CORR=r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_resultS3_0000001\pkl\B3_CORR_IRM.pkl"
    # PKL_CSE =r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module/baseline_resultS3_0000001\pkl\FULL_CISE.pkl"



    print("Loading data...")
    df = pd.read_csv(feature_path)
    X = df.select_dtypes(include=[np.number]).values
    positions = load_positions_csv(pos_csv)

    data_cse = load_pkl_dict(PKL_CSE)
    data_corr = load_pkl_dict(PKL_CORR)
    evaluator = SubGraphEvaluator(X, positions)

    top_nodes_cse = data_cse.get("meta", {}).get("top_nodes", [])
    top_nodes_corr = data_corr.get("meta", {}).get("top_nodes", [])
    if not top_nodes_cse: top_nodes_cse = list(parse_graph_structure(data_cse, positions).nodes())
    if not top_nodes_corr: top_nodes_corr = list(parse_graph_structure(data_corr, positions).nodes())

    results = {}

    for name, data, top_nodes, is_causal in [
        ("Causal", data_cse, top_nodes_cse, True),
        ("Correlation", data_corr, top_nodes_corr, False)
    ]:
        print(f"\n{'=' * 20} Evaluating {name} {'=' * 20}")

        G = parse_graph_structure(data, positions, is_causal=is_causal)

        # 1. 基础统计
        print(f"[Statistics]")
        print(f"  - Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

        # 2. 任务指标
        rmse = evaluator.compute_rmse(top_nodes)
        ent = evaluator.compute_entropy(top_nodes)
        print(f"[Task Performance (Standardized)]")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - Entropy: {ent:.4f}")

        # 3. 基础拓扑
        net_metrics = evaluator.compute_network_metrics(G)
        print(f"[General Topology]")
        for k, v in net_metrics.items(): print(f"  - {k:<25}: {v:.4f}")

        # 4. [新增] 维护能耗 (Physical Cost)
        energy_metrics = evaluator.compute_maintenance_energy(G)
        print(f"[Network Maintenance Cost (Proxy for Energy/Bandwidth)]")
        print(f"  - Total Energy Cost:   {energy_metrics['Total Maintenance Energy']:.4f}")
        print(f"  - Avg Energy per Link: {energy_metrics['Avg Energy per Link']:.4f}")

        # 5. 因果有向结构
        causal_metrics = evaluator.compute_causal_structure_metrics(G)
        print(f"[Directed/Causal Structure]")
        for k, v in causal_metrics.items():
            print(f"  - {k:<31}: {v:.4f}")

        # 6. 干预分析
        interv_metrics = evaluator.perform_intervention_analysis(G)
        print(f"[Intervention Analysis (The 'Do' Operator)]")
        print(f"  - Targets Identified:          {interv_metrics['Num Intervention Targets']}")
        print(f"  - Avg Impact Reach (Nodes):    {interv_metrics['Avg Impact Reach (Nodes)']:.2f}")
        print(f"  - Impact Specificity [0-1]:    {interv_metrics['Impact Specificity [0-1]']:.4f}")

        # 7. 弹性
        ax, ay = evaluator.simulate_attack(G, steps=10)
        results[name] = {"attack": (ax, ay)}

    # 绘图
    plt.figure(figsize=(8, 6))
    for name in results:
        x, y = results[name]["attack"]
        plt.plot(x, y, marker='o', label=name)
    plt.title("Resilience: Efficiency under Random Attack")
    plt.xlabel("Removed Fraction")
    plt.ylabel("Efficiency Retention")
    plt.legend()
    plt.grid(True)
    plt.show()
