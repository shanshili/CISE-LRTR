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
# 1. 基础数据加载 (保持不变)
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

        # [Modification 2] 数据预处理：标准化 (Z-Score)
        # 解决熵计算数值不稳定的问题
        clean_data = np.nan_to_num(full_data)
        scaler = StandardScaler()
        self.std_data = scaler.fit_transform(clean_data)

        # 计算标准化数据的协方差矩阵 (即相关系数矩阵，数值范围更可控)
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

    def compute_rmse(self, selected_nodes: List[int]) -> float:
        if not selected_nodes: return 999.0
        sel_indices = sorted(list(set(selected_nodes)))
        target_indices = np.setdiff1d(np.arange(self.num_nodes), sel_indices)
        if len(target_indices) == 0: return 0.0

        X_train = np.array([self.positions.get(i, (0, 0)) for i in sel_indices])
        X_test = np.array([self.positions.get(i, (0, 0)) for i in target_indices])

        # 使用标准化后的数据进行 KNN 训练，效果通常更稳定
        Y_train = self.std_data[:, sel_indices].T
        Y_test_true = self.std_data[:, target_indices].T

        knn = KNeighborsRegressor(n_neighbors=min(5, len(sel_indices)), weights='distance')
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)

        # 这里的 RMSE 是基于 standardized 数据的，便于跨数据集比较
        return np.sqrt(np.mean((Y_test_true - Y_pred) ** 2))

    def compute_entropy(self, selected_nodes: List[int]) -> float:
        """
        [Modification 2 Fix] 增强鲁棒性的熵计算
        """
        if not selected_nodes: return 0.0
        idx = list(set(selected_nodes))
        sub_cov = self.cov_matrix[np.ix_(idx, idx)]

        # 增加正则化项 alpha，防止奇异矩阵
        alpha = 1e-4
        sign, logdet = np.linalg.slogdet(sub_cov + np.eye(len(idx)) * alpha)

        # 如果行列式极小（logdet为负无穷），返回一个有意义的下界
        if sign <= 0 or logdet < -1000:
            return -100.0  # 截断值
        return 0.5 * logdet  # 忽略常数项

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

        # [Modification 1] 严格的 V-Structure (Collider) 检测
        # 定义：A -> C <- B，且 A 与 B 之间无任何连边 (Unshielded Collider)
        v_structures = 0
        nodes = list(G.nodes())

        for n in nodes:  # n 是 C
            preds = list(G.predecessors(n))
            if len(preds) < 2: continue

            # 检查父节点对
            # 使用 set 避免重复，并检查双向连通性
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    p1, p2 = preds[i], preds[j]

                    # 关键修正：必须确保 p1 和 p2 之间没有任何边 (有向或反向)
                    if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                        v_structures += 1

        return {
            "Reciprocity (Low is Causal)": reciprocity,
            "Flow Hierarchy (High is Causal)": hierarchy,
            "Source Nodes (Root Causes)": num_sources,
            "Sink Nodes (Effects)": num_sinks,
            "V-Structures (Valid Colliders)": v_structures
        }

    # [Modification 3] 干预分析 (Intervention Analysis)
    def perform_intervention_analysis(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        模拟干预：如果我改变“源头”节点，影响会扩散多远？
        """
        if len(G) == 0: return {}

        # 1. 识别潜在的干预目标 (Intervention Targets)
        # 优先选择入度为0的 Source；如果没有(如相关性图)，选择出度最大的 Hub
        sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        if not sources:
            # Fallback for Correlation graph: top 5 high degree nodes
            degrees = dict(G.degree())
            sources = sorted(degrees, key=degrees.get, reverse=True)[:5]

        impact_sizes = []
        for src in sources:
            # 计算受影响的下游节点 (Descendants)
            descendants = nx.descendants(G, src)
            impact_sizes.append(len(descendants))

        avg_impact = np.mean(impact_sizes) if impact_sizes else 0.0

        # 影响特异性 (Specificity): 1 - (Impact / Total Nodes)
        # 因果图应该只影响局部 (Specificity 高)
        # 相关性图通常影响全网 (Specificity 低)
        total_nodes = max(1, G.number_of_nodes())  # 注意：是图中节点数，不是全网
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

    # PKL_CSE = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
    PKL_CORR = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

    PKL_CSE = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    # # PKL_CORR = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3/threshold_0.8/subgraph_threshold_0.8.pkl"
    # PKL_CORR = r"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"


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
        print(f"  - Entropy: {ent:.4f} (Higher absolute value is informative)")

        # 3. 基础拓扑
        net_metrics = evaluator.compute_network_metrics(G)
        print(f"[General Topology]")
        for k, v in net_metrics.items(): print(f"  - {k:<25}: {v:.4f}")

        # 4. 因果有向结构
        causal_metrics = evaluator.compute_causal_structure_metrics(G)
        print(f"[Directed/Causal Structure]")
        for k, v in causal_metrics.items():
            print(f"  - {k:<31}: {v:.4f}")

        # 5. [新增] 干预分析
        interv_metrics = evaluator.perform_intervention_analysis(G)
        print(f"[Intervention Analysis (The 'Do' Operator)]")
        print(f"  - Targets Identified:          {interv_metrics['Num Intervention Targets']}")
        print(f"  - Avg Impact Reach (Nodes):    {interv_metrics['Avg Impact Reach (Nodes)']:.2f}")
        print(f"  - Impact Specificity [0-1]:    {interv_metrics['Impact Specificity [0-1]']:.4f}")
        print(f"    (High Specificity = Targeted Control; Low = Uncontrolled Spread)")

        # 6. 弹性
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
