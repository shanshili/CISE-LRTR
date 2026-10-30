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


# ... [load_positions_csv, load_pkl_dict, parse_graph_structure 保持不变] ...
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


def parse_graph_structure(data, positions, is_causal=True):
    adj = np.array(data["graph"])
    G = nx.DiGraph()
    mask = (adj == "-->").any(axis=2) if adj.ndim == 3 else (adj == "-->")
    srcs, dsts = np.where(mask)
    for s, d in zip(srcs, dsts):
        s, d = int(s), int(d)
        if s not in G: G.add_node(s, pos=positions.get(s, (0, 0)))
        if d not in G: G.add_node(d, pos=positions.get(d, (0, 0)))
        G.add_edge(s, d)
    if "meta" in data and data["meta"].get("top_nodes"):
        for n in data["meta"]["top_nodes"]:
            if int(n) not in G: G.add_node(int(n), pos=positions.get(int(n), (0, 0)))
    return G


class SubGraphEvaluator:
    def __init__(self, full_data, positions):
        self.raw_data = full_data
        self.positions = positions
        self.num_nodes = full_data.shape[1]
        self.std_data = StandardScaler().fit_transform(np.nan_to_num(full_data))
        self.cov_matrix = np.cov(self.std_data, rowvar=False)

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
        tgt = np.setdiff1d(np.arange(self.num_nodes), sel)
        if len(tgt) == 0: return 0.0
        X_tr = np.array([self.positions.get(i, (0, 0)) for i in sel])
        X_te = np.array([self.positions.get(i, (0, 0)) for i in tgt])
        Y_tr = self.std_data[:, sel].T
        Y_te = self.std_data[:, tgt].T
        knn = KNeighborsRegressor(min(5, len(sel)), weights='distance')
        knn.fit(X_tr, Y_tr)
        return np.sqrt(np.mean((Y_te - knn.predict(X_te)) ** 2))

    def compute_entropy(self, nodes):
        if not nodes: return 0.0
        idx = list(set(nodes))
        sub = self.cov_matrix[np.ix_(idx, idx)] + np.eye(len(idx)) * 1e-4
        s, ld = np.linalg.slogdet(sub)
        return 0.5 * ld if s > 0 and ld > -1000 else -100.0

    # [修正] 严格 V-Structure
    def compute_causal_metrics(self, G):
        if len(G) == 0: return {}
        ind = dict(G.in_degree());
        outd = dict(G.out_degree())

        v_structs = 0
        for n in G.nodes():
            preds = list(G.predecessors(n))
            # 严格筛选：必须只有单向入边 (p->n 且 not n->p)
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
            "Sinks": sum(1 for n in G if ind[n] > 0 and outd[n] == 0),
            "V-Structures (Strict)": v_structs
        }

    def compute_intervention(self, G):
        if len(G) == 0: return {}
        # 因果图找 Sources，相关性图找 Hubs
        srcs = [n for n in G if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        if not srcs:
            d = dict(G.degree())
            srcs = sorted(d, key=d.get, reverse=True)[:5]

        impacts = [len(nx.descendants(G, s)) for s in srcs]
        avg_imp = np.mean(impacts) if impacts else 0
        spec = 1.0 - (avg_imp / max(1, G.number_of_nodes()))
        return {"Targets": len(srcs), "Avg Reach": avg_imp, "Specificity": spec}

    def compute_topo(self, G):
        eff = self._eff(G)
        gc = len(max(nx.connected_components(G.to_undirected()), key=len)) / len(G) if len(G) else 0
        return {"Eff": eff, "GiantComp": gc, "Density": nx.density(G)}

    # [新增] 部署优化实验：增量部署
    def simulation_deployment(self, G, strategy='causal'):
        nodes = list(G.nodes())
        if not nodes: return [], []

        # 排序策略
        if strategy == 'causal':
            # 优先部署 Source，其次是出度大的
            ind = dict(G.in_degree());
            outd = dict(G.out_degree())
            # Source 权重给极大，保证排在前面
            score = {n: (1000 if ind[n] == 0 else 0) + outd[n] for n in nodes}
            ordered = sorted(nodes, key=lambda n: score[n], reverse=True)
        else:
            # Correlation: 传统的 Degree Centrality
            deg = dict(G.degree())
            ordered = sorted(nodes, key=lambda n: deg[n], reverse=True)

        x, y = [], []
        # 模拟从部署 1 个传感器到 N 个
        step = max(1, len(nodes) // 10)
        for k in range(1, len(nodes) + 1, step):
            sub = ordered[:k]
            rmse = self.compute_rmse(sub)
            x.append(k)
            y.append(rmse)
        return x, y


if __name__ == "__main__":
    # Path
    d_dir = "../dataset/pems_spatial_kmeans_topK"
    f_path = os.path.join(d_dir, "id_standardize_Env_A_timeseries.csv")
    pos_path = os.path.join(d_dir, "id_node_positions.csv")

    # Files
    pkl_cse = r"../dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final3/Env_A_gt_core_pcmci_tau5.pkl"
    # pkl_cse = r"..\mask_subgraph_visualization\results_pems_20260116_172104_2\threshold_0.5\subgraph_threshold_0.5.pkl"
    pkl_corr = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

    # pkl_cse = r"../mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260108_180245_subgraph0p3/threshold_0.8/subgraph_threshold_0.8.pkl"
    # pkl_corr = r"./outputs_importance/subgraphs_corr/corr_knn_top30_20260115_123922/top30_knn15_union_lag1_tau5/subgraph_top30_knn15_union_lag1_tau5.pkl"
    #


    print("Loading...")
    df = pd.read_csv(f_path)
    X = df.select_dtypes(include=[np.number]).values
    pos = load_positions_csv(pos_path)
    ev = SubGraphEvaluator(X, pos)

    d_cse = load_pkl_dict(pkl_cse)
    d_corr = load_pkl_dict(pkl_corr)

    # Top Nodes
    tn_cse = d_cse.get("meta", {}).get("top_nodes", [])
    tn_corr = d_corr.get("meta", {}).get("top_nodes", [])
    if not tn_cse: tn_cse = list(parse_graph_structure(d_cse, pos).nodes())
    if not tn_corr: tn_corr = list(parse_graph_structure(d_corr, pos).nodes())

    G_cse = parse_graph_structure(d_cse, pos, True)
    G_corr = parse_graph_structure(d_corr, pos, False)

    # 1. 指标打印
    for name, G, tn in [("Causal", G_cse, tn_cse), ("Correlation", G_corr, tn_corr)]:
        print(f"\n=== {name} ===")
        print(f"Stats: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
        print(f"Task : RMSE {ev.compute_rmse(tn):.4f}, Ent {ev.compute_entropy(tn):.4f}")

        topo = ev.compute_topo(G)
        print(f"Topo : Eff {topo['Eff']:.4f}, Density {topo['Density']:.4f}")

        causal = ev.compute_causal_metrics(G)
        print("Causal Metrics (Revised):")
        for k, v in causal.items(): print(f"  - {k}: {v}")

        intv = ev.compute_intervention(G)
        print("Intervention:")
        for k, v in intv.items(): print(f"  - {k}: {v}")

    # 2. 部署优化绘图 (Resilient Deployment)
    print("\nSimulating Deployment Strategy...")
    cx, cy = ev.simulation_deployment(G_cse, 'causal')
    ox, oy = ev.simulation_deployment(G_corr, 'degree')

    plt.figure(figsize=(8, 5))
    plt.plot(cx, cy, 'r-o', label='Causal Strategy (Source First)')
    plt.plot(ox, oy, 'b--s', label='Correlation Strategy (Hub First)')
    plt.title("Optimization: Sensor Deployment Efficiency")
    plt.xlabel("Number of Sensors Deployed")
    plt.ylabel("Network Reconstruction RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()
