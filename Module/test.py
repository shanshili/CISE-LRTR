import numpy as np, pandas as pd
pos = pd.read_csv("D:\Tjnu-p\Mp\GNN-Causal-Inference\dataset\pems_spatial_kmeans_topK/node_positions.csv")
feat = pd.read_csv("D:\Tjnu-p\Mp\GNN-Causal-Inference\dataset\pems_spatial_kmeans_topK/id_standardize_Env_A_timeseries.csv", nrows=1)
cols = [c for c in feat.columns if c.lower() not in ("time","timestamp","date","index")]
print(len(cols), len(pos))
print(cols[:5])                        # 特征列名
print(pos.node_id.astype(int).astype(str).tolist()[:5])   # 坐标行序
