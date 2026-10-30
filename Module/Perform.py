import os
import sys
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import networkx as nx
# 将项目根目录加入 sys.path
sys.path.append(project_root)
from utils.utils import (find_value_according_index_list,
                         natural_connectivity2,  DS2, calculate_aec, robustness_score,location_graph)
from utils.Causal_dataset_process import filter_pcmci_matrix_by_graph, normalize_matrix_with_sign, normalize_matrix_positive_only
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pickle
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒

# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/Causal_simulation_data_2/N100_T10000_t3')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs('./Reference indicators', exist_ok=True)

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

def create_causal_graph_from_matrices(pcmci_path, scmi_path, positions, n_nodes=100, tau_max=3):
    """
    根据PCMCI和SCMI因果矩阵构建时空因果图，并使用utils.Causal_dataset_process中的标准化方法

    Parameters:
    - pcmci_path: PCMCI结果文件路径
    - scmi_path: SCMI结果文件路径
    - positions: 节点位置数组
    - n_nodes: 节点数
    - tau_max: 最大时间延迟

    Returns:
    - G: 构建的因果图 (包含时空因果关系)
    """
    # 加载PCMCI结果
    with open(pcmci_path, 'rb') as f:
        pcmci_results = pickle.load(f)
    pcmci_graph = pcmci_results['graph']
    pcmci_val_matrix = pcmci_results['val_matrix']

    # 使用utils.Causal_dataset_process中的filter_pcmci_matrix_by_graph函数过滤PCMCI结果
    filtered_pcmci_val_matrix = filter_pcmci_matrix_by_graph(pcmci_val_matrix, pcmci_graph)

    # 使用utils.Causal_dataset_process中的normalize_matrix_with_sign函数标准化PCMCI结果
    normalized_pcmci = normalize_matrix_with_sign(filtered_pcmci_val_matrix)

    # 加载SCMI结果
    scmi_matrix = np.load(scmi_path)

    # 使用utils.Causal_dataset_process中的normalize_matrix_positive_only函数标准化SCMI结果
    normalized_scmi = normalize_matrix_positive_only(scmi_matrix)

    # 创建MultiDiGraph以支持多重边和自循环
    G = nx.MultiDiGraph()

    # 添加节点并设置位置信息
    for i in range(n_nodes):
        G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    # 添加时间因果边 (来自标准化后的PCMCI)
    for target in range(n_nodes):
        for source in range(n_nodes):
            for tau in range(1, tau_max + 1):
                if pcmci_graph[target, source, tau] == '-->':
                    causal_effect = normalized_pcmci[target, source, tau]
                    weight = abs(causal_effect)
                    # 添加时间因果边: source -> target (带时间滞后)
                    G.add_edge(source, target, key=f'time_lag_{tau}',
                               weight=weight, lag=tau, causal_effect=causal_effect,
                               edge_type='temporal')

    # 添加空间因果边 (来自标准化后的SCMI)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if scmi_matrix[i, j] > 0:  # 如果存在空间因果关系
                # 添加无向空间因果边 (使用两个方向表示无向)
                causal_effect = normalized_scmi[i, j]
                G.add_edge(i, j, key='spatial', weight=causal_effect,
                           edge_type='spatial', lag=0, causal_effect=causal_effect)
                G.add_edge(j, i, key='spatial', weight=causal_effect,
                           edge_type='spatial', lag=0, causal_effect=causal_effect)

    return G


"""
根据重要性依次选择选择节点加入，构成1-100点的100个图，观察各性能曲线趋势
很多鲁棒性的指标就是反着来，依次删掉节点，也是要对比100个图
先写衡量指标，依次计算100个数值的代码，然后画在一张图里
"""
# 参数设置
node = 400  # 扩展后的节点数
original_node = 100  # 原始节点数
data_num = 9997
Gpn = 62

# 加载数据
fea_ori = np.load('./perturbation_data/origin_data_fea_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz', allow_pickle=True)
fea_o = fea_ori['arr_0']

position_file = os.path.join(dataset_dir, 'node_positions.csv')
positions_df = pd.read_csv(position_file)
positions = positions_df[['x', 'y']].values
lon = positions[:, 0]
lat = positions[:, 1]

# if os.path.exists(pcmci_file_path) and os.path.exists(scmi_file_path):
#     G_causal = create_causal_graph_from_matrices(pcmci_file_path, scmi_file_path, positions)
#     print(f"因果图构建完成: {G_causal.number_of_nodes()} 节点, {G_causal.number_of_edges()} 边")
#

# 读取节点排序（使用PageRank生成的节点列表）
node_list = []
node_list_file = 'WPR_result/_node_list_20250908_173018_GP_62_node_400_data_9997.txt'# 'D:\Tjnu-p\Mp\GNN-Causal-Inference\dataset\Causal_simulation_data_2\out_weights_node_ranking.txt'#   # 100个节点
if os.path.exists(node_list_file):
    with open(node_list_file, 'r') as f:
        for line in f:
            node_list.append(int(list(line.strip('\n').split(','))[0]))
else:
    print(f"警告：节点列表文件 {node_list_file} 不存在，使用默认排序")
    node_list = list(range(original_node))

print(f"节点列表长度: {len(node_list)}")

# G_0 中最重要节点的拓扑特征编码
# encode_o = MSE_node_feature(G_causal,node_list[0])

selected_node = [[] for _ in range(len(node_list))]
g = [[] for _ in range(len(node_list))]
A = [[] for _ in range(len(node_list))]
natural_conn = [None] * (int(len(node_list)) + 1)
res_energy_avg = [None] * (int(len(node_list)) + 1)
communicate_circle = [None] * (int(len(node_list)) + 1)
AEC_2 = [None] * (int(len(node_list)) + 1)
mse = [None] * (int(len(node_list)) + 1)
mse_all_node = [None] * (int(len(node_list)) + 1)
Ds = [None] * (int(len(node_list)) + 1)
Rg = [None] * (int(len(node_list)) + 1)
x = [None] * (int(len(node_list)) + 1)
i = 0 # 采样计数
"""
小图密集采样显示
"""
for select_node in range(0,100,1):
    x[i] = select_node+1 # 已选择节点数目
    selected_node[i] = node_list[:select_node+1]
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(positions, selected_node[i])
    if x[i] > 1:
        g[i],A[i] = location_graph(location_list)

        natural_conn[i] = natural_connectivity2(g[i])
        # communicate_circle[i],res_energy_avg[i] = network_life(g[i])
        AEC_2[i] = calculate_aec(g[i],location_list)
        print(AEC_2[i])
        # encode = MSE_node_feature(g[i],0)
        # mse[i] = mean_squared_error(encode_o,encode)
        # mse_all_node[i] = MSE_all_node_feature(g[i],selected_node[i])
        # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
        # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
        Ds[i] = DS2(g[i], select_node + 1)
        # Ds[i] = DS3(location_list, select_node + 1)
        Rg[i] = robustness_score(g[i])
        # print(mse[i])
        # print(mse_all_node[i])
        # print(communicate_circle[i],res_energy_avg[i])
    else:  # 仅选择一个节点时
        test = NearestNeighbors(radius=0.05)
        test.fit(positions)  #
        A[i] = np.ones((1, 1))
        g[i] = nx.from_numpy_array(A[i])
        A[i] = nx.to_pandas_adjacency(g[i])

        # natural_conn[i] = natural_connectivity2(g[i])
        # communicate_circle[i], res_energy_avg[i] = network_life(g[i])
        AEC_2[i] = calculate_aec(g[i], location_list)
        # print(AEC_2[i])
        # encode = MSE_node_feature(g[i],0)
        # mse[i] = mean_squared_error(encode_o,encode)
        # mse_all_node[i] = MSE_all_node_feature(g[i], selected_node[i])
        # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
        # print(np.array(location_list)[:, 0])
        # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
        Ds[i] = DS2(g[i], select_node + 1)
        # Ds[i] = DS3(location_list, select_node + 1)
        # Rg[i] = robustness_score2(g[i])

        # print(mse[i])
        # print(mse_all_node[i])
        # nx.draw(g[i], pos=location_list, with_labels=True, alpha=0.4, node_size=10, font_size=5)
    print(x[i])
    i = i+1  # 采样计数
"""
大图粗略采样显示
"""
for select_node in range(101,int(len(node_list))+1,5):
    x[i] = select_node+1
    selected_node[i] = node_list[:select_node+1]
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(positions, selected_node[i])
    g[i],A[i] = location_graph(location_list)

    # natural_conn[i] = natural_connectivity2(g[i])
    # communicate_circle[i], res_energy_avg[i] = network_life(g[i])
    # AEC_2[i] = calculate_aec(g[i], location_list)
    # encode = MSE_node_feature(g[i],0)
    # mse[i] = mean_squared_error(encode_o,encode)
    # mse_all_node[i] = MSE_all_node_feature(g[i], selected_node[i])
    # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
    # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
    # Ds[i] = DS2(g[i], select_node + 1)
    # Ds[i] = DS3(location_list, select_node + 1)
    # print(mse[i])
    # print(mse_all_node[i])
    # print(x[i])
    i = i+1

## print(natural_conn)
# print(communicate_circle,res_energy_avg)
# natural_conn_cleaned = [value for value in natural_conn if value is not None]
# print(natural_conn_cleaned)
# max_natural_conn = natural_conn.index(max(natural_conn_cleaned))
# print(max(natural_conn_cleaned))
##  max_natural_conn = natural_conn.index(1.596723933094875)+1
# print(max_natural_conn)
# print(x)
# print(mse)

"""
多个节点特征怎么汇总
图特征怎么获得
"""





fig, conn = plt.subplots(figsize=(8, 4))
# AEC = conn.twinx()
# cc = conn.twinx()
mse_ax = conn.twinx()
aec_2 = conn.twinx()
ds_ax = conn.twinx()
rg_ax = conn.twinx()
# mse_all_ax = conn.twinx()

# cc.spines['right'].set_position(('outward', 40))
# mse_ax.spines['right'].set_position(('outward', 40))
aec_2.spines['right'].set_position(('outward', 30))
ds_ax.spines['right'].set_position(('outward', 75))
# mse_all_ax.spines['right'].set_position(('outward', 80))


line_CONN= conn.plot(x, natural_conn, marker = '.',markerfacecolor='white', label='NC', color='#d92523')
# line_AEC = AEC.plot(x, res_energy_avg, markerfacecolor='white', label='res_energy_avg', color='#2e7ebb')
# line_cc = cc.plot(x, communicate_circle,marker = '.', markerfacecolor='white', label='communicate_circle', color='#00FF00')
line_AEC2 = aec_2.plot(x, AEC_2, marker = '.',markerfacecolor='white', label='AEC', color='#2e7ebb')
# line_mse = mse_ax.plot(x, mse,marker = '.',markerfacecolor='white', label='MSE', color='#FFA500')
line_ds = ds_ax.plot(x, Ds,marker = '.',markerfacecolor='white', label='Ds', color='#00FFFF')
# line_mse_all = mse_all_ax.plot(x, mse_all_node,marker = '.', markerfacecolor='white', label='mse_all', color='#FF1493')
line_rg = ds_ax.plot(x, Rg,marker = '.',markerfacecolor='white', label='Rg', color='#FFA500')

# conn.axvline(6, c='#E89B9E', ls='--')
# plt.text(6,0,'6+')
# conn.axvline(30, c='#6495ED', ls='--')
# plt.text(30,0,'30-')
# conn.axvline(12, c='#E89B9E', ls='--')
# plt.text(12,0,'12+')
# conn.axvline(24, c='#00FF00', ls='--')
# plt.text(24,0,'24')
conn.axvline(12, c='#00FF00', ls='--')
plt.text(12,0,'12')

conn.set_ylabel('NC')
# AEC.set_ylabel('res_energy_avg')
# cc.set_ylabel('communicate_circle')
aec_2.set_ylabel('AEC')
# mse_ax.set_ylabel('MSE')
ds_ax.set_ylabel('Ds')
# mse_all_ax.set_ylabel(' mse2')
rg_ax.set_ylabel('rg')

# 设置坐标轴名
conn.set_xlabel('Number of nodes')
plt.title('Reference indicators')


# 设置图例
lines =  line_CONN+line_AEC2+line_ds+line_rg
labels = [l.get_label() for l in lines]
conn.legend(lines, labels, loc='right')
# 统一两个 y 轴的比例
# conn.set_ylim(0, 1.7)
# AEC.set_ylim(0, 1.7)
plt.savefig('.\Reference indicators\select-indicators'+timestamp+'.svg', format='svg',dpi=600)
plt.show()

