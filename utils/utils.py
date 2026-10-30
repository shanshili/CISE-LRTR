"""
utils.py 代码功能说明

概述
该文件是一个工具函数集合，包含图论分析、网络特性计算、能量消耗分析、连通性评估等多种功能。
主要用于传感器网络分析、图结构处理和网络性能评估。

主要功能模块：

一、图构建和拓扑特征
1. softsort(x, tau=0.1)
   功能: 使用软排序算法对输入进行排序
   参数: x - 输入张量, tau - 温度参数
   返回: 软排序后的张量

2. topological_features_construct(G)
   功能: 计算图的多种拓扑特征
   参数: G - 图对象
   返回: 包含6种中心性指标的DataFrame

3. location_graph(location)
   功能: 基于位置信息构建K近邻图
   参数: location - 节点位置坐标
   返回: 构建的图对象和邻接矩阵

二、索引和查找功能
4. find_value_according_index_list(aim_list, index_list)
   功能: 通过索引列表查找对应的值
   参数: aim_list - 目标列表, index_list - 索引列表
   返回: 查找结果列表

三、自然连通性计算
5. natural_connectivity(G)
   功能: 计算图的自然连通性（基于邻接矩阵特征值）
   参数: G - 图对象
   返回: 自然连通性值

6. natural_connectivity2(G)
   功能: 计算图的自然连通性（基于拉普拉斯矩阵特征值）
   参数: G - 图对象
   返回: 自然连通性值

四、网络能量消耗和寿命分析
7. get_h_hop_neighbors(G, node, hop=1)
   功能: 获取节点的h跳邻居
   参数: G - 图对象, node - 中心节点, hop - 跳数
   返回: 各跳邻居字典和指定跳数的邻居列表

8. communication_energy_loss(G, node, neighbors, E_0, E_elec=600, alpha=3, beta=120, bit=1000)
   功能: 计算通信能量损失
   参数: 图对象、节点、邻居、初始能量等参数
   返回: 剩余能量

9. network_life(G)
   功能: 计算网络寿命
   参数: G - 图对象
   返回: 网络寿命和剩余平均能量

10. calculate_etr(position, Eelec=600, beta=120, alpha=2, bit=1000)
    功能: 计算单次通信的发送能耗
    参数: position - 通信距离, 其他为能量参数
    返回: 发送能耗值

11. calculate_network_lifetime(g, position, E0, f=10, l=1000)
    功能: 计算网络寿命（简化版）
    参数: g - 图对象, position - 位置, E0 - 初始能量等
    返回: 网络寿命

12. dynamic_k(n)
    功能: 根据节点数动态调整k值
    参数: n - 节点数
    返回: 合适的k值

13. calculate_aec(g, position)
    功能: 计算平均能量消耗
    参数: g - 图对象, position - 节点位置
    返回: 平均能量消耗值

五、误差计算
14. mean_squared_error(y_true, y_pred)
    功能: 计算均方误差
    参数: y_true - 真实值, y_pred - 预测值
    返回: 均方误差值

六、空洞检测和距离敏感度分析
15. find_holes(graph)
    功能: 查找图中的空洞（连通分量）
    参数: graph - 图对象
    返回: 连通分量列表

16. compute_radius(hole, graph)
    功能: 计算空洞的半径（基于图论距离）
    参数: hole - 空洞节点集, graph - 图对象
    返回: 半径值

17. compute_radius2(hole, positions)
    功能: 计算空洞的半径（基于物理距离）
    参数: hole - 空洞节点集, positions - 节点位置
    返回: 半径值

18. maximum_hole_radius(graph)
    功能: 计算最大空洞半径（基于图论距离）
    参数: graph - 图对象
    返回: 最大半径值

19. maximum_hole_radius2(positions)
    功能: 计算最大空洞半径（基于物理距离，重构图）
    参数: positions - 节点位置
    返回: 最大半径值

20. maximum_hole_radius3(positions, k=4)
    功能: 计算最大空洞半径（基于物理距离，考虑邻居数限制）
    参数: positions - 节点位置, k - 近邻数
    返回: 最大半径值

21. DS3(position, num_selected)
    功能: 计算距离敏感度指标（基于物理距离重构图）
    参数: position - 位置, num_selected - 选择的节点数
    返回: 距离敏感度值

22. DS2(g, num_selected)
    功能: 计算距离敏感度指标（基于图论距离）
    参数: g - 图对象, num_selected - 选择的节点数
    返回: 距离敏感度值

七、鲁棒性评分
23. robustness_score(graph)
    功能: 计算图的鲁棒性评分Rg
    参数: graph - 图对象
    返回: 鲁棒性评分

24. robustness_score2(G)
    功能: 计算图的鲁棒性评分Rg（另一种实现）
    参数: G - 图对象
    返回: 鲁棒性评分

八、辅助功能
25. pad_vectors(low_dim_vectors, node_lists, max_dim=450)
    功能: 填充低维向量到指定维度
    参数: low_dim_vectors - 低维向量, node_lists - 节点索引, max_dim - 最大维度
    返回: 填充后的向量

应用场景：
- 传感器网络性能评估
- 图结构特性分析
- 网络寿命和能量消耗预测
- 节点选择优化
- 网络鲁棒性和连通性分析

注意事项：
- 部分函数依赖于networkx、numpy、scipy等库
- 能量计算函数中的参数可能需要根据具体场景调整
- 空洞检测函数适用于二维平面上的节点分布
"""
import networkx as nx
import numpy as np
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import torch.nn.functional as F

def softsort(x, tau=0.1):
    # 计算每对元素的差值
    pairwise_diff = x.unsqueeze(1) - x.unsqueeze(0)
    # 计算相似度矩阵
    similarity_matrix = -pairwise_diff.abs() / tau
    # 计算软排序矩阵
    # 构建一个相似性度量,两个数值越接近（即它们之间的差越小），得到的分数越高
    soft_permutation_matrix = F.softmax(similarity_matrix, dim=1)
    # 计算软排序后的张量
    soft_sorted_x = soft_permutation_matrix @ x
    return soft_sorted_x

def topological_features_construct(G):
    # print(G.degree())
    topological_features = pd.DataFrame((nx.degree_centrality(G), nx.harmonic_centrality(G), nx.closeness_centrality(G),
                                         nx.betweenness_centrality(G), nx.subgraph_centrality(G), nx.clustering(G)))
    return topological_features
def location_graph(location):
    test = NearestNeighbors(radius = 20)
    test.fit(location)  #?/?????
    """
    dis,ind = test.radius_neighbors(radius = 0.05)
    print("dis:",dis)
    print("ind:",ind)
    """

    # Epsilon neighbor
    #A = test.radius_neighbors_graph(radius = 0.08)
    # K neighbor
    if len(location) < 9:
        A = test.kneighbors_graph(n_neighbors=len(location)-1)
    else:
        A = test.kneighbors_graph(n_neighbors= 8)
    # print("A", A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)   # 双向图，但是在这里不影响
    # print("A2:",A)
    location_g = nx.from_numpy_array(A)
    A = nx.to_pandas_adjacency(location_g)
    # nx.draw(location_g,pos = location,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
    # nx.draw_networkx_nodes(location_g, pos=location, nodelist = [0,1,2], node_size = 20, node_color="r")
    # plt.title('Knn_4_graph')
    # plt.savefig('Knn_4_graph2'+'.svg', format='svg')
    # plt.show()
    return location_g,A

"""
通过节点索引查找数据
"""
def find_value_according_index_list(aim_list, index_list):  # 索引转换 从排序索引找对应节点
    i = 0
    reslut_list = []
    while i < len(index_list):
        reslut_list.append(aim_list[index_list[i]])
        i = i + 1
    return reslut_list



"""
自然连通性
"""
def natural_connectivity(G):
    adj_spec = nx.adjacency_spectrum(G)
    adj_spec_exp = np.exp(adj_spec)
    adj_spec_exp_sum = np.sum(adj_spec_exp)
    n_c = np.log(adj_spec_exp_sum / adj_spec.shape)
    return n_c

def natural_connectivity2(G):
    # 构建拉普拉斯矩阵
    L = nx.laplacian_matrix(G).toarray()
    # 计算拉普拉斯矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(L)
    # 去掉一个零特征值
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    # 计算自然连通性
    n = G.number_of_nodes()
    # n = 1
    # log_sum = np.log(np.prod(non_zero_eigenvalues))
    # print(np.prod(non_zero_eigenvalues))
    log_sum = np.sum(np.log(non_zero_eigenvalues))
    natural_conn = log_sum / n
    return natural_conn


"""
通信能量损失
"""
def get_h_hop_neighbors(G, node, hop=1):
    '''
    :param G: 图
    :param node: 中心节点
    :param hop: 跳数
    :return:
    '''
    output = {}
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=hop))  # 广度优先搜索（BFS）
    nodes = [node]
    for i in range(1, hop+1):  # 遍历每一跳
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output, output[hop]

def communication_energy_loss(G, node, neighbors, E_0, E_elec=600, alpha=3, beta=120, bit=1000):
    # Neighbor_number = len(neighbors)
    Energy = E_0
    for i in neighbors:
        dis = nx.shortest_path_length(G, node, i)  # 和邻居之间的最短路径
        Energy -= (2 * E_elec + beta * dis ** alpha) * 1e-9 * bit   # 和范围半径内的每个邻居通信
    return Energy

def network_life(G):
    """
    至少2个节点
    """
    n = G.number_of_nodes()
    # 初始能量
    energy_loss = [6] * nx.number_of_nodes(G)  ### 应该是节点初始能量
    res_energy_avg = 0
    i = 0
    """
    如果图不是连通的，nx.radius 函数会抛出 NetworkXError，因为半径的定义要求图必须是连通的。
    对于非连通图，你需要分别计算每个连通子图的半径。
    """
    try: # 连通图
        # 尝试计算整个图的半径
        radius = nx.radius(G)
        # 通信轮次
        for i in range(100000):
            # 节点
            for j in range(n): # 单个节点的剩余能量
                # 找到通信范围
                neighbor_dict, neighbor_list = get_h_hop_neighbors(G, j, radius)
                # 计算单个节点的剩余能量
                node_energy_loss = communication_energy_loss(G, j, neighbor_list, energy_loss[j])
                energy_loss[j] = node_energy_loss                                              #### 逻辑有问题，每一轮次对能量的计算有问题
                # print(j)
            # print(energy_loss)
            # energy_collect.append(energy_loss.copy())  # 使用浅拷贝避免覆盖

            # 轮到某节点通信时没有能量了
            if any(v <= 0 for v in energy_loss):  # 判断列表中是否有小于0的数
                rest_energy = [u for u in energy_loss if u > 0]  # 删除小于0的数
                # 最后一个轮次节点消耗能量的均值
                res_energy_avg = sum(rest_energy) / n
                # print('网络寿命为：{}'.format(i + 1))
                break
        # energy_collect = np.array(energy_collect)
        # print('i',i)

    except nx.NetworkXError as e:  # 非连通图
        # 获取所有连通子图
        connected_components = list(nx.connected_components(G))
        # 计算连通子图的个数
        num_connected_components = len(connected_components)
        # print('connected: ',n)
        # print('connected_components: ', num_connected_components)
        # 计算每个连通子图的半径
        radii = []
        ii = [[] for _ in range(num_connected_components)]
        res_energy_avg_sub = [[] for _ in range(num_connected_components)]
        com = 0
        for cc in connected_components:
            # print(cc)
            subgraph = G.subgraph(cc)
            subgraph_radius = nx.radius(subgraph)
            radii.append(subgraph_radius)
            # 通信轮次
            for x in range(100000):
                # 节点
                for j in range(n):
                    # 找到通信范围
                    neighbor_dict_sub, neighbor_list_sub = get_h_hop_neighbors(G, j, subgraph_radius)
                    # 计算剩余能量
                    node_energy_loss_sub = communication_energy_loss(G, j, neighbor_list_sub, energy_loss[j])
                    energy_loss[j] = node_energy_loss_sub
                    # print(j)
                # print(energy_loss)
                # energy_collect.append(energy_loss.copy())  # 使用浅拷贝避免覆盖

                if any(v <= 0 for v in energy_loss):  # 判断列表中是否有小于0的数
                    rest_energy = [u for u in energy_loss if u > 0]  # 删除小于0的数
                    res_energy_avg_sub[com] = sum(rest_energy) / n
                    # print('网络寿命为：{}'.format(i + 1))
                    break

            ii[com] = x + 1
            com = com+1

            # print('ii',ii)
            # print('res_energy_avg_sub',res_energy_avg_sub)
        i = max(ii)
        res_energy_avg = sum(res_energy_avg_sub)/len(res_energy_avg_sub)
        # print(ii)
        # print(i)
        # print(res_energy_avg_sub)
        # print(res_energy_avg)
            # energy_collect = np.array(energy_collect)
    # print('i',i + 1)
    # ('res_energy_avg_sub', res_energy_avg)

    return i + 1, res_energy_avg



"""
平均能量消耗
"""
def calculate_etr(position,Eelec = 600, beta = 120, alpha = 2, bit=1000):
    """
    计算单次通信中的发送能耗.
    参数:
    Eelec : float
        单位信息的电子损耗.
    beta : float
        功放损失系数.
    alpha : float
        传播衰减指数.
    distance : float
        通信距离.
    返回:
    float: 发送能耗.
    """
    etr = 2*(Eelec + beta * math.pow(position, alpha))* 1e-9 * bit
    return etr

# def calculate_etr(position, Eelec=600, beta=120, alpha=3, bit=1000):
#     """
#     计算单次通信中的发送能耗.
#     使用更合理的参数设置
#     """
#     # 添加输入验证
#     if position <= 0 or np.isnan(position) or np.isinf(position):
#         return 0
#
#     try:
#         # 使用更合理的传播模型参数
#         etr = 2 * (Eelec + beta * math.pow(position, alpha)) * 1e-9 * bit
#         return etr if etr > 0 and not np.isnan(etr) and not np.isinf(etr) else 0
#     except:
#         return 0



def calculate_network_lifetime(g, position,E0,f = 10, l = 1000):
    """
    # 没有进一步详细写代码，公式上来看是AEC的倒数
    计算网络寿命.
    参数:
    E0 : float
        每个传感器节点的初始能量 (焦耳).
    E_tx : float
        发送能耗 (nJ/比特).
    E_rx : float
        接收能耗 (nJ/比特).
    f : float
        每个节点的数据传输频率 (次/秒).
    l : int
        每个数据包的大小 (比特).
    返回:
    float: 网络寿命 (秒).
    """
    # 计算每个节点的总能耗
    E_tr = calculate_etr()
    E_total = f * l * E_tr * 1e-9  # 将 nJ 转换为 J
    # 计算网络寿命
    L = E0 / E_total
    return L

def dynamic_k(n):
    """根据节点数动态调整k值"""
    if n <= 10:
        return min(4, n-1)
    elif n <= 50:
        return 4 + (n-10)//20  # 逐步增加到6
    else:
        return min(10, n//10)   # 最大10

def calculate_aec(g, position):
    """
    计算平均能量消耗 (AEC).
    参数:
    N : int
        网络中的传感器节点数量.
    E0 : float
        每个传感器节点的初始能量.
    L : int
        网络寿命.
    ET : list of float
        每个传感器节点的发送能耗列表.
    ER : list of float
        每个传感器节点的接收能耗列表.
    返回:
    float: 平均能量消耗 (AEC).
    """
    n = g.number_of_nodes()
    k = 8
    # k = dynamic_k(n)
    # k = min(10, max(3, n // 10))
    E_tr = [[] for _ in range(n)]
    E_tr_j = [0 for _ in range(k)]
    distance_matrix = squareform(pdist(position))
    # print(distance_matrix)
    # print(distance_matrix)
    tree = cKDTree(position)
    _, indices = tree.query(position, k=k + 1)  # k+1 是因为最近的一个点是自己
    # print(indices[0][1:])
    for i in range(n):
        E_tr_j[0] = 0
        m = 0
        if n < k:
            for j in range(i+1,n):
                if distance_matrix[i, j-1] < 50:
                    E_tr_j[m] = calculate_etr(distance_matrix[i, j-1])
                else:
                    E_tr_j[m] = 0
                m += 1

        else:
            for j in indices[i][1:]:
                if distance_matrix[i, j-1] < 50:
                    E_tr_j[m] = calculate_etr(distance_matrix[i, j-1])
                else:
                    E_tr_j[m] = 0
                m +=1
        # (E_tr_j)
        E_tr[i] = sum(E_tr_j)
    aec = sum(E_tr) / n # * f = 10, l = 1000
    # print('sum(E_tr)',sum(E_tr))
    return aec

    # E_tr = [0] * n
    #
    # for i in range(n):
    #     energy_sum = 0
    #     for j in indices[i][1:]:  # 跳过自己
    #         if j < n:  # 确保索引有效
    #             distance = distance_matrix[i, j]  # 修复索引
    #             if distance < 5:
    #                 energy_sum += calculate_etr(distance)
    #     E_tr[i] = energy_sum
    #
    # return sum(E_tr) / n if n > 0 else 0

# def calculate_aec(g, position):
#     """
#     计算平均能量消耗 (AEC).
#     简化版本：直接与节点数相关
#     """
#     n = g.number_of_nodes()
#     if n <= 1:
#         return 0.1  # 非零基础值
#
#     try:
#         # 基础能耗（随节点数对数增长）
#         base_energy = 0.1 * np.log(n + 1)
#
#         # 连通性因子（图越连通，能耗越高）
#         if nx.is_connected(g):
#             connectivity_bonus = 0.05 * g.number_of_edges() / n
#         else:
#             connectivity_bonus = 0.01 * nx.number_connected_components(g)
#
#         # 密度因子（节点越密集，通信越频繁）
#         if len(position) > 1:
#             distances = pdist(position)
#             avg_distance = np.mean(distances)
#             density_factor = max(0.1, 1.0 / (avg_distance + 0.1))  # 距离越小，密度越高
#         else:
#             density_factor = 1.0
#
#         aec = base_energy * (1 + connectivity_bonus) * density_factor
#         return aec
#
#     except Exception as e:
#         print(f"计算AEC时出错: {e}")
#         return 0.1 * np.log(n + 1)  # 退化到基础版本


"""
单个节点MSE
"""
# def MSE_node_feature(g,node):
#     topological_features = topological_features_construct(g)
#     data = np.array(topological_features)
#     norm_scalar = MinMaxScaler()
#     data = np.transpose(norm_scalar.fit_transform(data))
#     data = torch.tensor(data, dtype=torch.float32)
#     model_path = '../MGC-RM/model_save/autoencoder.pth'
#     autoencoder = AutoEncoder()
#     autoencoder = torch.load(model_path)
#     autoencoder.eval()
#     loss_fun = nn.MSELoss()
#     loss_history = []
#     first_node_encode = []
#     # 节点特征:2000->1
#     with torch.no_grad():
#         encoded, decoded = autoencoder(data)
#         loss = loss_fun(decoded, data)
#         loss_history.append(loss.item())
#         # print(encoded)
#         # print(loss)
#         """
#         选择重要性第一名节点的特征
#         """
#         first_node_encode.append(encoded[node].item())
#         # print(' loss: ' + str(loss.item()))
#         # print(' encode: ' + str(first_node_encode))
#     return first_node_encode

"""
图MSE
"""
# 填充低维向量
def pad_vectors(low_dim_vectors, node_lists, max_dim = 450):
    padded_vectors = torch.zeros(max_dim)
    i = 0
    for index in node_lists:
        # print(index)
        padded_vectors[index] = low_dim_vectors[i]
        i =i+1
    # print(padded_vectors)
    return padded_vectors

# def MSE_all_node_feature(g,node_list):
#     """
#     只对填充后的值做MSE,未对填充后的值进行汇聚
#     """
#     topological_features = topological_features_construct(g)
#     data = np.array(topological_features)
#     norm_scalar = MinMaxScaler()
#     data = np.transpose(norm_scalar.fit_transform(data))
#     data = torch.tensor(data, dtype=torch.float32)
#     model_path = '../MGC-RM/model_save/autoencoder.pth'
#     autoencoder = AutoEncoder()
#     autoencoder = torch.load(model_path)
#     autoencoder.eval()
#     loss_fun = nn.MSELoss()
#     loss_history = []
#     # 节点特征:2000->1
#     with torch.no_grad():
#         encoded, decoded = autoencoder(data)
#         loss = loss_fun(decoded, data)
#         loss_history.append(loss.item())
#         node_encode = encoded.clone().detach()
#         # print(encoded)
#         # print(loss)
#         # print(' loss: ' + str(loss.item()))
#         # print(' encode: ' + str(first_node_encode))
#     filtered_list = list(filter(lambda x: x != [], node_list))
#     # print(node_encode,'\n',filtered_list)
#     padded_vectors = pad_vectors(node_encode, filtered_list)
#
#     return padded_vectors

def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 (MSE)
    参数:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值
    返回:
    float: 均方误差
    """
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算误差平方
    squared_errors = (y_true - y_pred) ** 2
    # 计算均方误差
    mse = np.mean(squared_errors)
    # 计算均方根误差（RMSE）
    # mse2 = np.sqrt(mse)
    # 计算欧几里得距离
    # euclidean_distance = np.linalg.norm(y_true - y_pred)
    return mse



"""
DS
"""
def find_holes(graph):
    # 使用 BFS 查找连通分量
    holes = list(nx.connected_components(graph))
    return holes

def compute_radius(hole, graph):
    # 计算空洞的半径
    max_radius = 0
    hole_graph = graph.subgraph(hole)
    eccentricities = nx.eccentricity(hole_graph)
    """
    使用 nx.eccentricity 函数来计算每个节点的偏心率（eccentricity），
    即从该节点到图中其他节点的最短路径的最大值。
    半径是所有偏心率中的最小值
    """
    radius = min(eccentricities.values())
    return radius

def compute_radius2(hole, positions):
    """
    考虑物理位置
    """
    # 计算空洞的半径
    # 提取空洞中的节点位置
    # print(hole)
    # print(positions)
    hole_positions = [positions[node] for node in hole]
    # 计算所有节点之间的距离
    if len(hole_positions) < 2:
        return 0  # 如果空洞中只有一个节点，半径为0
    distances = pdist(hole_positions)
    # 找到最大距离
    max_distance = np.max(distances)
    # 最大空洞半径是最大距离的一半
    max_radius = max_distance / 2
    return max_radius

def maximum_hole_radius(graph):
    """
    不考虑物理距离
    """
    holes = find_holes(graph)
    max_radius = 0
    for hole in holes:
        radius = compute_radius(hole, graph)
        max_radius = max(max_radius, radius)
    return max_radius

def maximum_hole_radius2(positions):
    """
    重新构图
    考虑物理距离，但是没有限制连接数量
    """
    # 构建图
    num_nodes = len(positions)
    Gn = nx.Graph()
    Gn.add_nodes_from(range(num_nodes))
    # 计算所有节点之间的距离
    distance_matrix = squareform(pdist(positions))
    # print(distance_matrix)
    # 添加边，如果两个节点之间的距离小于阈值，则认为它们是连通的
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i, j] < 50:
                Gn.add_edge(i, j)
    # print(Gn)
    # plt.figure()
    # nx.draw(Gn,pos = positions,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
    # plt.show()
    # 查找连通分量
    holes = find_holes(Gn)

    # 计算每个空洞的最大半径
    max_radius = 0
    for hole in holes:
        radius = compute_radius2(hole, positions)
        max_radius = max(max_radius, radius)

    return max_radius

def maximum_hole_radius3(positions, k=4):
    """
    重新构图
    考虑物理距离，考虑连接的邻居个数
    但是起始必须大于4个节点

    构图时考虑物理距离没有太大意义，与KNN类似
    计算最大半径时考虑物理距离也许有意义
    """
    # 构建图
    num_nodes = len(positions)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    if num_nodes < 5:
        # 计算所有节点之间的距离
        distance_matrix = squareform(pdist(positions))
        # print(distance_matrix)
        # 添加边，如果两个节点之间的距离小于阈值，则认为它们是连通的
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distance_matrix[i, j] < 1:
                    G.add_edge(i, j)
        # plt.figure()
        # nx.draw(G,pos = positions,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
        # plt.show()
    else:
        # 使用 cKDTree 找到每个节点的最近 k 个邻居
        tree = cKDTree(positions)
        _, indices = tree.query(positions, k=k + 1)  # k+1 是因为最近的一个点是自己
        # print(indices)
        # 添加边
        for i in range(num_nodes):
            for j in indices[i][1:]:  # 跳过第一个元素（即自身）
                G.add_edge(i, j)
    # 查找连通分量
    holes = find_holes(G)

    # 计算每个空洞的最大半径
    max_radius = 0
    for hole in holes:
        radius = compute_radius2(hole, positions)
        max_radius = max(max_radius, radius)

    return max_radius

def DS3(position,num_selected):
    """
    max_radius = maximum_hole_radius2(g) 或 maximum_hole_radius3(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    """
    m = 2
    n = num_selected
    max_radius = maximum_hole_radius2(position)
    # max_radius = maximum_hole_radius3(position)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    print('max_radius',max_radius)
    print('Rst',denominator)
    if max_radius > 0:
        DS = denominator / max_radius
    else:
        DS = denominator / 0.5
    return DS

def DS2(g,num_selected):
    """
    不包含只有一个点的情况
    max_radius = maximum_hole_radius(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    """
    m = 2
    n = num_selected
    max_radius = maximum_hole_radius(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    # print('max_radius',max_radius)
    # print('Rst',denominator)
    if max_radius > 0:
        DS = denominator / max_radius
    else:
        DS = denominator / 0.5   # 值不合适的话，不能取第一个点
    return DS



"""
R_g
"""
def robustness_score(graph):
    """Calculate the robustness metric Rg for a given graph."""
    N = graph.number_of_nodes()
    c = len(list(nx.connected_components(graph)))
    # 构建拉普拉斯矩阵
    laplacian = nx.laplacian_matrix(graph).toarray()
    # 计算拉普拉斯矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(laplacian)
    # 去掉零特征值
    non_zero_eigvals = eigenvalues[eigenvalues > 1e-10]
    sum_reciprocal_nonzero = np.sum(1 / non_zero_eigvals)
    Rg = (2 / (N - 1)) * ((N - c) / sum_reciprocal_nonzero)
    return Rg

def robustness_score2(G):
    """Calculate the robustness metric Rg for a given graph."""
    # Calculate the Laplacian matrix of the graph
    laplacian = nx.laplacian_matrix(G).toarray()
    # Compute the eigenvalues of the Laplacian matrix
    eigenvalues = np.linalg.eigvalsh(laplacian)
    # Filter out the zero eigenvalues
    non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
    N = G.number_of_nodes()
    sum_reciprocals = np.sum(1 / non_zero_eigenvalues)
    # Calculate the R_g score
    rg_score = (2 / (N - 1)) * sum_reciprocals
    return rg_score


"""
W_s
"""
# def Weighted_spectrum(graph):


