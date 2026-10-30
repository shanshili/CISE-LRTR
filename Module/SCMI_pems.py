import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm
import os

def haversine_km_matrix(lat_deg, lon_deg):
    # lat/lon: shape (N,)
    R = 6371.0088  # Earth radius in km
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat/2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # (N,N) km


class SCMIAlgorithm:
    def __init__(self, data_path, positions_path, delta=15.0, k_neighbors=5, n_bins=3):
        """
        初始化SCMI算法，适配固定坐标的传感器网络

        参数:
        data_path: str - 时间序列数据文件路径 (simulated_data.csv)
        positions_path: str - 节点位置文件路径 (node_positions.csv)
        delta: float - 距离阈值
        k_neighbors: int - K近邻数量
        n_bins: int - 离散化分箱数量

        """
        # 读取时间序列数据
        self.data_df = pd.read_csv(data_path)
        # 转换为numpy数组，形状为 (节点数, 时间步数)
        self.data = self.data_df.values.T  # 转置以获得 (节点数, 时间步数)
        self.data = np.expand_dims(self.data, axis=2)  # 添加维度以匹配原始代码格式

        # 读取位置数据
        self.positions = pd.read_csv(positions_path)

        self.delta = delta
        self.k_neighbors = k_neighbors
        self.n_bins = n_bins
        self.num_nodes = self.data.shape[0]
        self.time_steps = self.data.shape[1]

        print(f"数据加载完成: {self.num_nodes} 个节点, {self.time_steps} 个时间步")

        # 提取节点特征
        self.node_features = self._extract_node_features()

        # 构建图结构
        self.adj_matrix = self._build_graph()

        # 计算空间权重矩阵
        self.weight_matrix = self._compute_spatial_weights()

        # 计算全局Moran's I
        self.moran_i = self._compute_global_morans_i()

    def _extract_node_features(self):
        """(1) 单节点时间序列聚合 - 使用均值作为Com(s_i)"""
        return np.mean(self.data[:, :, 0], axis=1)  # shape = (节点数,)

    def _build_graph(self):
        """(2) 空间关联图构建 - 使用距离阈值法"""
        # 确保位置数据与节点一一对应
        assert len(self.positions) == self.num_nodes, "位置数据与节点数量不匹配"

        coords = self.positions[['y', 'x']].values  # 如果 y=Latitude, x=Longitude
        lat = coords[:, 0];
        lon = coords[:, 1]
        distances = haversine_km_matrix(lat, lon)

        # 距离阈值法构建邻接矩阵
        adj_matrix = (distances <= self.delta).astype(int)

        # 确保对角线为0(节点不与自身相连)
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix

    def _compute_spatial_weights(self):
        """(2) 空间权重矩阵 - 使用1/d_ij"""
        coords = self.positions[['y', 'x']].values  # 如果 y=Latitude, x=Longitude
        lat = coords[:, 0];
        lon = coords[:, 1]
        distances = haversine_km_matrix(lat, lon)

        # 将距离转换为空间权重
        weight_matrix = np.zeros_like(distances)
        # 只对距离小于阈值的节点计算权重
        mask = (distances <= self.delta) & (distances > 0)  # 排除自身距离为0的情况
        weight_matrix[mask] = 1 / distances[mask]

        # 确保对角线为0
        np.fill_diagonal(weight_matrix, 0)

        return weight_matrix

    def _compute_global_morans_i(self):
        """(3) 全局Moran's I计算"""
        # 计算均值
        com_mean = np.mean(self.node_features)

        # 计算分子
        numerator = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                numerator += self.weight_matrix[i, j] * (self.node_features[i] - com_mean) * (
                        self.node_features[j] - com_mean)

        # 计算分母
        denominator = np.sum((self.node_features - com_mean) ** 2)

        # 计算总权重
        total_weight = np.sum(self.weight_matrix)

        # 计算Moran's I
        if total_weight > 0 and denominator > 0:
            moran_i = (self.num_nodes / total_weight) * (numerator / denominator)
        else:
            moran_i = 0

        return moran_i

    def _get_conditioning_nodes(self, n_idx, o_idx):
        """(4) 条件节点选择 - 所有与n或o直接相连的节点"""
        connected_to_n = np.where(self.adj_matrix[n_idx] == 1)[0]
        connected_to_o = np.where(self.adj_matrix[o_idx] == 1)[0]

        # 合并并去重
        conditioning_nodes = np.unique(np.concatenate((connected_to_n, connected_to_o)))

        # 排除n和o自身
        conditioning_nodes = conditioning_nodes[~np.isin(conditioning_nodes, [n_idx, o_idx])]

        return conditioning_nodes

    def _discretize_data(self):
        """(4) 离散化 - 使用分箱"""
        # 将数据展平为(节点数*时间步数, 1)
        flat_data = self.data[:, :, 0].reshape(-1, 1)

        # 创建分箱器
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')

        # 应用分箱
        discretized_flat = discretizer.fit_transform(flat_data)

        # 恢复原始形状
        discretized_data = discretized_flat.reshape(self.num_nodes, self.time_steps)

        return discretized_data

    def _compute_spatial_conditional_probability(self, x_e_idx, x_target_idx, discretized_data):
        """
        (4) 空间修正概率计算
        计算 P(x_target|x_e) = (1 + I_M^*) * Conf(x_e -> x_target)
        """
        # 计算P(x_e)
        e_values = discretized_data[x_e_idx]
        e_counts = np.bincount(e_values.astype(int), minlength=self.n_bins)
        e_probs = e_counts / len(e_values)

        # 计算P(x_target, x_e)
        target_values = discretized_data[x_target_idx]
        joint_counts = np.zeros((self.n_bins, self.n_bins))

        for t in range(self.time_steps):
            e_bin = int(e_values[t])
            target_bin = int(target_values[t])
            joint_counts[e_bin, target_bin] += 1

        joint_probs = joint_counts / self.time_steps

        # 计算P(x_target|x_e) = P(x_target,x_e)/P(x_e)
        conditional_probs = np.zeros((self.n_bins, self.n_bins))
        for e in range(self.n_bins):
            if e_probs[e] > 0:
                conditional_probs[e] = joint_probs[e] / e_probs[e]

        # 应用空间修正
        spatial_conditional_probs = (1 + self.moran_i) * conditional_probs

        return spatial_conditional_probs

    def _compute_spatial_joint_conditional_probability(self, x_e_idx, x_n_idx, x_o_idx, discretized_data):
        """
        (4) 空间修正联合条件概率计算
        计算 P(x_n, x_o|x_e) = (1 + I_M^*) * Conf(x_e -> {x_n, x_o})
        """
        # 计算P(x_e)
        e_values = discretized_data[x_e_idx]
        e_counts = np.bincount(e_values.astype(int), minlength=self.n_bins)
        e_probs = e_counts / len(e_values)

        # 计算P(x_n, x_o, x_e)
        n_values = discretized_data[x_n_idx]
        o_values = discretized_data[x_o_idx]
        joint_counts = np.zeros((self.n_bins, self.n_bins, self.n_bins))

        for t in range(self.time_steps):
            e_bin = int(e_values[t])
            n_bin = int(n_values[t])
            o_bin = int(o_values[t])
            joint_counts[e_bin, n_bin, o_bin] += 1

        joint_probs = joint_counts / self.time_steps

        # 计算P(x_n, x_o|x_e) = P(x_n, x_o, x_e)/P(x_e)
        joint_conditional_probs = np.zeros((self.n_bins, self.n_bins, self.n_bins))
        for e in range(self.n_bins):
            if e_probs[e] > 0:
                joint_conditional_probs[e] = joint_probs[e] / e_probs[e]

        # 应用空间修正
        spatial_joint_conditional_probs = (1 + self.moran_i) * joint_conditional_probs

        return spatial_joint_conditional_probs

    def _compute_joint_distribution(self, x_n_idx, x_o_idx, x_e_idx, discretized_data):
        """(5) 三变量联合分布估计"""
        # 获取三个节点的离散值
        n_values = discretized_data[x_n_idx].astype(int)
        o_values = discretized_data[x_o_idx].astype(int)
        e_values = discretized_data[x_e_idx].astype(int)

        # 初始化联合分布计数
        joint_counts = np.zeros((self.n_bins, self.n_bins, self.n_bins))

        # 计数
        for t in range(self.time_steps):
            n_bin = n_values[t]
            o_bin = o_values[t]
            e_bin = e_values[t]
            joint_counts[e_bin, n_bin, o_bin] += 1

        # 归一化为概率
        joint_probs = joint_counts / self.time_steps

        return joint_probs

    def _compute_scmi(self, x_n_idx, x_o_idx, x_e_idx, discretized_data):
        """(5) SCMI计算"""
        # 获取联合分布
        joint_probs = self._compute_joint_distribution(x_n_idx, x_o_idx, x_e_idx, discretized_data)

        # 计算空间修正的条件概率
        spatial_cond_n_e = self._compute_spatial_conditional_probability(x_e_idx, x_n_idx, discretized_data)
        spatial_cond_o_e = self._compute_spatial_conditional_probability(x_e_idx, x_o_idx, discretized_data)
        spatial_joint_cond = self._compute_spatial_joint_conditional_probability(x_e_idx, x_n_idx, x_o_idx, discretized_data)

        # 计算SCMI
        scmi = 0
        for e in range(self.n_bins):
            for n in range(self.n_bins):
                for o in range(self.n_bins):
                    if joint_probs[e, n, o] > 0:
                        p_neo = joint_probs[e, n, o]

                        # 使用空间修正的概率
                        p_n_o_e = spatial_joint_cond[e, n, o] if spatial_joint_cond[e, n, o] > 0 else 1e-10
                        p_n_e = spatial_cond_n_e[e, n] if spatial_cond_n_e[e, n] > 0 else 1e-10
                        p_o_e = spatial_cond_o_e[e, o] if spatial_cond_o_e[e, o] > 0 else 1e-10

                        if p_n_e > 0 and p_o_e > 0:
                            ratio = (p_n_o_e / p_n_e) / p_o_e
                            if ratio > 0:  # 避免log(0)
                                scmi += p_neo * np.log(ratio)

        return scmi

    def compute_causal_graph(self, threshold=0.01):
        """
        (6) 因果图生成

        参数:
        threshold: float - SCMI阈值，低于此值的边将被过滤
        """
        # 离散化数据
        discretized_data = self._discretize_data()

        # 初始化因果图
        causal_graph = np.zeros((self.num_nodes, self.num_nodes))

        # 遍历所有节点对
        edges_computed = 0
        for n in tqdm(range(self.num_nodes), desc="计算因果图", total=self.num_nodes):
            for o in range(self.num_nodes):
                if n == o:
                    continue

                # 检查是否有边连接
                if self.adj_matrix[n, o] == 1:
                    # 获取条件节点
                    conditioning_nodes = self._get_conditioning_nodes(n, o)

                    # 如果没有条件节点，跳过
                    if len(conditioning_nodes) == 0:
                        continue

                    # 计算所有条件节点的SCMI
                    scmi_values = []
                    for e in conditioning_nodes:
                        try:
                            scmi = self._compute_scmi(n, o, e, discretized_data)
                            scmi_values.append(scmi)
                        except Exception as e:
                            print(f"计算SCMI时出错: 节点 {n}->{o} 条件节点 {e}, 错误: {e}")
                            scmi_values.append(0)

                    # 取最大值作为边的因果强度
                    if scmi_values:
                        max_scmi = max(scmi_values)

                        # 如果SCMI大于阈值，则保留边
                        if max_scmi >= threshold:
                            causal_graph[n, o] = max_scmi
                            edges_computed += 1

        print(f"共计算了 {edges_computed} 条边的SCMI值")

        return causal_graph

    def draw_causal_graph(self, causal_graph, output_dir, threshold=0.01):
        """
        绘制SCMI因果图

        参数:
        causal_graph: np.array - 因果图矩阵
        output_dir: str - 输出目录
        threshold: float - 阈值，用于过滤边
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取节点位置
        positions = self.positions[['x', 'y']].values
        N = positions.shape[0]

        # 创建无向图
        G = nx.Graph()  # 确保使用无向图

        # 添加节点和位置信息
        for i in range(N):
            G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

        # 添加边（无向）
        edge_count = 0
        processed_edges = set()  # 避免重复添加边

        for i in range(N):
            for j in range(N):
                if i != j and causal_graph[i, j] > threshold:
                    # 确保每条边只添加一次
                    edge_key = tuple(sorted([i, j]))
                    if edge_key not in processed_edges:
                        G.add_edge(i, j, weight=causal_graph[i, j])
                        processed_edges.add(edge_key)
                        edge_count += 1

        print(f"总共添加了 {edge_count} 条边")

        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')

        # 绘图
        # 1. 初始化画布，使用正方形尺寸
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        # 2. 强制 1:1 纵横比 (Equal Aspect Ratio)
        # 'box' 调整绘图框形状以适应数据比例
        ax.set_aspect('equal', adjustable='box')

        # 计算节点的度数
        degrees = np.array([G.degree(i) for i in range(N)])
        vmax = np.max(degrees) if np.max(degrees) > 0 else 1
        node_colors = degrees
        node_sizes = 100 + degrees * 10
        cmap_nodes = plt.cm.YlGn

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                               node_color=node_colors, cmap=cmap_nodes,
                               edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)

        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        if weights:
            # 根据权重设置边的宽度和颜色
            widths = [1 + abs(w) * 5 for w in weights]
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   width=widths,
                                   edge_color=weights,
                                   edge_cmap=plt.cm.Reds,
                                   alpha=0.7, ax=ax)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds,
                                   norm=plt.Normalize(vmin=min(weights) if weights else 0,
                                                      vmax=max(weights) if weights else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label('Causal Effect', rotation=270, labelpad=10)

        # 添加节点出度颜色图例
        sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn,
                                         norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.05, location='left')
        cbar_nodes.set_label('Number of Outgoing Edges', rotation=90, labelpad=7)

        plt.title("SCMI Causal Graph")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # 保存图像，文件名包含参数信息
        param_str = f"delta{self.delta}_k{self.k_neighbors}_bins{self.n_bins}"
        filename = f'scmi_causal_graph_{param_str}_threshold{threshold}.svg'
        # 1. 保存高清 SVG (矢量图，无损缩放)
        svg_path = os.path.join(output_dir, filename + '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        # 2. 保存 PNG (预览用)
        png_path = os.path.join(output_dir, filename + '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')  # 高DPI
        print(f"因果图已保存至：{os.path.join(output_dir, filename)}")
        plt.close()

    def draw_causal_matrix(self, causal_graph, output_dir, threshold=0.01):
        """
        绘制因果矩阵热力图

        参数:
        causal_graph: np.array - 因果图矩阵
        output_dir: str - 输出目录
        threshold: float - 阈值，用于过滤边
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建因果矩阵（二值化）
        N = causal_graph.shape[0]
        causal_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j and causal_graph[i, j] > threshold:
                    causal_matrix[i, j] = causal_graph[i, j]

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        # 只显示非零值，增强对比度
        masked_matrix = np.ma.masked_where(causal_matrix == 0, causal_matrix)
        # 使用更适合的色彩映射和范围
        vmax = np.max(causal_matrix) if np.max(causal_matrix) > 0 else 1
        vmin = np.min(causal_matrix[causal_matrix > 0]) if np.any(causal_matrix > 0) else 0

        cax = ax.imshow(causal_matrix, cmap='YlOrRd', interpolation='none', aspect='auto',
                        vmin=vmin, vmax=vmax)

        # 添加颜色条
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Causal Strength')

        # 设置标题和坐标轴
        ax.set_title("SCMI Causal Matrix", fontsize=14)
        ax.set_xlabel("Target Node", fontsize=12)
        ax.set_ylabel("Source Node", fontsize=12)

        # 设置坐标轴刻度
        ax.set_xticks(np.arange(0, N, 10))
        ax.set_yticks(np.arange(0, N, 10))
        ax.tick_params(axis='both', which='major', labelsize=5)

        plt.tight_layout()

        # 保存图像，文件名包含参数信息
        param_str = f"delta{self.delta}_k{self.k_neighbors}_bins{self.n_bins}"
        filename = f'scmi_causal_matrix_{param_str}_threshold{threshold}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename)}")
        plt.close()


def main():
    """主函数，应用SCMI算法到N100_T10000_t3数据集"""
    # 数据路径
    base_path = "C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\dataset\pems_spatial_kmeans_topK"
    data_path = os.path.join(base_path, "id_standardize_Env_A_timeseries.csv")
    positions_path = os.path.join(base_path, "id_node_positions.csv")

    # SCMI算法参数
    delta = 10.0      # 距离阈值设为15（根据数据分布调整）
    k_neighbors = 10   # K近邻数量
    n_bins = 5        # 离散化为3个箱

    # 创建参数标识字符串
    param_str = f"delta{delta}_k{k_neighbors}_bins{n_bins}"

    # 创建输出目录，包含参数信息
    output_dir = f"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\SCMI_PEMS/N100_1h_standardize_id_1"
    os.makedirs(output_dir, exist_ok=True)

    # 创建SCMI算法实例
    print("初始化SCMI算法...")
    scmi = SCMIAlgorithm(
        data_path=data_path,
        positions_path=positions_path,
        delta=delta,
        k_neighbors=k_neighbors,
        n_bins=n_bins
    )

    print(f"空间自相关 (Moran's I): {scmi.moran_i:.4f}")

    # 计算因果图
    threshold = 0.01
    print("开始计算因果图...")
    causal_graph = scmi.compute_causal_graph(threshold=threshold)
    # 保存结果，文件名包含参数信息
    output_path = os.path.join(output_dir, f"scmi_causal_graph_{param_str}_threshold{threshold}.npy")
    np.save(output_path, causal_graph)

    # 保存参数信息
    params_path = os.path.join(output_dir, f"scmi_params_{param_str}.txt")
    with open(params_path, 'w') as f:
        f.write(f"delta: {scmi.delta}\n")
        f.write(f"k_neighbors: {scmi.k_neighbors}\n")
        f.write(f"n_bins: {scmi.n_bins}\n")
        f.write(f"moran_i: {scmi.moran_i:.6f}\n")
        f.write(f"num_nodes: {scmi.num_nodes}\n")
        f.write(f"time_steps: {scmi.time_steps}\n")
        f.write(f"threshold: {threshold}\n")

    print(f"因果图已生成并保存到: {output_path}")
    print(f"参数信息已保存到: {params_path}")

    # 统计结果
    non_zero_edges = np.count_nonzero(causal_graph)
    print(f"检测到 {non_zero_edges} 条因果边")
    if non_zero_edges > 0:
        print(f"平均因果强度: {np.mean(causal_graph[causal_graph > 0]):.4f}")
        print(f"最大因果强度: {np.max(causal_graph):.4f}")

    # 绘制因果图
    print("开始绘制因果图...")
    scmi.draw_causal_graph(causal_graph, output_dir, threshold=threshold)

    # 绘制因果矩阵热力图
    print("开始绘制因果矩阵热力图...")
    scmi.draw_causal_matrix(causal_graph, output_dir, threshold=threshold)

if __name__ == "__main__":
    main()
