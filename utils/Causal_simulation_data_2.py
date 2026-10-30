"""
版本2
1. 复杂结构
每个节点有多个自回归项（多个时间滞后）
每个节点与邻近节点有多个因果连接（多个时间滞后）
包含正负效应和更复杂的连接控制
2. 复杂噪声处理
3. 复杂的因果图可视化（带坐标位置）
热力图显示节点间连接强度
详细的统计信息输出
多种可视化形式（全部节点和前10个节点的时间序列图
4. 连接控制机制:防止双向连接的机制
"""
import sys
import os
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录加入 sys.path
sys.path.append(project_root)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite import data_processing as pp
import networkx as nx
sys.path.append(project_root)
from Causal_simulation_perform import visualize_top_nodes,print_analysis_results,analyze_causal_links,print_top_nodes,plot_causal_importance_scatter,plot_causal_scatter_weights


# 设置项目目录结构
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, 'dataset', 'Causal_simulation_data_2')
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子确保可重复性
np.random.seed(123)

# 参数设置
N = 100  # 节点数量
T = 10000  # 时间步长
tau_max = 3  # 最大时间滞后
transient_fraction = 0.1  # 瞬态比例

""" 
生成节点坐标数据 (随机分布在100x100的平面内)
"""
node_positions = np.random.rand(N, 2) * 100
positions_df = pd.DataFrame({
    'node_id': np.arange(N),
    'x': node_positions[:, 0],
    'y': node_positions[:, 1]
})

"""
生成时间序列数据 (带空间相关性)
links_coeffs：
节点i: [((源节点, 滞后时间), 系数, 函数),
"""
links_coeffs = {}
# 用于记录每个节点的父节点，防止双向连接
parent_map = {i: set() for i in range(N)}
for i in range(N):
    links_coeffs[i] = []

    # 添加自回归项（多个时间滞后）
    # 为每个节点添加多个时间滞后的自相关连接
    num_self_lags = np.random.randint(1, tau_max+1)
    used_self_lags = set()

    for _ in range(num_self_lags):
        auto_coeff = np.random.uniform(0.1, 0.25)
        # 确保不会重复使用相同的时间滞后
        available_lags = [x for x in range(1, tau_max + 1) if x not in used_self_lags]
        # 偶尔添加负的自回归项
        if np.random.rand() < 0.2:  # 20%概率为负
            auto_coeff = -auto_coeff

        if available_lags:
            lag = np.random.choice(available_lags)
            if lag > 0 and lag <= tau_max:
                links_coeffs[i].append(((int(i), int(-lag)), auto_coeff, lambda x: x))
                used_self_lags.add(lag)

    # 添加空间邻近节点的相互影响
    distances = np.sqrt((node_positions[:, 0] - node_positions[i, 0]) ** 2 +
                        (node_positions[:, 1] - node_positions[i, 1]) ** 2)
    nearby_nodes = np.where(distances < 15)[0]

    # 控制每个节点的连接数 (最多3个连接)
    max_connections = 3
    potential_nodes = [j for j in nearby_nodes if j != i]
    np.random.shuffle(potential_nodes)

    # 只取前几个最近的节点
    nearby_nodes = potential_nodes[:max_connections]

    for j in nearby_nodes:
        if j != i:
            # 单向控制：如果 j 是 i 的父节点，则 i 不再作为 j 的父节点
            if i in parent_map.get(j, set()):
                continue  # 避免双向连接

            # 为每对节点添加多个时间滞后的连接（1-3个）
            num_lags = np.random.randint(1, tau_max)  # 1-3个时间滞后连接
            used_lags = set()  # 避免同一对节点在同一时间滞后上重复连接

            for _ in range(num_lags):
                # 随机选择影响系数，允许正负效应
                if np.random.rand() < 0.5:
                    # 正向效应
                    coeff = np.random.uniform(0.05, 0.15)
                else:
                    # 负向效应
                    coeff = -np.random.uniform(0.03, 0.1)

                # 获取可用的时间滞后
                available_lags = [x for x in range(1, tau_max + 1) if x not in used_lags]
                if not available_lags:
                    break  # 没有更多可用的滞后时间

                lag = np.random.choice(available_lags)
                if lag > 0 and lag <= tau_max:
                    links_coeffs[i].append(((int(j), int(-lag)), coeff, lambda x: x))
                    used_lags.add(lag)
                    parent_map[i].add(j)  # 记录 j 是 i 的父节点


# # 打印每个节点的滞后详情
# print("\n每个节点的滞后详情:")
# for i in range(min(10, N)):  # 只打印前10个节点以免输出过多
#     print(f"\n节点 {i}:")
#     if i in links_coeffs and links_coeffs[i]:
#         self_lags = []  # 自回归滞后
#         external_lags = []  # 外部连接滞后
#
#         for link in links_coeffs[i]:
#             (source_node, tau), coeff, func = link
#             lag = tau
#             if source_node == i:
#                 self_lags.append(lag)
#             else:
#                 external_lags.append(lag)
#
#         if self_lags:
#             print(f"  自回归滞后: {sorted(self_lags)}")
#         if external_lags:
#             print(f"  外部连接滞后: {sorted(external_lags)}")
#         print(f"  总连接数: {len(links_coeffs[i])}")
#     else:
#         print("  无连接")


# 在生成噪声前先计算 total_time_steps
total_time_steps = int((transient_fraction + 1) * T)
print(f"总时间步数: {total_time_steps} (T={T}, transient_fraction={transient_fraction})")


# 2. 生成噪声 - 使用绝对值和基线值确保正数
noises = np.zeros((total_time_steps, N))
for j in range(N):
    base_value = 0.01
    noise_scale = 0.01  # 噪声幅度

    # 使用正态分布而不是对数正态分布，更容易控制
    noise = np.random.normal(0, noise_scale, total_time_steps)
    noises[:, j] = base_value + noise  # 直接使用正态分布噪声

    # 严格限制噪声范围
    noises[:, j] = np.clip(noises[:, j], -0.05, 0.05)

    print(f"节点{j} - 均值: {noises[:,j].mean():.2f}, 标准差: {noises[:,j].std():.2f}")
    # # 使用混合噪声分布
    # noise_type = np.random.rand()
    # if noise_type < 0.7:  # 70% 正态分布
    #     noise = np.random.normal(0, noise_scale, total_time_steps)
    # elif noise_type < 0.9:  # 20% 对数正态
    #     noise = np.random.lognormal(0, noise_scale * 0.5, total_time_steps)
    # else:  # 10% 重尾分布
    #     noise = np.random.standard_cauchy(total_time_steps) * 0.5
    #
    # noises[:, j] = base_value + np.abs(noise)  # 确保正数

# # 3. 生成时间序列数据
# try:
#     data, _ = toys.structural_causal_process(
#         links=links_coeffs,
#         T=T,
#         noises=noises,
#         seed=42,
#         transient_fraction=transient_fraction
#     )
#     # 数据范围控制
#     data = np.clip(data, a_min=0, a_max=100)  # 设置数据上限为20
#     print(f"\n生成的数据形状: {data.shape}")
#     print(f"数据统计 - 最小值: {data.min():.2f}, 最大值: {data.max():.2f}, 均值: {data.mean():.2f}")
#     print(f"前5个节点的均值: {[data[:, i].mean() for i in range(5)]}")
# except Exception as e:
#     print(f"生成数据时出错: {e}")
#     raise


# # 调试代码：检查所有滞后值
# print("\n=== 调试信息：检查滞后值 ===")
# positive_lags = []
# all_lags = []
#
# for i in links_coeffs:
#     for link in links_coeffs[i]:
#         (source_node, tau), coeff, func = link
#         all_lags.append(tau)
#         if tau > 0:  # 找出正值
#             positive_lags.append((i, source_node, tau, coeff))
#
# print(f"总连接数: {len(all_lags)}")
# print(f"所有滞后值范围: {min(all_lags) if all_lags else 0} 到 {max(all_lags) if all_lags else 0}")
#
# if positive_lags:
#     print(f"发现 {len(positive_lags)} 个正滞后值:")
#     for target, source, lag, coeff in positive_lags[:10]:  # 只显示前10个
#         print(f"  节点{target} <- 节点{source}, 滞后={lag}, 系数={coeff:.3f}")
# else:
#     print("未发现正滞后值")
#
# # 检查是否有零值
# zero_lags = [tau for tau in all_lags if tau == 0]
# if zero_lags:
#     print(f"发现 {len(zero_lags)} 个零滞后值")
#
# print("=== 调试信息结束 ===\n")

# 3. 生成时间序列数据
try:
    data, _ = toys.structural_causal_process(
        links=links_coeffs,
        T=T,
        noises=noises,
        seed=1,
        transient_fraction=transient_fraction
    )

    # 数据后处理 - 优化
    data = np.abs(data)  # 确保正数
    # # 局部平滑而非全局裁剪
    # data = pd.DataFrame(data).rolling(window=3, min_periods=1).mean().values
    # # 标准化数据
    # data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 20  # 缩放到0-20范围


except Exception as e:
    print(f"生成数据时出错: {e}")
    raise

# 创建tigramite DataFrame
var_names = [f'Node_{i}' for i in range(N)]
dataframe = pp.DataFrame(
    data,
    var_names=var_names,
    datatime=np.arange(data.shape[0])
)

"""
可视化部分
"""
# 1. 绘制节点坐标分布图
plt.figure(figsize=(10, 8))
plt.scatter(positions_df['x'], positions_df['y'], s=50, alpha=0.7)
plt.title('Node Position Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'node_positions.png'))
plt.close()

# 2. 使用tigramite绘制时间序列 - 全部节点
plt.figure(figsize=(22, 12))
tp.plot_timeseries(
    dataframe=dataframe,
    figsize=(22, 12),
    label_fontsize=1     # 坐标轴标签字体大小
)
plt.suptitle('Simulated Time Series Data (All Nodes)', fontsize=6, y=0.98)
plt.xticks(fontsize=1)  # 手动设置刻度字体大小
plt.yticks(fontsize=1)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
plt.savefig(os.path.join(output_dir, 'timeseries_plot_all.png'), dpi=800)
plt.close()

# 3. 使用tigramite绘制时间序列 - 前10个节点
# 创建前10个节点的新DataFrame
subset_data = pp.DataFrame(
    data=data[:, :10],
    var_names=var_names[:10],
    datatime=np.arange(data.shape[0])
)

plt.figure(figsize=(14, 8))
tp.plot_timeseries(dataframe=subset_data)
plt.suptitle('Simulated Time Series Data (First 10 Nodes)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'timeseries_plot_first10.png'))
plt.close()

"""
保存数据到CSV
"""
# 保存数据
pd.DataFrame(data).to_csv(
    os.path.join(output_dir, 'simulated_data.csv'),
    index=False,
    header=var_names
)

# 保存节点坐标数据
positions_df.to_csv(
    os.path.join(output_dir, 'node_positions.csv'),
    index=False
)

print("\n数据生成完成:")
print(f"- 模拟数据已保存到: {os.path.join(output_dir, 'simulated_data.csv')}")
print(f"- 节点坐标已保存到: {os.path.join(output_dir, 'node_positions.csv')}")
print(f"- 节点分布图已保存到: {os.path.join(output_dir, 'node_positions.png')}")
print(f"- 全部节点时间序列图已保存到: {os.path.join(output_dir, 'timeseries_plot_all.png')}")
print(f"- 前10个节点时间序列图已保存到: {os.path.join(output_dir, 'timeseries_plot_first10.png')}")


def plot_causal_graph_with_positions(links_coeffs, node_positions, output_dir, title="Causal Graph with Node Positions", show=True):
    """
    使用真实坐标绘制因果图
    :param links_coeffs: 因果关系字典
    :param node_positions: 节点坐标数组 (N, 2)
    :param output_dir: 输出目录
    :param title: 图标题
    :param show: 是否显示图像
    """
    N = node_positions.shape[0]
    G = nx.MultiDiGraph()  # 改为MultiDiGraph

    # 添加节点
    for i in range(N):
        G.add_node(i)

    # 添加边并保留所有时间滞后信息
    for target in links_coeffs:
        for link in links_coeffs[target]:
            (source, tau), coeff, func = link
            if source is not None and source < N:
                lag = abs(tau)
                # 添加边到MultiDiGraph，使用lag作为key
                G.add_edge(source, target, key=lag, weight=coeff, lag=lag)

    # 使用真实坐标作为节点位置
    pos = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(N)}

    # 计算每个节点的出边数量（出边表示该节点影响其他节点的数量）
    out_degree = np.zeros(N)
    for source, target, key, data in G.edges(keys=True, data=True):
        out_degree[source] += 1  # 节点source有一条出边

    # 设置绘图参数
    plt.figure(figsize=(12, 10))
    # 绘制节点
    # nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgreen', alpha=0.8)
    # 绘制节点，颜色根据出边数量
    # node_sizes = 300 + out_degree * 20  # 基础大小300，根据出边数增加大小
    vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
    node_colors = plt.cm.autumn(out_degree / vmax) if vmax > 0 else plt.cm.autumn(np.zeros(N))

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=out_degree,
                           cmap=plt.cm.autumn, alpha=0.8)

    nx.draw_networkx_labels(G, pos, font_size=8)

    # 分别处理自相关边（自循环）和普通边
    self_edges = []  # 自相关边
    normal_edges = []  # 普通边

    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    # 获取所有权重用于颜色映射
    all_weights = [abs(data['weight']) for source, target, key, data in G.edges(keys=True, data=True)]
    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)

        # 确保颜色映射是对称的，以便更好地表示正负关系
        abs_max = max(abs(min_weight), abs(max_weight))
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1


    # 创建颜色映射（基于权重实际值，包括正负）
    cmap = plt.cm.seismic  # 红-黄-蓝反转，负值为蓝色，正值为红色
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # 绘制普通边（节点间连接）
    for source, target, key, data in normal_edges:
        lag = data['lag']
        weight = data['weight']  # 使用权重绝对值

        # 根据权重大小设置颜色
        color = cmap(norm(weight))

        # 根据权重绝对值设置边宽度
        width = 1 + abs(weight) * 8  # 基础宽度1，根据权重绝对值调整

        # 绘制单条边
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target, key)],
                               width=width,
                               edge_color=[color],
                               alpha=0.7, arrows=True, arrowsize=20)

    # 绘制自相关边（自循环）
    for source, target, key, data in self_edges:
        lag = data['lag']
        weight = data['weight']

        # 根据权重大小设置颜色
        color = cmap(norm(weight))

        # 根据权重绝对值设置边宽度
        width = 1 + abs(weight) * 8

        # 绘制自循环边
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target, key)],
                               width=width,
                               edge_color=[color],
                               alpha=0.7, arrows=True, arrowsize=20,
                               connectionstyle=f'arc3,rad=0.3')  # 弯曲的自循环

    # 添加边标签（包含时间滞后和效应符号）
    edge_labels = {}
    for source, target, key, data in G.edges(keys=True, data=True):
        lag = data['lag']
        weight = data['weight']
        sign = '+' if weight >= 0 else '-'
        edge_labels[(source, target, key)] = f"{sign}τ{lag}"

    # 绘制边滞后标签
    for (source, target, key), label in edge_labels.items():
        # 计算边的中点位置
        if source == target:  # 自循环
            # 自循环标签位置
            x, y = pos[source]
            label_pos = (x + 1.5, y + 1.5)
        else:  # 普通边
            # 普通边中点位置
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            label_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

            # 如果有多条边连接同一对节点，稍微偏移标签位置
            edge_count = len([k for uu, vv, k, d in G.edges(keys=True, data=True) if (uu, vv) == (source, target)])
            if edge_count > 1:
                label_pos = (label_pos[0] + 0.5, label_pos[1] + 0.5)

        # 根据权重符号设置标签颜色
        weight = None
        for uu, vv, kk, dd in G.edges(keys=True, data=True):
            if (uu, vv, kk) == (source, target, key):
                weight = dd['weight']
                break

        label_color = 'red' if weight is not None and weight >= 0 else 'blue'

        plt.text(label_pos[0], label_pos[1], label,
                 fontsize=2, ha='center', va='center', color=label_color,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 添加颜色图例（权重）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Connection Weight', rotation=270, labelpad=20)

    # 添加节点出边数量颜色图例
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.autumn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), shrink=0.8, location='left')
    cbar_nodes.set_label('Number of Outgoing Edges', rotation=270, labelpad=20)


    plt.title(title)
    plt.tight_layout()

    # 保存图像
    output_file = os.path.join(output_dir, 'causal_graph_with_positions.png')
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"\n带坐标信息的因果图已保存至: {output_file}")
    plt.close()

    # 打印连接统计信息
    print("\n因果连接统计:")
    lag_count = {}
    positive_count = 0
    negative_count = 0
    weight_ranges = {"<-0.3": 0, "-0.3--0.1": 0, "-0.1-0.1": 0, "0.1-0.3": 0, ">0.3": 0}

    for source, target, key, data in G.edges(keys=True, data=True):
        lag = data['lag']
        lag_count[lag] = lag_count.get(lag, 0) + 1

        # 统计正负权重
        weight = data['weight']
        if weight >= 0:
            positive_count += 1
        else:
            negative_count += 1

        # 统计权重分布
        if weight < -0.3:
            weight_ranges["<-0.3"] += 1
        elif weight < -0.1:
            weight_ranges["-0.3--0.1"] += 1
        elif weight < 0.1:
            weight_ranges["-0.1-0.1"] += 1
        elif weight < 0.3:
            weight_ranges["0.1-0.3"] += 1
        else:
            weight_ranges[">0.3"] += 1

    print(f"  正效应连接: {positive_count} 条")
    print(f"  负效应连接: {negative_count} 条")

    for lag in sorted(lag_count.keys()):
        print(f"  时间滞后 {lag}: {lag_count[lag]} 条连接")

    # print("\n权重分布:")
    # for range_key, count in weight_ranges.items():
    #     print(f"  权重 {range_key}: {count} 条连接")

    # 特别统计自相关连接
    self_connection_count = len(self_edges)
    print(f"\n自相关连接: {self_connection_count} 条")


def plot_causal_heatmap(links_coeffs, N, output_dir):
    """
    绘制因果连接热图，显示节点间因果链接的数量
    :param links_coeffs: 因果关系字典
    :param N: 节点数量
    :param output_dir: 输出目录
    """
    # 创建连接计数矩阵
    connection_count = np.zeros((N, N))

    # 统计每对节点之间的连接数
    for i in links_coeffs:
        for link in links_coeffs[i]:
            (j, tau), coeff, func = link
            if j is not None and j < N:
                # 增加从节点j到节点i的连接计数
                connection_count[j, i] += 1

    # 绘制热图
    plt.figure(figsize=(12, 10))
    # 使用'YlOrRd'颜色映射，避免太黑的颜色
    im = plt.imshow(connection_count, cmap='YlOrRd', interpolation='nearest', aspect='equal')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Causal Links', rotation=270, labelpad=20)

    # 设置标签和标题
    plt.title('Causal Connection Heatmap')
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')

    # 添加统计信息
    total_connections = np.sum(connection_count)
    max_connections = np.max(connection_count)
    avg_connections = np.mean(connection_count[connection_count > 0]) if np.sum(connection_count > 0) > 0 else 0
    non_zero_connections = np.sum(connection_count > 0)

    print(f"\n热图统计信息:")
    print(f"  总连接数: {total_connections}")
    print(f"  最大连接数(单对节点): {max_connections}")
    print(f"  矩阵大小: {N} x {N}")

    # 保存图像
    output_file = os.path.join(output_dir, 'causal_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"因果连接热图已保存至: {output_file}")
    plt.close()

def print_node_statistics(links_coeffs, N):
    """
    打印每个节点的自关联个数和非自关联出边个数
    :param links_coeffs: 因果关系字典
    :param N: 节点数量
    """
    # 初始化统计数组
    self_connections = np.zeros(N, dtype=int)  # 自关联个数
    out_connections = np.zeros(N, dtype=int)  # 非自关联出边个数

    # 统计每个节点的连接情况
    for i in links_coeffs:
        for link in links_coeffs[i]:
            (j, tau), coeff, func = link
            if j is not None and j < N:
                if j == i:
                    # 自关联连接
                    self_connections[i] += 1
                else:
                    # 非自关联出边
                    out_connections[j] += 1

    # 打印统计信息
    print(f"\n节点统计信息:")
    print(f"{'节点ID':<8} {'自关联个数':<12} {'非自关联出边个数':<18}")
    print("-" * 40)

    for i in range(N):
        print(f"{i:<8} {self_connections[i]:<12} {out_connections[i]:<18}")

    # 打印汇总统计
    total_self = np.sum(self_connections)
    total_out = np.sum(out_connections)
    avg_self = np.mean(self_connections)
    avg_out = np.mean(out_connections)

    print("-" * 40)
    print(f"{'总计':<8} {total_self:<12} {total_out:<18}")
    print(f"{'平均':<8} {avg_self:<12.2f} {avg_out:<18.2f}")
    print(f"{'最大':<8} {np.max(self_connections):<12} {np.max(out_connections):<18}")
    print(f"{'最小':<8} {np.min(self_connections):<12} {np.min(out_connections):<18}")

    # 找出特殊的节点
    max_self_node = np.argmax(self_connections)
    max_out_node = np.argmax(out_connections)

    print(f"\n特殊节点:")
    print(f"  自关联最多的节点: {max_self_node} ({self_connections[max_self_node]} 个)")
    print(f"  出边最多的节点: {max_out_node} ({out_connections[max_out_node]} 个)")

# print_node_statistics(links_coeffs, N)
# 调用函数绘制带坐标信息的因果图
# plot_causal_graph_with_positions(links_coeffs, node_positions, output_dir=output_dir, show=True)
# 在文件末尾调用热图绘制函数
# plot_causal_heatmap(links_coeffs, N, output_dir)


select = 18

# 分析因果连接
analysis_result = analyze_causal_links(links_coeffs, node_positions, N)

# 打印分析结果
# print_analysis_results(analysis_result, N)

# 打印最重要的节点
# print_top_nodes(analysis_result, N=select)

# 可视化最重要的select个节点
# visualize_top_nodes(links_coeffs, node_positions, analysis_result, output_dir, select_node=select)

# 绘制与PageRank.py样式一致的散点图
# plot_causal_importance_scatter(analysis_result, node_positions, output_dir, select_node=select)

# plot_causal_scatter_weights(analysis_result, node_positions, output_dir, select_node=select)

print(f"\n出边权重最高的{select}个节点:")
for i, node_id in enumerate(analysis_result['top_out_nodes']):
    print(f"{i+1:2d}. 节点{node_id:3d} (权重总和: {analysis_result['out_weights'][node_id]:.4f})")

print(f"\n入边权重最高的{select}个节点:")
for i, node_id in enumerate(analysis_result['top_in_nodes']):
    print(f"{i+1:2d}. 节点{node_id:3d} (权重总和: {analysis_result['in_weights'][node_id]:.4f})")



# 导出节点索引排序结果，用于perform.py
def export_node_rankings(analysis_result, output_dir):
    """
    导出节点索引排序结果到文件，用于perform.py

    Parameters:
    analysis_result: analyze_causal_links返回的分析结果
    output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 导出出边权重排序的节点索引
    out_weights_sorted_idx = analysis_result['out_weights_sorted_idx']
    np.savetxt(os.path.join(output_dir, 'out_weights_node_ranking.txt'),
               out_weights_sorted_idx, fmt='%d')

    # 导入入边权重排序的节点索引
    in_weights_sorted_idx = analysis_result['in_weights_sorted_idx']
    np.savetxt(os.path.join(output_dir, 'in_weights_node_ranking.txt'),
               in_weights_sorted_idx, fmt='%d')

    print(f"节点索引排序结果已导出到: {output_dir}")


# 在文件末尾调用导出函数
export_node_rankings(analysis_result, output_dir)

print(f"\n所有因果分析结果已保存到: {output_dir}")
