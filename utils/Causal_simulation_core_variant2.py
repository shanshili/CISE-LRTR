"""

# 流程特点：模块化设计，外部依赖可视化
- 拓扑构建 → 参数生成 → 数据仿真 → 外部模块可视化
- 节点数：50，时间步：3000，核心枢纽：5个
- 调用外部pcmci_draw_MultiDiGraph2模块
- 专注于数据生成，可视化委托外部

"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp

# ==============================================================================
# 1. 路径设置与模块导入 (修改部分)
# ==============================================================================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设当前脚本在项目根目录的某个子文件夹中，例如 /scripts)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 构建 Module 目录的绝对路径
module_dir = os.path.join(project_root, 'Module')

# 将 Module 目录加入系统路径，以便导入其中的 pcmci_draw_MultiDiGraph2
if module_dir not in sys.path:
    sys.path.append(module_dir)

print(f"已添加模块搜索路径: {module_dir}")

try:
    import pcmci_draw_MultiDiGraph2 as viz

    print("✅ 成功导入可视化模块: pcmci_draw_MultiDiGraph2")
except ImportError as e:
    print(f"❌ 错误: 无法导入 pcmci_draw_MultiDiGraph2。")
    print(f"   请检查该文件是否存在于: {module_dir}")
    print(f"   详细错误: {e}")


# ================= 配置与文件管理 =================
N = 50  # 节点总数
T = 3000  # 时间步长
num_hubs = 5  # 核心枢纽节点数量
transient_frac = 0.2  # 预热比例
TAU_MAX = 5  # 可视化时的最大滞后

# 设置随机种子
seed = 42
np.random.seed(seed)

# 动态生成文件夹名称
folder_name = f"simulation_hubs_core/N{N}_T{T}_Hubs{num_hubs}_Viz_Integrated"

# 设置输出路径 (dataset 也在项目根目录下)
output_dir = os.path.join(project_root, 'dataset', folder_name)
os.makedirs(output_dir, exist_ok=True)

print(f"输出目录已创建: {output_dir}")


# ================= 0. 核心函数 (数据生成用) =================
def saturation_func(x):
    """防止数值爆炸的饱和函数"""
    return np.tanh(x)


# ================= 1. 生成地理位置与识别 Hubs =================
print("1. 生成地理分布...")
node_positions = np.random.rand(N, 2) * 100
pos_dict = {i: node_positions[i] for i in range(N)}

# 保存坐标 (绘图模块需要用到这个格式)
pos_df = pd.DataFrame(node_positions, columns=['x', 'y'])
pos_df['node_id'] = range(N)
pos_df.to_csv(os.path.join(output_dir, 'node_positions.csv'), index=False)

# 计算距离矩阵
dist_matrix = squareform(pdist(node_positions))

# 随机选定 Hubs
hub_nodes = np.random.choice(N, num_hubs, replace=False).tolist()
print(f"Core Hubs: {hub_nodes}")

# ================= 2. 基于空间距离构建拓扑 =================
print("2. 构建空间受限的因果拓扑...")
G_structure = nx.DiGraph()
G_structure.add_nodes_from(range(N))

core_radius = 80.0
local_radius = 20.0
prob_connect_local = 0.5
prob_connect_core = 0.8

for i in range(N):
    for j in range(N):
        if i == j: continue
        dist = dist_matrix[i, j]
        is_i_hub = i in hub_nodes
        is_j_hub = j in hub_nodes

        # Core Links (Hub -> Hub, 长程)
        if is_i_hub and is_j_hub and dist < core_radius:
            if np.random.rand() < prob_connect_core:
                G_structure.add_edge(i, j, type='core')
        # Hub -> Leaf (辐射)
        elif is_i_hub and not is_j_hub and dist < local_radius * 1.5:
            if np.random.rand() < prob_connect_local:
                G_structure.add_edge(i, j, type='variant')
        # Leaf -> Leaf (局部)
        elif not is_i_hub and not is_j_hub and dist < local_radius:
            if np.random.rand() < prob_connect_local:
                G_structure.add_edge(i, j, type='variant')


# ================= 3. 定义因果连接参数生成器 =================
def generate_causal_params(env_type):
    """
    生成 Tigramite 所需的 links 字典 (Ground Truth)
    """
    links = {i: [] for i in range(N)}

    # 1. 自回归 (Inertia)
    for i in range(N):
        coeff = 0.8 if i in hub_nodes else 0.5
        links[i].append(((i, -1), coeff, saturation_func))

    # 2. 交互连接 (Interaction)
    for u, v, data in G_structure.edges(data=True):
        edge_type = data['type']

        # 默认参数
        weight = 0.0
        lag = 1
        active = False

        if edge_type == 'core':
            # Core 结构稳定
            active = True
            weight = 0.6
            lag = 1
        else:
            # 环境调制逻辑
            target_x = pos_dict[v][0]
            is_west = target_x < 50

            if env_type == 'Env_A':
                if is_west:  # 西边活跃
                    active = True;
                    weight = 0.3;
                    lag = 1
                else:  # 东边拥堵(弱连接，高延迟)
                    active = True;
                    weight = 0.02;
                    lag = 3
            elif env_type == 'Env_B':
                if not is_west:  # 东边活跃
                    active = True;
                    weight = 0.3;
                    lag = 1
                else:  # 西边抑制
                    active = True;
                    weight = 0.02;
                    lag = 3

        if active:
            # 随机正负号 (模拟促进或抑制)
            # 保持随机性一致以便调试：基于节点ID做随机种子
            local_seed = (u * N + v + (1 if env_type == 'Env_A' else 2))
            rng = np.random.RandomState(local_seed)
            final_weight = weight if rng.rand() > 0.3 else -weight

            links[v].append(((u, -lag), final_weight, saturation_func))

    return links


# ================= 4. [适配器] Links -> Matrix =================
def convert_links_to_pcmci_format(links_dict, N, max_lag):
    """
    将生成模型的 links 字典转换为可视化模块需要的 graph 和 val_matrix
    """
    # graph: 字符串矩阵，'-->' 表示存在连接
    graph = np.full((N, N, max_lag + 1), "", dtype=object)
    # val_matrix: 浮点数矩阵，存储连接强度(带正负号)
    val_matrix = np.zeros((N, N, max_lag + 1))

    for target_node, parents in links_dict.items():
        for parent_tuple in parents:
            # parent_tuple 结构: ((source_node, -lag), coeff, func)
            (source_node, lag_val), coeff, _ = parent_tuple

            # Tigramite 中 lag 是负数 (e.g., -1)，我们需要转为正整数索引
            abs_lag = abs(lag_val)

            if abs_lag <= max_lag:
                graph[target_node, source_node, abs_lag] = "-->"
                val_matrix[target_node, source_node, abs_lag] = coeff

    return graph, val_matrix


# ================= 5. 数据生成与可视化流程 =================

def process_environment(env_name, seed_val):
    print(f"\n🚀 处理环境: {env_name} ...")

    # 1. 生成参数
    links = generate_causal_params(env_name)

    # 2. 生成时间序列数据
    total_steps = int((1 + transient_frac) * T)
    noises = np.zeros((total_steps, N))
    for i in range(N):
        scale = 0.1 if i in hub_nodes else 0.3
        noises[:, i] = np.random.normal(0, scale, total_steps)

    data, _ = toys.structural_causal_process(
        links=links, T=T, noises=noises, seed=seed_val, transient_fraction=transient_frac
    )

    # 3. 保存时间序列 (关键步骤)
    df = pd.DataFrame(data, columns=[f"Node_{i}" for i in range(N)])
    csv_path = os.path.join(output_dir, f'{env_name}_timeseries.csv')
    df.to_csv(csv_path, index=False)
    print(f"  💾 时间序列已保存: {csv_path}")

    # 4. 转换 Ground Truth 为可视化格式
    print("  🔄 转换 Ground Truth 结构为绘图格式...")
    gt_graph, gt_val_matrix = convert_links_to_pcmci_format(links, N, TAU_MAX)

    # 5. 调用外部模块进行可视化
    print("  🎨 调用 pcmci_draw_MultiDiGraph2 进行绘图...")

    # 步骤 5.1: 构建 MultiDiGraph 对象
    # 注意：这里需要传入 numpy 格式的 positions
    G, edge_counts, self_counts = viz.build_multi_di_graph(
        gt_graph, gt_val_matrix, N, TAU_MAX, node_positions
    )

    # 步骤 5.2: 绘制网络图 (保存为 png)
    plot_title = f"Ground Truth: {env_name} (N={N})\nHubs={num_hubs}, Red=Pos, Blue=Neg, Width=Strength"

    # 这里我们借用 alpha_str 参数来作为环境标识 (GT_Env_A)
    viz.plot_causal_graph(
        G=G,
        positions=node_positions,
        edge_counts=edge_counts,
        node_self_counts=self_counts,
        output_dir=output_dir,
        tau_str=f"tau{TAU_MAX}",
        alpha_str=f"GT_{env_name}",
        title=plot_title
    )

    # 步骤 5.3: 绘制矩阵热力图
    viz.plot_causal_matrix(
        edge_counts=edge_counts,
        N=N,
        output_dir=output_dir,
        tau_str=f"tau{TAU_MAX}",
        alpha_str=f"GT_{env_name}"
    )


# --- 执行主流程 ---
if __name__ == "__main__":
    process_environment('Env_A', seed_val=10)
    process_environment('Env_B', seed_val=20)

    print("\n=== 全部完成 ===")
    print(f"📁 结果保存在: {output_dir}")
    print("生成内容包括：")
    print("1. env_A_timeseries.csv, env_B_timeseries.csv (原始数据)")
    print("2. network_causal_graph_*.png (高清因果图，含正负颜色和滞后标签)")
    print("3. causal_matrix_*.png (因果强度矩阵)")
