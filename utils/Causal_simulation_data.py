"""
版本1
1. 简单结构
每个节点只有1个自回归项（滞后1）
每个节点只与邻近节点有1个因果连接
连接系数相对简单固定
2. 简单噪声
3. 可视化和输出
基本的时间序列图（前5个节点）
简单的节点分布图
基础的CSV输出
"""
import sys
import os
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（GNN-Causal-Inference）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 sys.path
sys.path.append(project_root)
import numpy as np
import pandas as pd
from tigramite.toymodels import structural_causal_processes as toys
import matplotlib.pyplot as plt


"""
数据读取
"""
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(dataset_dir, 'Causal_simulation_data')
# 如果文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子确保可重复性
np.random.seed(42)

# 参数设置
N = 100  # 节点数量
T = 1000  # 时间步长
tau_max = 5  # 最大时间滞后
transient_fraction = 0.1  # 默认的瞬态比例

""" 
生成节点坐标数据 (随机分布在100x100的平面内)
"""
node_positions = np.random.rand(N, 2) * 100  # 生成0-100范围内的坐标
positions_df = pd.DataFrame({
    'node_id': np.arange(N),
    'x': node_positions[:, 0],
    'y': node_positions[:, 1]
})

"""
生成温度时间序列数据 (带空间相关性)
"""
# 1. 创建基础因果结构
links_coeffs = {}
for i in range(N):
    # 每个节点自相关
    links_coeffs[i] = [((i, -1), 0.6, lambda x: x)]  # 自回归系数0.6

    # 添加空间邻近节点的相互影响
    distances = np.sqrt((node_positions[:, 0] - node_positions[i, 0]) ** 2 +
                        (node_positions[:, 1] - node_positions[i, 1]) ** 2)
    nearby_nodes = np.where(distances < 15)[0]  # 15米范围内的节点

    for j in nearby_nodes:
        if j != i:
            # 随机添加影响 (0.1-0.3的系数)
            coeff = 0.1 + 0.2 * np.random.rand()
            lag = np.random.randint(1, tau_max + 1)
            links_coeffs[i].append(((j, -lag), coeff, lambda x: x))

# 2. 生成噪声 (修正维度)
total_time_steps = int((transient_fraction + 1) * T)
print(f"需要的时间步数: {total_time_steps} (T={T}, transient_fraction={transient_fraction})")

noises = np.zeros((total_time_steps, N))
for j in range(N):
    noises[:, j] = (1. + 0.1 * j) * np.random.randn(total_time_steps)

print(f"噪声矩阵形状: {noises.shape} (应为({total_time_steps}, {N}))")

# 3. 生成时间序列数据
try:
    data, _ = toys.structural_causal_process(
        links_coeffs,
        T=T,
        noises=noises,
        seed=42,
        transient_fraction = transient_fraction  # 明确传递这个参数
    )
    print(f"生成的数据形状: {data.shape} (应为({T}, {N}))")
except Exception as e:
    print(f"生成数据时出错: {e}")
    raise


# 转置为(时间×节点)格式
temperature_data = pd.DataFrame(data.T)

"""
保存数据到CSV
"""
# 保存温度数据 (1000×100)
temp_data_path = os.path.join(output_dir, 'simulated_temperature_data.csv')
temperature_data.to_csv(temp_data_path, index=False, header=False)

# 保存节点坐标数据
pos_data_path = os.path.join(output_dir, 'simulated_node_positions.csv')
positions_df.to_csv(pos_data_path, index=False)

print("数据生成完成:")
print(f"- 温度数据已保存到: {temp_data_path}")
print(f"- 节点坐标已保存到: {pos_data_path}")


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

# 2. 绘制部分节点的时间序列波形图
plt.figure(figsize=(12, 6))
for i in range(min(5, N)):  # 只绘制前5个节点的时间序列
    plt.plot(temperature_data[i], label=f'Node {i}', alpha=0.7)
plt.title('Temperature Time Series (First 5 Nodes)')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'temperature_time_series.png'))
plt.close()

print(f"- 节点分布图已保存到: {os.path.join(output_dir, 'node_positions.png')}")
print(f"- 时间序列图已保存到: {os.path.join(output_dir, 'temperature_time_series.png')}")