"""
2022 TJ dataprocess
构建【节点数, 时间步数, 特征维度】三位npy
"""

import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
数据读取
"""
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/TJ')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)

# 读取 TJ_position.csv
position_file = os.path.join(dataset_dir, 'TJ_position.csv')
positions_df = pd.read_csv(position_file)

# TJ 文件夹路径
tj_dir = os.path.join(dataset_dir, 'TJ')

# 选择要使用的测量变量
measurement_columns = ['RHU'] #TEM	PRS	WIN	RHU
# 提取测量变量名称用于文件夹和文件命名
measurement_name = measurement_columns[0]  # 提取实际的测量变量名称
# 创建一个字典来存储每个节点的时序数据
node_timeseries = {}
file_list = [f for f in os.listdir(tj_dir) if f.startswith('TJ_') and f.endswith('.csv')]

for filename in tqdm(file_list, desc="Reading Node Data", unit="file"):
    file_path = os.path.join(tj_dir, filename)
    node_data = pd.read_csv(file_path)
    node_id = filename[3:-4]  # 提取节点编号
    node_data['node_id'] = node_id
    # 保存为 numpy array：shape = (时间步数, 特征维度)
    node_timeseries[node_id] = node_data[measurement_columns].values

# 将数据合并为三维张量 shape = (节点数, 时间步数, 特征维度)
node_ids = sorted(node_timeseries.keys())
num_nodes = len(node_ids)
time_steps = node_timeseries[node_ids[0]].shape[0]
num_features = len(measurement_columns)

# 检查所有节点的时间步数是否一致
assert all(node_timeseries[nid].shape[0] == time_steps for nid in node_ids), "所有节点的时间步数必须一致"

# 创建三维张量
tensor_data = np.stack([node_timeseries[nid] for nid in node_ids], axis=0)

# tensor_data.shape = (节点数, 时间步数, 特征维度)
print(f"三维张量 shape: {tensor_data.shape}")

"""
数据抽样
"""
k = 5  # 每隔 K 个时间步采样一次
sampled_tensor = tensor_data[:, ::k, :]  # shape = (num_nodes, sampled_time_steps, num_features)
# sampled_tensor = tensor_data
print("采样前张量尺寸:", tensor_data.shape)  # 添加打印
print("采样后张量尺寸:", sampled_tensor.shape)  # 添加打印

# 构建输出文件路径
output_folder = f'dataprocess/k{k}_{measurement_name}'
output_dir = os.path.join(dataset_dir, output_folder)
os.makedirs(output_dir, exist_ok=True)

# 构建输出文件路径，文件名也包含k值和测量变量信息
output_filename = f'sampled_tensor_data_k{k}_{measurement_name}.npy'
output_path = os.path.join(output_dir, output_filename)
np.save(output_path, sampled_tensor)
print(f"数据已保存至：{output_path}")



"""
数据标准化
"""
# 创建标准化器字典，用于保存每个节点的标准化参数（可选）
scalers = {}
# 创建一个与 tensor_data 形状相同的空数组用于保存标准化后的数据
standardized_tensor = np.zeros_like(sampled_tensor)
# 对每个节点的时间序列分别标准化
for i, node_id in enumerate(node_ids):
    scaler = StandardScaler()
    standardized_tensor[i, :, :] = scaler.fit_transform(sampled_tensor[i, :, :])
    scalers[node_id] = scaler  # 保存标准化器（用于后续反标准化）

# 构建标准化数据的输出文件路径
standardized_filename = f'sampled_and_standardized_tensor_data_k{k}_{measurement_name}.npy'
output_path = os.path.join(output_dir, standardized_filename)
np.save(output_path, standardized_tensor)
print(f"数据已保存至：{output_path}")
print("标准化后张量尺寸:", standardized_tensor.shape)

# 保存标准化器
scaler_filename = f'scalers_k{k}_{measurement_name}.pkl'
joblib.dump(scalers, os.path.join(output_dir, scaler_filename))

"""
三维张量 shape: (301, 8761, 1)
采样前张量尺寸: (301, 8761, 1)
采样后张量尺寸: (301, 1753, 1)
"""