"""
2022 TJ dataprocess
所有节点合并为一个CSV
"""
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from sklearn.preprocessing import MinMaxScaler

"""
数据读取
"""
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)

# 读取 TJ_position.csv
position_file = os.path.join(dataset_dir, 'TJ_position.csv')
positions_df = pd.read_csv(position_file)

# TJ 文件夹路径
tj_dir = os.path.join(dataset_dir, 'TJ')

# 读取 TJ 文件夹下的所有文件，并显示进度条
all_data = []
file_list = [f for f in os.listdir(tj_dir) if f.startswith('TJ_') and f.endswith('.csv')]

for filename in tqdm(file_list, desc="Reading Node Data", unit="file"):
    file_path = os.path.join(tj_dir, filename)
    node_data = pd.read_csv(file_path)
    node_id = filename[3:-4]  # 提取节点编号
    node_data['node_id'] = node_id
    all_data.append(node_data)

# 合并为一个 DataFrame
combined_data = pd.concat(all_data, ignore_index=True)


"""
数据抽样
"""
# 选择特定测量数据（如 TEM）
selected_measurement = 'TEM'
measurement_data = combined_data[['time', 'node_id', selected_measurement]]
print(measurement_data)

# 时间裁剪与抽样
# 时间转换（使用 assign 避免 FutureWarning）
measurement_data = measurement_data.assign(
    time=pd.to_datetime(measurement_data['time'], format='%Y%m%d%H')
)
# 按照时间排序（确保时间顺序正确）
# measurement_data.sort_values(by='time', inplace=True)

# 系统抽样：每 K=5 个数据点抽取一个样本
k = 5
sampled_data = measurement_data.iloc[::k, :]
print(sampled_data)

# 查看抽样结果
print(f"原始数据数量: {len(combined_data)}")
print(f"抽样后数据数量: {len(sampled_data)}")

# 可视化抽样时间点
plt.figure(figsize=(12, 4))
plt.scatter(sampled_data['time'], [1]*len(sampled_data), c='blue', label='Sampled Time Points')
plt.xlabel('Time')
plt.title('Systematic Sampling (k=5)')
plt.yticks([])
plt.legend()
plt.grid(True)
plt.show()

# 构建输出文件路径
output_path = os.path.join(dataset_dir, 'sampled_data_tem.csv')
sampled_data.to_csv(output_path, index=False)
print(f"数据已保存至：{output_path}")


"""
数据标准化
"""
scaler = StandardScaler()
# 选择要标准化的列（如 TEM, PRS, WIN, RHU）
measurement_columns = ['TEM']
# 标准化并保存为新的列
sampled_data.loc[:, measurement_columns] = scaler.fit_transform(sampled_data[measurement_columns])
# 查看标准化后的数据
print(sampled_data.head())

# 构建输出文件路径
output_path = os.path.join(dataset_dir, 'sampled_and_standardized_data_tem.csv')
sampled_data.to_csv(output_path, index=False)
print(f"数据已保存至：{output_path}")

# 转换为 Tensor
tensor_data = torch.tensor(sampled_data[measurement_columns].values, dtype=torch.float32)

"""
保存标准化器
"""
# 保存标准化器，用于反标准化
# 保存 scaler
joblib.dump(scaler, 'scaler.pkl')

# 加载 scaler
# scaler = joblib.load('scaler.pkl')
