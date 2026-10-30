"""
节点*时间戳
7个特征单独输出CSV文件

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
数据读取
"""
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)

# 加载数据
data_path = os.path.join(dataset_dir, 'sensor_data_3d.npy')
data = np.load(data_path)

# 2. 调整维度：将特征维度放到第一维
data_transposed = np.transpose(data, (2, 1, 0))  # 新形状：(特征, 时间步, 节点)

# 3. 保存每个特征为单独的 CSV 文件
for feature_idx in range(data_transposed.shape[0]):
    feature_data = data_transposed[feature_idx]  # 获取当前特征的二维数据 (时间步×节点)
    df = pd.DataFrame(feature_data)
    df.to_csv(f'feature_{feature_idx}.csv', index=False, header=False)  # 不保存行列标签