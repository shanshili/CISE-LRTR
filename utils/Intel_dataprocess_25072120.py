"""
Intel 实验室传感器数据处理工具
===============================

本脚本提供Intel实验室传感器数据的完整预处理流程，将原始文本数据转换为结构化三维数组，包含以下核心功能：

1. 数据预处理流程：
   - 自动读取传感器原始数据（温度/湿度/光照/电压）和节点位置数据
   - 智能处理多种时间格式（支持混合格式自动识别）
   - 完善的数据校验与异常处理机制
   - 自动保存处理失败的样本供调试

2. 数据结构化输出：
   - 生成三维NumPy数组（节点×时间×特征）
   - 保存标准化时间戳数据
   - 保留节点坐标信息
   - 输出各节点时间步数统计报表

3. 关键特性：
   - 采用高效字典映射替代循环查找，提升处理速度
   - 实时进度显示（tqdm进度条）
   - 详细的运行时统计信息输出
   - 自动识别数据异常节点

4. 输出文件说明：
   - sensor_data_3d.npy：三维传感器数据数组
   - timestamps.npy：标准化时间戳数据
   - mote_ids.npy：节点ID列表

使用注意：执行前请确保原始数据文件(data.txt/location.txt)位于../dataset/Intel Lab Data.txt目录下
"""



import os
import pandas as pd
from tqdm import tqdm
import numpy as np


"""
数据读取
"""
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)


def process_intel_lab_data():
    """处理Intel Lab数据集，生成包含ID信息的三维数组"""
    # 文件路径
    data_path = os.path.join(dataset_dir, 'Intel Lab Data.txt', 'data.txt')
    location_path = os.path.join(dataset_dir, 'Intel Lab Data.txt', 'location.txt')

    # 1. 读取并处理坐标数据
    print("正在处理坐标数据...")
    loc_df = pd.read_csv(location_path, sep='\s+', header=None,
                         names=['moteid', 'x', 'y'], dtype={'moteid': int})

    # 2. 增强错误处理的传感器数据读取
    print("正在处理传感器数据...")
    col_names = ['date', 'time', 'epoch', 'moteid',
                 'temperature', 'humidity', 'light', 'voltage']

    try:
        # 读取数据时不指定dtype，先保持原始格式
        sensor_df = pd.read_csv(data_path, sep='\s+', header=None, names=col_names)

        # 打印数据前5行供检查
        print("\n原始数据前5行样本:")
        print(sensor_df.head())

        # 转换moteid为int
        sensor_df['moteid'] = pd.to_numeric(sensor_df['moteid'], errors='coerce')
        if sensor_df['moteid'].isna().any():
            bad_rows = sensor_df[sensor_df['moteid'].isna()]
            print(f"\n错误: moteid列发现{len(bad_rows)}个无效值，首行示例:")
            print(bad_rows.iloc[0])
            sensor_df['moteid'] = sensor_df['moteid'].fillna(-1).astype(int)

        # 转换传感器数值
        for col in ['temperature', 'humidity', 'light', 'voltage']:
            sensor_df[col] = pd.to_numeric(sensor_df[col], errors='coerce')

    except Exception as e:
        print(f"\n数据读取失败: {str(e)}")
        print("请检查数据文件格式是否正确")
        # 保存出错时的数据快照
        if 'sensor_df' in locals():
            error_path = os.path.join(dataset_dir, 'error_sample.csv')
            sensor_df.head(100).to_csv(error_path, index=False)
            print(f"已保存前100行数据到: {error_path}")
        raise

    # 修改时间合并方式 - 关键修改点
    print("\n正在处理时间戳...")
    try:
        # 尝试两种时间格式
        combined_time = sensor_df['date'] + ' ' + sensor_df['time']
        sensor_df['timestamp'] = pd.to_datetime(combined_time, format='mixed', errors='coerce')

        # 检查无效时间
        if sensor_df['timestamp'].isna().any():
            bad_time = sensor_df[sensor_df['timestamp'].isna()]
            print(f"警告: 发现{len(bad_time)}个无效时间格式，首行示例:")
            print(bad_time.iloc[0][['date', 'time']])
            print("尝试替代解析方式...")

            # 尝试ISO8601格式
            sensor_df['timestamp'] = pd.to_datetime(combined_time, format='ISO8601', errors='coerce')

            if sensor_df['timestamp'].isna().any():
                print("仍有无效时间，使用最后手段:")
                # 强制转换，忽略错误
                sensor_df['timestamp'] = pd.to_datetime(combined_time, errors='coerce')

        print("时间处理完成，无效时间数量:", sensor_df['timestamp'].isna().sum())

    except Exception as e:
        print(f"时间转换失败: {str(e)}")
        raise

    sensor_df.drop(['date', 'time'], axis=1, inplace=True)

    # 剩余处理代码保持不变...
    # [原有代码继续...]

    # 3. 数据对齐与排序
    print("合并数据并创建3D数组...")
    unique_motes = sorted(sensor_df['moteid'].unique())
    unique_times = sorted(sensor_df['timestamp'].unique())

    # 初始化3D数组 (节点数×时间步数×特征数)
    # 特征顺序: moteid, temperature, humidity, light, voltage, x, y
    num_features = 7  # 增加ID作为第一个特征
    data_3d = np.full((len(unique_motes), len(unique_times), num_features), np.nan)

    # 创建映射字典
    loc_dict = dict(zip(loc_df['moteid'], zip(loc_df['x'], loc_df['y'])))
    mote_idx = {mote: i for i, mote in enumerate(unique_motes)}

    # 修改填充3D数组的部分
    print("填充3D数组...")
    # 创建时间到索引的映射字典
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    for _, row in tqdm(sensor_df.iterrows(), total=len(sensor_df)):
        i = mote_idx[row['moteid']]
        try:
            j = time_to_idx[row['timestamp']]  # 使用字典查找替代np.where
            x, y = loc_dict.get(row['moteid'], (np.nan, np.nan))
            data_3d[i, j, :] = [
                row['moteid'],
                row['temperature'],
                row['humidity'],
                row['light'],
                row['voltage'],
                x,
                y
            ]
        except KeyError:
            print(f"警告: 未找到时间戳 {row['timestamp']} 在时间列表中")
            continue

    # 保存结果
    print("保存处理结果...")
    np.save(os.path.join(dataset_dir, 'sensor_data_3d.npy'), data_3d)
    print(f"数据处理完成！最终数据形状: {data_3d.shape}")
    print(f"特征顺序: [moteid, temperature, humidity, light, voltage, x, y]")

    # 保存时间戳数据
    timestamps_path = os.path.join(dataset_dir, 'timestamps.npy')
    np.save(timestamps_path, unique_times)
    print(f"时间戳数据已保存到: {timestamps_path}")

    # 保存节点ID数据（可选）
    moteids_path = os.path.join(dataset_dir, 'mote_ids.npy')
    np.save(moteids_path, unique_motes)
    print(f"节点ID数据已保存到: {moteids_path}")

    print("\n=== 各节点时间步数详细统计 ===")
    # 计算每个节点的有效时间步数（非NaN值）
    node_time_steps = np.sum(~np.isnan(data_3d[:, :, 0]), axis=1)  # 使用moteid列检查
    # 创建统计表格
    node_stats = pd.DataFrame({
        '节点ID': unique_motes,
        '时间步数': node_time_steps,
    })

    # 打印统计信息
    print("\n各节点时间步数统计表:")
    print(node_stats)
    # 打印统计摘要
    print("\n时间步数统计摘要:")
    print(node_stats['时间步数'].describe())
    # 识别异常节点（时间步数明显少于其他节点）
    median_steps = node_stats['时间步数'].median()
    q1 = node_stats['时间步数'].quantile(0.25)
    outliers = node_stats[node_stats['时间步数'] < q1 - 1.5 * (median_steps - q1)]
    if not outliers.empty:
        print("\n警告: 以下节点时间步数异常偏低:")
        print(outliers)
    else:
        print("\n未发现明显异常节点")

if __name__ == '__main__':
    process_intel_lab_data()


