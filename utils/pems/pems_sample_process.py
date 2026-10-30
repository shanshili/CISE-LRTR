"""
传感器节点数据提取脚本

功能:
1. 读取采样传感器节点的坐标信息 (location2_sampled200.csv)
2. 读取完整的交通数据 (hour_extracted_data.csv)
3. 根据传感器ID匹配，提取采样节点对应的交通数据
4. 保存提取的数据为新的CSV文件

输入:
- location2_sampled200.csv: 采样传感器节点的坐标信息
- hour_extracted_data.csv: 完整的交通数据

输出:
- sampled_sensor_data.csv: 采样节点对应的交通数据
"""

import pandas as pd
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 读取采样传感器节点的坐标信息
sampled_location_file = os.path.join(script_dir, "location2_sampled100.csv")
print("正在读取采样传感器节点坐标信息...")
try:
    sampled_sensors = pd.read_csv(sampled_location_file)
    print(f"成功读取采样传感器节点信息，共 {len(sampled_sensors)} 个节点")
    print("采样节点数据预览:")
    print(sampled_sensors.head())
except Exception as e:
    print(f"读取采样传感器节点文件时出错: {e}")
    exit(1)

# 2. 读取完整的交通数据
traffic_data_file = os.path.join(script_dir, "hour/hour_extracted_data.csv")
print("\n正在读取交通数据...")
try:
    traffic_data = pd.read_csv(traffic_data_file)
    print(f"成功读取交通数据，共 {len(traffic_data)} 条记录")
    print("交通数据预览:")
    print(traffic_data.head())
except FileNotFoundError:
    print(f"未找到交通数据文件: {traffic_data_file}")
    print("请确保 hour_extracted_data.csv 文件存在于脚本目录中")
    exit(1)
except Exception as e:
    print(f"读取交通数据文件时出错: {e}")
    exit(1)

# 3. 提取采样传感器节点对应的交通数据
print("\n正在提取采样节点对应的交通数据...")

# 获取采样传感器的ID列表
sampled_sensor_ids = sampled_sensors['ID'].tolist()
print(f"需要提取数据的传感器ID数量: {len(sampled_sensor_ids)}")

# 在交通数据中筛选出采样传感器节点的数据
sampled_traffic_data = traffic_data[traffic_data['Station'].isin(sampled_sensor_ids)]

print(f"提取到 {len(sampled_traffic_data)} 条与采样节点相关的交通数据")

# 4. 显示统计信息
print("\n=== 提取结果统计 ===")
print(f"采样传感器节点总数: {len(sampled_sensor_ids)}")
print(f"在交通数据中找到记录的传感器节点数: {sampled_traffic_data['Station'].nunique()}")

# 计算未找到记录的传感器节点
found_sensor_ids = sampled_traffic_data['Station'].unique().tolist()
missing_sensor_ids = [sid for sid in sampled_sensor_ids if sid not in found_sensor_ids]
print(f"未在交通数据中找到记录的传感器节点数: {len(missing_sensor_ids)}")

if len(missing_sensor_ids) > 0:
    print("未找到记录的传感器ID (前10个):")
    print(missing_sensor_ids[:10])

# 5. 保存提取的数据
if len(sampled_traffic_data) > 0:
    output_file = os.path.join(script_dir, "hour/hour_sampled100_sensor_data.csv")

    # 保存数据
    sampled_traffic_data.to_csv(output_file, index=False)
    print(f"\n数据提取完成，结果已保存到: {output_file}")

    # 显示数据示例
    print("\n提取数据示例 (前5行):")
    print(sampled_traffic_data.head())

    # 显示数据基本信息
    print(f"\n数据基本信息:")
    print(f"  总行数: {len(sampled_traffic_data)}")
    print(f"  列数: {len(sampled_traffic_data.columns)}")
    print(f"  列名: {list(sampled_traffic_data.columns)}")

    # 按传感器ID统计记录数
    station_counts = sampled_traffic_data['Station'].value_counts()
    print(f"\n传感器记录数统计:")
    print(f"  最多记录数: {station_counts.max()} (传感器 {station_counts.idxmax()})")
    print(f"  最少记录数: {station_counts.min()} (传感器 {station_counts.idxmin()})")
    print(f"  平均记录数: {station_counts.mean():.2f}")
else:
    print("\n警告: 未提取到任何数据，请检查:")
    print("1. 传感器ID是否匹配")
    print("2. 交通数据文件是否正确")
    print("3. 采样节点文件是否正确")

# 6. 可选：保存未找到记录的传感器ID
if len(missing_sensor_ids) > 0:
    missing_ids_file = os.path.join(script_dir, "missing_sensor_ids.txt")
    with open(missing_ids_file, 'w') as f:
        f.write("在交通数据中未找到记录的传感器ID:\n")
        for sid in missing_sensor_ids:
            f.write(f"{sid}\n")
    print(f"\n未找到记录的传感器ID已保存到: {missing_ids_file}")
