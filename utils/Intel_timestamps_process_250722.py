"""
Intel 实验室传感器数据半小时粒度处理工具
=======================================

以下是该数据处理脚本的核心功能概述：

1. **数据加载与预处理**
   - 读取3D传感器数据（[sensor_data_3d.npy]）和时间戳（[timestamps.npy]）
   - 自动转换时间戳为Pandas时间格式
   - 提供数据完整性检查与错误处理机制

2. **时间粒度处理**
   - 支持半小时粒度（[process_half_hour_data]）和10分钟粒度（[process_10min_data]）数据重采样
   - 自动处理时间边界对齐问题
   - 保留原始特征顺序：[moteid, temperature, humidity, light, voltage, x, y]

3. **数据质量评估**
   - 连贯性分析（[evaluate_data_coherence]）：基于节点覆盖率和缺失率筛选有效时间段
   - 连续零值检测（[evaluate_by_zero_consecutive]）：识别异常数据节点
   - 生成数据质量报告（[save_quality_report]）

4. **节点筛选与处理**
   - 缺失值填补（[filter_and_impute]）：线性插值处理NaN值
   - 异常节点剔除（[filter_abnormal_nodes]）
   - 位置数据匹配（节点ID与坐标关联）

5. **可视化输出**
   - 温度热力图（[plot_temperature_heatmap]）
   - 节点位置分布图（[plot_node_locations]），支持坐标轴翻转和整数ID标注
   - 自动保存图像和筛选后的节点坐标数据（CSV格式）

6. **核心输出**
   - 时间聚合后的数据文件（*.npy）
   - 筛选后的节点位置数据（filtered_node_locations.csv）
   - 可视化图表（PNG格式）

特点：全流程自动化处理，支持中文字符显示，包含详细日志输出和异常处理机制。
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
except:
    pass  # 如果绘图库未使用，忽略字体设置


# 设置数据路径
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, '..', 'dataset')

# 检查输入文件是否存在
sensor_data_path = os.path.join(dataset_dir, 'sensor_data_3d.npy')
timestamp_path = os.path.join(dataset_dir, 'timestamps.npy')

# 加载原始数据
try:
    data_3d = np.load(sensor_data_path)
    timestamps = np.load(timestamp_path, allow_pickle=True)
except Exception as e:
    raise RuntimeError(f"加载数据文件时发生错误: {str(e)}")

print(f"原始数据形状: {data_3d.shape}")
print(f"时间戳数量: {len(timestamps)}")

# 转换为DataFrame处理时间重采样
try:
    timestamps = pd.to_datetime(timestamps)
except Exception as e:
    raise RuntimeError(f"时间戳转换失败: {str(e)}")


def evaluate_data_coherence(data_3d, timestamps, node_threshold=0.7, time_threshold=0.5):
    """
    评估每个传感器节点的半小时数据连贯性，返回最佳时间段

    参数:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        timestamps: 对应的时间戳
        node_threshold: 节点筛选阈值(缺失率>此值则剔除)
        time_threshold: 时间连贯性阈值(节点覆盖率>此值则保留)
    """

    # ========== 新增数据诊断代码 ==========
    print("\n=== 数据诊断信息 ===")
    print(f"数据形状 (节点数×时间点数×特征数): {data_3d.shape}")

    # 转换为半小时粒度时间标签
    half_hour_labels = pd.to_datetime(timestamps).floor('30min')
    unique_intervals = pd.Series(half_hour_labels).unique()

    # 用温度列评估节点质量
    node_missing = np.mean(np.isnan(data_3d[:, :, 1]), axis=1)
    valid_nodes = node_missing < node_threshold
    filtered_data = data_3d[valid_nodes, :, :]
    print(f"\n节点筛选结果: 保留{np.sum(valid_nodes)}/{len(valid_nodes)}个节点")

    # 2. 评估每个半小时区间的节点覆盖率
    interval_quality = []
    for interval in unique_intervals:
        # 获取当前半小时的所有数据点
        mask = (half_hour_labels == interval)
        interval_data = filtered_data[:, mask, 0]  # 使用moteid列评估

        # 计算有多少节点在此半小时有数据
        node_has_data = ~np.isnan(interval_data).all(axis=1)
        coverage = np.mean(node_has_data)
        interval_quality.append((interval, coverage))

    # 3. 寻找最佳连续时间段 (节点覆盖率>threshold的最长连续段)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0

    for i in range(len(interval_quality)):
        _, coverage = interval_quality[i]
        if coverage > time_threshold:
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_start = i + 1
            current_length = 0

    # 获取最佳时间段
    best_intervals = [interval_quality[i][0]
                      for i in range(best_start, best_start + best_length)]
    best_mask = np.isin(half_hour_labels, best_intervals)

    # 输出统计信息
    print("\n=== 半小时数据分析结果 ===")
    print(f"总时间段数: {len(unique_intervals)}")
    print(f"最佳连续时间段数: {best_length}")
    if best_length > 0:
        print(
            f"时间段覆盖节点比例: {np.mean([iq[1] for iq in interval_quality[best_start:best_start + best_length]]):.2%}")
        print(f"开始时间: {best_intervals[0]}")
        print(f"结束时间: {best_intervals[-1]}")
    else:
        print("警告: 未找到满足条件的时间段，请降低阈值或检查数据质量")

    return best_mask, best_intervals


def filter_and_impute(data_3d, threshold=0.5):
    """筛选传感器节点并进行缺失值填补"""
    num_nodes = data_3d.shape[0]

    # 计算每个节点的缺失率
    missing_rates = np.mean(np.isnan(data_3d[:, :, 0]), axis=1)  # 使用moteid列

    # 筛选节点
    keep_nodes = missing_rates < threshold
    filtered_data = data_3d[keep_nodes, :, :]

    print(f"\n节点筛选结果: 保留{np.sum(keep_nodes)}/{num_nodes}个节点")
    print(f"剔除节点编号: {np.where(~keep_nodes)[0].tolist()}")
    print(f"剔除节点缺失率: {np.mean(missing_rates[~keep_nodes]):.2%}")

    # 对保留的节点进行缺失值填补
    for i in range(filtered_data.shape[0]):
        for j in range(filtered_data.shape[2]):
            if np.any(np.isnan(filtered_data[i, :, j])):
                # 使用前后值的线性插值
                valid_mask = ~np.isnan(filtered_data[i, :, j])
                if np.sum(valid_mask) > 1:  # 只有足够数据时才插值
                    indices = np.arange(len(valid_mask))
                    filtered_data[i, :, j] = np.interp(
                        indices,
                        indices[valid_mask],
                        filtered_data[i, valid_mask, j]
                    )
                else:
                    filtered_data[i, :, j] = 0  # 数据太少则填0

    return filtered_data, keep_nodes


def save_quality_report(data_3d, timestamps, output_dir):
    """生成并保存数据质量报告"""
    report_path = os.path.join(output_dir, 'quality_report.txt')

    with open(report_path, 'w') as f:
        f.write("传感器数据质量评估报告\n")
        f.write("=======================\n\n")

        # 时间范围信息
        f.write(f"时间范围: {timestamps[0]} 至 {timestamps[-1]}\n")
        f.write(f"总时间点数: {len(timestamps)}\n\n")

        # 节点完整性统计
        node_completeness = 1 - np.mean(np.isnan(data_3d[:, :, 0]), axis=1)
        f.write("节点数据完整性统计:\n")
        f.write(f"- 平均完整性: {np.mean(node_completeness):.2%}\n")
        f.write(f"- 最低完整性: {np.min(node_completeness):.2%}\n")
        f.write(f"- 最高完整性: {np.max(node_completeness):.2%}\n\n")

        # 特征缺失统计
        f.write("各特征缺失情况:\n")
        features = ['moteid', 'temperature', 'humidity', 'light', 'voltage', 'x', 'y']
        for i, feat in enumerate(features):
            missing = np.mean(np.isnan(data_3d[:, :, i]))
            f.write(f"- {feat}: {missing:.2%}缺失\n")

        # 时间连续性分析
        time_gaps = np.diff(timestamps).astype('timedelta64[h]')
        f.write("\n时间连续性分析:\n")
        f.write(f"- 最大时间间隔: {np.max(time_gaps)}小时\n")
        f.write(f"- 平均时间间隔: {np.mean(time_gaps):.1f}小时\n")


def process_half_hour_data(data_3d,timestamps):
    """处理半小时粒度传感器数据"""
    min_time, max_time = timestamps.min(), timestamps.max()

    # 创建半小时时间 bins
    print("\n创建半小时时间区间...")
    try:
        time_bins = pd.date_range(
            start=min_time.floor('h'),  # 使用'h'代替'H'
            end=max_time.ceil('h'),  # 使用'h'代替'H'
            freq='30min'
        )
    except Exception as e:
        raise RuntimeError(f"创建时间区间失败: {str(e)}")

    # 初始化结果数组
    num_nodes = data_3d.shape[0]
    num_features = data_3d.shape[2]
    half_hour_data = np.full((num_nodes, len(time_bins) - 1, num_features), np.nan)

    print("\n处理每个半小时区间...")
    print("注意: 开始处理时间区间，总进度如下:")
    for i in tqdm(range(len(time_bins) - 1), total=len(time_bins) - 1, desc="处理时间区间", unit="interval"):
        # 获取当前半小时区间
        start_time = time_bins[i]
        end_time = time_bins[i + 1]

        # 找出在此区间内的原始数据点
        time_mask = (timestamps >= start_time) & (timestamps < end_time)
        interval_data = data_3d[:, time_mask, :]

        # 计算半小时平均值 (忽略NaN)
        with np.errstate(invalid='ignore'):
            # 检查是否有有效数据
            if interval_data.shape[1] > 0 and not np.all(np.isnan(interval_data)):
                result = np.nanmean(interval_data, axis=1)
                # 确保结果维度匹配
                if result.shape[0] == num_nodes and result.shape[1] == num_features:
                    half_hour_data[:, i, :] = result

    # 保存结果
    output_path = os.path.join(dataset_dir, 'sensor_data_3d_30min.npy')
    try:
        np.save(output_path, half_hour_data)
    except Exception as e:
        raise RuntimeError(f"保存传感器数据失败: {str(e)}")

    # 保存时间戳
    time_labels = [f"{time_bins[i].strftime('%Y-%m-%d %H:%M')}"
                   for i in range(len(time_bins) - 1)]
    time_output_path = os.path.join(dataset_dir, 'timestamps_30min.npy')
    try:
        np.save(time_output_path, time_labels)
    except Exception as e:
        raise RuntimeError(f"保存时间戳失败: {str(e)}")

    # 添加数据质量报告
    print("\n=== 数据质量报告 ===")
    print(f"总时间区间数: {len(time_bins) - 1}")
    print(f"无数据的时间区间数: {np.isnan(half_hour_data).all(axis=(0, 2)).sum()}")
    print(f"每个传感器节点的非空值数量: {np.sum(~np.isnan(half_hour_data).all(axis=2), axis=1)}")

    print("\n处理完成!")
    print(f"半小时粒度数据形状: {half_hour_data.shape}")
    print(f"特征顺序保持原始顺序: [moteid, temperature, humidity, light, voltage, x, y]")
    print(f"数据已保存到: {output_path}")

    return half_hour_data,time_labels


def process_10min_data(data_3d, timestamps):
    """处理10分钟粒度传感器数据"""
    min_time, max_time = timestamps.min(), timestamps.max()

    # 创建10分钟时间bins
    print("\n创建10分钟时间区间...")
    try:
        time_bins = pd.date_range(
            start=min_time.floor('10min'),
            end=max_time.ceil('10min'),
            freq='10min'
        )
    except Exception as e:
        raise RuntimeError(f"创建时间区间失败: {str(e)}")

    # 初始化结果数组
    num_nodes = data_3d.shape[0]
    num_features = data_3d.shape[2]
    ten_min_data = np.full((num_nodes, len(time_bins) - 1, num_features), np.nan)

    print("\n处理每个10分钟区间...")
    for i in tqdm(range(len(time_bins) - 1), desc="处理时间区间", unit="interval"):
        start_time = time_bins[i]
        end_time = time_bins[i + 1]

        # 找出在此区间内的原始数据点
        time_mask = (timestamps >= start_time) & (timestamps < end_time)
        interval_data = data_3d[:, time_mask, :]

        # 计算10分钟平均值 (忽略NaN)
        with np.errstate(invalid='ignore'):
            if interval_data.shape[1] > 0 and not np.all(np.isnan(interval_data)):
                result = np.nanmean(interval_data, axis=1)
                if result.shape[0] == num_nodes and result.shape[1] == num_features:
                    ten_min_data[:, i, :] = result

    # 保存结果
    output_path = os.path.join(dataset_dir, 'sensor_data_3d_10min.npy')
    np.save(output_path, ten_min_data)

    time_labels = [f"{time_bins[i].strftime('%Y-%m-%d %H:%M')}"
                   for i in range(len(time_bins) - 1)]
    np.save(os.path.join(dataset_dir, 'timestamps_10min.npy'), time_labels)

    print(f"\n10分钟粒度数据形状: {ten_min_data.shape}")
    return ten_min_data, time_labels


def evaluate_by_zero_consecutive(data_3d, timestamps, max_zero_consecutive=3):
    """
    基于半小时颗粒上连续0值的连贯性评估

    Args:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        timestamps: 对应的时间戳
        max_zero_consecutive: 允许的最大连续0值半小时颗粒数

    Returns:
        valid_nodes: 有效节点索引列表
        time_mask: 有效时间点掩码
    """
    print("\n=== 基于连续0值的连贯性评估 ===")
    num_nodes, num_times, _ = data_3d.shape

    # 使用温度特征(假设是第1列)
    temperature_data = data_3d[:, :, 1]

    # 1. 评估每个节点的连续0值情况
    node_quality = []
    for node in range(num_nodes):
        zero_streaks = []
        current_streak = 0

        # 检测连续0值
        for t in range(num_times):
            if temperature_data[node, t] == 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    zero_streaks.append(current_streak)
                current_streak = 0

        # 记录最长连续0值
        max_streak = max(zero_streaks) if zero_streaks else 0
        node_quality.append({
            'node_id': node,
            'max_zero_streak': max_streak,
            'valid': max_streak <= max_zero_consecutive
        })

    # 2. 筛选合格节点(连续0值不超过阈值)
    valid_nodes = [q['node_id'] for q in node_quality if q['valid']]
    print(f"节点筛选结果: 保留{len(valid_nodes)}/{num_nodes}个节点")
    print(f"淘汰节点数(连续0值>{max_zero_consecutive}): "
          f"{sum(1 for q in node_quality if not q['valid'])}")

    # 3. 时间维度筛选(保留至少有一个有效节点的时间点)
    time_mask = ~np.all(np.isnan(data_3d[valid_nodes, :, 1]), axis=0)
    print(f"时间点筛选结果: 保留{np.sum(time_mask)}/{num_times}个时间点")

    # 4. 检测连续时间段
    time_gaps = np.diff(np.where(time_mask)[0])
    break_points = np.where(time_gaps > 1)[0] + 1
    time_segments = np.split(np.where(time_mask)[0], break_points)

    # 打印所有连续时间段
    print("\n=== 检测到的连续时间段 ===")
    for i, seg in enumerate(time_segments):
        if len(seg) > 0:
            start_idx, end_idx = seg[0], seg[-1]
            print(f"时间段 {i + 1}: {timestamps[start_idx]} 至 {timestamps[end_idx]} "
                  f"(持续{len(seg)}个时间点)")

    # 5. 选择最长的连续时间段
    longest_seg = max(time_segments, key=len) if time_segments else np.array([], dtype=int)
    longest_time_mask = np.zeros_like(time_mask, dtype=bool)
    if len(longest_seg) > 0:
        longest_time_mask[longest_seg] = True
        print(f"\n选择最长连续时间段: {len(longest_seg)}个时间点")
        print(f"开始时间: {timestamps[longest_seg[0]]}")
        print(f"结束时间: {timestamps[longest_seg[-1]]}")
    else:
        print("警告: 未找到有效连续时间段")

    return valid_nodes, time_mask


def select_time_period(data_3d, timestamps, start_time, end_time):
    """
    选择指定时间段的数据

    Args:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        timestamps: 对应的时间戳列表
        start_time: 开始时间字符串 (格式: 'YYYY-MM-DD HH:MM')
        end_time: 结束时间字符串 (格式: 'YYYY-MM-DD HH:MM')

    Returns:
        selected_data: 选择的数据
        selected_timestamps: 选择的时间戳
        time_mask: 时间点选择掩码
    """
    # 转换为datetime类型
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # 确保timestamps是Pandas DatetimeIndex
    if isinstance(timestamps, np.ndarray):
        timestamps_dt = pd.to_datetime(timestamps)
    else:
        timestamps_dt = timestamps

    # 创建时间选择掩码
    time_mask = (timestamps_dt >= start_dt) & (timestamps_dt <= end_dt)

    # 筛选数据
    selected_data = data_3d[:, time_mask, :]
    selected_timestamps = np.array(timestamps)[time_mask]

    print(f"\n时间段选择结果: {len(selected_timestamps)}/{len(timestamps)}个时间点")
    print(f"开始时间: {selected_timestamps[0]}")
    print(f"结束时间: {selected_timestamps[-1]}")

    return selected_data, selected_timestamps, time_mask


def plot_temperature_heatmap(data_3d, timestamps, output_dir):
    """
    绘制温度特征热图并保存

    Args:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        timestamps: 对应的时间戳列表 (可以是字符串或datetime对象)
        output_dir: 输出目录
    """
    # 温度是第1列(假设)
    temperature_data = np.ma.masked_invalid(data_3d[:, :, 1])  # 使用masked_invalid处理NaN
    csv_path = os.path.join(output_dir, 'temperature_data.csv')
    # 创建DataFrame，时间戳作为列名，节点ID作为行索引
    df = pd.DataFrame(temperature_data,
                      columns=[t.strftime('%Y-%m-%d %H:%M') for t in pd.to_datetime(timestamps)],
                      index=np.arange(temperature_data.shape[0]))

    df.to_csv(csv_path)
    print(f"温度数据已保存为CSV: {csv_path}")

    # 确保timestamps是datetime类型
    if isinstance(timestamps[0], (np.str_, str)):
        timestamps_dt = pd.to_datetime(timestamps)
    else:
        timestamps_dt = timestamps

    # 设置图形大小
    plt.figure(figsize=(20, 10))

    # 创建热图 - 关键修改：添加origin='lower'保持正确方向
    plt.imshow(temperature_data, aspect='auto', cmap='jet', origin='lower',
              extent=[0, len(timestamps_dt)-1, 0, data_3d.shape[0]-1])

    # 设置坐标轴
    time_labels = [t.strftime('%m-%d %H:%M') for t in timestamps_dt]

    # 获取有效节点(非全NaN的节点)
    valid_nodes = ~np.all(np.isnan(data_3d[:, :, 1]), axis=1)
    node_ids = np.arange(data_3d.shape[0])[valid_nodes]  # 真实节点编号

    # # 设置y轴 - 关键修改：使用真实节点编号且保持方向
    # plt.yticks(
    #     np.where(valid_nodes)[0],  # 位置
    #     node_ids,  # 标签(真实节点编号)
    # )

    # 修改y轴设置 - 现在直接用0-based连续编号
    plt.yticks(
        np.arange(data_3d.shape[0]),
        np.arange(data_3d.shape[0])  # 显示连续的节点序号
    )
    plt.ylabel("节点序号(0-based)")


    # 减少x轴标签密度 (每6个时间点显示一个)
    step = max(1, len(timestamps_dt) // 20)  # 确保至少显示20个标签
    plt.xticks(np.arange(0, len(timestamps_dt), step),
               [time_labels[i] for i in range(0, len(time_labels), step)],
               rotation=45)

    # 添加标签和颜色条
    plt.xlabel('时间点')
    plt.ylabel('节点ID')
    plt.colorbar(label='温度值')
    plt.title('节点温度变化热图')

    # 调整布局并保存
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'temperature_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n温度热图已保存到: {output_path}")
    plt.close()


def filter_abnormal_nodes(data_3d, node_ids_to_remove):
    """
    剔除指定的异常节点

    Args:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        node_ids_to_remove: 需要剔除的节点ID列表

    Returns:
        filtered_data: 过滤后的数据
        keep_mask: 保留节点的掩码
    """
    num_nodes = data_3d.shape[0]
    keep_mask = np.ones(num_nodes, dtype=bool)

    for node_id in node_ids_to_remove:
        if node_id < num_nodes:
            keep_mask[node_id] = False

    # 创建结果数组 (保持原始形状，用NaN填充被剔除的节点)
    # filtered_data = data_3d.copy()
    # for node_id in node_ids_to_remove:
    #     if node_id < num_nodes:
    #         filtered_data[node_id, :, :] = np.nan

    # 直接删除节点而不是设为NaN
    filtered_data = data_3d[keep_mask, :, :]

    print(f"\n节点剔除结果: 保留{np.sum(keep_mask)}/{num_nodes}个节点")
    print(f"剔除的节点ID: {[i for i in node_ids_to_remove if i < num_nodes]}")
    # print("注意: 保留原始节点编号，被剔除节点的数据已设为NaN")
    print("注意: 已完全删除被剔除节点，不保留其位置")

    return filtered_data, keep_mask


def read_locations(dataset_dir):
    """读取节点位置数据并返回DataFrame"""
    location_path = os.path.join(dataset_dir, 'Intel Lab Data.txt', 'location.txt')
    try:
        # 读取位置数据，假设格式为：节点ID x y
        locations = pd.read_csv(location_path, sep='\s+', header=None,
                             names=['moteid', 'x', 'y'], dtype={'moteid': int})

        print("\n位置数据加载成功")
        print(f"原始位置数据节点数: {len(locations)}")
        return locations
    except Exception as e:
        raise RuntimeError(f"加载位置数据失败: {str(e)}")


def plot_node_locations(locations, output_dir, title="传感器节点位置分布"):
    """
    绘制节点位置分布图

    Args:
        locations: 包含节点位置数据的DataFrame (columns: node_id, x, y)
        output_dir: 输出目录
        title: 图标题
    """
    plt.figure(figsize=(10, 8))

    # 获取当前坐标轴
    ax = plt.gca()

    # 翻转x轴和y轴
    ax.invert_xaxis()  # 左右翻转
    ax.invert_yaxis()  # 上下翻转

    # 绘制所有节点位置
    plt.scatter(locations['x'], locations['y'], c='blue', alpha=0.6,
                label=f'节点 (总数: {len(locations)})')

    # 标注所有节点的真实编号
    for _, row in locations.iterrows():
        plt.text(row['x'], row['y'], f"{int(row['moteid'])}",
                 fontsize=8, ha='center', va='bottom')

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存图像
    output_path = os.path.join(output_dir, 'node_locations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"节点位置图已保存到: {output_path}")

    # 保存筛选后的节点数据 (moteid, x, y)
    output_data_path = os.path.join(output_dir, 'filtered_node_locations.csv')
    locations[['moteid', 'x', 'y']].to_csv(output_data_path, index=False)
    print(f"筛选后的节点位置数据已保存到: {output_data_path}")

if __name__ == '__main__':
    # 步骤1: 先进行半小时粒度采样（原始数据）
    hourly_data, hourly_timestamps = process_half_hour_data(data_3d, timestamps)
    # hourly_data, hourly_timestamps = process_10min_data(data_3d, timestamps)

    # 步骤2: 在聚合后的数据上评估连贯性
    # best_mask, best_dates = evaluate_data_coherence(hourly_data, hourly_timestamps)
    # coherent_data = hourly_data[:, best_mask, :]
    # coherent_timestamps = np.array(hourly_timestamps)[best_mask]

    # 步骤2: 新连贯性评估
    good_nodes, time_mask = evaluate_by_zero_consecutive(
        hourly_data,
        hourly_timestamps,
        max_zero_consecutive=3  # 允许最多连续3个半小时颗粒为0
    )
    coherent_data = hourly_data[good_nodes, :, :][:, time_mask, :]
    coherent_timestamps = np.array(hourly_timestamps)[time_mask]

    # 步骤4: 选择特定时间段 (2004-02-28 00:30 至 2004-03-10 09:00)
    selected_data, selected_timestamps, _ = select_time_period(
        coherent_data,
        coherent_timestamps,
        start_time='2004-02-28 00:30',
        end_time='2004-03-10 09:00'
    )

    # 步骤5: 筛选节点并填补
    filtered_data, keep_nodes_from_impute = filter_and_impute(selected_data)

    # 新增步骤6: 剔除异常节点
    filtered_data, keep_mask = filter_abnormal_nodes(filtered_data, [15])

    # 获取最终保留的节点ID
    # 需要结合两个筛选步骤的结果
    final_node_ids = np.array(good_nodes)[keep_nodes_from_impute]  # 从filter_and_impute中筛选
    final_node_ids = final_node_ids[keep_mask]  # 再从filter_abnormal_nodes中筛选

    # 新增步骤7: 读取和处理位置数据
    locations = read_locations(dataset_dir)

    # 保存结果
    output_path = os.path.join(dataset_dir, 'filtered_sensor_data_3d.npy')
    np.save(output_path, filtered_data)
    np.save(os.path.join(dataset_dir, 'filtered_timestamps.npy'), coherent_timestamps)
    plot_temperature_heatmap(filtered_data, selected_timestamps, dataset_dir)

    # 筛选位置数据
    # 注意位置数据中的moteid比数组索引大1
    filtered_locations = locations[locations['moteid'].isin(final_node_ids + 1)]

    print(f"\n位置数据筛选结果: 保留{len(filtered_locations)}/{len(locations)}个节点位置")

    # 绘制节点位置图
    plot_node_locations(filtered_locations, dataset_dir,
                        title=f"有效传感器节点位置分布 (共{len(filtered_locations)}个节点)")