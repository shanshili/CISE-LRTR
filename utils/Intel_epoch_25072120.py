"""
Sensor Data Analysis and Visualization Tool
------------------------------------------

This script provides comprehensive analysis and visualization of 3D sensor data with features:

1. **Data Processing**:
   - Loads 3D sensor arrays (.npy format) with timestamps
   - Validates data consistency across sensors
   - Provides dimensional statistics

2. **Multi-resolution Heatmaps**:
   - Daily (24-hour aggregates)
   - 6-hour intervals
   - 3-hour intervals
   - Hourly resolution
   - 30-minute resolution (high granularity)

3. **Visualization Features**:
   - Custom colormaps (white for zero values)
   - Adaptive cell sizing and labeling
   - Automatic label density adjustment
   - High-quality SVG output (300DPI)

4. **Output Details**:
   - Per-sensor total readings
   - Time-bin specific counts
   - Color intensity indicates activity level

Usage: Modify the analyze_sensor_data() call to enable/disable specific heatmap generations.
"""
"""
传感器时间戳(epoch)分布分析与可视化工具
==========================

本脚本提供三维传感器数据的全面分析和可视化功能：

1. 数据处理功能：
   - 自动加载.npy格式的三维传感器数据
   - 检查各传感器时间步数一致性
   - 提供各维度特征的统计信息

2. 多时间粒度热图生成：
   - 每日热图（全天汇总）
   - 6小时间隔热图（每日4时段）
   - 3小时间隔热图（每日8时段） 
   - 小时级热图（每日24时段）
   - 半小时级热图（每日48时段，高精度）

3. 可视化优化特性：
   - 零值白色高亮显示
   - 自适应单元格尺寸和字体大小
   - 智能调整坐标轴标签密度
   - 输出300DPI高清矢量图(SVG)

4. 输出信息包含：
   - 每个传感器的总记录数
   - 分时段详细记录数（热图单元格中显示）
   - 颜色渐变表示活动强度（黄-橙-红色系）

使用说明：在analyze_sensor_data()函数中注释/取消注释相应热图生成函数调用
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


def plot_daily_heatmap(data_3d, timestamps):
    """
    Draw daily sensor activity heatmap with optimized cell dimensions and font sizes
    """
    # Convert timestamps to date format
    dates = pd.to_datetime(timestamps).date
    daily_labels = sorted(set(dates))

    # Calculate daily activity matrix (nodes × days)
    daily_activity = np.zeros((data_3d.shape[0], len(daily_labels)))
    daily_counts = np.zeros_like(daily_activity, dtype=int)

    for day_idx, day in enumerate(daily_labels):
        day_mask = (dates == day)
        daily_activity[:, day_idx] = np.sum(~np.isnan(data_3d[:, day_mask, 0]), axis=1)
        daily_counts[:, day_idx] = daily_activity[:, day_idx].astype(int)

    # Calculate total records per sensor
    total_records = np.sum(daily_activity, axis=1).astype(int)

    # Create figure with adjusted aspect ratio
    cell_height = 0.3  # Height per sensor row (in inches)
    cell_width = 0.8  # Width per day column (in inches)
    fig_width = max(10, len(daily_labels) * cell_width)
    fig_height = max(6, data_3d.shape[0] * cell_height)
    plt.figure(figsize=(fig_width, fig_height))

    cmap = plt.cm.YlOrRd
    cmap.set_under('white')  # 低于最小值的颜色设为白色

    # 在heatmap调用中使用这个cmap并设置vmin
    ax = sns.heatmap(
        daily_activity,
        cmap=cmap,
        vmin=0.1,  # 任何小于0.1的值将显示为白色
        xticklabels=[day.strftime('%Y-%m-%d') for day in daily_labels],
        yticklabels=[f"ID {i + 1} ({total})" for i, total in enumerate(total_records)],
        cbar_kws={'label': 'Records', 'shrink': 0.5},
        annot=daily_counts,
        fmt='d',
        annot_kws={
            'size': 8,
            'color': 'black',
            'weight': 'normal'
        },
        square=False,
        linewidths=0.3,
        linecolor='grey'
    )

    # Adjust font sizes and layout
    plt.title('Sensor Daily Records', pad=15, fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Sensor Node (Total)', fontsize=10)

    # Rotate x-axis labels more vertically
    plt.xticks(
        rotation=90,  # More vertical rotation
        ha='center',  # Center aligned
        fontsize=8  # Smaller date font
    )
    plt.yticks(fontsize=8)  # Smaller y-axis font

    # Adjust colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Records', fontsize=10)

    # Tight layout with minimal padding
    plt.tight_layout(pad=1.0)

    # Save as SVG
    svg_path = os.path.join(dataset_dir, 'sensor_daily_activity.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nHeatmap saved to: {svg_path}")
    plt.close()


def plot_6hour_heatmap(data_3d, timestamps):
    """
    绘制每6小时传感器活动热图
    """
    # 转换时间戳为datetime格式
    timestamps = pd.to_datetime(timestamps)

    # 创建6小时间隔
    time_bins = pd.date_range(
        start=timestamps.min().floor('D'),
        end=timestamps.max().ceil('D'),
        freq='6h'
    )

    # 计算每个6小时区间的活动量
    period_activity = np.zeros((data_3d.shape[0], len(time_bins) - 1))
    period_counts = np.zeros_like(period_activity, dtype=int)

    for i in range(len(time_bins) - 1):
        time_mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
        period_activity[:, i] = np.sum(~np.isnan(data_3d[:, time_mask, 0]), axis=1)
        period_counts[:, i] = period_activity[:, i].astype(int)

    # 计算总记录数
    total_records = np.sum(period_activity, axis=1).astype(int)

    # 设置图形尺寸
    cell_height = 0.4  # 每个传感器行的高度
    cell_width = 0.6  # 每个时间区间的宽度
    fig_width = max(12, (len(time_bins) - 1) * cell_width)
    fig_height = max(8, data_3d.shape[0] * cell_height)
    plt.figure(figsize=(fig_width, fig_height))

    # 创建热图
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')

    ax = sns.heatmap(
        period_activity,
        cmap=cmap,
        vmin=0.1,  # 小于0.1的值显示为白色
        xticklabels=[f"{time_bins[i].strftime('%m-%d %H:%M')}\nto\n{time_bins[i + 1].strftime('%H:%M')}"
                     for i in range(len(time_bins) - 1)],
        yticklabels=[f"ID {i + 1} ({total})" for i, total in enumerate(total_records)],
        cbar_kws={'label': 'Records per 6h', 'shrink': 0.5},
        annot=period_counts,
        fmt='d',
        annot_kws={'size': 8, 'color': 'black'},
        square=False,
        linewidths=0.3,
        linecolor='grey'
    )

    # 设置标题和标签
    plt.title('Sensor Activity per 6 Hours', pad=20, fontsize=14)
    plt.xlabel('6 Hour Time Intervals', fontsize=12)
    plt.ylabel('Sensor Node (Total Records)', fontsize=12)

    # 调整x轴标签
    plt.xticks(
        rotation=45,
        ha='right',
        fontsize=8
    )
    plt.yticks(fontsize=9)

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Records per 6h', fontsize=11)

    plt.tight_layout()

    # 保存图像
    svg_path = os.path.join(dataset_dir, 'sensor_6hour_activity.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\n6-hour heatmap saved to: {svg_path}")
    plt.close()


def plot_3hour_heatmap(data_3d, timestamps):
    """
    绘制每3小时传感器活动热图
    """
    # 转换时间戳为datetime格式
    timestamps = pd.to_datetime(timestamps)

    # 创建3小时间隔
    time_bins = pd.date_range(
        start=timestamps.min().floor('D'),
        end=timestamps.max().ceil('D'),
        freq='3h'
    )

    # 计算每个3小时区间的活动量
    period_activity = np.zeros((data_3d.shape[0], len(time_bins) - 1))
    period_counts = np.zeros_like(period_activity, dtype=int)

    for i in range(len(time_bins) - 1):
        time_mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
        period_activity[:, i] = np.sum(~np.isnan(data_3d[:, time_mask, 0]), axis=1)
        period_counts[:, i] = period_activity[:, i].astype(int)

    # 计算总记录数
    total_records = np.sum(period_activity, axis=1).astype(int)

    # 设置图形尺寸
    cell_height = 0.4  # 每个传感器行的高度
    cell_width = 0.4  # 每个时间区间的宽度(3小时比6小时更窄)
    fig_width = max(12, (len(time_bins) - 1) * cell_width)
    fig_height = max(8, data_3d.shape[0] * cell_height)
    plt.figure(figsize=(fig_width, fig_height))

    # 创建自定义颜色映射，0值显示为白色
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')

    # 创建热图
    ax = sns.heatmap(
        period_activity,
        cmap=cmap,
        vmin=0.1,  # 小于0.1的值显示为白色
        xticklabels=[f"{time_bins[i].strftime('%H:%M')}"  # 只显示时间，更简洁
                     for i in range(len(time_bins) - 1)],
        yticklabels=[f"ID {i + 1} ({total})" for i, total in enumerate(total_records)],
        cbar_kws={'label': 'Records per 3h', 'shrink': 0.5},
        annot=period_counts,
        fmt='d',
        annot_kws={'size': 8, 'color': 'black'},
        square=False,
        linewidths=0.2,  # 更细的网格线
        linecolor='grey'
    )

    # 设置标题和标签
    plt.title('Sensor Activity per 3 Hours', pad=20, fontsize=14)
    plt.xlabel('3 Hour Time Intervals (Date: ' + timestamps.min().strftime('%Y-%m-%d') + ')',
               fontsize=10)  # 在x轴标题显示日期
    plt.ylabel('Sensor Node (Total Records)', fontsize=10)

    # 调整x轴标签
    plt.xticks(
        rotation=90,  # 垂直旋转
        ha='center',  # 居中
        fontsize=7  # 更小的字体
    )
    plt.yticks(fontsize=8)

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Records per 3h', fontsize=10)

    plt.tight_layout()

    # 保存图像
    svg_path = os.path.join(dataset_dir, 'sensor_3hour_activity.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\n3-hour heatmap saved to: {svg_path}")
    plt.close()

def plot_hourly_heatmap(data_3d, timestamps):
    """
    绘制每小时传感器活动热图
    """
    # 转换时间戳为datetime格式
    timestamps = pd.to_datetime(timestamps)

    # 创建1小时间隔
    time_bins = pd.date_range(
        start=timestamps.min().floor('D'),
        end=timestamps.max().ceil('D'),
        freq='1h'
    )

    # 计算每个小时区间的活动量
    period_activity = np.zeros((data_3d.shape[0], len(time_bins) - 1))
    period_counts = np.zeros_like(period_activity, dtype=int)

    for i in range(len(time_bins) - 1):
        time_mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
        period_activity[:, i] = np.sum(~np.isnan(data_3d[:, time_mask, 0]), axis=1)
        period_counts[:, i] = period_activity[:, i].astype(int)

    # 计算总记录数
    total_records = np.sum(period_activity, axis=1).astype(int)

    # 设置图形尺寸
    cell_height = 0.4  # 每个传感器行的高度
    cell_width = 0.2   # 每个时间区间的宽度更窄
    fig_width = max(15, (len(time_bins) - 1) * cell_width)
    fig_height = max(8, data_3d.shape[0] * cell_height)
    plt.figure(figsize=(fig_width, fig_height))

    # 创建自定义颜色映射，0值显示为白色
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')

    # 创建热图
    ax = sns.heatmap(
        period_activity,
        cmap=cmap,
        vmin=0.1,  # 小于0.1的值显示为白色
        xticklabels=[f"{time_bins[i].strftime('%H:%M')}"
                    for i in range(len(time_bins) - 1)],
        yticklabels=[f"ID {i + 1} ({total})" for i, total in enumerate(total_records)],
        cbar_kws={'label': 'Records per hour', 'shrink': 0.5},
        annot=period_counts,
        fmt='d',
        annot_kws={'size': 6, 'color': 'black'},  # 更小的标注字体
        square=False,
        linewidths=0.1,  # 更细的网格线
        linecolor='lightgrey'  # 更浅的网格线颜色
    )

    # 设置标题和标签
    plt.title(f'Sensor Hourly Activity ({timestamps.min().date()} to {timestamps.max().date()})',
             pad=20, fontsize=12)
    plt.xlabel('Hourly Time Intervals', fontsize=10)
    plt.ylabel('Sensor Node (Total Records)', fontsize=10)

    # 调整x轴标签 - 每6小时显示一个标签
    tick_interval = 6  # 每6小时显示一个标签
    plt.xticks(
        ticks=np.arange(0, len(time_bins)-1, tick_interval),
        labels=[f"{time_bins[i].strftime('%H:%M')}"
               for i in range(0, len(time_bins)-1, tick_interval)],
        rotation=90,
        ha='center',
        fontsize=7
    )
    plt.yticks(fontsize=8)

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Records per hour', fontsize=10)

    plt.tight_layout()

    # 保存图像
    svg_path = os.path.join(dataset_dir, 'sensor_hourly_activity.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\nHourly heatmap saved to: {svg_path}")
    plt.close()


def plot_30min_heatmap(data_3d, timestamps):
    """
    绘制每半小时传感器活动热图
    """
    # 转换时间戳为datetime格式
    timestamps = pd.to_datetime(timestamps)

    # 创建半小时间隔
    time_bins = pd.date_range(
        start=timestamps.min().floor('D'),
        end=timestamps.max().ceil('D'),
        freq='30min'
    )

    # 计算每个半小时区间的活动量
    period_activity = np.zeros((data_3d.shape[0], len(time_bins) - 1))
    period_counts = np.zeros_like(period_activity, dtype=int)

    for i in range(len(time_bins) - 1):
        time_mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
        period_activity[:, i] = np.sum(~np.isnan(data_3d[:, time_mask, 0]), axis=1)
        period_counts[:, i] = period_activity[:, i].astype(int)

    # 计算总记录数
    total_records = np.sum(period_activity, axis=1).astype(int)

    # 设置图形尺寸
    cell_height = 0.4  # 每个传感器行的高度
    cell_width = 0.15  # 每个时间区间的宽度更窄
    fig_width = max(20, (len(time_bins) - 1) * cell_width)  # 更宽的图形
    fig_height = max(8, data_3d.shape[0] * cell_height)
    plt.figure(figsize=(fig_width, fig_height))

    # 创建自定义颜色映射，0值显示为白色
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')

    # 创建热图
    ax = sns.heatmap(
        period_activity,
        cmap=cmap,
        vmin=0.1,  # 小于0.1的值显示为白色
        xticklabels=[f"{time_bins[i].strftime('%H:%M')}"
                    for i in range(len(time_bins) - 1)],
        yticklabels=[f"ID {i + 1} ({total})" for i, total in enumerate(total_records)],
        cbar_kws={'label': 'Records per 30min', 'shrink': 0.5},
        annot=False,  # 不显示数值，因单元格太小
        square=False,
        linewidths=0.05,  # 更细的网格线
        linecolor='lightgrey'  # 更浅的网格线颜色
    )

    # 设置标题和标签
    plt.title(f'Sensor Activity per 30 Minutes ({timestamps.min().date()} to {timestamps.max().date()})',
             pad=20, fontsize=12)
    plt.xlabel('30 Minute Time Intervals', fontsize=10)
    plt.ylabel('Sensor Node (Total Records)', fontsize=10)

    # 调整x轴标签 - 每2小时显示一个标签
    tick_interval = 4  # 每4个半小时(即每2小时)显示一个标签
    plt.xticks(
        ticks=np.arange(0, len(time_bins)-1, tick_interval),
        labels=[f"{time_bins[i].strftime('%H:%M')}"
               for i in range(0, len(time_bins)-1, tick_interval)],
        rotation=90,
        ha='center',
        fontsize=6  # 更小的字体
    )
    plt.yticks(fontsize=8)

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Records per 30min', fontsize=10)

    plt.tight_layout()

    # 保存图像
    svg_path = os.path.join(dataset_dir, 'sensor_30min_activity.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"\n30-minute heatmap saved to: {svg_path}")
    plt.close()


def analyze_sensor_data(data_3d):
    """分析三维传感器数据"""
    # 检查时间步数一致性
    print("\n=== 时间步数检查 ===")
    time_steps_per_sensor = np.sum(~np.isnan(data_3d[:, :, 0]), axis=1)  # 使用moteid列检查
    unique_time_steps = np.unique(time_steps_per_sensor)

    if len(unique_time_steps) == 1:
        print(f"所有传感器具有相同时间步数: {unique_time_steps[0]}")
    else:
        print("警告: 传感器时间步数不一致")
        print(f"时间步数分布: {np.unique(time_steps_per_sensor, return_counts=True)}")
        print(f"最少时间步数: {np.min(time_steps_per_sensor)}")
        print(f"最多时间步数: {np.max(time_steps_per_sensor)}")

    # 打印特征维度信息
    print("\n=== 特征维度 ===")
    feature_names = [
        'moteid (节点ID)',
        'temperature (温度)',
        'humidity (湿度)',
        'light (光照)',
        'voltage (电压)',
        'x (x坐标)',
        'y (y坐标)'
    ]

    for i, name in enumerate(feature_names):
        print(f"维度 {i}: {name}")

    # # 各特征统计信息
    # print("\n=== 特征统计 ===")
    # stats = pd.DataFrame({
    #     '特征': feature_names,
    #     '平均值': np.nanmean(data_3d, axis=(0, 1)),
    #     '标准差': np.nanstd(data_3d, axis=(0, 1)),
    #     '最小值': np.nanmin(data_3d, axis=(0, 1)),
    #     '最大值': np.nanmax(data_3d, axis=(0, 1)),
    #     '缺失值数': np.isnan(data_3d).sum(axis=(0, 1))
    # })
    # print(stats)

    # 修改加载时间戳的代码部分
    try:
        timestamps = np.load(os.path.join(dataset_dir, 'timestamps.npy'), allow_pickle=True)
        print("\n=== 生成每日热图 ===")
        # plot_daily_heatmap(data_3d, timestamps)

        print("\n=== 生成6小时热图 ===")
        # plot_6hour_heatmap(data_3d, timestamps)

        print("\n=== 生成3小时热图 ===")
        # plot_3hour_heatmap(data_3d, timestamps)

        print("\n=== 生成每小时热图 ===")
        # plot_hourly_heatmap(data_3d, timestamps)

        print("\n=== 生成半小时热图 ===")
        plot_30min_heatmap(data_3d, timestamps)
    except FileNotFoundError:
        print("\n无法生成热图: 未找到timestamps.npy文件")
    except Exception as e:
        print(f"\n加载时间戳时出错: {str(e)}")

if __name__ == '__main__':
    data = np.load(os.path.join(dataset_dir, 'sensor_data_3d.npy'))
    analyze_sensor_data(data)