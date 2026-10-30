"""
传感器节点分布图绘制和采样脚本

功能:
1. 读取location2.csv文件中的传感器节点坐标
2. 绘制完整的传感器节点分布图
3. 按County区分颜色绘制节点分布图
4. 按City区分颜色绘制节点分布图
5. 可选采样模块，根据目标节点数量进行采样
6. 保存采样节点坐标信息
7. 绘制采样后图片

输入:
- location2.csv: 包含传感器节点ID和坐标的文件

输出:
- sensor_distribution_full.png: 完整传感器分布图
- sensor_distribution_by_county.png: 按County区分的传感器分布图
- sensor_distribution_by_city.png: 按City区分的传感器分布图
- sensor_distribution_sampled.png: 采样后传感器分布图（如果启用采样）
- location2_sampled.csv: 采样后的节点坐标信息（如果启用采样）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans

# 配置参数
ENABLE_SAMPLING = True  # 是否启用采样功能
TARGET_SAMPLE_SIZE = 100  # 目标采样节点数量

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取传感器节点位置数据
location_file = os.path.join(script_dir, "location2.csv")
df = pd.read_csv(location_file)

print(f"读取到 {len(df)} 个传感器节点")

# 检查数据
print("数据前5行:")
print(df.head())

# 创建图像保存目录
fig_dir = os.path.join(script_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 设置高分辨率
DPI = 600

# 1. 绘制完整的传感器节点分布图
print("正在绘制完整传感器节点分布图...")

plt.figure(figsize=(12, 10))
plt.scatter(df['Longitude'], df['Latitude'], alpha=0.6, s=1, c='blue', label=f'Sensor Nodes ({len(df)})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('PeMS Sensor Node Spatial Distribution (Full)')
plt.grid(True, alpha=0.3)
plt.legend()

# 保存完整分布图
full_plot_path = os.path.join(fig_dir, "sensor_distribution_full.png")
plt.savefig(full_plot_path, dpi=DPI, bbox_inches='tight')
plt.show()
plt.close()

print(f"完整传感器分布图已保存到: {full_plot_path}")

# 2. 按County绘制传感器节点分布图
print("正在绘制按County区分的传感器节点分布图...")

# 获取唯一的County值
counties = df['County'].unique()
counties.sort()

# 为每个County分配颜色
colors = plt.cm.Set3(np.linspace(0, 1, len(counties)))
county_color_map = dict(zip(counties, colors))

plt.figure(figsize=(12, 10))
for county in counties:
    county_data = df[df['County'] == county]
    plt.scatter(county_data['Longitude'], county_data['Latitude'],
                alpha=0.6, s=1, c=[county_color_map[county]],
                label=f'County {county} ({len(county_data)} nodes)')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('PeMS Sensor Node Spatial Distribution by County')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存按County区分的分布图
county_plot_path = os.path.join(fig_dir, "sensor_distribution_by_county.png")
plt.savefig(county_plot_path, dpi=DPI, bbox_inches='tight')
plt.show()
plt.close()

print(f"按County区分的传感器分布图已保存到: {county_plot_path}")

# 3. 按City绘制传感器节点分布图
print("正在绘制按City区分的传感器节点分布图...")

# 由于City数量可能较多，我们只显示前20个City，其余归为"Other"
city_counts = df['City'].value_counts()
top_cities = city_counts.head(20).index
other_cities = city_counts.tail(len(city_counts) - 20).index if len(city_counts) > 20 else []

# 为前20个City分配颜色
top_colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(city_counts))))
city_color_map = dict(zip(top_cities, top_colors))

plt.figure(figsize=(12, 10))
# 绘制前20个City
for city in top_cities:
    city_data = df[df['City'] == city]
    plt.scatter(city_data['Longitude'], city_data['Latitude'],
                alpha=0.6, s=1, c=[city_color_map[city]],
                label=f'City {city} ({len(city_data)} nodes)')

# 绘制其他City（如果有的话）
if len(other_cities) > 0:
    other_data = df[df['City'].isin(other_cities)]
    plt.scatter(other_data['Longitude'], other_data['Latitude'],
                alpha=0.6, s=1, c='gray', label=f'Other Cities ({len(other_data)} nodes)')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('PeMS Sensor Node Spatial Distribution by City (Top 20)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存按City区分的分布图
city_plot_path = os.path.join(fig_dir, "sensor_distribution_by_city.png")
plt.savefig(city_plot_path, dpi=DPI, bbox_inches='tight')
plt.show()
plt.close()

print(f"按City区分的传感器分布图已保存到: {city_plot_path}")


# 4. 采样模块
def kmeans_sampling(df, n_clusters):
    """
    基于K-means聚类的采样方法
    """
    # 提取坐标数据
    coordinates = df[['Latitude', 'Longitude']].values

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coordinates)

    # 为每个聚类选择最近的点
    sampled_indices = []
    for i in range(n_clusters):
        cluster_points = np.where(clusters == i)[0]
        if len(cluster_points) > 0:
            # 选择聚类中心最近的点
            cluster_center = kmeans.cluster_centers_[i]
            cluster_coords = coordinates[cluster_points]
            distances = np.sum((cluster_coords - cluster_center) ** 2, axis=1)
            closest_point_idx = cluster_points[np.argmin(distances)]
            sampled_indices.append(closest_point_idx)

    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)
    return sampled_df


# 如果启用采样功能
if ENABLE_SAMPLING:
    # 确保目标采样数量不超过总节点数
    target_sample_size = min(TARGET_SAMPLE_SIZE, len(df))

    print(f"\n正在进行采样，目标节点数量: {target_sample_size}")

    # 执行采样
    sampled_df = kmeans_sampling(df, target_sample_size)

    print(f"采样完成，实际获得 {len(sampled_df)} 个节点")

    # 5. 绘制采样后的传感器节点分布图
    print("正在绘制采样后的传感器节点分布图...")

    plt.figure(figsize=(12, 10))
    plt.scatter(sampled_df['Longitude'], sampled_df['Latitude'],
                alpha=0.7, s=2, c='red', label=f'Sampled Nodes ({len(sampled_df)})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(
        f'PeMS Sensor Node Spatial Distribution (Sampled)\nTarget Size: {target_sample_size}, Actual Size: {len(sampled_df)}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存采样分布图
    sampled_plot_path = os.path.join(fig_dir, "sensor_distribution_sampled.png")
    plt.savefig(sampled_plot_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"采样后的传感器分布图已保存到: {sampled_plot_path}")

    # 6. 保存采样后的节点坐标信息
    sampled_data_path = os.path.join(script_dir, "location2_sampled.csv")
    sampled_df.to_csv(sampled_data_path, index=False)
    print(f"采样后的节点坐标信息已保存到: {sampled_data_path}")

    # 7. 绘制对比图
    print("正在绘制采样前后对比图...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 完整数据图
    ax1.scatter(df['Longitude'], df['Latitude'], alpha=0.6, s=0.5, c='blue')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Full Sensor Distribution ({len(df)} nodes)')
    ax1.grid(True, alpha=0.3)

    # 采样数据图
    ax2.scatter(sampled_df['Longitude'], sampled_df['Latitude'], alpha=0.7, s=2, c='red')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Sampled Sensor Distribution ({len(sampled_df)} nodes)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = os.path.join(fig_dir, "sensor_distribution_comparison.png")
    plt.savefig(comparison_plot_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"采样前后对比图已保存到: {comparison_plot_path}")

# 输出统计信息
print("\n=== 处理结果统计 ===")
print(f"总节点数: {len(df)}")
print(f"County数量: {len(counties)}")
print("各County节点数:")
for county in counties:
    county_data = df[df['County'] == county]
    print(f"  County {county}: {len(county_data)} 个节点")

print(f"\nCity数量: {len(city_counts)}")
print("前10个City节点数:")
for city, count in city_counts.head(10).items():
    print(f"  City {city}: {count} 个节点")
