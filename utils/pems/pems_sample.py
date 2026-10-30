"""
传感器节点分布图绘制脚本

功能:
1. 读取location.csv文件中的传感器节点坐标
2. 绘制完整的传感器节点分布图
3. 按County区分颜色绘制节点分布图
4. 按City区分颜色绘制节点分布图

输入:
- location.csv: 包含传感器节点ID和坐标的文件

输出:
- sensor_distribution_full.png: 完整传感器分布图
- sensor_distribution_by_county.png: 按County区分的传感器分布图
- sensor_distribution_by_city.png: 按City区分的传感器分布图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
