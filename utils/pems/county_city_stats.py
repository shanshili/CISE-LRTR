import pandas as pd
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取传感器节点位置数据
location_file = os.path.join(script_dir, "location2.csv")

# 检查文件是否存在
if not os.path.exists(location_file):
    print(f"错误: 找不到文件 {location_file}")
else:
    # 读取数据
    df = pd.read_csv(location_file)

    print(f"读取到 {len(df)} 个传感器节点")

    # 统计 County 和 City 的数量及各自的节点数目
    county_counts = df['County'].value_counts().sort_index()
    city_counts = df['City'].value_counts().sort_index()

    # 准备输出内容
    output_lines = []
    output_lines.append("PeMS 传感器节点统计信息")
    output_lines.append("=" * 50)
    output_lines.append(f"总节点数: {len(df)}")
    output_lines.append(f"County 数量: {df['County'].nunique()}")
    output_lines.append(f"City 数量: {df['City'].nunique()}")
    output_lines.append("")

    output_lines.append("各 County 节点数目:")
    output_lines.append("-" * 30)
    for county, count in county_counts.items():
        output_lines.append(f"County {county}: {count} 个节点")
    output_lines.append("")

    output_lines.append("各 City 节点数目:")
    output_lines.append("-" * 30)
    for city, count in city_counts.items():
        output_lines.append(f"City {city}: {count} 个节点")

    # 打印统计信息到控制台
    for line in output_lines:
        print(line)

    # 保存统计信息到 txt 文件
    stats_output = os.path.join(script_dir, "county_city_statistics.txt")
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n统计信息已保存到: {stats_output}")
