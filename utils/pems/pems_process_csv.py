"""
数据处理脚本说明:

输入:
- 数据文件: extracted 目录下的所有.txt文件
- 文件格式: 无列标题的CSV格式，使用逗号分隔符
- 数据结构: 每行代表监测站5分钟交通数据，包含时间戳、监测站ID、区域编号、
           高速公路编号、行车方向、车道类型、路段长度、样本数、观测比例、
           总流量、平均占有率、平均速度等字段，以及可选的多车道数据

功能:
- 批量处理: 自动查找并处理目录下所有txt文件
- 数据解析: 解析无标题行CSV数据并根据文档分配列名
- 字段提取: 从原始数据中提取7个关键字段:
  * Timestamp: 数据时间戳
  * Station: 监测站ID
  * District: 区域编号
  * Freeway #: 高速公路编号
  * Direction of Travel: 行车方向
  * Total Flow: 总流量
  * Station Length: 监测站路段长度
- 进度追踪: 显示处理进度条和详细处理信息
- 错误处理: 捕获并显示处理过程中的异常
- 中间验证: 处理前几个文件后显示示例数据验证正确性

输出:
- 文件路径: extracted_data.csv
- 文件格式: 标准CSV格式，包含列标题
- 数据内容: 合并所有输入文件中提取的指定字段数据
- 控制台输出: 处理统计信息、中间和最终数据示例、错误警告信息

用途: 将PeMS交通监测数据从原始格式转换为简化格式，便于后续分析处理
"""

import pandas as pd
import os
import glob
from tqdm import tqdm

# 定义实际的列名（根据文档说明）
actual_columns = [
    'Timestamp',
    'Station',
    'District',
    'Freeway #',
    'Direction of Travel',
    'Lane Type',
    'Station Length',
    'Samples',
    '% Observed',
    'Total Flow',
    'Avg Occupancy',
    'Avg Speed'
    # 后面还有车道相关的列，根据数据看是多车道的数据
]
# 根据你提供的示例数据，看起来还有多条车道的数据
# 我们先处理前12列，这些是固定的
# 需要提取的列
required_columns = [
    'Timestamp',
    'Station',
    'District',
    'Freeway #',
    'Direction of Travel',
    'Total Flow',
    'Station Length'
]

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 或者使用固定的目录
extracted_dir = r"D:\Tjnu-p\Mp\PeMSD7\hour"

# 查找脚本目录下所有txt文件
txt_files = glob.glob(os.path.join(extracted_dir, "*.txt"))

print(f"找到 {len(txt_files)} 个txt文件")

if len(txt_files) == 0:
    print("警告: 没有找到txt文件，请检查路径是否正确")
else:
    print("前3个文件:")
    for i, file in enumerate(txt_files[:3]):
        print(f"  {i + 1}. {os.path.basename(file)}")

# 创建一个空的DataFrame来存储所有数据
combined_data = pd.DataFrame()

# 计数器，用于跟踪处理的文件数量
file_count = 0

# 使用tqdm创建进度条，遍历所有txt文件
for file_path in tqdm(txt_files, desc="处理文件进度", unit="文件"):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None)

        # 显示前几个文件的信息，帮助诊断
        if file_count < 3:
            tqdm.write(f"\n文件 {os.path.basename(file_path)} 的数据形状: {df.shape}")
            tqdm.write(f"  列数: {df.shape[1]}")
            tqdm.write(f"  行数: {df.shape[0]}")
            tqdm.write("  前几列的数据示例:")
            for i in range(min(12, df.shape[1])):
                tqdm.write(f"    列 {i}: {df.iloc[0, i]}")

        # 根据文档说明，为数据指定列名
        # 先处理前12列（固定列）
        base_columns = [
            'Timestamp',  # 0
            'Station',  # 1
            'District',  # 2
            'Freeway #',  # 3
            'Direction of Travel',  # 4
            'Lane Type',  # 5
            'Station Length',  # 6
            'Samples',  # 7
            '% Observed',  # 8
            'Total Flow',  # 9
            'Avg Occupancy',  # 10
            'Avg Speed'  # 11
        ]

        # 如果列数超过12，说明有多车道数据
        lane_columns = []
        if df.shape[1] > 12:
            lane_count = (df.shape[1] - 12) // 5  # 每条车道有5个数据列
            for i in range(lane_count):
                lane_columns.extend([
                    f'Lane {i + 1} Samples',
                    f'Lane {i + 1} Flow',
                    f'Lane {i + 1} Avg Occ',
                    f'Lane {i + 1} Avg Speed',
                    f'Lane {i + 1} Observed'
                ])
        # 合并所有列名
        all_columns = base_columns + lane_columns

        # 如果实际列数与推断的列数不匹配，调整列名列表
        if len(all_columns) != df.shape[1]:
            # 如果列数不匹配，只使用基础列名，并为多余列添加默认名称
            if len(all_columns) < df.shape[1]:
                all_columns = all_columns + [f'Unnamed: {i}' for i in range(len(all_columns), df.shape[1])]
            else:
                all_columns = all_columns[:df.shape[1]]

        # 为DataFrame分配列名
        df.columns = all_columns

        if file_count < 3:
            tqdm.write(f"  分配的列名: {all_columns[:min(12, len(all_columns))]}")

            # 检查所需列是否存在于当前文件中
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        # 显示前几个文件的列匹配情况
        if file_count < 3:
            tqdm.write(f"  匹配的列: {available_columns}")
            if missing_columns:
                tqdm.write(f"  缺少的列: {missing_columns}")

        # 如果所有必需列都存在
        if len(available_columns) == len(required_columns):
            extracted_df = df[required_columns]
            combined_data = pd.concat([combined_data, extracted_df], ignore_index=True)
            if file_count < 3:
                tqdm.write(f"  成功提取所有列")
        # 如果部分列存在，也可以选择性处理
        elif len(available_columns) > 0:
            extracted_df = df[available_columns]
            combined_data = pd.concat([combined_data, extracted_df], ignore_index=True)
            if file_count < 3:
                tqdm.write(f"  部分提取列: {available_columns}")
        else:
            if file_count < 3:
                tqdm.write(f"  未找到任何需要的列，跳过该文件")

        # 增加文件计数器
        file_count += 1

        # 处理3个文件后打印示例数据
        if file_count == 3 and len(combined_data) > 0:
            tqdm.write("\n" + "=" * 60)
            tqdm.write("前3个文件处理示例")
            tqdm.write("=" * 60)
            tqdm.write("前10行数据:")
            tqdm.write(str(combined_data.head(10)))
            tqdm.write(f"\n当前数据形状: {combined_data.shape}")
            tqdm.write("=" * 60 + "\n")

    except Exception as e:
        tqdm.write(f"\n处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        # 打印更多错误信息帮助诊断
        import traceback
        tqdm.write(f"详细错误信息: {traceback.format_exc()}")

# 保存为新的CSV文件（保存在脚本所在目录）
output_path = os.path.join(script_dir, "extracted_data.csv")

if len(combined_data) > 0:
    combined_data.to_csv(output_path, index=False)
    tqdm.write(f"\n数据提取完成:")
    tqdm.write(f"- 总共处理了 {len(combined_data)} 行数据")
    tqdm.write(f"- 提取的字段: {', '.join(required_columns)}")
    tqdm.write(f"- 输出文件保存在: {output_path}")

    # 最终数据示例
    tqdm.write("\n" + "=" * 50)
    tqdm.write("最终数据示例")
    tqdm.write("=" * 50)
    tqdm.write("前5行数据:")
    tqdm.write(str(combined_data.head()))
else:
    tqdm.write(f"\n警告: 没有成功提取到数据 (总共处理了0行数据)")
    tqdm.write("可能的原因:")
    tqdm.write("1. 文件格式与预期不符")
    tqdm.write("2. 列名与预期不匹配")
    tqdm.write("3. 文件路径中没有txt文件")
    tqdm.write("4. 文件权限问题")
