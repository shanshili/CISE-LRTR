"""
元数据处理脚本说明:

输入:
- 数据文件: d07_text_meta_2023_12_22.txt
- 文件格式: 制表符分隔的文本文件，第一行为列名
- 数据结构: 每行代表一个监测站的元数据，包含ID、高速公路编号、方向、 county标识、
           城市、州里程标、绝对里程标、经纬度、长度、类型、车道数、名称等字段

功能:
- 解析元数据文件，第一行为列名，从第二行开始读取数据
- 提取关键字段: ID, Latitude, Longitude
- 生成简化的位置信息文件

输出:
- 文件路径: location.csv
- 文件格式: CSV格式，包含列标题
- 数据内容: 站点ID和对应的经纬度坐标
- 控制台输出: 处理统计信息和示例数据

用途: 提取PeMS监测站的地理位置信息，便于后续的空间分析和可视化
"""

import pandas as pd
import os
from tqdm import tqdm

# 需要提取的列
required_columns = [
    'ID',
    'County',
    'City',
    'Latitude',
    'Longitude'
]

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 输入文件路径
input_file = os.path.join(script_dir, "d07_text_meta_2023_12_22.txt")

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误: 找不到输入文件 {input_file}")
    print("请确保元数据文件位于脚本同一目录下")
else:
    print(f"找到元数据文件: {os.path.basename(input_file)}")

    try:
        # 读取元数据文件（制表符分隔，第一行为列名）
        print("正在读取元数据文件...")
        df = pd.read_csv(input_file, delimiter='\t')

        print(f"文件读取完成，数据形状: {df.shape}")
        print(f"  原始行数: {df.shape[0]}")
        print(f"  列数: {df.shape[1]}")

        # 显示列名
        print(f"\n列名: {list(df.columns)}")

        # 显示前几行数据示例
        print("\n前5行数据示例:")
        print(df.head())

        # 检查所需列是否存在于数据中
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        print(f"\n需要提取的列: {required_columns}")
        print(f"匹配的列: {available_columns}")
        if missing_columns:
            print(f"缺少的列: {missing_columns}")

        # 如果所有必需列都存在
        if len(available_columns) == len(required_columns):
            # 提取所需的列
            extracted_df = df[required_columns].copy()

            # 显示缺失值统计
            print(f"\n缺失值统计:")
            missing_info = []
            for col in required_columns:
                missing_count = extracted_df[col].isnull().sum()
                print(f"  {col}: {missing_count} 个缺失值")
                if missing_count > 0:
                    # 记录有缺失值的ID
                    missing_ids = extracted_df[extracted_df[col].isnull()]['ID'].tolist()
                    missing_info.append({
                        'column': col,
                        'count': missing_count,
                        'ids': missing_ids
                    })

            # 总体统计
            total_rows = len(extracted_df)
            rows_with_missing = extracted_df.isnull().any(axis=1).sum()
            rows_without_missing = total_rows - rows_with_missing

            print(f"\n数据质量统计:")
            print(f"  总行数: {total_rows}")
            print(f"  包含缺失值的行数: {rows_with_missing}")
            print(f"  完整行数: {rows_without_missing}")
            print(f"  数据完整率: {rows_without_missing / total_rows * 100:.2f}%")

            # 输出缺失数据的ID详情
            if missing_info:
                print(f"\n缺失数据详情:")
                for info in missing_info:
                    print(f"  {info['column']} 字段缺失的ID: {info['ids']}")

                # 将缺失数据的ID保存到文件
                missing_ids_output = os.path.join(script_dir, "missing_ids.txt")
                with open(missing_ids_output, 'w', encoding='utf-8') as f:
                    f.write("缺失数据的ID详情:\n")
                    f.write("=" * 50 + "\n")
                    for info in missing_info:
                        f.write(f"{info['column']} 字段缺失 ({info['count']} 个):\n")
                        f.write(f"{info['ids']}\n\n")
                print(f"\n缺失数据ID详情已保存到: {missing_ids_output}")

            # 默认行为：删除包含缺失值的行
            cleaned_df = extracted_df.dropna().copy()

            print(f"\n采用默认处理方式 (删除包含缺失值的行):")
            print(f"  删除前: {len(extracted_df)} 行")
            print(f"  删除后: {len(cleaned_df)} 行")
            print(f"  删除了: {len(extracted_df) - len(cleaned_df)} 行")

            # 检查ID重复情况
            duplicate_ids = cleaned_df.duplicated(subset=['ID']).sum()
            if duplicate_ids > 0:
                print(f"  发现 {duplicate_ids} 个重复的ID")
                # 保留第一个出现的重复项
                before_dedup = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates(subset=['ID'], keep='first')
                print(f"  去重后: {len(cleaned_df)} 行 (删除了 {before_dedup - len(cleaned_df)} 个重复项)")

            # 保存为CSV文件
            output_path = os.path.join(script_dir, "location2.csv")
            cleaned_df.to_csv(output_path, index=False)

            print(f"\n数据提取完成:")
            print(f"- 原始数据: {df.shape[0]} 行")
            print(f"- 提取字段: {', '.join(required_columns)}")
            print(f"- 有效数据: {len(cleaned_df)} 行")
            print(f"- 输出文件: {output_path}")

            # 显示最终数据示例
            print("\n" + "=" * 50)
            print("最终数据示例 (前10行)")
            print("=" * 50)
            print(cleaned_df.head(10))
            print("=" * 50)

            # 显示数据范围
            if len(cleaned_df) > 0:
                print(f"\n数据范围:")
                print(f"  ID 范围: {cleaned_df['ID'].min()} - {cleaned_df['ID'].max()}")
                print(f"  纬度范围: {cleaned_df['Latitude'].min()} - {cleaned_df['Latitude'].max()}")
                print(f"  经度范围: {cleaned_df['Longitude'].min()} - {cleaned_df['Longitude'].max()}")

        else:
            print(f"错误: 缺少必需的列，无法提取数据")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback

        print(f"详细错误信息: {traceback.format_exc()}")
