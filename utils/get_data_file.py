"""
概述
该文件提供文件路径查找功能，用于定位项目中的数据文件和结果文件。
支持动态查找因果图、相关图和特征数据文件。

函数说明

1. get_project_root()
功能: 获取项目根目录路径
返回: 项目根目录的Path对象

2. find_causal_adj_file(project_root, params)
功能: 在Combined_Results目录下查找因果图邻接矩阵文件
参数:
  - project_root: 项目根目录
  - params: 参数字典，包含N, T, t等参数
返回: 找到的因果图邻接矩阵文件路径

3. find_corr_adj_file(project_root)
功能: 在Correlation_Graph目录下查找相关图邻接矩阵文件
参数:
  - project_root: 项目根目录
返回: 相关图邻接矩阵文件路径

4. find_features_file(project_root, params)
功能: 在dataset目录下查找特征文件
参数:
  - project_root: 项目根目录
  - params: 参数字典，包含N, T, t等参数
返回: 特征文件路径

查找规则
1. 因果图文件: 在Combined_Results目录下按参数模式匹配
2. 相关图文件: 在Correlation_Graph目录下的固定文件名
3. 特征文件: 在dataset/Causal_simulation_data_2目录下按参数路径查找
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
def get_project_root():
    """
    获取项目根目录路径
    """
    # 获取当前脚本的目录
    current_dir = Path(__file__).parent.resolve()

    # 向上查找直到找到项目根目录（包含特定标识的目录）
    # 这里假设项目根目录包含requirements.txt或.git目录
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent

    # 如果没有找到标识，返回当前目录的父目录
    return current_dir.parent


def find_causal_adj_file(project_root, params):
    """
    在Combined_Results目录下查找因果图邻接矩阵文件

    Args:
        project_root: 项目根目录
        params: 参数字典，包含N, T, t等参数

    Returns:
        因果图邻接矩阵文件路径
    """
    combined_results_dir = project_root / "Combined_Results/N100_T10000_Hubs15_t3_Viz_Final_2"
    if not combined_results_dir.exists():
        raise FileNotFoundError(f"Combined_Results directory not found: {combined_results_dir}")

    # 构建文件名模式
    pattern = f"combined_causal_matrix_N{params['N']}_T{params['T']}_t{params['t']}_*"

    # 查找匹配的文件
    matching_files = list(combined_results_dir.glob(pattern + ".npy"))
    if not matching_files:
        # 如果没有精确匹配，尝试更宽松的匹配
        pattern_relaxed = f"*N{params['N']}*T{params['T']}*t{params['t']}*.npy"
        matching_files = list(combined_results_dir.glob(pattern_relaxed))

    if not matching_files:
        raise FileNotFoundError(
            f"No causal adjacency matrix found in {combined_results_dir} matching pattern: {pattern}")

    # 如果有多个匹配文件，返回第一个
    return matching_files[0]


def find_corr_adj_file(project_root):
    """
    在Correlation_Graph目录下查找相关图邻接矩阵文件

    Args:
        project_root: 项目根目录

    Returns:
        相关图邻接矩阵文件路径
    """
    corr_graph_dir = project_root / "Correlation_Graph"
    if not corr_graph_dir.exists():
        raise FileNotFoundError(f"Correlation_Graph directory not found: {corr_graph_dir}")

    corr_adj_path = corr_graph_dir / "corr_adjacency_matrix.npy"
    if not corr_adj_path.exists():
        raise FileNotFoundError(f"Correlation adjacency matrix not found: {corr_adj_path}")

    return corr_adj_path


def find_features_file(project_root, params):
    """
    在dataset目录下查找特征文件

    Args:
        project_root: 项目根目录
        params: 参数字典，包含N, T, t等参数

    Returns:
        特征文件路径
    """
    dataset_dir = project_root / "dataset" / f"simulation_hubs_core/N100_T10000_Hubs15_Viz_Final"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    features_path = dataset_dir / "Env_A_timeseries.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    return features_path