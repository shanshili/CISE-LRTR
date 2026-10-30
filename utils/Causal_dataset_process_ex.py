"""
Causal_dataset_process.py 代码功能说明

概述
该文件用于处理因果发现算法（PCMCI和SCMI）的结果数据，、包括路径构建矩阵扩展、标准化、
组合以及可视化等功能。

函数说明

1. build_scmi_paths
功能: 构建SCMI算法数据文件的路径
参数:
  - n: 节点数
  - t: 时间序列长度
  - tau: 时间延迟
  - delta: delta参数
  - k: k参数
  - bins: bins参数
  - threshold: 阈值参数
  - algorithm_name: 算法文件夹名称
返回: SCMI因果图文件路径

2. build_pcmci_paths
功能: 构建PCMCI算法数据文件的路径
参数:
  - n: 节点数
  - t: 时间序列长度
  - tau: 时间延迟
  - alpha: 显著性水平
  - algorithm_name: 算法文件夹名称
返回: PCMCI结果文件路径

3. filter_pcmci_matrix_by_graph
功能: 根据graph中的连接关系过滤val_matrix，只保留有连接的元素
参数:
  - val_matrix: 原始PCMCI结果矩阵，形状为(n_nodes, n_nodes, tau_max+1)
  - graph: 因果关系图，形状为(n_nodes, n_nodes, tau_max+1)
返回: 过滤后的矩阵，只保留有连接的元素

4. expand_pcmci_matrix
功能: 将PCMCI的val_matrix时间展开为(400,400)的矩阵
参数:
  - val_matrix: 原始PCMCI结果矩阵
  - n_nodes: 节点数
  - tau_max: 最大时间延迟
返回: 展开后的矩阵，形状为(4*n_nodes, 4*n_nodes)

5. expand_scmi_matrix
功能: 改进版：SCMI仅在时间层对角线位置填充因果关系
参数:
  - scmi_matrix: SCMI原始矩阵
  - n_nodes: 节点数
  - tau_max: 最大时间延迟
返回: 扩展后的SCMI矩阵

6. normalize_matrix_with_sign
功能: 标准化矩阵保留正负号信息
参数: matrix: 输入矩阵
返回: 标准化后的矩阵（保留正负号）

7. normalize_matrix_positive_only
功能: 优化版：对SCMI仅考虑非零对角块的标准化
参数: matrix: 输入矩阵
返回: 标准化后的矩阵

8. weighted_sum_matrices_normalized
功能: 对展开后的PCMCI和SCMI矩阵进行标准化后加权求和
参数:
  - pcmci_expanded: 展开后的PCMCI矩阵
  - scmi_expanded: 展开后的SCMI矩阵
  - pcmci_weight: PCMCI权重
  - scmi_weight: SCMI权重
返回: 加权求和结果及各标准化矩阵

9. save_combined_matrix
功能: 保存加权求和后的结果矩阵
参数: 包括矩阵数据和各种参数
返回: 保存的文件路径

10. plot_matrix_heatmap
功能: 绘制矩阵热图，并在左上角第一块的上方、左方边框处标注节点序号
参数:
  - matrix: 要绘制的矩阵
  - title: 图标题
  - filename: 保存的文件名
  - n_nodes: 每个时间层的节点数
  - tau_max: 最大时间延迟
返回: PNG和SVG文件路径

主函数说明
主执行流程包含以下步骤：
1. 参数配置: 配置SCMI和PCMCI参数
2. 路径构建: 使用build_pcmci_paths和build_scmi_paths构建文件路径
3. 数据加载: 读取PCMCI和SCMI数据文件并进行处理
4. 矩阵处理: 对两个算法的结果进行标准化和加权求和
5. 可视化: 绘制各种矩阵的热图
6. 结果输出: 打印统计信息并保存结果

主要功能特点
- 算法融合: 将PCMCI和SCMI两种因果发现算法的结果进行融合
- 时间扩展: 将因果关系扩展到多个时间层
- 标准化处理: 对不同算法的输出进行标准化处理
- 可视化支持: 提供矩阵热图绘制功能
- 文件管理: 自动构建文件路径并管理结果保存

"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_scmi_paths(n, t, tau, Hubs, delta, k, bins, threshold, algorithm_name="SCMI"):
    """
    构建SCMI算法数据文件的路径

    Parameters:
    - n: 节点数
    - t: 时间序列长度
    - tau: 时间延迟
    - delta: delta参数
    - k: k参数
    - bins: bins参数
    - threshold: 阈值参数
    - algorithm_name: 算法文件夹名称
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 构建算法文件夹路径
    # folder_name = f"{algorithm_name}_N{n}_T{t}_t{tau}_delta{delta}_k{k}_bins{bins}"
    folder_name = f"{algorithm_name}_N{n}_T{t}_Hubs{Hubs}_delta{delta}_k{k}_bins{bins}"
    algorithm_dir = os.path.join(project_root, 'Module', folder_name)

    # 构建文件路径
    graph_filename = f"scmi_causal_graph_delta{delta}_k{k}_bins{bins}_threshold{threshold}.npy"
    graph_path = os.path.join(algorithm_dir, graph_filename)

    return graph_path


def build_pcmci_paths(n, t, tau, Hubs, alpha, algorithm_name="PCMCI_"):
    """
    构建PCMCI算法数据文件的路径

    Parameters:
    - n: 节点数
    - t: 时间序列长度
    - tau: 时间延迟
    - alpha: 显著性水平
    - algorithm_name: 算法文件夹名称
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 构建算法文件夹路径
    # folder_name = f"{algorithm_name}/N{n}_T{t}_t{tau}"
    folder_name = f"{algorithm_name}/N{n}_T{t}_Hubs{Hubs}_Viz_Final"
    algorithm_dir = os.path.join(project_root, 'Module', folder_name)

    # 构建文件路径
    results_filename = f"pcmci_results_tau{tau}_alpha{alpha}.pkl"
    results_path = os.path.join(algorithm_dir, results_filename)

    return results_path


def filter_pcmci_matrix_by_graph(val_matrix, graph):
    """
    根据graph中的连接关系过滤val_matrix，只保留有连接的元素

    Parameters:
    - val_matrix: 原始PCMCI结果矩阵，形状为(n_nodes, n_nodes, tau_max+1)
                 val_matrix[target, source, tau] 表示 source 在 t-tau 对 target 在 t 的因果效应
    - graph: 因果关系图，形状为(n_nodes, n_nodes, tau_max+1)

    Returns:
    - filtered_matrix: 过滤后的矩阵，只保留有连接的元素
    """
    # 创建过滤后的矩阵，初始化为0
    filtered_matrix = np.zeros_like(val_matrix)

    # 遍历所有元素，只保留graph中有连接的元素
    for target in range(val_matrix.shape[0]):  # 目标节点（结果）
        for source in range(val_matrix.shape[1]):  # 源节点（原因）
            for tau in range(val_matrix.shape[2]):  # 时间延迟
                # 如果graph中存在连接关系，则保留val_matrix中的值
                # graph[target, source, tau] == '-->' 表示 source 在 t-tau 对 target 在 t 有因果关系
                if graph[target, source, tau] == '-->':
                    filtered_matrix[target, source, tau] = val_matrix[target, source, tau]

    return filtered_matrix


def expand_pcmci_matrix(val_matrix, n_nodes, tau_max):
    """
    将PCMCI的val_matrix时间展开为(400,400)的矩阵
    修改为：横坐标是target(被影响)，纵坐标是source(影响)
    - 列表示目标时间层（被影响）- Target
    - 行表示源时间层（影响）- Source

    Parameters:
    - val_matrix: 原始PCMCI结果矩阵，形状为(n_nodes, n_nodes, tau_max+1)
                 val_matrix[target, source, tau] 表示 source 在 t-tau 对 target 在 t 的因果效应
    - n_nodes: 节点数（例如100）
    - tau_max: 最大时间延迟（例如3）

    Returns:
    - expanded_matrix: 展开后的矩阵，形状为(4*n_nodes, 4*n_nodes)
                      行: [t层节点, t-1层节点, t-2层节点, t-3层节点] - Source
                      列: [t层节点, t-1层节点, t-2层节点, t-3层节点] - Target
    """
    # 总时间层数
    n_time_layers = tau_max + 1  # t, t-1, t-2, t-3 => 4层
    expanded_size = n_time_layers * n_nodes
    expanded_matrix = np.zeros((expanded_size, expanded_size))

    # 填充展开矩阵
    # i: 源时间层索引 (行方向) - Source
    # j: 目标时间层索引 (列方向) - Target
    for i in range(n_time_layers):  # 源时间层 (行方向) - Source
        for j in range(n_time_layers):  # 目标时间层 (列方向) - Target
            # 计算时间延迟 tau = 目标时间层 - 源时间层
            # 例如: j=0(当前时刻), i=1(过去时刻) => tau = 0-1 = -1 => tau=1(延迟1个时间步)
            tau = j - i

            # 检查是否有对应的val_matrix数据 (tau应该是正数，表示过去的事件影响现在)
            if 0 <= tau <= tau_max:
                # 计算在展开矩阵中的位置
                row_start = i * n_nodes  # 源时间层的起始行 (Source)
                row_end = (i + 1) * n_nodes  # 源时间层的结束行 (Source)
                col_start = j * n_nodes  # 目标时间层的起始列 (Target)
                col_end = (j + 1) * n_nodes  # 目标时间层的结束列 (Target)

                # 将val_matrix[target, source, tau]填入对应位置
                # 注意：val_matrix[target, source, tau] 中的source和target需要交换位置以适应新的坐标系
                expanded_matrix[row_start:row_end, col_start:col_end] = val_matrix[:, :, tau].T

    return expanded_matrix


def expand_scmi_matrix(scmi_matrix, n_nodes, tau_max):
    """
    改进版：SCMI仅在时间层对角线位置填充因果关系
    """
    n_time_layers = tau_max + 1
    expanded_size = n_time_layers * n_nodes
    expanded_matrix = np.zeros((expanded_size, expanded_size))

    # 仅在对角块填充SCMI矩阵
    for t in range(n_time_layers):  # t表示时间层索引
        row_start = t * n_nodes
        row_end = (t + 1) * n_nodes
        col_start = t * n_nodes
        col_end = (t + 1) * n_nodes

        # 填充SCMI矩阵的转置到对角块
        expanded_matrix[row_start:row_end, col_start:col_end] = scmi_matrix.T

    return expanded_matrix


def normalize_matrix_with_sign(matrix):
    """
    标准化矩阵保留正负号信息
    对于有正负值的矩阵(如PCMCI)，使用最大绝对值进行标准化
    """
    # 获取非零元素
    non_zero_elements = matrix[matrix != 0]

    if len(non_zero_elements) == 0:
        # 如果所有元素都是0，返回原矩阵
        return matrix

    # 获取最大绝对值
    max_abs_val = np.max(np.abs(non_zero_elements))

    if max_abs_val == 0:
        return matrix

    # 标准化：除以最大绝对值
    normalized = np.where(matrix != 0, matrix / max_abs_val, 0.0)
    return normalized


def normalize_matrix_positive_only(matrix):
    """
    优化版：对SCMI仅考虑非零对角块的标准化
    """
    # 提取对角块数据
    n_time_layers = matrix.shape[0] // matrix.shape[1]  # 假设矩阵是方阵
    n_nodes = matrix.shape[0] // n_time_layers
    diag_blocks = np.vstack([
        matrix[i * n_nodes:(i + 1) * n_nodes, i * n_nodes:(i + 1) * n_nodes]
        for i in range(n_time_layers)
    ])

    non_zero_elements = diag_blocks[diag_blocks != 0]
    if len(non_zero_elements) == 0:
        return matrix

    min_val = np.min(non_zero_elements)
    max_val = np.max(non_zero_elements)

    if max_val == min_val:
        normalized = np.where(matrix != 0, 1.0, 0.0)
        return normalized

    # 仅对对角块标准化
    for t in range(n_time_layers):
        row_start = t * n_nodes
        row_end = (t + 1) * n_nodes
        col_start = t * n_nodes
        col_end = (t + 1) * n_nodes

        block = matrix[row_start:row_end, col_start:col_end]
        matrix[row_start:row_end, col_start:col_end] = np.where(
            block != 0,
            (block - min_val) / (max_val - min_val),
            0.0
        )

    return matrix


def weighted_sum_matrices_normalized(pcmci_expanded, scmi_expanded, pcmci_weight=0.5, scmi_weight=0.5):
    """
    对展开后的PCMCI和SCMI矩阵进行标准化后加权求和
    """
    # 标准化两个矩阵，保留PCMCI的正负信息
    pcmci_normalized = normalize_matrix_with_sign(pcmci_expanded)
    scmi_normalized = normalize_matrix_positive_only(scmi_expanded)

    # 确保权重之和为1
    total_weight = pcmci_weight + scmi_weight
    if total_weight > 0:
        pcmci_weight /= total_weight
        scmi_weight /= total_weight

    # 加权求和
    weighted_sum = pcmci_normalized +scmi_normalized
    return weighted_sum, pcmci_normalized, scmi_normalized


def save_combined_matrix(matrix, n, t, tau, pcmci_alpha, scmi_delta, scmi_k,
                         scmi_bins, scmi_threshold, pcmci_weight, scmi_weight):
    """
    保存加权求和后的结果矩阵

    Parameters:
    - matrix: 要保存的矩阵
    - n, t, tau: 参数
    - pcmci_alpha: PCMCI的alpha参数
    - scmi_delta, scmi_k, scmi_bins, scmi_threshold: SCMI参数
    - pcmci_weight, scmi_weight: 权重参数
    """
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 创建保存目录
    save_dir = os.path.join(project_root, 'Combined_Results_ex/n1000')
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名
    filename = f"combined_causal_matrix_N{n}_T{t}_t{tau}_" \
               f"pcmci_alpha{pcmci_alpha}_" \
               f"scmi_delta{scmi_delta}_k{scmi_k}_bins{scmi_bins}_threshold{scmi_threshold}_" \
               f"pcmci_w{pcmci_weight}_scmi_w{scmi_weight}.npy"

    # 保存文件
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, matrix)
    print(f"组合矩阵已保存至: {save_path}")
    return save_path


def plot_matrix_heatmap(matrix, title, filename, n_nodes=100, tau_max=3):
    """
    绘制矩阵热图，并在左上角第一块的上方、左方边框处标注节点序号
    修改坐标轴标注：横坐标是target(被影响)，纵坐标是source(影响)

    Parameters:
    - matrix: 要绘制的矩阵
    - title: 图标题
    - filename: 保存的文件名
    - n_nodes: 每个时间层的节点数
    - tau_max: 最大时间延迟
    """
    # 创建图形和轴
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 设置背景为白色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 打印矩阵统计信息
    print(f"{title}统计信息:")
    print(f"  最小值: {np.min(matrix)}")
    print(f"  最大值: {np.max(matrix)}")
    print(f"  平均值: {np.mean(matrix)}")
    print(f"  非零元素数量: {np.count_nonzero(matrix)}")

    # 根据矩阵是否有负值选择颜色映射
    has_negative = np.min(matrix) < 0
    if has_negative:
        # 有负值时使用RdBu颜色映射，中心为白色
        im = ax.imshow(matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
    else:
        # 只有正值时使用hot_r颜色映射
        im = ax.imshow(matrix, cmap='hot_r', interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Causal Strength', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)

    # 设置标题
    ax.set_title(title, fontsize=16, pad=20)

    # 添加分块边框
    n_time_layers = tau_max + 1
    for i in range(n_time_layers):
        for j in range(n_time_layers):
            # 计算矩形位置
            x_start = j * n_nodes
            y_start = i * n_nodes
            width = n_nodes
            height = n_nodes

            # 创建并添加矩形边框
            rect = patches.Rectangle(
                (x_start - 0.5, y_start - 0.5),
                width,
                height,
                linewidth=1,
                edgecolor='black',
                facecolor='none'
            )
            ax.add_patch(rect)

    # 设置坐标轴标签 (明确因果关系方向)
    time_labels = ['t', 't-1', 't-2', 't-3']

    # x轴：目标时间层 (Effect/Target) - 列方向
    x_ticks = [i * n_nodes + n_nodes // 2 for i in range(n_time_layers)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel('Target Time Layers (Effect)', fontsize=12)

    # y轴：源时间层 (Cause/Source) - 行方向
    y_ticks = [i * n_nodes + n_nodes // 2 for i in range(n_time_layers)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(time_labels)
    ax.set_ylabel('Source Time Layers (Cause)', fontsize=12)

    # 只在左上角第一块的上方、左方边框处标注节点序号
    step = 20  # 每20个节点标注一次，避免过于密集

    # 在上方边框处标注列节点序号（x轴方向 - 目标节点）
    for j in range(0, n_nodes, step):
        # 标注在左上角块的上方边框
        x_pos = j
        y_pos = -5  # 标注在上方边框外侧
        ax.text(x_pos, y_pos, str(j),
                verticalalignment='bottom', horizontalalignment='center',
                fontsize=6, color='black')

    # 在左方边框处标注行节点序号（y轴方向 - 源节点）
    for i in range(0, n_nodes, step):
        # 标注在左上角块的左方边框
        x_pos = -5  # 标注在左方边框外侧
        y_pos = i
        ax.text(x_pos, y_pos, str(i),
                verticalalignment='center', horizontalalignment='right',
                fontsize=6, color='black')

    # 添加网格线
    ax.set_xticks([i * n_nodes for i in range(n_time_layers + 1)], minor=True)
    ax.set_yticks([i * n_nodes for i in range(n_time_layers + 1)], minor=True)
    ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_dir = os.path.join(project_root, 'Combined_Results_ex/n1000')
    os.makedirs(save_dir, exist_ok=True)

    # 保存为PNG格式（高分辨率）
    png_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"{title} PNG热图已保存至: {png_path}")

    # 保存为SVG格式
    svg_path = os.path.join(save_dir, f"{filename}.svg")
    plt.savefig(svg_path, dpi=600, format='svg', bbox_inches='tight', facecolor='white')
    print(f"{title} SVG热图已保存至: {svg_path}")

    # 显示图像
    plt.show()

    return png_path, svg_path


if __name__ == "__main__":
    # 示例使用：根据参数自由定制
    # SCMI参数
    scmi_params = {
         'n': 1000,
         't': 8000,
         'tau': 3,
    #     'Hubs': 15,
         'delta': 10.0,
         'k': 5,
         'bins': 10,
         'threshold': 0.25
     }
    #
    # PCMCI参数
    pcmci_params = {
         'n': 1000,
         't': 8000,
         'tau': 3,
    #     'Hubs': 15,
         'alpha': '0p0001'  # 注意这里用'0p0001'代替'0.0001'以匹配文件名
     }
    # 直接指定文件路径
    pcmci_file_path = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\PCMCI_ex\N1000_T10000\pcmci_results_tau3_alpha0p0001.pkl"
    scmi_file_path = r"C:\Work-lcr\Tjnu-p\Mp\GNN-Causal-Inference\Module\\SCMI_ex\N1000_delta15\scmi_causal_graph_delta10.0_k5_bins10_threshold0.25.npy"

    # 权重参数
    weights = {
        'pcmci_weight': 0.5,
        'scmi_weight': 0.5
    }

    # 构建路径
    # pcmci_file_path = build_pcmci_paths(
    #     pcmci_params['n'],
    #     pcmci_params['t'],
    #     pcmci_params['tau'],
    #     pcmci_params['Hubs'],
    #     pcmci_params['alpha']
    # )
    #
    # scmi_file_path = build_scmi_paths(
    #     scmi_params['n'],
    #     scmi_params['t'],
    #     scmi_params['tau'],
    #     scmi_params['Hubs'],
    #     scmi_params['delta'],
    #     scmi_params['k'],
    #     scmi_params['bins'],
    #     scmi_params['threshold']
    # )

    # 初始化变量
    pcmci_expanded = None
    scmi_expanded = None

    # 检查并读取PCMCI结果
    if os.path.exists(pcmci_file_path):
        with open(pcmci_file_path, 'rb') as f:
            loaded = pickle.load(f)
        print(f"PCMCI键值：{loaded.keys()}")  # 查看所有键
        print(f"PCMCI val_matrix:{loaded['val_matrix'].shape}")  # (100, 100, 4)

        # 获取val_matrix和graph
        val_matrix = loaded['val_matrix']
        graph = loaded['graph']

        # 根据graph过滤val_matrix，只保留有连接的元素
        filtered_val_matrix = filter_pcmci_matrix_by_graph(val_matrix, graph)

        print(f"过滤前非零元素数量: {np.count_nonzero(val_matrix)}")
        print(f"过滤后非零元素数量: {np.count_nonzero(filtered_val_matrix)}")

        # 显示一些示例因果关系
        print("\n示例因果关系 (前5个):")
        count = 0
        for target in range(min(5, val_matrix.shape[0])):
            for source in range(min(5, val_matrix.shape[1])):
                for tau in range(min(2, val_matrix.shape[2])):
                    if graph[target, source, tau] == '-->':
                        print(f"  节点{source}(Source)在t-{tau} → 节点{target}(Target)在t (效应值: {val_matrix[target, source, tau]:.4f})")
                        count += 1
                        if count >= 5:
                            break
                if count >= 5:
                    break
            if count >= 5:
                break

        # 展开过滤后的PCMCI矩阵
        pcmci_expanded = expand_pcmci_matrix(filtered_val_matrix, pcmci_params['n'], pcmci_params['tau'])
        print(f"展开后矩阵形状: {pcmci_expanded.shape}")  # (400, 400)

        # 验证展开矩阵的因果关系方向
        print(f"\n展开矩阵维度说明:")
        print(f"  行方向 (Target/被影响): [t层, t-1层, t-2层, t-3层]")
        print(f"  列方向 (Source/影响): [t层, t-1层, t-2层, t-3层]")
        print(f"  例如: 子块(0,1)表示 t层节点(目标) 受 t-1层节点(源) 的影响")

    else:
        print(f"PCMCI文件不存在: {pcmci_file_path}")

    # 检查并读取SCMI因果图
    if os.path.exists(scmi_file_path):
        scmi_graph = np.load(scmi_file_path)
        print(f"SCMI原始矩阵形状: {scmi_graph.shape}")  # (100, 100)

        # 扩展SCMI矩阵
        scmi_expanded = expand_scmi_matrix(scmi_graph, scmi_params['n'], scmi_params['tau'])
        print(f"SCMI扩展后矩阵形状: {scmi_expanded.shape}")  # (400, 400)

    else:
        print(f"SCMI文件不存在: {scmi_file_path}")

    # 如果两个矩阵都成功加载，则进行加权求和
    if pcmci_expanded is not None and scmi_expanded is not None:
        # 加权求和
        combined_matrix, pcmci_normalized, scmi_normalized = weighted_sum_matrices_normalized(
            pcmci_expanded,
            scmi_expanded,
            weights['pcmci_weight'],
            weights['scmi_weight']
        )

        print(f"组合矩阵形状: {combined_matrix.shape}")  # (400, 400)
        print(f"组合矩阵统计信息:")
        print(f"  最小值: {np.min(combined_matrix)}")
        print(f"  最大值: {np.max(combined_matrix)}")
        print(f"  平均值: {np.mean(combined_matrix)}")
        print(f"  非零元素数量: {np.count_nonzero(combined_matrix)}")

        # 保存结果
        save_path = save_combined_matrix(
            combined_matrix,
            scmi_params['n'],
            scmi_params['t'],
            scmi_params['tau'],
            pcmci_params['alpha'],
            scmi_params['delta'],
            scmi_params['k'],
            scmi_params['bins'],
            scmi_params['threshold'],
            weights['pcmci_weight'],
            weights['scmi_weight']
        )

        print(f"标准化后PCMCI矩阵统计信息:")
        print(f"  最小值: {np.min(pcmci_normalized)}")
        print(f"  最大值: {np.max(pcmci_normalized)}")
        print(f"  平均值: {np.mean(pcmci_normalized)}")
        print(f"  是否包含负值: {np.min(pcmci_normalized) < 0}")

        print(f"标准化后SCMI矩阵统计信息:")
        print(f"  最小值: {np.min(scmi_normalized)}")
        print(f"  最大值: {np.max(scmi_normalized)}")
        print(f"  平均值: {np.mean(scmi_normalized)}")
        print(f"  是否包含负值: {np.min(scmi_normalized) < 0}")

        # 绘制热图
        plot_matrix_heatmap(
            combined_matrix,
            f"Combined Causal Matrix Heatmap (PCMCI weight:{weights['pcmci_weight']}, SCMI weight:{weights['scmi_weight']})",
            "combined_causal_matrix_heatmap",
            n_nodes=scmi_params['n'],
            tau_max=scmi_params['tau']
        )
        # 分别绘制扩展后的PCMCI和SCMI热图
        plot_matrix_heatmap(
            pcmci_expanded,
            "Expanded PCMCI Matrix (Filtered by Graph)",
            "expanded_pcmci_matrix",
            n_nodes=scmi_params['n'],
            tau_max=scmi_params['tau']
        )

        plot_matrix_heatmap(
            scmi_expanded,
            "Expanded SCMI Matrix",
            "expanded_scmi_matrix",
            n_nodes=scmi_params['n'],
            tau_max=scmi_params['tau']
        )
        # 分别绘制标准化后的PCMCI和SCMI热图
        plot_matrix_heatmap(
            pcmci_normalized,
            "Normalized PCMCI Matrix (Preserving Sign)",
            "normalized_pcmci_matrix",
            n_nodes=scmi_params['n'],
            tau_max=scmi_params['tau']
        )
        plot_matrix_heatmap(
            scmi_normalized,
            "Normalized SCMI Matrix",
            "normalized_scmi_matrix",
            n_nodes=scmi_params['n'],
            tau_max=scmi_params['tau']
        )

    else:
        print("无法生成组合矩阵，因为缺少必要的输入数据")
