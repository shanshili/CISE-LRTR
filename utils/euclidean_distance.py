"""
euclidean_distance.py 代码功能说明

概述
该文件提供欧氏距离计算功能，用于计算平面或三维空间中两点之间的直线距离。

函数说明

1. compute_euclidean_matrix(positions, scale=1.0)
功能: 计算欧氏距离矩阵（平面距离）
参数:
  - positions: 节点坐标数组，形状为(N, 2)或(N, 3)
  - scale: 单位缩放系数（例如：1=米，0.001=千米）
返回: 距离矩阵，形状为(N, N)，单位与输入相同

计算原理
- 对于二维坐标: distance = sqrt((x1-x2)² + (y1-y2)²)
- 对于三维坐标: distance = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)

应用场景
- 平面坐标系统的距离计算
- 三维空间的距离测量
- 图形学和计算机视觉中的距离计算
"""
import numpy as np


def compute_euclidean_matrix(positions, scale=1.0):
    """
    计算欧氏距离矩阵（平面距离）

    参数:
        positions: 节点坐标数组，shape (N, 2) 或 (N, 3)
        scale: 单位缩放系数（例如：1=米，0.001=千米）

    返回:
        distance_matrix: 距离矩阵 (N, N)，单位与输入相同
    """
    N = positions.shape[0]
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff = positions[i] - positions[j]
            distance_matrix[i, j] = np.sqrt(np.sum(diff ** 2)) * scale

    return distance_matrix
