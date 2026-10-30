"""
概述
该文件提供Haversine距离计算功能，用于计算地球表面两点之间的球面距离。
适用于地理坐标（经纬度）的距离计算。

函数说明

1. haversine_distance(lat1, lon1, lat2, lon2, R=6371, unit='km')
功能: 计算两个经纬度点之间的Haversine距离
参数:
  - lat1, lon1: 第一个点的纬度和经度（度）
  - lat2, lon2: 第二个点的纬度和经度（度）
  - R: 地球半径，默认为6371千米
  - unit: 距离单位，可选'km'（千米）或'm'（米）
返回: 两点之间的距离（根据unit参数返回千米或米）

2. compute_haversine_matrix(coords)
功能: 计算所有点对之间的Haversine距离矩阵
参数:
  - coords: 坐标数组，形状为(N, 2)，每行包含[纬度, 经度]
返回: 对称的距离矩阵，形状为(N, N)，单位为千米

应用场景
- 地理信息系统中的距离计算
- 地球表面两点间的实际距离测量
- 空间数据分析中的距离矩阵构建
"""
import numpy as np
import os
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from tigramite import plotting as tp


def haversine_distance(lat1, lon1, lat2, lon2, R=6371,unit='km'):
    """
    计算两个经纬度点之间的 Haversine 距离（单位：千米）
    """
    # 转换为弧度
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # 差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine 公式
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = R * c
    # 根据要求返回单位
    if unit.lower() == 'm':
        return distance_km * 1000  # 转换为米
    else:
        return distance_km  # 默认返回千米


def compute_haversine_matrix(coords):
    """
    输入: coords = (N, 2) 的经纬度坐标数组
    输出: (N, N) 的 Haversine 距离矩阵（单位：千米）
    """
    N = coords.shape[0]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist = haversine_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 对称矩阵
    return dist_matrix



