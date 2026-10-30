# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tigramite
from sympy.codegen.cnodes import sizeof
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.toymodels import surrogate_generator
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import Models, Prediction
import math
import sklearn
from sklearn.linear_model import LinearRegression


#正常显示坐标轴负号
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(14)     # Fix random seed
lin_f = lambda x: x
"""
因果关系定义
Define the links and coefficients for the structural causal process
"""
links_coeffs = {
                0: [((0, -1), 0.7, lin_f)],
                # X⁰_t = 0.7 * X⁰_{t-1} + ε⁰_t  表示变量X⁰在时间t的值取决于它自己滞后1期的值（X⁰_{t-1}）和一个噪声项ε⁰_t
                1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, lin_f)],
                # X¹_t = 0.8 * X¹_{t-1} + 0.3 * X⁰_{t-1} + ε¹_t
                # 表示变量X¹在时间t的值取决于它自己滞后1期的值（X¹_{t-1}）、变量X⁰滞后1期的值（X⁰_{t-1}）和一个噪声项ε¹_t
                2: [((2, -1), 0.5, lin_f), ((0, -2), -0.5, lin_f)],
                3: [((3, -1), 0., lin_f)], #, ((4, -1), 0.4, lin_f)],
                4: [((4, -1), 0., lin_f), ((3, 0), 0.5, lin_f)], #, ((3, -1), 0.3, lin_f)],
                }
T = 200     # time series length
# Make some noise with different variance, alternatively just noises=None
# 结构性因果过程（structural causal process）的扰动项
noises = np.array([(1. + 0.2*float(j))*np.random.randn((T + int(math.floor(0.2*T))))
                   for j in range(len(links_coeffs))]).T


"""
时间序列生成
Generate the structural causal process

该函数根据给定的结构方程模型（SEM）定义生成时间序列数据。它实现了结构性因果过程（Structural Causal Process, SCP），
按照指定的因果关系和噪声项生成符合这些关系的合成时间序列数据。

links_coeffs: 因果关系定义，包含每个变量的父节点及其对应的系数和函数
T: 时间序列长度（200）
noises: 噪声数据，形状为(240,5)，具有不同方差的高斯噪声
seed: 随机种子（14），确保结果可复现
"""

data, _ = toys.structural_causal_process(links_coeffs, T=T, noises=noises, seed=14)
T, N = data.shape

# For generality, we include some masking
# mask = np.zeros(data.shape, dtype='int')
# mask[:int(T/2)] = True
mask=None

"""
时间序列图生成（观测数据）
"""
# Initialize dataframe object, specify time axis and variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
dataframe = pp.DataFrame(data,
                         mask=mask,
                         datatime = {
            0:np.arange(len(data))},
                         var_names=var_names)
tp.plot_timeseries(dataframe=dataframe) # 将生成的数据可视化为时间序列图
plt.suptitle("Observed Time Series Data", fontsize=16)
plt.tight_layout()
# plt.savefig('timeseries_plot.png', dpi=300, bbox_inches='tight')
# plt.show()


"""
*****************************************************************
滞后函数图
"""
tau_max = 10
parcorr = ParCorr(significance='analytic',
#                   mask_type='y'
                 )
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=0)
original_correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']

# 先创建一个空白画布
lag_func_matrix = tp.setup_matrix(
    N=N,
    tau_max=tau_max,
    x_base=5,
    figsize=(10, 10),
    var_names=var_names,
    y_base=5
)

# 然后手动添加滞后函数曲线
lag_func_matrix.add_lagfuncs(
    val_matrix=original_correlations,
    color='black'
)

# 添加总标题
plt.suptitle("Lagged Correlation Functions", fontsize=16, y=0.98)
# lag_func_matrix.savefig('lag_functions_plot.png')

"""
*****************************************************************
推测因果图生成
不是真实的因果图，而是通过PCMCI+算法从观测数据中学习到的因果图
"""
results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.01)
print(results['val_matrix'].shape)

graph = results['graph']
val_matrix = results['val_matrix']

edge_counts = {}
for i in range(N):
    for j in range(N):
        # if i == j:  # 排除自关联
        #     continue
        count = 0
        for tau in range(0, tau_max + 1):
            if graph[i, j, tau] == '-->' or graph[i, j, tau] == 'o-o' :  # 节点 i 在时间 t-τ 时刻对节点 j 在时间 t 时刻有因果影响
                print(graph[i, j, tau])
                print(f"节点{i}与{j}因果效应{val_matrix[i, j, tau]}")
                count += 1
        if count > 0:
            edge_counts[(i, j)] = count
            print(f"节点{i}与{j}链接{count}条")
"""
-->
节点0与0因果效应0.5578951835632324
节点0与0链接1条
-->
节点0与1因果效应0.27064645290374756
节点0与1链接1条
-->
节点0与2因果效应-0.3540237247943878
节点0与2链接1条
-->
节点1与1因果效应0.6784783601760864
节点1与1链接1条
-->
节点2与2因果效应0.4284769892692566
节点2与2链接1条
o-o
节点3与4因果效应0.3270717263221741
节点3与4链接1条
o-o
节点4与3因果效应0.3270717263221741
节点4与3链接1条
"""


"""
直线链接表示同期依赖关系， o-o
曲线箭头表示滞后依赖关系。
"""
tp.plot_graph(results['graph'],
              val_matrix=results['val_matrix'],
              var_names=var_names,
              link_colorbar_label='MCI',
              node_colorbar_label='auto-MCI',
              link_label_fontsize=14,
              label_fontsize=14,
              tick_label_size=14,
              node_label_size=14,
              edge_ticks=0.5,
              node_ticks=0.5,
              node_size=0.3
              )
# 添加标题
plt.title("Estimated Causal Graph using PCMCI+", fontsize=14)
# plt.savefig('causal_graph.png', dpi=300, bbox_inches='tight')
# plt.show()

# tp.plot_time_series_graph(
#         figsize=(4, 4),
#         val_matrix=results['val_matrix'],
#         graph = results['graph'],
#         var_names=var_names,
#         link_colorbar_label='MCI',
#         label_fontsize=20,
#         tick_label_size=20
# )
# # 添加标题
# plt.title("Estimated Time Series Causal Graph using PCMCI+", fontsize=14)

"""
*****************************************************************
"""
# 正确版本的问题代码段
true_graph = toys.links_to_graph(links_coeffs)  # 从link定义创建真实图

# # 从 links_coeffs 提取信息构建 val_matrix
# def build_val_matrix(links_coeffs, tau_max=10):
#     N = len(links_coeffs)  # 变量数量
#     true_val_matrix = np.zeros((N, N, tau_max + 1))  # 初始化全零矩阵
#
#     for j in links_coeffs:
#         for ((i, tau), coeff, _) in links_coeffs[j]:
#             if tau < 0 and abs(tau) <= tau_max:
#                 lag = abs(tau)
#                 true_val_matrix[i, j, lag] = coeff  # 填充对应位置的系数
#
#     return true_val_matrix
#
# # 构建 val_matrix
# true_val_matrix_from_links = build_val_matrix(links_coeffs, tau_max=10)
# print(true_val_matrix_from_links)
# tp.plot_graph(true_graph,
#               val_matrix=true_val_matrix_from_links,
#               var_names=var_names,
#               link_colorbar_label='Weight',
#               node_colorbar_label='Weight',
#               link_label_fontsize=14,
#               label_fontsize=14,
#               tick_label_size=14,
#               node_label_size=14,
#               edge_ticks=0.5,
#               node_ticks=0.5,
#               node_size=0.3
#               )
# plt.title("True Graph", fontsize=14)
# plt.savefig('true_graph.png', dpi=300, bbox_inches='tight')
# plt.show()
parents = toys.dag_to_links(true_graph)  # 获取父节点关系
print("True parents:", parents)


realizations = 100

"""
Generate surrogate data
基于观测数据和已知的因果结构生成代理数据集（surrogate data）。
这些代理数据保留了原始数据的基本统计特性（如协方差结构），但打乱或保持了变量间的特定关系，用于假设检验、置信区间估计或模型验证。
"""
generator = surrogate_generator.generate_linear_model_from_data(dataframe, parents, tau_max, realizations=realizations,
                generate_noise_from='covariance')
datasets = {
            }
for r in range(realizations):
    datasets[r] = next(generator)


"""
相关性对比
"""
# 对每个替代数据集计算滞后相关
correlations = np.zeros((realizations, N, N, tau_max + 1))
for r in range(realizations):
    pcmci = PCMCI(
        dataframe=pp.DataFrame(datasets[r]),
        cond_ind_test=ParCorr(),
        verbosity=0)
    correlations[r] = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']
# Get mean and 5th and 95th quantile
correlation_mean = correlations.mean(axis=0)
correlation_interval = np.percentile(correlations, q=[5, 95], axis=0) # 置信区间
# Plot lag functions of mean and 5th and 95th quantile together with original correlation in one plot
lag_func_matrix = tp.setup_matrix(N=N, tau_max=tau_max, x_base=5, figsize=(10, 10), var_names=var_names)
lag_func_matrix.add_lagfuncs(val_matrix=correlation_mean, color='black')
lag_func_matrix.add_lagfuncs(val_matrix=correlation_interval[0], color='grey')
lag_func_matrix.add_lagfuncs(val_matrix=correlation_interval[1], color='grey')
lag_func_matrix.add_lagfuncs(val_matrix=original_correlations, color='red')
# lag_func_matrix.savefig('compare_lag_functions.png')
# 添加标题
plt.suptitle("Lagged Correlation Comparison", fontsize=16, y=0.95)
# plt.gcf().savefig('compare_lag_functions.png', dpi=300, bbox_inches='tight')
# plt.show()


"""
*****************************************************************
绘制显著相关图
"""
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=0)
#original_correlations_pvals = pcmci.get_lagged_dependencies(tau_max=tau_max)['p_matrix']  # 计算滞后依赖的 p 值（显著性检验）  越小越相关
# 获取偏相关系数和对应的 p 值
dependencies = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=False)
original_correlations = dependencies['val_matrix']
original_correlations_pvals = dependencies['p_matrix']
# print(original_correlations)
tp.plot_graph(graph=original_correlations_pvals<=0.01,
              val_matrix=original_correlations,
              var_names=var_names,
              link_colorbar_label='Partial correlation',
              node_colorbar_label='auto-correlation',
              link_label_fontsize=14,
              label_fontsize=14,
              tick_label_size=14,
              node_label_size=14,
              edge_ticks=0.5,
              node_ticks=0.5,
              node_size=0.3
              ) # 绘制显著相关性图（根据 p 值 ≤ 0.01 的阈值）
# 将 p 值小于等于 0.01 的边标记为显著相关，绘制这些显著关系构成的图结构
plt.title("Significant correlation graph", fontsize=14, pad=20)
plt.tight_layout()
# plt.savefig('significant_correlations_graph.png', dpi=300, bbox_inches='tight')
# plt.show()

# tp.plot_time_series_graph(
#         figsize=(4, 4),
#         val_matrix=original_correlations,
#         graph = original_correlations_pvals<=0.01,
#         var_names=var_names,
#         link_colorbar_label='Partial correlation',
#         label_fontsize=20,
#         tick_label_size=20
# )
# # 添加标题
# plt.title("Significant Time Series correlation graph", fontsize=14)
# plt.show()