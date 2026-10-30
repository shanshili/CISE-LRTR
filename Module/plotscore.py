import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 将项目根目录添加到 Python 的搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.GraphConstruct import data_color_graph2
from utils.dataprocess import normalization
import numpy as np
import networkx as nx
from matplotlib import rcParams

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

node = 450
data_num = 2000
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
fea_ori = np.load('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
G_o = nx.Graph(adj_ori['arr_0'])
# print('location',location_file.files,location_file['arr_0'].shape)
location = location_file['arr_0']
# print(location)
# f = open('similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt','r')
# a = list(f)
# print(a)
# print(type(a[0]))
# f.close()


score=[]
gpn = 75
error_point = 328
# 标号为数组下标
adjust =  1.463619555579498410e-04#75-329的数据 #1.949527258053421974e-05#159-328# #1.664528177166357636e-04#310-21#
# with open('similarity score\gp_159_z_node_450_data_2000model_20241102_165518.txt','r') as f:
# with open('similarity score\gp_310_z_node_450_data_2000model_20241102_165317.txt','r') as f:
with open('similarity score/zGP_75_node_450_data_2000model_20241102_211834.txt', 'r') as f:
	for line in f:
		score.append(float(list(line.strip('\n').split(','))[0]))

# print(score)
# print(type(score))
# print(type(score[0]))
# print(score[0])

score_n = normalization(score,error_point,adjust)
print(score_n)

data_color_graph2(score_n,G_o,location,gpn,'rainbow',False)


