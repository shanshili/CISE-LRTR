
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import numpy as np
import torch
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from matplotlib import rcParams
import sys
from pathlib import Path

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(str(Path(__file__).parent.parent))

# 从 CADGNN 导入相关组件
from Module.CADGNN import CADGNN

# 导入必要的模块
from utils.utils import find_value_according_index_list, robustness_score, location_graph, softsort
import pandas as pd
from datetime import datetime

# 导入因果分析流水线
from causal_analysis_pipeline import run_resilience_causal_analysis

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒

# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/Causal_simulation_data_2/N100_T10000_t3')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs('./Reference indicators', exist_ok=True)

# 全局修改字体
config = {
    "font.family": 'Times New Roman',
    "font.size": 10.5,
    'axes.unicode_minus': False
}
rcParams.update(config)


class RegressionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


class CausalResilienceModel(nn.Module):
    def __init__(self, causal_feature_dim, hidden_dim, output_dim, args):
        super(CausalResilienceModel, self).__init__()
        self.regression_module = RegressionModule(causal_feature_dim, hidden_dim, 1)
        self.args = args

    def forward(self, causal_features):
        """
        直接使用因果特征进行回归，跳过图嵌入步骤
        causal_features: 从CADGNN获得的时空因果特征 (batch_size, causal_feature_dim)
        """
        scores = self.regression_module(causal_features)
        return scores


def load_cadgnn_model(model_path, device):
    """
    加载训练好的 CADGNN 模型（增强版）
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 检查模型配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # 如果没有model_config，使用默认配置（根据文件名推断）
            model_config = {
                'input_dim': 8000,  # 与特征选择一致
                'hidden_dim': 128,
                'num_layers': 5,
                'lambda_adv': 0.1,
                'lambda_weight': 0.02
            }
            print("Warning: Using inferred model config based on filename")

        # 创建 CADGNN 模型
        model = CADGNN(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            lambda_adv=model_config['lambda_adv'],
            lambda_weight=model_config['lambda_weight']
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        return model, model_config
    except Exception as e:
        print(f"Error loading CADGNN model: {e}")
        raise e


# 解析参数
parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--max_epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--hidden_dim", default=512, type=int)  # 调整隐藏层维度
parser.add_argument("--causal_feature_dim", default=128, type=int)  # CADGNN输出维度
parser.add_argument("--weight_decay", type=int, default=1e-4)
parser.add_argument("--cadgnn_model_path", type=str,
                    default="./CADGNN_results/best_model.pth",
                    help="CADGNN模型路径")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

# 读取数据（保持原有数据读取逻辑）
wpr_rank = []
with open('WPR_result/_node_list_20250908_173018_GP_62_node_400_data_9997.txt', 'r') as f:
    for line in f:
        wpr_rank.append(int(list(line.strip('\n').split(','))[0]))

Gpn = 62
node = 400
data_num = 9997
select_node = 18
fea_ori = np.load('./perturbation_data/origin_data_fea_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz',
                  allow_pickle=True)
fea_o = fea_ori['arr_0']
position_file = os.path.join(dataset_dir, 'node_positions.csv')
positions_df = pd.read_csv(position_file)
positions = positions_df[['x', 'y']].values

# PageRank选点结果
selected_node = wpr_rank[:select_node]
fea_list = find_value_according_index_list(fea_o, selected_node)
location_list = find_value_according_index_list(positions, selected_node)
unselected_node = wpr_rank[select_node + 1:114]
un_fea_list = find_value_according_index_list(fea_o, unselected_node)
un_location_list = find_value_according_index_list(positions, unselected_node)

# 加载训练好的 CADGNN 模型
try:
    cadgnn_model, cadgnn_config = load_cadgnn_model(args.cadgnn_model_path, device)
    args.cadgnn_hidden_dim = cadgnn_config['hidden_dim']
    print(f"Loaded CADGNN model with hidden dim: {args.cadgnn_hidden_dim}")
    # 确保模型处于评估模式
    cadgnn_model.eval()
except Exception as e:
    print(f"Error loading CADGNN model: {e}")
    print("Please make sure you have a trained CADGNN model in the CADGNN_results folder")
    cadgnn_model = None

# 初始化直接回归模型（不再包含CADGNN）
RP_model = CausalResilienceModel(args.causal_feature_dim, args.hidden_dim, 1, args).to(device)
optimizer = torch.optim.Adam(RP_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_CrossEntropy = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, cooldown=1, verbose=True)

# 计算真实鲁棒性分数
R_Rg = []
location_list_with_placeholder = location_list + [None]
fea_list_with_placeholder = fea_list + [None]

for i, (un_location, un_fea) in enumerate(zip(un_location_list, un_fea_list)):
    location_list_with_placeholder[select_node] = un_location
    fea_list_with_placeholder[select_node] = un_fea
    temp_G, _ = location_graph(np.array(location_list_with_placeholder[:-1]))  # 移除占位符
    R_Rg.append(torch.tensor(robustness_score(temp_G)))

R_Rg_tensor = torch.stack(R_Rg, dim=0).requires_grad_(True).to(device)

# 真实值处理
criticality_scores = softsort(R_Rg_tensor)
criticality_scores_normal = (criticality_scores - torch.min(criticality_scores)) / (
        torch.max(criticality_scores) - torch.min(criticality_scores))

loss_history = []

# 训练过程
for epoch in range(args.max_epoch):
    tensors = []

    # 为每个候选节点构建因果图并提取特征
    print(f"Epoch {epoch + 1}/{args.max_epoch}: 构建候选节点因果图...")

    # 运行因果分析流水线 - 为每个候选节点构建包含已选节点和该候选节点的因果图
    cadgnn_inputs = run_resilience_causal_analysis(
        dataset_dir=dataset_dir,
        selected_node_features=fea_list,  # 已选节点特征
        selected_node_positions=location_list,  # 已选节点位置
        unselected_node_features=un_fea_list,  # 候选节点特征列表
        unselected_node_positions=un_location_list,  # 候选节点位置列表
        pcmci_params={
            'tau_max': 3,
            'pc_alpha': 0.0001
        },
        scmi_params={
            'delta': 15.0,
            'k_neighbors': 5,
            'n_bins': 10,
            'threshold': 0.01,
            'tau_max': 3
        }
    )

    # 为每个候选节点提取因果特征
    for i, cadgnn_input in enumerate(cadgnn_inputs):
        if cadgnn_input is None or cadgnn_model is None:
            # 如果没有因果信息或CADGNN模型不可用，使用随机特征
            print(f"  候选节点 {i + 1}: 使用随机特征")
            causal_features = torch.randn(args.causal_feature_dim, requires_grad=True).to(device)
        else:
            # 使用CADGNN模型提取因果特征
            try:
                print(f"  候选节点 {i + 1}: 使用CADGNN提取特征")
                # 构建时空因果图
                adjacency_matrix = torch.FloatTensor(cadgnn_input['adjacency_matrix']).to(device)
                node_features = torch.FloatTensor(cadgnn_input['node_features']).to(device)

                # 使用CADGNN提取特征
                with torch.no_grad():
                    causal_features = cadgnn_model(adjacency_matrix, node_features)
                    # 如果输出是元组，取第一个元素
                    if isinstance(causal_features, tuple):
                        causal_features = causal_features[0]
            except Exception as e:
                print(f"  候选节点 {i + 1}: 使用CADGNN提取特征时出错: {e}")
                # 出错时使用随机特征
                causal_features = torch.randn(args.causal_feature_dim, requires_grad=True).to(device)

        # 直接回归，跳过图嵌入
        RP_model.train()
        score = RP_model(causal_features.unsqueeze(0))  # 添加batch维度
        tensors.append(score)
        print(f"  候选节点 {i + 1}: 特征提取完成")

    scores_tensor = torch.stack(tensors, dim=0).requires_grad_(True).to(device)
    scores_tensor_scores = softsort(scores_tensor)
    optimizer.zero_grad()

    # 计算损失
    scores_tensor_normal = (scores_tensor_scores - torch.min(scores_tensor_scores)) / (
            torch.max(scores_tensor_scores) - torch.min(scores_tensor_scores))

    r_ij = []
    y_hat_ij = []
    for ii in range(len(scores_tensor_normal) - 1):
        for jj in range(ii + 1, len(scores_tensor_normal) - 1):
            r_ij.append(criticality_scores_normal[ii] - criticality_scores_normal[jj])
            y_hat_ij.append(scores_tensor_normal[ii] - scores_tensor_normal[jj])

    r_ij_tensor = torch.stack(r_ij, dim=0).requires_grad_(True).to(device)
    y_hat_ij_tensor = torch.stack(y_hat_ij, dim=0).requires_grad_(True).to(device)
    loss = loss_CrossEntropy(y_hat_ij_tensor, r_ij_tensor)

    loss.backward(retain_graph=True)
    optimizer.step()
    loss_history.append(loss.item())
    scheduler.step(loss.item())
    print('epoch:{}, loss:{}'.format(epoch, loss.item()))

# 后续保存和可视化代码保持不变...
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history)), loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch : {}'.format(args.max_epoch))
plt.title('Training Loss: epoch' + str(args.max_epoch) + ' lr' + str(args.lr) + ' ' + str(device))
plt.text(0, loss_history[0], str(format(loss_history[0], '.8f')))
print(format(loss_history[(args.max_epoch - 1)], '.15f'))
plt.text(round(args.max_epoch / 10 * 9), loss_history[(args.max_epoch - 1)],
         str(format(loss_history[(args.max_epoch - 1)], '.10f')),
         horizontalalignment='left')

# time stamp
timestamp = time.time()
localtime = time.localtime(timestamp)
formatted_time = time.strftime('%Y%m%d_%H%M%S', localtime)

# 保存目录
os.makedirs('./training_loss', exist_ok=True)
os.makedirs('./model_save', exist_ok=True)
os.makedirs('./scores_save', exist_ok=True)

# Save
plt.savefig('./training_loss/_Training_Loss_epoch_' + str(args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(
    formatted_time) + '.svg', format='svg')
plt.show()

torch.save(RP_model,
           './model_save/_e_' + str(args.max_epoch) + '_l_' + str(args.lr) + '_' + str(formatted_time) + '.pth')

np.savetxt(
    './scores_save/scores_epoch_' + str(args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.txt',
    scores_tensor.cpu().detach().numpy())

np.savetxt('./scores_save/softsort_normal_epoch_' + str(args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(
    formatted_time) + '.txt', scores_tensor_scores.cpu().detach().numpy())

# 可视化结果
location_list_a = np.array(location_list)
un_location_list_a = np.array(un_location_list)
plt.scatter(un_location_list_a[:, 0], un_location_list_a[:, 1], s=15, c=scores_tensor_scores.cpu().detach().numpy(),
            cmap='Greens')
plt.scatter(location_list_a[:select_node, 0], location_list_a[:select_node, 1], s=20, c='#f44336')  # selected_node
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('softsort_normal')
plt.savefig('./scores_save/_softsort_normal_epoch_' + str(args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(
    formatted_time) + '.svg', format='svg')
plt.show()
