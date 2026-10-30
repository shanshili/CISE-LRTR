import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# 从 CADGNN 导入相关组件
from Module.CADGNN import CADGNN
import argparse
import networkx as nx
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import os


class MFC_RMF(nn.Module):
    def __init__(self, cadgnn_model, args):
        super(MFC_RMF, self).__init__()
        torch.manual_seed(1234)
        self.args = args
        # 使用训练好的 CADGNN 模型
        self.cadgnn = cadgnn_model
        self.readout_rep = readout(args.cadgnn_hidden_dim)  # 使用 CADGNN 的隐藏维度
        self.readout_match = readout(args.match_size)
        self.match = matching(args.cadgnn_hidden_dim, args.match_size)
        self.mlp = MLP(args.match_size)

    # 使用 CADGNN 获取因果表示
    def get_causal_representation(self, x, edge_index, edge_weight):
        with torch.no_grad():
            z_c = self.cadgnn.causal_encoder(x, edge_index, edge_weight)
            return z_c

    def forward(self, graph_pair):
        # 图对信息
        edge_index_1 = graph_pair.edge_index_s
        edge_index_2 = graph_pair.edge_index_t

        # 标准化
        feature_1 = graph_pair.x_s.clone().detach()
        feature_2 = graph_pair.x_t.clone().detach()
        # 修改标准化方式
        feature_1 = (feature_1 - feature_1.min()) / (feature_1.max() - feature_1.min() + 1e-8)
        feature_2 = (feature_2 - feature_2.min()) / (feature_2.max() - feature_2.min() + 1e-8)

        # 调试信息
        if torch.isnan(feature_1).any() or torch.isnan(feature_2).any():
            print("Warning: NaN values in input features")

        # print(f"Feature stats - feat1: mean={mean_1.item():.4f}, std={std_1.item():.4f}")
        # print(f"Feature stats - feat2: mean={mean_2.item():.4f}, std={std_2.item():.4f}")

        # 使用 CADGNN 获取因果表示
        edge_weight_1 = graph_pair.edge_weight_s
        edge_weight_2 = graph_pair.edge_weight_t
        h_1_self = self.get_causal_representation(feature_1, edge_index_1, edge_weight_1)
        h_2_self = self.get_causal_representation(feature_2, edge_index_2, edge_weight_2)
        # print(f"Causal representations - h1: {h_1_self.shape}, h2: {h_2_self.shape}")

        # 全局自表示
        h_1_self_global = self.readout_rep(h_1_self)
        h_2_self_global = self.readout_rep(h_2_self)

        # 局部互表示
        h_1_mutual = self.mutual_representation(h_1_self, h_2_self)
        h_2_mutual = self.mutual_representation(h_2_self, h_1_self)

        # 全局互表示
        h_1_mutual_global = self.readout_rep(h_1_mutual)
        h_2_mutual_global = self.readout_rep(h_2_mutual)

        # 多粒度交叉匹配
        # 局部-局部匹配
        miu_1 = self.match(h_1_self, h_1_mutual)
        miu_2 = self.match(h_2_self, h_2_mutual)

        # 读出全局匹配向量
        miu_1_global = self.readout_rep(miu_1)
        miu_2_global = self.readout_rep(miu_2)

        # 局部-全局匹配
        phi_1 = self.match(h_1_self, h_1_mutual_global)
        phi_2 = self.match(h_2_self, h_2_mutual_global)

        # 读出全局匹配向量
        phi_1_global = self.readout_rep(phi_1)
        phi_2_global = self.readout_rep(phi_2)

        # 全局-局部匹配
        psi_1 = self.match(h_1_self_global, h_1_mutual)
        psi_2 = self.match(h_2_self_global, h_2_mutual)

        # 读出全局匹配向量
        psi_1_global = self.readout_rep(psi_1)
        psi_2_global = self.readout_rep(psi_2)

        # 全局-全局匹配
        omega_1 = self.match(h_1_self_global, h_1_mutual_global)
        omega_2 = self.match(h_2_self_global, h_2_mutual_global)

        # 匹配特征融合
        z_1 = torch.cat([miu_1_global, phi_1_global, psi_1_global, omega_1], dim=1).t()
        z_2 = torch.cat([miu_2_global, phi_2_global, psi_2_global, omega_2], dim=1).t()
        z = torch.cat([z_1, z_2], dim=0)
        z = self.mlp(z.t())

        # 正确返回图对的标签信息
        label = graph_pair.label
        label_exp = graph_pair.label  # 如果需要扩展标签，可以在这里处理

        return z, label, label_exp

    # 互表示操作：输出互表示特征
    def mutual_representation(self, h_1, h_2):
        cos_sim = torch.cosine_similarity(h_1.unsqueeze(1), h_2.unsqueeze(0), dim=-1)
        cos_ceo = cos_sim / torch.sum(cos_sim, dim=0)
        h_mutual = torch.mm(cos_ceo, h_2)
        return h_mutual


# 全局读出操作：输出全局特征
class readout(nn.Module):
    def __init__(self, output_size):
        super(readout, self).__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h):
        global_h = torch.matmul(torch.mean(h, dim=0), self.weight)
        transformed_global = torch.tanh(global_h)
        sigmoid_scores = torch.sigmoid(torch.mm(h, transformed_global.view(-1, 1)).sum(dim=1))
        h_global = sigmoid_scores.unsqueeze(-1) * h
        return h_global.sum(dim=0).unsqueeze(0)


# 匹配操作：输出匹配特征
class matching(nn.Module):
    def __init__(self, output_size, match_size):
        super(matching, self).__init__()
        self.output_size = output_size
        self.match_size = match_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.match_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h_1, h_2):
        h_1 = h_1.unsqueeze(1)
        h_2 = h_2.unsqueeze(1)
        weight = self.weight.unsqueeze(0)
        h_1_w = torch.mul(h_1, weight)
        h_2_w = torch.mul(h_2, weight)
        match = torch.cosine_similarity(h_1_w, h_2_w)
        return match


class MLP(nn.Module):
    def __init__(self, match_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(match_size * 8, match_size * 2)
        self.linear2 = nn.Linear(match_size * 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(match_size * 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear1(z)
        z = self.layer_norm1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return self.sigmoid(z)


class GraphPair:
    def __init__(self, edge_index_s, edge_index_t, x_s, x_t, edge_weight_s, edge_weight_t, label):
        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.x_s = x_s
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.label = label

    def save(self, GraphPair_file):
        torch.save({
            'edge_index_s': self.edge_index_s,
            'edge_index_t': self.edge_index_t,
            'x_s': self.x_s,
            'x_t': self.x_t,
            'edge_weight_s': self.edge_weight_s,
            'edge_weight_t': self.edge_weight_t,
            'label': self.label,
        }, GraphPair_file)

    @staticmethod
    def load(GraphPair_file):
        loaded_data = torch.load(GraphPair_file)
        return GraphPair(
            edge_index_s=loaded_data['edge_index_s'],
            edge_index_t=loaded_data['edge_index_t'],
            x_s=loaded_data['x_s'],
            x_t=loaded_data['x_t'],
            edge_weight_s=loaded_data['edge_weight_s'],
            edge_weight_t=loaded_data['edge_weight_t'],
            label=loaded_data['label']
        )


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

os.makedirs('./MGC-RM_result',exist_ok=True)
os.makedirs('./MGC-RM_result/training_loss', exist_ok=True)
os.makedirs('./MGC-RM_result/model_save', exist_ok=True)
os.makedirs('./MGC-RM_result/prediction_loss', exist_ok=True)
os.makedirs('./MGC-RM_result/similarity score', exist_ok=True)

parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--max_epoch", type=int, default=150)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--match_size", default=128, type=int)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--cadgnn_model_path", type=str, default="./CADGNN_results/N100_T10000_hid128_lay5_ladv0.1_lw0.02_20250902_203304_model.pth",
                    help="Path to the trained CADGNN model")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

# 设置数据参数
node = 400
data_num = 9997

# 从 perturbation_data 目录加载数据
weight_ori = np.load('./perturbation_data/origin_data_weight_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz', allow_pickle=True)
fea_ori = np.load('./perturbation_data/origin_data_fea_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz', allow_pickle=True)
perturbed_graph_weight = np.load(
    './perturbation_data/perturbed_graph_weight' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz')
perturbed_label = np.load(
    './perturbation_data/perturbed_label' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz')
perturbed_fea = np.load('./perturbation_data/perturbed_fea' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz')
print('perturbed_graph_weight', perturbed_graph_weight.files, perturbed_graph_weight['arr_0'].shape)

perturbed_w = perturbed_graph_weight['arr_0']
perturbed_l = perturbed_label['arr_0']
perturbed_fe = perturbed_fea['arr_0']

# 构建原始图
# 正确构造带权重的图
weight_matrix = weight_ori['arr_0']  # 假设这是 NumPy 邻接矩阵
G_o = nx.from_numpy_array(weight_matrix, create_using=nx.Graph())  # 自动将矩阵值作为 'weight' 属性
edge_index_s = np.array(list(G_o.edges)).T
# 从图中提取边权重
edge_weight_s = torch.tensor([G_o[u][v]['weight'] for u, v in G_o.edges], dtype=torch.float32).to(device)# 检查边权重

# for u, v, data in G_o.edges(data=True):
#     print(f"Edge ({u}, {v}), Weight: {data.get('weight', 1.0)}")  # 默认权重为1.0


# 处理特定的图对
j = 39  # 图对编号
print('======= perturbed：', j, '========================')
perturbed_graph_w = perturbed_w[j]
perturbed_f = perturbed_fe[j]
perturbed_graph = nx.from_numpy_array(perturbed_graph_w, create_using=nx.Graph())
edge_index_t = np.array(list(perturbed_graph.edges)).T
edge_weight_t = torch.tensor([perturbed_graph[u][v]['weight'] for u, v in perturbed_graph.edges], dtype=torch.float32).to(device)
# for u, v, data in perturbed_graph.edges(data=True):
#      print(f"Edge ({u}, {v}), Weight: {data.get('weight', 1.0)}")  # 默认权重为1.0

# 在MGC-RM.py中添加特征选择函数
def select_features(features, target_dim=8000):
    """
    特征选择函数，参考CADGNN中的特征选择方法
    选择前target_dim个特征维度以匹配训练模型的输入维度
    """
    if features.shape[1] > target_dim:
        # 选择前target_dim个时间步特征（与CADGNN训练时一致）
        selected_features = features[:, :target_dim]
        # print(f"Feature selection: {features.shape[1]} -> {selected_features.shape[1]} dimensions")
        return selected_features
    else:
        # print(f"Feature dimension {features.shape[1]} <= target {target_dim}, no selection needed")
        return features


# 修改张量创建部分
# 转换成 PyTorch 张量并处理维度 (参考CADGNN的特征选择方法)
raw_x_s_tensor = torch.tensor(fea_ori['arr_0'], dtype=torch.float32).to(device)
raw_x_t_tensor = torch.tensor(perturbed_f, dtype=torch.float32).to(device)

# print(f"Raw feature shapes - x_s: {raw_x_s_tensor.shape}, x_t: {raw_x_t_tensor.shape}")

# 特征选择：从9997维选择到8000维（与CADGNN训练时使用的维度一致）
x_s_tensor = select_features(raw_x_s_tensor, 8000)
x_t_tensor = select_features(raw_x_t_tensor, 8000)

# print(f"Selected feature shapes - x_s: {x_s_tensor.shape}, x_t: {x_t_tensor.shape}")

edge_index_s_tensor = torch.tensor(edge_index_s, dtype=torch.int64).to(device)
edge_index_t_tensor = torch.tensor(edge_index_t, dtype=torch.int64).to(device)
label_tensor = torch.tensor(perturbed_l, dtype=torch.int64).to(device)
edge_weight_s_tensor = edge_weight_s.to(device)  # 权重张量
edge_weight_t_tensor = edge_weight_t.to(device)  # 权重张量

# 创建图对
# 创建图对（包含权重信息）
graph_pair = GraphPair(
    edge_index_s_tensor,
    edge_index_t_tensor,
    x_s_tensor,
    x_t_tensor,
    edge_weight_s_tensor,  # 添加权重张量
    edge_weight_t_tensor,  # 添加权重张量
    label_tensor
)
# 加载训练好的 CADGNN 模型
try:
    cadgnn_model, cadgnn_config = load_cadgnn_model(args.cadgnn_model_path, device)
    args.cadgnn_hidden_dim = cadgnn_config['hidden_dim']
    print(f"Loaded CADGNN model with hidden dim: {args.cadgnn_hidden_dim}")
except Exception as e:
    print(f"Error loading CADGNN model: {e}")
    print("Please make sure you have a trained CADGNN model in the CADGNN_results folder")
    exit()


# 模型初始化
model = MFC_RMF(cadgnn_model, args).to(device)
print('CADGNN hidden dim: ', args.cadgnn_hidden_dim)
print('Expected input dim for CADGNN: ', cadgnn_config['input_dim'])
print('edge_index_s_tensor: ', edge_index_s_tensor.shape)
print('edge_index_t_tensor: ', edge_index_t_tensor.shape)
print('x_s_tensor: ', x_s_tensor.shape)
print('x_t_tensor: ', x_t_tensor.shape)
print(args)

# 检查模型参数
print("Model parameters check:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")


# 简化版图编辑距离计算
# 修改目标相似度计算，使用线性映射
def simple_graph_distance(G1, G2):
    node_diff = abs(G1.number_of_nodes() - G2.number_of_nodes())
    edge_diff = abs(G1.number_of_edges() - G2.number_of_edges())
    distance = node_diff + edge_diff * 0.1
    return distance

# 使用线性映射
graph_edit_distance = simple_graph_distance(G_o, perturbed_graph)
# 动态计算最大距离（基于训练数据）
max_distance = np.max([simple_graph_distance(G_o, nx.from_numpy_array(perturbed_w[i], create_using=nx.Graph())) for i in range(min(50, perturbed_w.shape[0]))])
# 使用线性映射替代sigmoid
similarity = 1.0 - (graph_edit_distance / max_distance)  # 线性映射到[0.1, 0.9]
z_targe = torch.tensor([similarity], dtype=torch.float32).to(device)
print(f"Simple graph distance: {graph_edit_distance}")
print(f"Target similarity (clipped): {similarity}")


# 模型训练
# 修改优化器参数
optimizer = Adam(model.parameters(), lr=0.00002, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.3,   # 更激进的衰减
#     patience=10,  # 更宽松的触发阈值
#     min_lr=1e-6 # 设置学习率下限
# )
# 推荐配置（余弦退火+线性预热）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.max_epoch,  # 总周期
    eta_min=1e-6  # 最小学习率
)
mse_loss = nn.MSELoss()
loss_history = []

print(f"Target similarity: {z_targe.item()}")



for epoch in range(args.max_epoch):
    model.train()
    z, label, label_exp = model.forward(graph_pair)
    loss = mse_loss(z[0], z_targe)

    # 检查预测值是否有效
    if torch.isnan(z[0]) or torch.isinf(z[0]):
        print(f"Warning: Invalid prediction at epoch {epoch}: {z[0].item()}")
        continue

    optimizer.zero_grad()
    loss.backward()

    # 加强梯度检查
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Invalid loss at epoch {epoch}: {loss.item()}")
        continue

    # ========== 梯度裁剪应 ==========
    # 动态调整梯度裁剪阈值
    if epoch < 100:
        gradient_clip_value = 1.0  # 初期使用较小的裁剪值
    else:
        gradient_clip_value = 0.5  # 后期使用更小的裁剪值

    # 修改梯度裁剪逻辑，仅裁剪过大的梯度
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
    if grad_norm > 10.0:  # 仅当梯度范数过大时跳过更新
        print(f"Skipping update at epoch {epoch}, grad_norm={grad_norm:.6f}")
        continue

    optimizer.step()
    scheduler.step(loss.item())
    # ========== 梯度裁剪结束 ==========

    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.15f}, prediction={z[0].item():.15f}, grad_norm={grad_norm:.8f}")

    # 每10个epoch打印一次详细信息
    if epoch % 10 == 0 or epoch == args.max_epoch - 1:
        print(f'epoch:{epoch:3d}, loss:{loss.item():.8f}, prediction:{z[0].item():.8f}')



# 损失显示
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history)), loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch : {}'.format(args.max_epoch))
plt.title('GraphPair_' + str(j) + ' Training Loss: epoch' + str(args.max_epoch) + ' lr' + str(args.lr))
plt.text(0, loss_history[0], str(format(loss_history[0], '.8f')))
print(format(loss_history[(args.max_epoch - 1)], '.15f'))
plt.text(round(args.max_epoch / 10 * 9), loss_history[(args.max_epoch - 1)],
         str(format(loss_history[(args.max_epoch - 1)], '.10f')),
         horizontalalignment='left')

# 时间戳
timestamp = time.time()
localtime = time.localtime(timestamp)
formatted_time = time.strftime('%Y%m%d_%H%M%S', localtime)

# 保存
plt.savefig(
    './MGC-RM_result/training_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Training_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.svg', format='svg')
plt.show()
torch.save(model, './MGC-RM_result/model_save/model_GP_' + str(j) + '_n_' + str(node) + '_d_' + str(data_num) + '_e_' + str(
    args.max_epoch) + '_l_' + str(args.lr) + '_' + str(formatted_time) + '.pth')

# 模型评估
model.eval()
z_p = []
loss_history2 = []

for i in range(min(50, perturbed_w.shape[0])):  # 限制评估数量以节省时间
    print('GP：', i, '===============================')
    perturbed_graph_w2 = perturbed_w[i]
    perturbed_f2 = perturbed_fe[i]
    perturbed_graph2 = nx.from_numpy_array(perturbed_graph_w2, create_using=nx.Graph())
    edge_index_t2 = np.array(list(perturbed_graph2.edges)).T
    edge_weight_t2 = torch.tensor([perturbed_graph2[u][v]['weight'] for u, v in perturbed_graph2.edges],
                                  dtype=torch.float32).to(device)

    # 转换成 PyTorch 张量并进行特征选择
    raw_x_t_tensor2 = torch.tensor(perturbed_f2, dtype=torch.float32).to(device)
    x_t_tensor2 = select_features(raw_x_t_tensor2, 8000)  # 添加特征选择

    edge_index_t_tensor2 = torch.tensor(edge_index_t2, dtype=torch.int64).to(device)
    edge_weight_t_tensor2 = edge_weight_t2.to(device)  # 权重张量

    # 创建包含权重的图对
    graph_pair2 = GraphPair(
        edge_index_s_tensor,
        edge_index_t_tensor2,
        x_s_tensor,
        x_t_tensor2,
        edge_weight_s_tensor,  # 原始图权重
        edge_weight_t_tensor2,  # 扰动图权重
        label_tensor
    )

    # 计算 ground-truth
    graph_edit_distance2 = nx.graph_edit_distance(G_o, perturbed_graph2)
    similarity2 = math.exp(-graph_edit_distance2 / ((G_o.number_of_nodes() + perturbed_graph2.number_of_nodes()) / 2))
    z_targe2 = torch.Tensor([similarity2]).to(device)

    with torch.no_grad():
        z_prediction, label2, label_exp2 = model(graph_pair2)
        loss2 = mse_loss(z_prediction[0], z_targe2)
        loss_history2.append(loss2.item())
        z_p.append(z_prediction[0].item())
        print(' loss: ' + str(loss2.item()))
        print(' score: ' + str(z_prediction[0].item()))

# 保存结果
np.savetxt('./MGC-RM_result/similarity score/z' + 'GP_' + str(j) + '_node_' + str(node) + '_data_' + str(data_num) + 'model_' + str(
    formatted_time) + '.txt', z_p)
np.savetxt(
    './MGC-RM_result/prediction_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Prediction_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.txt', loss_history2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history2)), loss_history2)
plt.ylabel('Loss')
plt.xlabel('GraphPair_ : {}'.format(len(loss_history2)))
plt.title('Similarity Loss_' + 'GP_' + str(j) + '_n_' + str(node) + '_d_' + str(data_num) + '_e_' + str(
    args.max_epoch) + '_l_' + str(args.lr))
plt.savefig(
    './MGC-RM_result/prediction_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Similarity_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.svg', format='svg')
plt.show()
