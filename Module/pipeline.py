# -*- coding: utf-8 -*-
r"""
pipeline.py —— 编码器 / 提取器 / 训练 / 数据视图 / 掩码打分

五组方法（每组与 FULL 相差恰好一个变量）
------------------------------------------------------------------
  方法          图源     编码器            投影      选点分支  连边方式      与 FULL 的差异
  --------------------------------------------------------------------------------------
  MGC-RM        corr     不训练(复用产物)  —         node      corr_knn        相关性基线
  B2_GCN_LRTR   causal   PlainGCN(可训)    CGBP      edge      pcmci_mask      对抗解耦
  B3_CORR_INV   corr     PlainGCN(可训)    CGBP      edge      corr_knn_mask   因果图
  B4_NOCGBP     causal   CADGNN(冻结)      NodeHead  node      pcmci_all       CGBP
  FULL_CISE     causal   CADGNN(冻结)      CGBP      edge      pcmci_mask      —（复用产物）

CGBP 与它的消融
------------------------------------------------------------------
  FULL（CGBP，边级）
      Z^(k) = Enc_CADGNN(X, A~^(k) ⊙ M)
      Phi_k = A~^(k)_sub ⊙ (Z^(k) W_p Z^(k)T)   ∈ R^{N×N}

  B4_NOCGBP（无 CGBP，节点级）
      H^(k) = LN(GConv(Z^(k), A~^(k) ⊙ M))
      Phi_k = s^(k) = H^(k) w                   ∈ R^{N×1}   节点因果分数

  两者的 LRTR 是同一个式子，只是作用层级不同：
      FULL：不变性约束作用在边级 -> 每个 m_ij 拿到属于自己那条边的梯度 -> 边可排序
      B4  ：不变性约束作用在节点分数 -> 同一节点的入边梯度同号同向 -> 边不可分辨

  NodeHead 用"图卷积 + LayerNorm + 线性头"而不是直接取 Z：FULL 有可训的 W_p(d×d)，
  若 B4 直接取 Z 则一个可训参数都没有，FULL−B4 的差异会混进"有没有可训投影"这第二个
  变量。补平参数量后，唯一变量回到"输出是边级还是节点级"。

  可主张的是"把不变性约束抬到边级"，不是"双线性" —— CADGNN.CausalDecoder 里已有
  W_hat = theta_w(Z) Z^T，CGBP 复用的就是它的结构。
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import subgraph as SG          # 节点预算的唯一来源

__version__ = "1.0"


# ==========================================================
# 0. 工具
# ==========================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_module(path, alias, extra_paths=None):
    path = Path(path).resolve()
    for p in (extra_paths or []):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def search_upward(fn, start=None, depth=4):
    start = start or Path(__file__).resolve().parent
    for d in [start] + list(start.parents)[:depth]:
        if (d / fn).exists():
            return d / fn
        for sub in d.glob(f"*/{fn}"):
            return sub
    return None


_CISE = None


def cise(path=None):
    """从 CausalInvariantSubgraphExtractor2_3_1_1.py 复用数据处理函数（不用它的模型）。"""
    global _CISE
    if _CISE is None:
        p = Path(path or os.environ.get("CISE_PATH")
                 or search_upward("CausalInvariantSubgraphExtractor2_3_1_1.py")).resolve()
        _CISE = load_module(p, "cise_ref", [str(p.parent), str(p.parent.parent)])
    return _CISE


# ==========================================================
# 1. 编码器
# ==========================================================

class WeightedGraphConv(nn.Module):
    r"""h' = Wh + D^{-1} A~ (Wh),  D = diag(sum_j |A_ij|)

    不用对称归一化：PCMCI 权重有负值，sqrt(deg) 会出 NaN。
    消息按 edge_weight 加权 -> 梯度直达 edge_weight -> 直达掩码。
    """

    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o)
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, ei, ew):
        h = self.lin(x)
        msg = h.index_select(0, ei[0]) * ew.unsqueeze(-1)
        agg = torch.zeros_like(h).index_add_(0, ei[1], msg)
        deg = torch.zeros(x.size(0), device=x.device, dtype=h.dtype).index_add_(0, ei[1], ew.abs())
        return h + agg / (deg.unsqueeze(-1) + 1e-6)


class PlainGCN(nn.Module):
    """B2 / B3 的编码器：与 CADGNN.causal_encoder 同规模，但没有对抗解耦分支
    （无 z_s、无判别器、无 L_adv）。

    LayerNorm 必须有：8000 维输入 + 5 层残差若不归一化，std(Phi) 会到 1e12 量级，
    LRTR 的比值恒定、掩码收不到有效梯度。
    """

    def __init__(self, in_dim=8000, hid=128, layers=5, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        d = [in_dim] + [hid] * layers
        self.convs = nn.ModuleList([WeightedGraphConv(d[i], d[i + 1]) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

    def forward(self, x, ei, ew):
        h = x
        for i, c in enumerate(self.convs):
            h = self.norms[i](c(h, ei, ew))
            if i < len(self.convs) - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h


class FrozenCADGNN(nn.Module):
    """B4 / FULL 的编码器：冻结的 CADGNN.causal_encoder。"""

    def __init__(self, cad):
        super().__init__()
        self.cad = cad
        for p in self.cad.parameters():
            p.requires_grad = False

    def forward(self, x, ei, ew):
        return self.cad.causal_encoder(x, ei, ew)


class NodeHead(nn.Module):
    r"""CGBP 消融分支：图卷积 + LayerNorm + 线性头 -> 节点因果分数 s ∈ R^{N×1}。
    可训参数量与 FULL 的 W_p(d×d) 同量级。"""

    def __init__(self, hid):
        super().__init__()
        self.conv = WeightedGraphConv(hid, hid)
        self.norm = nn.LayerNorm(hid)
        self.lin = nn.Linear(hid, 1, bias=False)

    def forward(self, Z, ei, ew):
        H = self.norm(self.conv(Z, ei, ew))
        return self.lin(F.relu(H))          # [N,1]


# ==========================================================
# 2. 提取器
# ==========================================================

class Extractor(nn.Module):
    r"""
    平稳掩码核：m_ij^(l) = sigmoid(Theta_ijl / T)，按 lag 平铺到 M[tN+i, (t+l)N+j]。
    投影：cgbp=True  -> Phi = A~ ⊙ (W_p Z Z^T)     边级   [N,N]
          cgbp=False -> Phi = Linear(GConv(Z))     节点级 [N,1]
    """

    def __init__(self, encoder, hid, N, tau_max, prior, cgbp=True,
                 freeze=True, floor=0.05):
        super().__init__()
        self.encoder, self.N, self.tau_max = encoder, N, tau_max
        self.target = N * (tau_max + 1)
        self.cgbp, self.floor, self.temperature = cgbp, floor, 2.0
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if isinstance(prior, np.ndarray):
            prior = torch.from_numpy(prior).float()
        # prior=1 -> sigmoid(2)=0.88（保留高置信）；prior=0 -> sigmoid(-1)=0.27（留探索空间）
        init = prior.float() * 3.0 - 1.0
        self.causal_mask_kernel = nn.Parameter(init + torch.randn_like(init) * 0.1)
        if cgbp:
            self.W_p = nn.Linear(hid, hid, bias=False)
            self.head = None
        else:
            self.W_p = None
            self.head = NodeHead(hid)

    @property
    def proj(self):
        return self.W_p if self.cgbp else self.head

    def set_temperature(self, t): self.temperature = t

    def small_mask(self):
        return torch.sigmoid(self.causal_mask_kernel / self.temperature)

    def full_mask(self):
        m, N, L = self.small_mask(), self.N, self.tau_max + 1
        out = self.floor * torch.ones(self.target, self.target, device=m.device, dtype=m.dtype)
        for lag in range(L):
            for t in range(L - lag):
                out[t * N:(t + 1) * N, (t + lag) * N:(t + lag + 1) * N] = m[:, :, lag]
        return out

    def forward(self, batch):
        M = self.full_mask()
        feats = []
        for (x, ei, ew) in batch:
            adj = torch.zeros(x.size(0), x.size(0), device=x.device)
            adj[ei[0], ei[1]] = ew
            am = adj * M
            idx = torch.nonzero(torch.abs(am) > 1e-8).t()
            if idx.size(1) == 0:
                idx = torch.tensor([[0], [0]], device=x.device)
                w = torch.zeros(1, device=x.device)
            else:
                w = am[idx[0], idx[1]]
            Z = self.encoder(x, idx, w)
            if self.cgbp:
                feats.append(am * torch.matmul(self.W_p(Z), Z.t()))   # [N,N]
            else:
                feats.append(self.head(Z, idx, w))                    # [N,1]
        return torch.stack(feats, dim=2), M


# ==========================================================
# 3. 损失 + 训练
# ==========================================================

def lrtr(Phi):
    r"""尺度不变 LRTR：
        L_inv = sum_k ||Phi_k - Phi_bar||_F^2 / (sum_k ||Phi_k||_F^2 + eps)  ∈ [0,1]

    分母不可省：未归一化的 sum_k||Phi_k - Phi_bar||^2 存在捷径解 —— M->0 则 Phi≡0、
    L->0，且与稀疏项、二值项同时取下界，空掩码成为全局最优。归一化后分子分母同阶趋零。
    Phi 是 [N,N,K] 还是 [N,1,K] 都适用。
    """
    if Phi.numel() == 0 or Phi.size(2) <= 1:
        return torch.zeros((), device=Phi.device, requires_grad=True)
    mean = Phi.mean(2, keepdim=True).expand_as(Phi)
    return torch.sum((Phi - mean) ** 2) / (torch.sum(Phi ** 2) + 1e-8)


CFG = dict(num_epochs=100, lr=0.005, batch_size=4, lambda_inv=1.0,
           budget_coeff=10.0, binary_coeff=2.0, target_density=0.0265,
           sparsity_start=50, grad_clip=1.0, weight_decay=1e-5,
           mask_lr_mult=2.0, encoder_lr_mult=0.2)


def train(model, data, device="cuda", verbose=True, **kw):
    r"""
    总损失：L = lambda_inv · L_inv + budget_coeff · (m_bar - rho)^2 · ramp
                + binary_coeff · mean(m(1-m))

    掩码不设 weight_decay：一旦 L_inv 的梯度接近 0，L2 会成为唯一梯度源，Adam 归一化后
    步长约等于满学习率，十几个 epoch 就能把 logits 全推到 0，掩码退化成常数 0.5。

    稀疏项用预算靶向 (m_bar - rho)^2 而不是 L1：L1 只单向下压，会与二值项合谋把掩码压空。
    预算项对每个条目的梯度相同（是均匀下压，不是选择器），必须与二值项同量级联立
    —— 只有在 m_bar=rho 且 m∈{0,1} 的约束下，"选哪些"才由 L_inv 决定。
    """
    cfg = dict(CFG); cfg.update({k: v for k, v in kw.items() if k in CFG})
    model = model.to(device)

    g = [{"params": [model.causal_mask_kernel], "lr": cfg["lr"] * cfg["mask_lr_mult"],
          "weight_decay": 0.0},
         {"params": list(model.proj.parameters()), "lr": cfg["lr"],
          "weight_decay": cfg["weight_decay"]}]
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc:
        g.append({"params": enc, "lr": cfg["lr"] * cfg["encoder_lr_mult"],
                  "weight_decay": cfg["weight_decay"]})
    opt = Adam(g)

    hist = {k: [] for k in ("lrtr", "spar", "binv", "mask_mean", "mask_std",
                            "mask_density", "phi_std")}
    n = len(data)
    for ep in range(cfg["num_epochs"]):
        model.set_temperature(1.0 if ep < 60 else max(0.1, 1.0 - 0.4 * (ep - 60) / 40.0))
        model.train()
        perm = torch.randperm(n)
        acc = dict(lrtr=0.0, spar=0.0, binv=0.0, pstd=0.0); steps = 0
        for i in range(0, n, cfg["batch_size"]):
            b = [(f.to(device), e.to(device), w.to(device))
                 for f, e, w in (data[int(j)] for j in perm[i:i + cfg["batch_size"]])]
            if len(b) < 2:                       # LRTR 至少要两个视图
                continue
            opt.zero_grad()
            Phi, _ = model(b)
            Li = lrtr(Phi)
            m = model.small_mask()
            if ep < cfg["sparsity_start"]:
                spar = torch.zeros((), device=device); bw = cfg["binary_coeff"] * 0.01
            else:
                ramp = min(1.0, (ep - cfg["sparsity_start"]) / 10.0 + 0.1)
                spar = cfg["budget_coeff"] * (m.mean() - cfg["target_density"]) ** 2 * ramp
                bw = cfg["binary_coeff"] * min(1.0, 0.05 + (ep - cfg["sparsity_start"]) / 20.0)
            binl = bw * torch.mean(m * (1 - m))
            tot = Li * cfg["lambda_inv"] + spar + binl
            tot.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            opt.step()
            acc["lrtr"] += float(Li); acc["spar"] += float(spar)
            acc["binv"] += float(binl); acc["pstd"] += float(Phi.detach().std())
            steps += 1
            del b, Phi, tot
            if device != "cpu":
                torch.cuda.empty_cache()
        steps = max(1, steps)
        with torch.no_grad():
            m = model.small_mask()
            hist["mask_mean"].append(float(m.mean()))
            hist["mask_std"].append(float(m.std()))
            hist["mask_density"].append(float((m > 0.5).float().mean()))
        for k, s in (("lrtr", "lrtr"), ("spar", "spar"), ("binv", "binv"), ("phi_std", "pstd")):
            hist[k].append(acc[s] / steps)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d} | LRTR {hist['lrtr'][-1]:.3e} | m mean {hist['mask_mean'][-1]:.3f} "
                  f"std {hist['mask_std'][-1]:.4f} | std(Phi) {hist['phi_std'][-1]:.3e}")
    if hist["mask_std"][-1] < 1e-3:
        print(f"  [WARN] mask_std={hist['mask_std'][-1]:.2e} -> 掩码几乎无分化，排序无意义，"
              f"该结果不可用。先跑 README 的 LRTR 判决实验。")
    return model, hist


@torch.no_grad()
def node_head_scores(model, data, device="cuda"):
    """B4 一致性检查：head 输出的 |mean_k s_i|。不作为读出（读出统一走掩码）。"""
    if model.cgbp:
        return None
    model.eval()
    ss = []
    for (x, ei, ew) in data[:16]:
        b = [(x.to(device), ei.to(device), ew.to(device))]
        Phi, _ = model(b)
        ss.append(Phi[:, 0, 0].detach().cpu().numpy())
    return np.abs(np.mean(np.stack(ss, 0), 0))


# ==========================================================
# 4. 数据视图
# ==========================================================

_CACHE: Dict[str, Any] = {}


def causal_views(cfg):
    """因果扰动视图（B2 / B4 / FULL）：100 个 leave-one-out 视图。
    CISE 内部把 PCMCI 的时间因果边（lag 1..tau）与 SCMI 的空间边（lag 0）融合成
    [N(tau+1), N(tau+1)] 邻接。注意这些视图只是训练输入，导出底图是 --pcmci-full-pkl。"""
    k = f"causal::{cfg['pcmci_dir']}::{cfg['scmi_dir']}"
    if k in _CACHE:
        return _CACHE[k]
    C = cise(cfg.get("cise_path"))
    sc = C.build_global_scaler(cfg["feature_file"], target_dim=cfg["input_dim"],
                              original_nodes=cfg["num_nodes"])
    _CACHE[k] = C.load_causal_perturbed_graphs(
        pcmci_dir=cfg["pcmci_dir"], scmi_dir=cfg["scmi_dir"],
        feature_file_path=str(cfg["feature_file"]), num_nodes=cfg["num_nodes"],
        tau_max=cfg["tau_max"], scaler=sc)
    return _CACHE[k]


def sym_lagged_corr(X, tau_max):
    r"""C_bar[i,j,l] = 0.5*(corr(x_i^t, x_j^{t+l}) + corr(x_j^t, x_i^{t+l}))

    必须显式对称化：corr(x_i^t, x_j^{t+l}) 在 l>=1 时不对称，那是滞后互相关
    （无条件 Granger），是有向量。B3 声明无向，就必须拿对称张量。
    """
    T, n = X.shape
    Z = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    C = np.zeros((n, n, tau_max + 1))
    for l in range(tau_max + 1):
        A, B = (Z[:T - l], Z[l:]) if l > 0 else (Z, Z)
        Cl = (A.T @ B) / max(1, A.shape[0])
        C[:, :, l] = 0.5 * (Cl + Cl.T)
    return np.nan_to_num(C)


def knn_binary(A_abs, D, k=15, radius=10.0):
    """半径 r 内取 k 个最强邻居、union -> 对称二值矩阵（相关图的构图方式）。"""
    C = np.array(A_abs, float, copy=True)
    n = C.shape[0]
    np.fill_diagonal(C, -np.inf)
    if radius is not None:
        C[D > radius] = -np.inf
    np.fill_diagonal(C, -np.inf)
    A = np.zeros((n, n), bool)
    for i in range(n):
        v = np.isfinite(C[i])
        if not v.any():
            continue
        kk = min(int(k), int(v.sum()))
        idx = np.argpartition(-C[i], kk - 1)[:kk]
        A[i, idx[np.isfinite(C[i, idx])]] = True
    A = A | A.T
    np.fill_diagonal(A, False)
    return A.astype(float)


def corr_knn_skeleton(corr_adj, pos, knn_k=15, radius=10.0):
    """全 N 节点上的 lag-0 相关 KNN 骨架（无向无权），B3 打分的底图。
    有 --corr-edges-npy 时应优先用那份权威骨架，本函数是回退路径。"""
    p = np.asarray(pos, float)
    D = np.sqrt(((p[:, None, :2] - p[None, :, :2]) ** 2).sum(-1))
    return knn_binary(np.abs(np.asarray(corr_adj, float)), D, knn_k, radius)


def corr_views(cfg):
    r"""相关扰动视图（B3）：与因果路径结构对位的 100 个 leave-one-out 视图。

        W^(v)[:,:,l] = KNN_{k,r}( |C_bar^(v)[:,:,l]| )     对称二值 -> 无向无权

    - 构图方式用 KNN，与 MGC-RM 同 k 同 r：这是相关性方法的原生构图，也堵住
      "只是给基线更差的稀疏度"这一质疑。
    - lag-0 走 s0 槽位（空间统计量）、lag>=1 走 W（时间统计量），与因果路径
      "SCMI 占 lag-0、PCMCI 占 lag-k"对位。若把相关矩阵直接塞进 lag-0，
      expand_and_merge 内部的 combined_lag0 = pcmci_lag0 + scmi_norm 会把负相关抵消。
    - 除图本身外，scaler / 节点特征 / expand_and_merge / 视图数量全部与因果路径一致，
      这样 "FULL − B3 = 因果图的贡献" 才成立。
    """
    k = f"corr::{cfg['feature_file']}::{cfg['tau_max']}::k{cfg['knn_k']}::r{cfg['radius']}"
    if k in _CACHE:
        return _CACHE[k]
    C = cise(cfg.get("cise_path"))
    N, tau = cfg["num_nodes"], cfg["tau_max"]
    df = pd.read_csv(cfg["feature_file"])
    X = df.select_dtypes(include=[np.number]).values
    if X.shape[0] == N and X.shape[1] != N:
        X = X.T
    pos = pd.read_csv(cfg["position_csv"])[["x", "y"]].values
    Dall = np.sqrt(((pos[:, None, :2] - pos[None, :, :2]) ** 2).sum(-1))
    sc = C.build_global_scaler(cfg["feature_file"], target_dim=cfg["input_dim"], original_nodes=N)
    base = C.process_features_from_file(cfg["feature_file"], target_nodes=N * (tau + 1),
                                        target_dim=cfg["input_dim"], original_nodes=N,
                                        tau_max=tau, scaler=sc)

    out = []
    for v in range(N):
        Cv = sym_lagged_corr(np.delete(X, v, axis=1), tau)
        Dv = np.delete(np.delete(Dall, v, axis=0), v, axis=1)
        W = np.zeros_like(Cv)
        for l in range(1, tau + 1):
            W[:, :, l] = knn_binary(np.abs(Cv[:, :, l]), Dv, cfg["knn_k"], cfg["radius"])
        W[:, :, 0] = 0.0
        s0 = knn_binary(np.abs(Cv[:, :, 0]), Dv, cfg["knn_k"], cfg["radius"])
        W = np.insert(np.insert(W, v, 0., 0), v, 0., 1)
        s0 = np.insert(np.insert(s0, v, 0., 0), v, 0., 1)
        ex = C.expand_and_merge_perturbed_matrix(W, s0, N, tau, 1.0, 1.0)
        adj = torch.FloatTensor(ex)
        msk = torch.abs(adj) > 1e-6
        ei = torch.nonzero(msk).t()
        ew = adj[msk]
        if ei.size(1) == 0:
            ei = torch.arange(adj.size(0)).unsqueeze(0).repeat(2, 1)
            ew = torch.ones(adj.size(0))
        out.append((base.clone(), ei, ew))
        if (v + 1) % 25 == 0:
            print(f"  [B3] {v+1}/{N} corr-KNN views (k={cfg['knn_k']}, r={cfg['radius']}, "
                  f"|E|={int(msk.sum())})")
    _CACHE[k] = out
    return out


# ==========================================================
# 5. 注册表
# ==========================================================

METHODS: Dict[str, Dict[str, Any]] = {
    "B2_GCN_LRTR": dict(views="causal", encoder="gcn", cgbp=True, freeze=False,
                        branch="edge", induce="pcmci_mask", rank_edges=True,
                        desc="STC-GCM + GCN + LRTR（无对抗解耦）| FULL−B2 = CADGNN 的贡献"),
    # B3 与 B2/FULL 同为边级方法，只是边的候选列表来自 KNN 构图而不是 PCMCI，
    # 因此 FULL−B3 的唯一变量仍是"图源"。
    "B3_CORR_INV": dict(views="corr", encoder="gcn", cgbp=True, freeze=False,
                        branch="edge", induce="corr_knn_mask", rank_edges=True,
                        desc="相关扰动多视图(KNN 构图) + GCN + 不变学习（无向无权）"
                             "| FULL−B3 = 因果图的贡献"),
    "B4_NOCGBP":   dict(views="causal", encoder="cadgnn", cgbp=False, freeze=True,
                        branch="node", induce="pcmci_all", rank_edges=False,
                        desc="去 CGBP（NodeHead -> 节点因果分数）| FULL−B4 = CGBP 的贡献"),
    "FULL_CISE":   dict(views="causal", encoder="cadgnn", cgbp=True, freeze=True,
                        branch="edge", induce="pcmci_mask", rank_edges=True,
                        desc="完整 CISE-LRTR（默认复用已有不变子图结果 --full-mask-dir）"),
    "PRIOR_ONLY":  dict(views="causal", encoder="cadgnn", cgbp=True, freeze=True,
                        branch="edge", induce="pcmci_mask", rank_edges=True, skip_train=True,
                        desc="[诊断，不进主表] 不训练，掩码=平稳先验核"),
}

DEFAULT_CFG = dict(
    cise_path=None, vis_path=None,
    feature_file=None, pcmci_dir=None, scmi_dir=None,
    pcmci_full_pkl=None, position_csv=None, cadgnn_path=None,
    out_dir="./results",
    num_nodes=100, tau_max=3, input_dim=8000, hidden_dim=128,
    keep_frac=0.0265,                 # 掩码边密度目标，与节点预算无关
    node_budget=SG.NODE_BUDGET,       # 唯一来源在 subgraph.NODE_BUDGET
    knn_k=15, radius=10.0,
    mask_transposed=True,             # 见 mask_to_scores
)


def build_encoder(kind, cfg, dev):
    if kind == "gcn":
        return PlainGCN(cfg["input_dim"], cfg["hidden_dim"]).to(dev)
    C = cise(cfg.get("cise_path"))
    p = cfg["cadgnn_path"]
    if os.path.exists(p):
        ck = torch.load(p, map_location=dev)
        mc = ck.get("model_config", {"input_dim": cfg["input_dim"], "hidden_dim": cfg["hidden_dim"]})
        cad = C.CADGNN(mc["input_dim"], mc["hidden_dim"], 5, 0.1, 0.02).to(dev)
        cad.load_state_dict(ck["model_state_dict"])
    else:
        print(f"[WARN] CADGNN ckpt 缺失: {p} -> 随机初始化（仅调试）")
        cad = C.CADGNN(cfg["input_dim"], cfg["hidden_dim"], 5, 0.1, 0.02).to(dev)
    return FrozenCADGNN(cad).to(dev)


# ==========================================================
# 6. 掩码 -> 分数
# ==========================================================

def _fold_mask(full_mask, N, tau_max, transposed=True):
    """[N(L), N(L)] -> [N,N,L]：同一 lag 的各时间块取均值；必要时转置回底图方向。"""
    L = tau_max + 1
    M = np.zeros((N, N, L))
    for lag in range(L):
        M[:, :, lag] = np.mean(np.stack(
            [full_mask[t * N:(t + 1) * N, (t + lag) * N:(t + lag + 1) * N]
             for t in range(L - lag)], 0), 0)
    if transposed:
        M = np.transpose(M, (1, 0, 2))
    return M


def to_lagged_mask(arr, N, tau_max, transposed=True):
    """把掩码统一成 [N,N,L]：接受 [N,N,L]（直接用）或 [N(L),N(L)]（折叠）。"""
    a = np.asarray(arr, float)
    L = tau_max + 1
    if a.shape == (N, N, L):
        return np.transpose(a, (1, 0, 2)) if transposed else a
    if a.shape == (N * L, N * L):
        return _fold_mask(a, N, tau_max, transposed)
    raise ValueError(f"掩码形状无法解析: {a.shape}（期望 ({N},{N},{L}) 或 ({N*L},{N*L})）")


def _pick_mask(obj, N, L, temperature):
    """在任意嵌套结构里找出掩码数组；键名含 kernel 的做 sigmoid(x/T)。"""
    want = {(N, N, L), (N * L, N * L)}
    if isinstance(obj, dict):
        for k, v in obj.items():
            r = _pick_mask(v, N, L, temperature)
            if r is not None:
                a, tag = r
                if "kernel" in str(k).lower():
                    a = 1.0 / (1.0 + np.exp(-a / temperature))
                    tag = f"{k}(sigmoid/T={temperature})"
                else:
                    tag = str(k)
                return a, tag
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = _pick_mask(v, N, L, temperature)
            if r is not None:
                return r
        return None
    try:
        import torch as _t
        if isinstance(obj, _t.Tensor):
            obj = obj.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(obj, np.ndarray) and obj.shape in want:
        return obj.astype(float), "array"
    return None


def load_mask_dir(path, N, tau_max, temperature=0.6):
    """从已有训练产物目录里取出掩码，返回原始数组（[N,N,L] 或 [N(L),N(L)]）。

    扫描 *.pkl / *.npy / *.npz / *.pth / *.pt，认形状 (N,N,L) 或 (N(L),N(L))；
    若来自 causal_mask_kernel（logits），按 sigmoid(kernel/T) 还原，T 取训练末期的温度
    （默认 0.6，可用 --mask-temperature 改）。会打印命中的文件与键名，跑完核对一下。
    """
    p = Path(path)
    L = tau_max + 1
    files = sorted(list(p.rglob("*.pkl")) + list(p.rglob("*.npy")) + list(p.rglob("*.npz"))
                   + list(p.rglob("*.pth")) + list(p.rglob("*.pt")))
    if not files:
        raise FileNotFoundError(f"{p} 下没有 pkl/npy/npz/pth/pt")
    for f in files:
        try:
            if f.suffix == ".npy":
                obj = np.load(f, allow_pickle=True)
                if obj.dtype == object and obj.ndim == 0:
                    obj = obj.item()
            elif f.suffix == ".npz":
                obj = {k: np.load(f, allow_pickle=True)[k] for k in np.load(f, allow_pickle=True).files}
            elif f.suffix in (".pth", ".pt"):
                obj = torch.load(f, map_location="cpu")
            else:
                import pickle as _pk
                with open(f, "rb") as fh:
                    obj = _pk.load(fh)
        except Exception as e:
            print(f"  [mask-dir] 跳过 {f.name} ({e})")
            continue
        r = _pick_mask(obj, N, L, temperature)
        if r is not None:
            a, tag = r
            print(f"[FULL] 掩码取自 {f}  key={tag}  shape={a.shape}  "
                  f"range=[{a.min():.3f},{a.max():.3f}]  std={a.std():.4f}")
            return a
    raise ValueError(f"{p} 下没找到形状为 ({N},{N},{L}) 或 ({N*L},{N*L}) 的掩码")


def mask_to_scores(full_mask, base_graph, base_val, N, tau_max,
                   keep_frac=0.0265, transposed=True):
    r"""
    因果路径（B2 / B4 / FULL）的打分。底图是原始 PCMCI 图（--pcmci-full-pkl）：

        s_E[i,j,l] = 1[M > eta] · |val[i,j,l]| · 1[graph[i,j,l] == '-->']
        s_V        = s_E 的入 + 出

    transposed：expand_and_merge_perturbed_matrix 写入时带转置
        （expanded[a+tN, b+(t+l)N] = P[b,a,l]），而 get_causal_mask 不带，
        因此训练中 m[a,b,l] 门控的是底图的边 b->a，导出必须转置回来。
        跑之前用 README 里的三行验证确认，不对就加 --mask-not-transposed。

    阈值一律用 quantile：掩码的绝对取值受预算项影响，可能整体停在 0.49 或被压到 0.11，
        用绝对阈值 0.5 会导致 strong 全零 -> 节点分数全并列 -> argsort 退化为按索引。

    B4 也走这里：读出统一从掩码派生，FULL 与 B4 必须同一条读出路径，否则"唯一变量"
        会变成"读掩码 vs 读 head 输出"。B4 的 s_E 只用于派生 s_V，不用于排边 ——
        SG.build(induce='pcmci_all') 会把 edge_score 置 None。
    """
    M = to_lagged_mask(full_mask, N, tau_max, transposed)
    L = tau_max + 1

    thr = float(np.quantile(M, 1.0 - keep_frac))
    strong = (M > thr).astype(float)
    if float(M.std()) < 1e-3:
        print(f"  [WARN] 掩码 std={M.std():.2e} -> 排序无意义")

    V = np.abs(np.asarray(base_val, float))
    is_edge = (np.asarray(base_graph) == "-->")
    sE = strong * V * is_edge
    for l in range(L):
        np.fill_diagonal(sE[:, :, l], 0.0)
    sV = sE.sum((1, 2)) + sE.sum((0, 2))
    return sE, sV, thr


def mask_to_scores_corr(full_mask, skel, N, tau_max, keep_frac=0.0265):
    r"""
    相关路径（B3）的打分。掩码打在相关 KNN 骨架上，不碰 PCMCI：

        M_bar      = (M + M^T)/2                      相关图无向 -> 掩码先对称化
        s_E[i,j,l] = 1[M_bar > eta] · M_bar[i,j,l] · skel[i,j]
        s_V        = s_E 的入 + 出

    与因果路径的口径差异（正文需交代）：因果图有底图权 |val| 可乘，相关 KNN 图无权、
    没有底图权可用，所以边分数只能是掩码强度本身 —— B3 的边分数唯一的信息来源就是
    不变性学习学到的掩码。

    不做转置：相关图对称，转与不转等价。
    """
    M = to_lagged_mask(full_mask, N, tau_max, transposed=False)
    M = 0.5 * (M + np.transpose(M, (1, 0, 2)))
    thr = float(np.quantile(M, 1.0 - keep_frac))
    if float(M.std()) < 1e-3:
        print(f"  [WARN] 掩码 std={M.std():.2e} -> 排序无意义")
    S = np.asarray(skel, float)
    sE = (M > thr).astype(float) * M * S[:, :, None]
    for l in range(tau_max + 1):
        np.fill_diagonal(sE[:, :, l], 0.0)
    sV = sE.sum((1, 2)) + sE.sum((0, 2))
    return sE, sV, thr
