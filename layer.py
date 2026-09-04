import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, TransformerConv
from config import args


# ------------------------------------------------------------------ #
# GCN layer
# ------------------------------------------------------------------ #
class GCNNet(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, hid_dim=args.hidden, bias=True):
        super(GCNNet, self).__init__()
        self.res1 = GCNConv(in_dim, hid_dim, bias=bias)
        self.res2 = GCNConv(hid_dim, hid_dim, bias=bias)
        self.res3 = GCNConv(hid_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(p=args.dropout)
        self.dropout2 = nn.Dropout(p=args.dropout)

    def forward(self, x, edge_index):
        h = self.res1(x, edge_index)
        h = F.relu(h)
        h = self.dropout1(h)          # 第一层后 Dropout
        h = self.res2(h, edge_index)
        h = F.relu(h)
        h = self.dropout2(h)
        output = self.res3(h, edge_index)
        return output


# ------------------------------------------------------------------ #
# GAT Network
# ------------------------------------------------------------------ #
class GATNet(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, heads=8, hid_dim=args.hidden, bias=True):
        super(GATNet, self).__init__()
        self.res1 = GATv2Conv(in_dim, out_dim * 70, heads, bias=bias, concat=False)
        self.res2 = GATv2Conv(out_dim * 70, out_dim, heads, bias=bias, concat=False)

    def forward(self, x, edge_index):  # g: adj, z: feature
        h = self.res1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=args.dropout, training=self.training)
        output = self.res2(h, edge_index)
        return output


# ------------------------------------------------------------------ #
# Hybrid GNN
# ------------------------------------------------------------------ #
class HybridGNN(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, heads=8, hid_dim=args.hidden, bias=True):
        super(HybridGNN, self).__init__()
        self.gcn1 = GCNConv(in_dim, out_dim * 70, bias=bias)
        self.gat1 = GATv2Conv(out_dim * 70, out_dim * 70, heads, bias=bias, concat=False)
        self.gat2 = GATv2Conv(out_dim * 70, out_dim, heads, bias=bias, concat=False)

    def forward(self, x, edge_index):  # g: adj, z: feature
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=args.dropout, training=self.training)
        output = self.gat2(h, edge_index)
        return output


# ------------------------------------------------------------------ #
# Transformer GNN
# ------------------------------------------------------------------ #
class TransformerGNN(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, hid_dim=args.hidden, bias=True):
        super(TransformerGNN, self).__init__()
        self.conv1 = TransformerConv(in_dim, out_dim * 70, heads=4, bias=bias, concat=False)
        self.conv2 = TransformerConv(out_dim * 70, out_dim, heads=4, bias=bias, concat=False)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=args.dropout, training=self.training)
        output = self.conv2(h, edge_index)
        return output


# ------------------------------------------------------------------ #
# 1-D CNN baseline（无图结构，edge_index 仅为保持接口一致）
# ------------------------------------------------------------------ #
class CNNNet(nn.Module):
    """Treats each spectrum as a 1-D signal of length `in_dim`."""

    def __init__(self, in_dim=args.d, out_dim=args.c, hid_dim=args.hidden, bias=True):
        super(CNNNet, self).__init__()
        self.in_dim = in_dim
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=bias)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=bias)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, out_dim, bias=bias)

    def forward(self, x, edge_index=None):
        h = x.unsqueeze(1)                 # (N, 1, in_dim)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)       # (N, 32)
        return self.fc(h)


# ------------------------------------------------------------------ #
# Fully-connected baseline（同样忽略 edge_index）
# ------------------------------------------------------------------ #
class FCNet(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, hid_dim=args.hidden, bias=True):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim, bias=bias)
        self.fc2 = nn.Linear(hid_dim, out_dim, bias=bias)

    def forward(self, x, edge_index=None):
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=args.dropout, training=self.training)
        return self.fc2(h)
