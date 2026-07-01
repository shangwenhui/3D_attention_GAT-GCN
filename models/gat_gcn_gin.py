import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GAT_GCN_GIN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_edge=1, output_dim=128, dropout=0.2):
        super(GAT_GCN_GIN, self).__init__()

        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # ---------- GAT layers (using edge_attr) ----------
        self.gat_conv1 = GATConv(num_features_xd, num_features_xd, heads=10, edge_dim=num_features_edge)
        self.gat_conv2 = GATConv(num_features_xd * 10, num_features_xd * 5, edge_dim=num_features_edge)

        # ---------- GCN layers (using edge_weight) ----------
        self.gcn_conv1 = GCNConv(num_features_xd * 5, num_features_xd * 10)
        self.gcn_conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)

        # ---------- GIN layers ----------
        gin_dim = 32
        nn1 = Sequential(Linear(num_features_xd * 10, gin_dim), ReLU(), Linear(gin_dim, gin_dim))
        self.gin_conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(gin_dim)

        nn2 = Sequential(Linear(gin_dim, gin_dim), ReLU(), Linear(gin_dim, gin_dim))
        self.gin_conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(gin_dim)

        nn3 = Sequential(Linear(gin_dim, gin_dim), ReLU(), Linear(gin_dim, gin_dim))
        self.gin_conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(gin_dim)

        # ---------- Fully connected layers ----------
        self.fc_g1 = torch.nn.Linear(gin_dim * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)

        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Extract distance features
        if hasattr(data, 'edge_s') and data.edge_s is not None:
            edge_attr = data.edge_s          # [E, 1]
            edge_weight = data.edge_s.squeeze()  # [E]
        else:
            edge_attr = None
            edge_weight = None

        # ---------- GAT layers ----------
        x = self.gat_conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.gat_conv2(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)

        # ---------- GCN layers ----------
        x = self.gcn_conv1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.gcn_conv2(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)

        # ---------- GIN layers ----------
        x = self.gin_conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gin_conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.gin_conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)

        # ---------- Global pooling ----------
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # ---------- Fully connected ----------
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out