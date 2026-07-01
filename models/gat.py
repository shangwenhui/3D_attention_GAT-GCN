import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool as gmp

class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, num_features_edge=1, n_output=1, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout, edge_dim=num_features_edge)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout, edge_dim=num_features_edge)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Safely obtain edge_s，if it doesn't 'None'
        edge_attr = getattr(data, 'edge_s', None)
        # GATConv allow edge_attr=None
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out