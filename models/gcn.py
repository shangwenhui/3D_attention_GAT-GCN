import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gap

class GCNNet(torch.nn.Module):
    """
    Graph convolutional network using standard GCNConv.
    Supports passing inter-atomic spatial distances (3D information) via edge_weight.
    """
    def __init__(self, n_output=1, num_features_xd=78, output_dim=128, dropout=0.2):
        super(GCNNet, self).__init__()

        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # GCN layers (optionally use edge_weight)
        self.conv1 = GCNConv(num_features_xd, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Fully connected layers
        self.fc_g1 = nn.Linear(256 * 2, 1024)  # Because both max and mean pooling are used
        self.fc_g2 = nn.Linear(1024, output_dim)

        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # If distance feature edge_s exists, convert it to edge weight (e.g., using exp(-d) or directly using distance)
        if hasattr(data, 'edge_s') and data.edge_s is not None:
            # edge_s has shape [E, 1], convert to [E] and flatten
            edge_weight = data.edge_s.squeeze()  # Directly use normalized distance as weight
            # Alternative transformation: edge_weight = torch.exp(-data.edge_s.squeeze())  # Gaussian kernel
        else:
            edge_weight = None

        # GCN layers (pass edge_weight)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)

        # Global pooling: using both max and mean pooling to enhance representational capacity
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fully connected part
        x = self.fc_g1(x)
        x = self.relu(x)
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