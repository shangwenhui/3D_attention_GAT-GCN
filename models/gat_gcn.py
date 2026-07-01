import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class AtomicAttentionLayer(nn.Module):
    """
    Atomic attention layer: Calculate an attention weight for each atom and weight the node features.
    input:
        x: Node feature matrix [num_nodes, feature_dim]
        batch: Batch vector [num_nodes], indicating which graph each node belongs to.
    output:
        weighted_x: The weighted node features [num_nodes, feature_dim]
        attention_weights: Attention weights [num_nodes, 1]
    """
    def __init__(self, input_dim):
        super(AtomicAttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, batch):
        # Calculate the original attention scores.
        attention_scores = self.attention_weights(x)  # [num_nodes, 1]

        # softmax
        unique_batches = torch.unique(batch)
        attention_weights_list = []
        for batch_idx in unique_batches:
            mask = (batch == batch_idx)
            batch_scores = attention_scores[mask]
            if len(batch_scores) > 0:
                batch_weights = self.softmax(batch_scores)
                attention_weights_list.append(batch_weights)

        # Concatenate the attention weights of all batches.
        if attention_weights_list:
            attention_weights = torch.cat(attention_weights_list, dim=0)
        else:
            # Safe fallback: If there is no valid batch, return all-1 weights.
            attention_weights = torch.ones_like(attention_scores)

        # Weighted node features
        weighted_x = x * attention_weights

        return weighted_x, attention_weights


class GAT_GCN(torch.nn.Module):
    """
    GAT + GCN hybrid model, supporting 3D distance features (edge_s),
    and incorporating an atomic attention mechanism after convolution.
    """
    def __init__(self, n_output=1, num_features_xd=78, output_dim=128, dropout=0.2, edge_dim=1):
        super(GAT_GCN, self).__init__()

        self.n_output = n_output

        # First layer GAT: use edge features (edge_attr)
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10, edge_dim=edge_dim)
        # Second layer GCN: use edge weights (edge_weight)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)

        # ---- Newly added: Atomic attention layer ----
        # Input dimension is the output dimension of conv2
        self.atomic_attention = AtomicAttentionLayer(num_features_xd * 10)

        # Fully connected layers (consistent with original structure)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Optionally store attention weights for analysis
        self.attention_weights = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- Extract distance features (keep original logic) ----
        if hasattr(data, 'edge_s') and data.edge_s is not None:
            edge_attr = data.edge_s          # [E, 1] for GAT
            edge_weight = data.edge_s.squeeze()  # [E] for GCN
        else:
            edge_attr = None
            edge_weight = None

        # ---- First layer GAT: use edge_attr ----
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)

        # ---- Second layer GCN: use edge_weight ----
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)

        # ---- Newly added: Atomic attention layer (weighting node features) ----
        x, attention_weights = self.atomic_attention(x, batch)
        # Save attention weights for post-analysis
        self.attention_weights = attention_weights

        # ---- Global pooling (max + mean) ----
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # ---- Fully connected part (keep unchanged) ----
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

    def get_attention_weights(self):
        """Returns the most recent atomic attention weights for interpretability analysis."""
        return self.attention_weights