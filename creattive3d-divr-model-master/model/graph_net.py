import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv, SAGEConv


class GCNGraphEmbedding(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GCNGraphEmbedding, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global Mean Pooling to get graph-level embedding
        x = global_mean_pool(x, batch)  # Aggregate features from all nodes in a graph

        # Optional: A linear layer to transform the graph embedding
        x = self.linear(x)

        return x

class TemporalGCNGraphEmbedding_het(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels, num_timestamps):
        super(TemporalGCNGraphEmbedding_het, self).__init__()
        self.num_timestamps = num_timestamps

        # MLP that will be used to compute edge weights from edge features
        mlp = Sequential(
            Linear(num_edge_features, hidden_channels),  # First layer maps from edge features to hidden_channels
            ReLU(),
            Linear(hidden_channels, num_node_features * hidden_channels)  # Adjusted to match expected size
        )
        self.convs = torch.nn.ModuleList([
            NNConv(num_node_features, hidden_channels, mlp, aggr='mean') for _ in range(num_timestamps)
        ])

        mlp2 = Sequential(
            Linear(num_edge_features, hidden_channels),  # First layer maps from edge features to hidden_channels
            ReLU(),
            Linear(hidden_channels, hidden_channels * hidden_channels)  # Adjusted to match expected size
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, mlp2, aggr='mean')
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, data):
        batch = data.batch
        x_all_timestamps = []

        for t in range(self.num_timestamps):
            x = getattr(data, f'x_t{t}')
            edge_index = getattr(data, f'edge_index_t{t}')
            edge_attr = getattr(data, f'edge_feat_t{t}')

            x = self.convs[t](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
            x = global_mean_pool(x, batch)  # Aggregate features for each graph in the batch
            x_all_timestamps.append(x)

        # Concatenate features from all timestamps
        x_concat = torch.stack(x_all_timestamps, dim=1)

        # Final linear transformation
        x_out = self.linear(x_concat)

        return x_out


