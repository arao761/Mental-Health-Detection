import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class GNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_size, hidden_size)
        self.conv2 = pyg_nn.GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x
    