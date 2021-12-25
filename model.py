import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

## GRAPH ATTENTION LAYER


class GNN(torch.nn.Module):
    def __init__(self, in_features,model_params=None):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = model_params['model_out_features']
        self.aggregation_technique = model_params['model_aggregation']

        if self.aggregation_technique == 'GATConv':
            # Extract Graph Attention parameters
            heads = model_params['model_heads']
            concat = model_params['model_concat']
            edge_dim = model_params['model_edge_dim']
            # first GAT Layer
            self.AGG_1 = GATConv(in_channels=self.in_features, out_channels=self.out_features, heads=heads,
                                 concat=concat, edge_dim=edge_dim)
            # Second GAT Layer
            self.AGG_2 = GATConv(in_channels=self.out_features, out_channels=self.out_features, heads=heads,
                                 concat=concat)

        elif self.aggregation_technique == 'SAGEConv':
            # Extract SAGEConv parameters
            normalize = model_params['model_normalize']
            self.AGG_1 = SAGEConv(in_channels= self.in_features, out_channels = self.out_features, normalize=normalize)
            self.AGG_2 = SAGEConv(in_channels=self.out_features, out_channels=self.out_features, normalize=normalize)
        else:
            print('Error: No Valid Aggregation Technique')

        # Standard Feed forward NN for edge detection on the fully connected graph
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edges, edge_feature):
        #
        x = self.AGG_1(x, edges, edge_feature)
        x = self.AGG_2(x, edges)

        # construct N^2 x 2D matrix for every edge, for every edge we have the feature vector as described in the paper
        N = x.size()[0]**2
        D = x.size()[1]*2

        x_edges = torch.zeros((N, D))
        for index, edge_cur in enumerate(edges.T):
            x_edges[index, :] = torch.concat((x[edge_cur[0]] , x[edge_cur][1]))

        ##

        # MLP Layers with Relu and softmax to get probanility of every edge being true or not
        x_edges = F.relu(self.fc1(x_edges))
        y = F.softmax(self.fc2(x_edges), dim=1)

        return y
