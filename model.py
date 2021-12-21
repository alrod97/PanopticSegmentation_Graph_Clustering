import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

## GRAPH ATTENTION LAYER


class GNN(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # first GAT Layer
        #self.GAT_1 = GATLayer(in_features=self.in_features , out_features=self.out_features , dropout=dropout, alpha=alpha)
        self.GAT_1 = GATConv(in_channels=3, out_channels=32, heads=4, concat=False, edge_dim=1)
        # Second GAT Layer
        self.GAT_2 = GATConv(in_channels=32, out_channels=32, heads=4, concat=False)

        # Standard Feed forward NN
        self.fc1 = nn.Linear(128 , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x, edges, edge_feature):
        #
        x_0 = self.GAT_1(x, edges, edge_feature)
        x = self.GAT_2(x_0, edges)

        ## construct N^2 x 2D matrix for every edge
        N = x.size()[0]**2
        D = (x.size()[1] + x_0.size()[1])*2

        x_edges = torch.zeros((N, D))
        for index, edge_cur in enumerate(edges.T):
            x_edges[index, :] = torch.concat((x_0[edge_cur[0]] , x_0[edge_cur][1], x[edge_cur][0], x[edge_cur][1]))

        ##

        # MLP Layers with Relu and softmax to get probanility of every edge being true or not
        x_edges = F.relu(self.fc1(x_edges))
        x_edges = F.relu(self.fc2(x_edges))
        y = F.softmax(self.fc3(x_edges), dim=1)

        return y

