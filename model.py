import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

## GRAPH ATTENTION LAYER

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)  # matrix multiplication
        N = h.size()[0]
        print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GNN(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # first GAT Layer
        #self.GAT_1 = GATLayer(in_features=self.in_features , out_features=self.out_features , dropout=dropout, alpha=alpha)
        self.GAT_1 = GATConv(in_channels=3, out_channels=32, heads=4, concat=False, edge_dim=1)
        # Second GAT Layer
        self.GAT_2 = GATLayer(in_features=self.out_features, out_features=self.out_features, dropout=dropout, alpha=alpha)

        # Standard Feed forward NN
        self.fc1 = nn.Linear(self.out_features + 3, int(self.out_features/2))
        self.fc2 = nn.Linear(int(self.out_features / 2), 1)

    def forward(self, x, edges, edge_feature):
        #
        x = self.GAT_1(x, edges, edge_feature)
        #x = self.GAT_2(x, adj)
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = F.sigmoid(x)

        return x

model = GNN(in_features=3, out_features=32, dropout=0, alpha=0.2)
