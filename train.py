import numpy as np
import tqdm
import torch
from model import GNN
from dataset import PanopticClusteringDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                                batch.edge_attr.float(),
                                batch.edge_index,
                                batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    #calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

train_dataset = PanopticClusteringDataset(root='graph_data/')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
model = GNN(in_features=3, out_features=32, dropout=0, alpha=0.2)

print(train_loader)

for i in train_loader:
    # get xyz points
    points = i.x
    edge_index = i.edge_index
    label = i.y
    edge_feature = i.edge_attr
    edge_feature = edge_feature[:, None]

    print(edge_feature.size())

    Adj = torch.reshape(label, (points.size()[0], points.size()[0]))
    Adj = torch.ones((points.size()[0], points.size()[0]))
    row, col = np.where(Adj)
    coo = np.array(list(zip(row, col)))
    edges = torch.from_numpy(np.reshape(coo, (2, -1)))
    print(edges.size())
    output = model(points, edges, edge_feature)

    print(points.size())
    print(output.size())
    print('--')

model = GNN(in_features=3, out_features=32, dropout=0, alpha=0.2)
