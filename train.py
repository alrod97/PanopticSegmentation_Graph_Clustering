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
    for _, batch in enumerate(tqdm.tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        Adj = torch.ones((batch.x.size()[0], batch.x.size()[0]))
        row, col = np.where(Adj)
        coo = np.array(list(zip(row, col)))
        edges = torch.from_numpy(np.reshape(coo, (2, -1)))

        pred = model(batch.x , edges, batch.edge_attr[:, None])
        # Calculating the loss and gradients
        # get ground truth labels edges
        Adj_gt = torch.reshape(batch.y, (batch.x.size()[0], batch.x.size()[0]))

        labels_gt = torch.zeros(edges.size()[1])
        # print('--')
        for index, edge_cur in enumerate(batch.edge_index.T):
            if Adj_gt[edge_cur[0], edge_cur[1]] == 1:
                labels_gt[index] = 1
            else:
                labels_gt[index] = 0

        loss = loss_fn(torch.squeeze(pred[:, 0]), labels_gt.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        print(loss.item())
        #all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        #all_labels.append(batch.y.cpu().detach().numpy())

    #all_preds = np.concatenate(all_preds).ravel()
    #all_labels = np.concatenate(all_labels).ravel()
    #calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

train_dataset = PanopticClusteringDataset(root='graph_data/')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
model = GNN(in_features=3, out_features=32, dropout=0, alpha=0.2)
loss_fn  = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs_total = np.arange(10)
for epoch in epochs_total:
    train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)

