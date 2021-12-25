import numpy as np
import tqdm
import torch
from model import GNN
from dataset import PanopticClusteringDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
#import mlflow.pytorch
import mlflow

from config import HYPERPARAMETERS, SIGNATURE

#mlflow.set_tracking_uri("http://localhost:5000")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

        #all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        #all_labels.append(batch.y.cpu().detach().numpy())

    #all_preds = np.concatenate(all_preds).ravel()
    #all_labels = np.concatenate(all_labels).ravel()
    #calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        # Passing the node features and the connection info
        Adj = torch.ones((batch.x.size()[0], batch.x.size()[0]))
        row, col = np.where(Adj)
        coo = np.array(list(zip(row, col)))
        edges = torch.from_numpy(np.reshape(coo, (2, -1)))

        pred = model(batch.x, edges, batch.edge_attr[:, None])
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

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
   # calculate_metrics(all_preds, all_labels, epoch, "test")
   # log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss / step

import mlflow
def run_one_training(params_list):
    params = params_list[0]
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        print('eeee')
        for key in params.keys():
            print(key)
            mlflow.log_param(key, params[key])

        # Loading the dataset
        print("Loading dataset...")
        train_dataset = PanopticClusteringDataset(root='graph_data/')
        #test_dataset =
        #params["model_edge_dim"] = train_dataset[0].edge_attr.shape[0]

        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader =  DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = GNN(in_features=3, model_params=model_params) # dropout and alpha modelparams
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # < 1 increases precision, > 1 recall
        #weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        # Start training
        best_loss = 10000
        early_stopping_counter = 0
        for epoch in range(50):
            if early_stopping_counter <= 10:  # = x * 5
                # Training
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]

from mango import scheduler, Tuner
# %% Hyperparameter search
print("Running hyperparameter search...")

config = dict()
config['model_aggregation'] = 'GATConv'
#config['model_out_features'] = 32
#config['model_heads'] = 4
config['model_concat'] = False
config['model_edge_dim'] = 1
config['batch_size'] = 1
#config['learning_rate'] = 0.01
#config['sgd_momentum'] = 0.001
#config['weight_decay'] = 0.1
#config['scheduler_gamma'] = 0.1


config = dict()
config["optimizer"] = "Bayesian"
config["num_iteration"] = 100

params = []
params.append(config)

tuner = Tuner(HYPERPARAMETERS,
              objective=run_one_training,
              conf_dict=config)

results = tuner.minimize()
print(results)
#run_one_training(params_list=params)