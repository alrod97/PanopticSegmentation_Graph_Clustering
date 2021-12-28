from scipy.sparse.csgraph import connected_components
import numpy as np
import torch
from scipy.sparse import csr_matrix

def get_clusters(pred, info_path):
    # pred is the N^2 x 1 array with 1 if edge and 0 if no edge

    # edges is N^2 x 2 array with edges (0,0), (0,1) ...
    # pred is N^2 x 1 array with prob of true edge 1 for true and 0 for no edge
    # info_path is path to dictionary with indexes cluster information

    # extract points of things
    info_clusters = torch.load(info_path)
    points = info_clusters['points']
    indexes = np.zeros(points.shape[0])
    # construct Adj_pred matrix from pred
    N = int(np.sqrt(pred.shape[0]))

    Adj_pred = np.reshape(pred, (N, N))

    graph = csr_matrix(Adj_pred)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    print(n_components)
    for component in range(n_components):
        # get which sub-clusters belong to this cluster
        corresponding_sub_clusters = np.argwhere(np.asarray(labels) == component)
        # go over each points of subcluster and assigns them the same isntance id
        for id in corresponding_sub_clusters:
            # read indexes to pc
            indexes[info_clusters[id[0]]] = component

    return indexes

pred = torch.load('graph_data/predictions/pred.pt').detach().numpy()
pred = np.argmax(pred, axis=1)

final_pred = np.zeros_like(pred)
final_pred[pred == 0] = 1

print(final_pred.sum())
get_clusters(final_pred, 'graph_data/processed/graph_data_0_ALL_INFO.pt')
