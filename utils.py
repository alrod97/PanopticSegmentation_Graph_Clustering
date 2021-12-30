from scipy.sparse.csgraph import connected_components
import numpy as np
import torch
from scipy.sparse import csr_matrix
import open3d as o3d

def get_clusters(pred, info_path, gt_path):
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
    print(N)
    for component in range(n_components):
        # get which sub-clusters belong to this cluster
        corresponding_sub_clusters = np.argwhere(np.asarray(labels) == component)
        # go over each points of subcluster and assigns them the same isntance id
        for id in corresponding_sub_clusters:
            # read indexes to pc
            indexes[info_clusters[id[0]]] = component

    # load GT
    data = torch.load(gt_path)
    y_gt = data.y.numpy()
    Adj_gt = np.reshape(y_gt, (N, N))

    graph_gt = csr_matrix(Adj_gt)
    n_components_gt, labels_gt = connected_components(csgraph=graph_gt, directed=False, return_labels=True)
    indexes_gt = np.zeros(points.shape[0])
    for component in range(n_components_gt):
        # get which sub-clusters belong to this cluster
        corresponding_sub_clusters = np.argwhere(np.asarray(labels_gt) == component)
        # go over each points of subcluster and assigns them the same isntance id
        for id in corresponding_sub_clusters:
            # read indexes to pc
            indexes_gt[info_clusters[id[0]]] = component

    # overcluster indexes
    indexes_overcluster =  np.zeros(points.shape[0])

    for int_over in range(len(info_clusters)-1):
        indexes_overcluster[info_clusters[int_over]] = int_over

    return indexes, points, indexes_gt, indexes_overcluster

def get_overclusters(info_path):
    # pred is the N^2 x 1 array with 1 if edge and 0 if no edge

    # edges is N^2 x 2 array with edges (0,0), (0,1) ...
    # pred is N^2 x 1 array with prob of true edge 1 for true and 0 for no edge
    # info_path is path to dictionary with indexes cluster information

    # extract points of things
    info_clusters = torch.load(info_path)
    points = info_clusters['points']

    # overcluster indexes
    indexes_overcluster =  np.zeros(points.shape[0])

    print(info_clusters.keys())
    j = 0

    for int_over in info_clusters:
        if int_over == 'points':
            pass
        else:
            indexes_overcluster[info_clusters[int_over]] = j
            j += 1

    instance_colors = {
        -1: [180, 30, 80],
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [245, 150, 100],
        3: [245, 230, 100],
        4: [250, 80, 100],  # 100, 80, 250
        5: [150, 60, 30],
        6: [255, 0, 0],
        7: [180, 30, 80],
        8: [255, 0, 0],
        9: [30, 30, 255],
        10: [200, 40, 255],
        11: [90, 30, 150],
        12: [255, 0, 255],
        13: [255, 150, 255],
        14: [75, 0, 75],
        15: [75, 0, 175],
        16: [0, 200, 255],
        17: [50, 120, 255],
        18: [0, 150, 255],
        19: [170, 255, 150],
        20: [0, 175, 0],
        21: [0, 60, 135],
        22: [245, 150, 100],
        23: [255, 0, 0],
        24: [200, 40, 255],
        25: [30, 30, 255],
        26: [90, 30, 150],
        27: [250, 80, 100],
        28: [180, 30, 80],
    }

    color_inst = np.zeros_like(points)
    for instance_cur in np.unique(indexes_overcluster):
        color_cur = instance_colors[instance_cur]

        color_inst[indexes_overcluster == instance_cur, 0] = color_cur[2]
        color_inst[indexes_overcluster == instance_cur, 1] = color_cur[1]
        color_inst[indexes_overcluster == instance_cur, 2] = color_cur[0]


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color_inst.astype(np.float) / 255.0)
    o3d.visualization.draw_geometries([pcd])


pred = torch.load('graph_data/predictions/pred.pt').detach().numpy()
pred = np.argmax(pred, axis=1)

final_pred = np.zeros_like(pred)
final_pred[pred == 0] = 1

#print(final_pred.sum())
indexes, points, indexes_gt, indexes_overcluster= get_clusters(final_pred, 'graph_data/processed/graph_data_0_ALL_INFO.pt', 'graph_data/processed/graph_data_0_ALL_.pt')

get_overclusters(info_path='graph_data/processed/graph_data_0_ALL_INFO.pt')


color_inst = np.zeros_like(points)

instance_colors = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [245, 150, 100],
            3: [245, 230, 100],
            4: [250, 80, 100],  # 100, 80, 250
            5: [150, 60, 30],
            6: [255, 0, 0],
            7: [180, 30, 80],
            8: [255, 0, 0],
            9: [30, 30, 255],
            10: [200, 40, 255],
            11: [90, 30, 150],
            12: [255, 0, 255],
            13: [255, 150, 255],
            14: [75, 0, 75],
            15: [75, 0, 175],
            16: [0, 200, 255],
            17: [50, 120, 255],
            18: [0, 150, 255],
            19: [170, 255, 150],
            20: [0, 175, 0],
            21: [0, 60, 135],
            22: [245, 150, 100],
            23: [255, 0, 0],
            24: [200, 40, 255],
            25: [30, 30, 255],
            26: [90, 30, 150],
            27: [250, 80, 100],
            28: [180, 30, 80],
        }
i = 0

for instance_cur in np.unique(indexes_gt):
    color_cur = instance_colors[instance_cur]

    color_inst[indexes == instance_cur, 0] = color_cur[2]
    color_inst[indexes == instance_cur, 1] = color_cur[1]
    color_inst[indexes == instance_cur, 2] = color_cur[0]

    i += 1
print(i)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(color_inst.astype(np.float) / 255.0)
o3d.visualization.draw_geometries([pcd])
print(points.shape)
print(indexes.shape)
