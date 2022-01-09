import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from sklearn.neighbors import kneighbors_graph
import yaml
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsTransformer
from scipy import linalg
from scipy.sparse import csgraph
from torch_geometric.data import Data
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import hdbscan
from utils import get_instances

# This is the dataloader for our Panoptic Clustering apporach.
class PanopticClusteringDatasetAll(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, clustering='kmeans'):
        self.clustering = clustering
        super(PanopticClusteringDatasetAll, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        all_files = os.listdir(os.path.join(self.root, 'raw'))
        val_files = [val for val in all_files if not val.endswith(".label")]
        if '.DS_Store' in val_files:
            val_files.remove('.DS_Store')
        return val_files

    @property
    def processed_file_names(self):
        number = 2
        file_names = []
        for i in range(number+1):
            file_names.append('graph_data_'+str(i)+'_ALL_.pt')
        return file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass
        ...

    def process(self):
        idx = 0

        # load sematic info dictionary
        with open('/Users/albertorodriguez/Desktop/RCI_Master_TUM/WiSe21:22/ADL4CV/4D-PLS/data/SemanticKitti/semantic-kitti.yaml') as f:
            DATA = yaml.safe_load(f)

        class_remap = DATA["learning_map"]
        class_inv_remap = DATA["learning_map_inv"]

        number_graphs = 0

        total = 0
        false = 0
        true = 0

        sum = 0

        missmatches = 0

        for raw_path in self.raw_paths:
            # raw path is path to bin Point Cloud file
            # label file
            raw_name = raw_path.split('/')[-1]
            label_path = os.path.join(*raw_path.split('/')[0:-1], raw_name.split('.')[0]+'.label')

            # Read pointcloud from `raw_path`
            scan = np.fromfile(raw_path, dtype=np.float32)

            scan = scan.reshape((-1, 4))
            # extract xyz coordinates = points and reflectance
            points = scan[:, 0:3]
            reflectance = scan[:, 3]

            # get labels
            label = np.fromfile(label_path, dtype=np.uint32)

            u_label_sem_class_orig = label & 0xFFFF  # remap to xentropy format
            u_label_inst_orig = label >> 16

            # get new inst label
            u_label_inst = get_instances(inst_label=u_label_inst_orig, seg_label=u_label_sem_class_orig)
            print(np.unique(u_label_inst))
            # map class labels to our learned class labels where 1-8 are things
            u_label_sem_class = [class_remap[sem_class] for sem_class in u_label_sem_class_orig]
            # create graph


            # only overcluster thing classes
            sem_class_thing_mask = np.logical_and(np.greater(u_label_sem_class, 0),
                                                  np.less(u_label_sem_class, 9))
            points_class = points[sem_class_thing_mask, :]

            instances_class = u_label_inst[sem_class_thing_mask]
            reflectance_class = reflectance[sem_class_thing_mask]

            # n components using ground truth
            n_components = np.shape(np.unique(instances_class))[0]
            print('n components')
            print(np.unique(instances_class))
            print(n_components)

            if (n_components + 30) > points_class.shape[0]:
                n_clusters = points_class.shape[0]
            else:
                n_clusters = n_components + 30

            # K means Over Clustering
            if self.clustering == 'kmeans':
                km = KMeans(n_clusters=n_clusters)
                km.fit(points_class)
                km_clusters = km.predict(points_class)

            else:
                # HBDSCAN CLUSTERING
                clusterer = hdbscan.HDBSCAN(min_cluster_size=n_clusters)
                clusterer.fit(points_class)
                km_clusters = clusterer.labels_


            sorted_unique_clusters = np.sort(np.unique(km_clusters))

            # construct for every cluster a node feature

            clusters_instance = []
            node_features = None

            for i in sorted_unique_clusters:
                instances_correspondent = instances_class[km_clusters == i]
                instances_correspondent_unique = np.sort(np.unique(instances_correspondent))
                # compute node cluster feature == centroid point of the cluster in xyz
                cluster_center = np.mean(points_class[km_clusters == i, :], axis=0)
                reflectance_cluster = np.mean(reflectance_class[km_clusters == i])
                feature = np.append(cluster_center, reflectance_cluster)
                if node_features is None:
                    node_features = feature
                else:
                    node_features = np.vstack((node_features, feature))

                total += 1

                # for every cluster we need a correspondent instance e.g cluster 1 consitst of points with all
                # instance 1, then instance(cluster 1) = 1.
                # If a cluster consists of points with different ground truth instances, we assign the instance
                # cluster with majority vote.
                if instances_correspondent_unique.shape[0] > 1:
                    instances_info = []
                    for j in instances_correspondent_unique:
                        instances_info.append(instances_correspondent[instances_correspondent == j].shape[0])

                    # assign isntance with majority vote
                    clusters_instance.append(instances_correspondent_unique[np.argmax(np.asarray(instances_info))])

                    false += 1
                else:
                    clusters_instance.append(instances_correspondent_unique[0])
                    true += 1

            # Full Connected Graph Init adj matrix where all entries = 1, all nodes==subclusters connected to each
            # other at the beginning

            Adj = np.zeros((n_clusters, n_clusters))

            edges_attr = []

            for cluster in sorted_unique_clusters:
                cluster_inst_cur = clusters_instance[cluster]
                for cluster_next in sorted_unique_clusters:
                    cluster_next_inst = clusters_instance[cluster_next]

                    # cluster centroid distance as edge attribute
                    edges_attr.append(
                        np.sum(np.sqrt((node_features[cluster, :] - node_features[cluster_next, :]) ** 2)))

                    # Adj matrix set to 1 when two clusters share the same instance ID
                    if cluster_inst_cur == cluster_next_inst:
                        Adj[cluster, cluster_next] = 1
                        #Adj[cluster_next, cluster] = 1
                    else:
                        pass

            # get components in Adj matrix
            graph = csr_matrix(Adj)
            n_components_graph, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            # Edges in COO format for fully connected graph
            Adj_fully_connected = np.ones_like(Adj)
            row, col = np.where(Adj_fully_connected)
            coo = np.array(list(zip(row, col))).T

            if (n_components - n_components_graph) != 0:
                #print('MISMATCH')
                missmatches += 1

            data = Data(x=torch.from_numpy(node_features), edge_index=torch.tensor(coo),
                        y=torch.tensor(Adj.ravel(), dtype=torch.int64), edge_attr=torch.Tensor(edges_attr))


            edd = data.edge_index

            #print(data)

            torch.save(data, osp.join(self.processed_dir, 'graph_data_' + str(idx) + '_ALL_.pt'))

            # create dictionary with clusters indexes
            cluster_indexes = {}
            for current_cluster in sorted_unique_clusters:
                cluster_indexes[current_cluster] = np.argwhere(km_clusters == current_cluster)

            cluster_indexes['points'] = points_class

            torch.save(cluster_indexes, 'graph_data/processed/graph_data_' + str(idx) + '_ALL_INFO.pt')

            idx += 1

            # visulisation
            visualisation = True
            if visualisation:
                # load GT
                graph_gt = csr_matrix(Adj)
                n_components_gt, labels_gt = connected_components(csgraph=graph_gt, directed=False, return_labels=True)
                print('n components')
                print(n_components_gt)
                indexes_gt = np.zeros(points_class.shape[0])

                for component in range(n_components_gt):
                    # get which sub-clusters belong to this cluster
                    corresponding_sub_clusters = np.argwhere(labels_gt == component)
                    # go over each points of subcluster and assigns them the same isntance id
                    for id in corresponding_sub_clusters:
                        # read indexes to pc
                        indexes_gt[cluster_indexes[id[0]]] = component

                color_inst = np.zeros_like(points_class)
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
                    29: [110, 30, 80],
                    30: [180, 90, 80],
                }
                i = 0

                for instance_cur in np.unique(indexes_gt):
                    color_cur = instance_colors[instance_cur]

                    color_inst[indexes_gt == instance_cur, 0] = color_cur[2]
                    color_inst[indexes_gt == instance_cur, 1] = color_cur[1]
                    color_inst[indexes_gt == instance_cur, 2] = color_cur[0]

                    i += 1

                # ground truth instances visualisation
                color_inst_gt = np.zeros_like(points_class)
                for instance_cur in np.unique(instances_class):
                    color_cur = instance_colors[instance_cur]

                    color_inst_gt[instances_class == instance_cur, 0] = color_cur[2]
                    color_inst_gt[instances_class == instance_cur, 1] = color_cur[1]
                    color_inst_gt[instances_class == instance_cur, 2] = color_cur[0]

                    i += 1

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_class)
                pcd.colors = o3d.utility.Vector3dVector(color_inst.astype(np.float) / 255.0)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(points_class)
                pcd_gt.colors = o3d.utility.Vector3dVector(color_inst_gt.astype(np.float) / 255.0)

                #o3d.visualization.draw_geometries([pcd])
                o3d.visualization.draw_geometries([pcd_gt])

        print('division')
        print(false / total)
        print(true / total)
        print('Missmathces')
        print(missmatches)
        print(sum)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_data_'+str(idx)+'_ALL_.pt'))
        return data


train_dataset = PanopticClusteringDatasetAll(root='graph_data/')

#data = torch.load(osp.join('graph_data/processed', 'graph_data_'+str(0)+'_ALL_.pt'))

#for e in data.edge_index.T:
#    print(e)