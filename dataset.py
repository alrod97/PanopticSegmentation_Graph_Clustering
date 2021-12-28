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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import hdbscan


# This is the dataloader for our Panoptic Clustering apporach.

class PanopticClusteringDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, clustering='kmeans'):
        self.clustering = clustering
        super(PanopticClusteringDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        all_files = os.listdir(os.path.join(self.root, 'raw'))
        val_files = [val for val in all_files if not val.endswith(".label")]
        return val_files

    @property
    def processed_file_names(self):
        number = 188
        file_names = []
        for i in range(number+1):
            file_names.append('graph_data_'+str(i)+'_.pt')
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
        class_ignore = DATA["learning_ignore"]
        nr_classes = len(class_inv_remap)  # 19 in in our case including  stuff and things
        class_strings = DATA["labels"]


        number_graphs = 0

        total = 0
        false = 0
        true = 0

        sum = 0

        missmatches = 0
        for raw_path in self.raw_paths:
            #print(raw_path)
            # raw path is path to bin Point Cloud file
            # label file
            raw_name = raw_path.split('/')[-1]
            label_path = os.path.join(*raw_path.split('/')[0:-1], raw_name.split('.')[0]+'.label')

            # Read pointcloud from `raw_path`
            scan = np.fromfile(raw_path, dtype=np.float32)

            scan = scan.reshape((-1, 4))
            # extract xyz coordinates = points
            points = scan[:, 0:3]

            # get labels
            label = np.fromfile(label_path, dtype=np.uint32)

            u_label_sem_class = label & 0xFFFF  # remap to xentropy format
            u_label_inst = label >> 16

            # create graphs
            for sem_class in np.unique(u_label_sem_class):
                # learning id
                id_learning = class_remap[sem_class]
                label = class_strings[sem_class]
                # only overcluster if sem_class  is a thing class
                if id_learning < 9 and id_learning != 0:
                    points_class = points[u_label_sem_class == sem_class, :]
                    # get instance information
                    instances_class = u_label_inst[u_label_sem_class == sem_class]

                    # n components
                    n_components = np.shape(np.unique(instances_class))[0]

                    if (n_components + 25) > points_class.shape[0]:
                        n_clusters = points_class.shape[0]
                        print(n_clusters)
                    else:
                        n_clusters = n_components + 25

                    # K means Over Clustering
                    if self.clustering == 'kmeans':
                        km = KMeans(n_clusters=n_clusters)
                        km.fit(points_class)
                        km_clusters = km.predict(points_class)
                    else:
                        # HBDSCAN CLUSTERING
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=n_components + 15, min_samples=60)
                        clusterer.fit(points_class)
                        km_clusters = clusterer.labels_

                    sorted_unique_clusters = np.sort(np.unique(km_clusters))

                    clusters_instance = []
                    node_features = None
                    for i in sorted_unique_clusters:
                        instances_correspondent = instances_class[km_clusters == i]
                        instances_correspondent_unique = np.sort(np.unique(instances_class[km_clusters == i]))
                        # compute node cluster feature == centroid point of the cluster in xyz
                        if node_features is None:
                            node_features = np.mean(points_class[km_clusters == i, :], axis=0)

                        else:
                            node_features = np.vstack((node_features, np.mean(points_class[km_clusters == i, :], axis=0)))

                        total += 1

                        if instances_correspondent_unique.shape[0] > 1:
                            # print('UUUU')
                            #print(label)
                            #print(instances_correspondent.shape[0])
                            # print(instances_correspondent)
                            instances_info = []
                            for j in instances_correspondent_unique:
                               # print(instances_correspondent[instances_correspondent == j].shape[0] /
                               #       instances_correspondent.shape[0])
                                instances_info.append(instances_correspondent[instances_correspondent == j].shape[0])
                                # print('--')
                                # print(np.argwhere(km_clusters == j).shape[0]/)
                                # print(instances_correspondent.shape[0])
                            # assign isntance with majority vote
                            clusters_instance.append(instances_correspondent_unique[np.argmax(np.asarray(instances_info))])

                            #print('--')
                            false += 1
                        else:
                            clusters_instance.append(instances_correspondent_unique[0])
                            true += 1

                    # Full Connected Graph Init adj matrix where all entries = 1, all nodes==subclusters connected to each
                    # other at the beginning

                    Adj = np.zeros((n_clusters, n_clusters))

                    edegs_attr = []

                    for cluster in sorted_unique_clusters:
                        cluster_inst_cur = clusters_instance[cluster]
                        for cluster_next in sorted_unique_clusters:
                            cluster_next_inst = clusters_instance[cluster_next]
                            if points_class.shape[0] > 1:
                                edegs_attr.append(np.sum(np.sqrt( (node_features[cluster, :] - node_features[cluster_next, :])**2 )))
                            else:
                                edegs_attr.append(0)

                            if cluster != cluster_next and cluster_inst_cur == cluster_next_inst:
                                Adj[cluster, cluster_next] = 1
                                Adj[cluster_next, cluster] = 1
                            else:
                                pass

                   # print(Adj)
                    graph = csr_matrix(Adj)
                    n_components_graph, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

                    # Edges in COO format
                    row, col = np.where(Adj)
                    coo = np.array(list(zip(row, col)))
                    coo = np.reshape(coo, (2, -1))

                    if (n_components - n_components_graph)!= 0:
                        print('MISMATCH')
                        missmatches += 1

                    print(len(edegs_attr))
                    print(Adj.shape)
                    data = Data(x=torch.from_numpy(node_features), edge_index=torch.tensor(coo, dtype=torch.long),
                                y=torch.tensor(Adj.ravel(), dtype=torch.int64), edge_attr=torch.Tensor(edegs_attr))

                    print(data)

                    if len(edegs_attr) == 1:
                        pass
                    else:
                        torch.save(data, osp.join(self.processed_dir, 'graph_data_' + str(idx) + '_.pt'))
                        print(Adj)
                        sum += np.sum(Adj)
                        print('--')
                        idx += 1
                    #print(node_features.shape)
                    #print(Adj.shape)
                    #print(np.unique(np.asarray(clusters_instance)))
                    #print(label)
                    #print(n_components)
                    #print('--')

                else:
                    pass

                    data = 0


        #torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        #idx += 1

        print('division')
        print(false/total)
        print(true/total)
        print('Missmathces')
        print(missmatches)
        print(sum)
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_data_'+str(idx)+'_.pt'))
        return data

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
        number = 0
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
            u_label_inst = label >> 16

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

            if (n_components + 25) > points_class.shape[0]:
                n_clusters = points_class.shape[0]
            else:
                n_clusters = n_components + 25

            # K means Over Clustering
            if self.clustering == 'kmeans':
                km = KMeans(n_clusters=n_clusters)
                km.fit(points_class)
                km_clusters = km.predict(points_class)
            else:
                # HBDSCAN CLUSTERING
                clusterer = hdbscan.HDBSCAN(min_cluster_size=n_components + 15, min_samples=60)
                clusterer.fit(points_class)
                km_clusters = clusterer.labels_

            sorted_unique_clusters = np.sort(np.unique(km_clusters))

            # construct for every cluster a node feature

            clusters_instance = []
            node_features = None

            for i in sorted_unique_clusters:
                instances_correspondent = instances_class[km_clusters == i]
                instances_correspondent_unique = np.sort(np.unique(instances_class[km_clusters == i]))
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
                    if cluster != cluster_next and cluster_inst_cur == cluster_next_inst:
                        Adj[cluster, cluster_next] = 1
                        Adj[cluster_next, cluster] = 1
                    else:
                        pass

            # get components in Adj matrix
            graph = csr_matrix(Adj)
            n_components_graph, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            # Edges in COO format
            row, col = np.where(Adj)
            coo = np.array(list(zip(row, col)))
            coo = np.reshape(coo, (2, -1))


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