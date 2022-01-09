import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from sklearn.neighbors import kneighbors_graph
import yaml
from sklearn.neighbors import RadiusNeighborsTransformer
from scipy import linalg
from scipy.sparse import csgraph
from torch_geometric.data import Data

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# This is the dataloader for our Panoptic Clustering apporach.

def get_instances(seg_label, inst_label):
    # unique classes segmentation
    seg_classes = np.unique(seg_label)
    # unique instances
    instances = np.unique(inst_label)

    instances_new_label = np.zeros_like(inst_label)

    inst_counter = 1

    for inst_cur in instances:
        # corresponding classes
        if inst_cur == 0:
            pass
        else:
            indexes_inst = np.argwhere(inst_label == inst_cur)
            seg_class_inst = seg_label[indexes_inst]
            for seg_class in np.unique(seg_class_inst):
                indexes_seg = np.argwhere(seg_class_inst == seg_class)
                instances_new_label[indexes_inst[indexes_seg]] = inst_counter
                inst_counter += 1

    return instances_new_label


class PanopticClusteringDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, radius=None, k =None):
        self.radius = radius
        self.k = k
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

        all_true_edges = 0
        all_false_edges = 0

        more_clusters_than_gt = 0
        less_clusters_than_gt = 0

        number_graphs = 0

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

            # get new inst label
            u_label_inst_new = get_instances(inst_label=u_label_inst, seg_label=u_label_sem_class)
            print(np.unique(u_label_inst_new))
            # create graphs

            for sem_class in np.unique(u_label_sem_class):
                # learning id
                id_learning = class_remap[sem_class]
                label = class_strings[sem_class]

                # only create graphs if sem_class  is a thing class
                if id_learning < 9 and id_learning != 0:
                    points_class = points[u_label_sem_class == sem_class, :]
                    # get instance information
                    instances_class = u_label_inst[u_label_sem_class == sem_class]

                    #rad_neighb = RadiusNeighborsTransformer(radius=self.radius, mode='distance')
                    #rad_neighb.fit(points_class)
                    #A = rad_neighb.radius_neighbors_graph(X=points_class)
                    #print(points_class.shape)
                    #print(self.k)
                    if points_class.shape[0] <= self.k:
                     #   print('aa')
                        k_use = points_class.shape[0] - 1
                        #print(label)
                    else:
                        k_use = self.k

                    if points_class.shape[0] > 1 :

                        A = kneighbors_graph(points_class, n_neighbors=k_use)
                        A_dense = A.todense()
                        A_dense[A_dense != 0] = 1
                        graph = csr_matrix(A_dense)
                        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

                        if n_components >= np.unique(instances_class).shape[0]:
                            more_clusters_than_gt += (n_components - np.unique(instances_class).shape[0])
                        else:
                            less_clusters_than_gt += (n_components - np.unique(instances_class).shape[0])
                        # get edges == graph connectivity in COO format
                        row, col = np.where(A_dense)
                        coo = np.array(list(zip(row, col)))
                        coo = np.reshape(coo, (2, -1))

                        edges = []
                        edges_symm = []
                        edges_number = 0
                        True_edges = 0
                        False_edges = 0

                        edges_label = [] # 1 for trues edge and 0 for zero edge==false edge
                        edge_feature = []
                        for pos in coo.T:
                            edges_number += 1

                            # check if true edge or not
                            if instances_class[pos[0]] == instances_class[pos[1]]:
                                True_edges += 1
                                edges_label.append(1)
                            else:
                                False_edges += 1
                                edges_label.append(0)

                            edge_feature.append(np.sqrt(np.sum((points_class[pos[0]] - points_class[pos[1]]) ** 2)))

                        # print('Radius:' + str(self.radius))
                        # print('True edges: ' + str(True_edges))
                        # print('False edges: ' + str(False_edges))
                        # print('True edges/ False edges: '+str(True_edges/False_edges))

                        all_true_edges += True_edges
                        all_false_edges += False_edges
                        number_graphs += 1

                        #edges_matrix = np.reshape(np.asarray(edges), (edges_number, 2))
                        edge_feature_array = np.asarray(edge_feature)


                        #data = Data(x=torch.from_numpy(points_class), edge_index=torch.tensor(coo, dtype=torch.long),
                        #            edge_attr=torch.Tensor(edge_feature_array), y=torch.tensor(edges_label, dtype=torch.int64))

                        #torch.save(data, osp.join(self.processed_dir, 'graph_data_'+str(idx)+'_.pt'))

                        idx += 1

                    else:
                        pass



                    #print('n components: ' + str(n_components))
                    #print('A matrix for segmentation class:' + label)
                    #print('Instances: ' + str(np.unique(instances_class).shape[0]))
                    #print('---')

                    data = 0

        #print('Radius:' + str(self.radius))
        print('K:' + str(self.k))
        print('True edges/number_graphs: ' + str(all_true_edges/number_graphs))
        print('False edges/number_graphs: ' + str(all_false_edges/number_graphs))
        #print('True edges/ False edges: '+str(all_true_edges/all_false_edges))
        print('More Clusters than gt: '+ str(more_clusters_than_gt))
        print('Less Clusters than gt: ' + str(less_clusters_than_gt))
        print('---')
        #torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        #idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_data_'+str(idx)+'_.pt'))
        return data

dataset = PanopticClusteringDataset(root='graph_data/', k=33)

#radiuses = np.arange(1.8,3,0.1)
#k = np.arange(20,40)
#print(radiuses)
#for i in k:
#    print(i)
#    dataset = PanopticClusteringDataset(root='graph_data/', radius=i,k=i)

