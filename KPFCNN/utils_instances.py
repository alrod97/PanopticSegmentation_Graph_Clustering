import numpy as np
import torch.nn as nn
import torch
import open3d as o3d


def color_instances(ins_color, ins_prediction_class):
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
    cluster_number = 1
    for inst_cur in np.unique(ins_prediction_class):
        if ins_color[ins_prediction_class == inst_cur, :].shape[0] < 10:
            pass
        else:
            ins_color[ins_prediction_class == inst_cur, :] = np.asarray(instance_colors[cluster_number])
        print('Cluster  points:  ' + str(ins_color[ins_prediction_class == inst_cur, :].shape[0]))
        cluster_number += 1
    return ins_color


def display_instances(points_class, ins_colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_class)
    pcd.colors = o3d.utility.Vector3dVector(ins_colors.astype(np.float) / 255.0)
    o3d.visualization.draw_geometries([pcd])

def new_pdf_normal(x, mean, var):
    """
    Computes instance belonging probability values
    :param x: embeddings values of all points NxD
    :param mean: instance embedding 1XD
    :param var: instance variance value 1XD
    :return: probability scores for all points Nx1
    """
    eps = torch.ones_like(var, requires_grad=True, device=x.device) * 1e-5
    var_eps = var + eps
    var_seq = var_eps.squeeze()
    inv_var = torch.diag(1 / var_seq)
    mean_rep = mean.repeat(x.shape[0], 1)
    dif = x - mean_rep
    d = torch.pow(dif, 2)
    e = torch.matmul(d, inv_var)
    probs = torch.exp(e * -0.5)
    probs = torch.sum(probs, 1) / torch.sum(var_eps)
    return probs


def ins_pre_2_pass(predicted, centers_output, var_output, embedding, next_ins_id, points=None, times=None):
    """
    Calculate instance probabilities for each point with considering old predictions also
    :param predicted: class labels for each point
    :param centers_output: center predictions
    :param var_output : variance predictions
    :param embedding : embeddings for all points
    :param points: xyz location of points
    :return: instance ids for all points, and new instances and next available ins_id
    """
    ins_prediction = torch.zeros_like(predicted)

    if var_output.shape[1] - embedding.shape[1] > 4:
        global_emb, _ = torch.max(embedding, 0, keepdim=True)
        embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

    if var_output.shape[1] - embedding.shape[1] == 3:
        embedding = torch.cat((embedding, points[0]), 1)
    if var_output.shape[1] - embedding.shape[1] == 4:
        embedding = torch.cat((embedding, points[0], times), 1)

    counter = 0
    ins_id = next_ins_id
    centers = []

    while True:
        ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
        # ins_idxs: 1x1 tuple containing ?x1 vector containing values from 0 to N
        # example ?=4308
        if len(ins_idxs[0]) == 0:
            break
        ins_centers = centers_output[ins_idxs]  # ?x1 containing values from 0 to 1
        ins_embeddings = embedding[ins_idxs]  # ?xD containing random emb values
        ins_variances = var_output[ins_idxs]  # ?xD containing random var values
        ins_points = points[0][ins_idxs]  # ?x3 containing random x,y,z values
        if counter == 0:
            sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            # sorted: ?x1 containing values from 0 to 1 in descending order
            # indices: ?x1 containing values from 0 to ?
        if sorted[0 + counter] < 0.1 or (sorted[0] < 0.7):
            break
        idx = indices[0 + counter]  # 1x1 integer giving index of center point with high objectness score
        centers.append(idx)
        mean = ins_embeddings[idx]  # 1xD embedding of center
        var = ins_variances[idx]  # 1xD variance of center

        center = points[0][ins_idxs][idx]  # 1x3 containing x,y,z of center
        distances = torch.sum((ins_points - center) ** 2, 1)  # distance of other points to center
        if torch.cuda.device_count() > 1:
            new_device = torch.device("cuda:1")
            probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            # probs: ?x1 containing probabilites of points belonging to center
        else:
            probs = new_pdf_normal(ins_embeddings, mean, var)

        probs[distances > 20] = 0
        ins_points = torch.where(probs >= 0.5)
        # ins_points: 1x1 tuple containing ??x1 containing values from 0 to ? indicating where points belong to center
        # example ??=31
        if ins_points[0].size()[0] < 2:
            counter += 1
            if counter == sorted.shape[0]:
                break
            continue

        ids = ins_idxs[0][ins_points[0]]
        ins_prediction[ids] = ins_id
        counter = 0
        ins_id += 1

    ins_prediction = torch.zeros_like(predicted)
    ins_idxs = torch.where((predicted < 9) & (predicted != 0))
    ins_embeddings = embedding[ins_idxs]  # ?xD containing random emb values
    ins_variances = var_output[ins_idxs]  # ?xD containing random var values
    probs_for_centers = torch.empty((ins_embeddings.shape[0], len(centers)))
    column = 0
    for center in centers:
        # ?xCenters
        mean = ins_embeddings[center]  # 1xD embedding of center
        var = ins_variances[center]
        if torch.cuda.device_count() > 1:
            new_device = torch.device("cuda:1")
            probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            # probs: ?x1 containing probabilites of points belonging to center
        else:
            probs = new_pdf_normal(ins_embeddings, mean, var)
        probs_for_centers[:, column] = probs
        column += 1

    _, ins_prediction = torch.max(probs_for_centers)

    return ins_prediction

def knn_ins_pred(self, predicted, centers_output, var_output, embedding, prev_instances={}, next_ins_id=1,
                 points=None, times=None):
    """A function to predict the instance association for a semantically segmented point-cloud.
    Args:
        predicted: The perdicted class of each point                            (torch.int64 vector)
        centers_output: "Centerness" of each point                              (torch.float32 vector)
        var_output: The variances of each point                                 (torch.float32 matrix)
        embedding: An embedding for each point                                  (torch.float32 matrix)
        prev_instances: ?                                                       (dict)
        prev_instances["04"]: ?                                                 (dict)
        next_ins_id: The next ID which an new instace will be assigned          (int)
        points: x,y,z coordinates of different time steps                       (list)
        points[0]: The x,y,z coordinates of current time step for each point    (torch.float32 matrix)
        times: ?                                                                (torch.float32 vector)

    Args example shapes:
        predicted:              82781
        centers_output:         torch.Size([82781, 1])
        var_output:             torch.Size([82781, 260])
        embedding:              torch.Size([82781, 256])
        prev_instances:         1
        prev_instances["04"]:   0
        next_ins_id:            1
        points:                 6
        points[0]:              torch.Size([82781, 3])
        times:                  82781

    Output:
        ins_prediction: The predicted instance for each point                               (torch.int64)
        new_instances: Information about current(?) instaces                                (dict)
        new_instance["1"]: contains mean, var, life, bbox, bbox_proj, tracker, kalman_bbox  (dict)
        ins_id: The next ID which an new instace will be assigned                           (int)

    Output example shapes:
        ins_prediction:     82781
        new_isntances:      9
        new_instance["1"]:  7
        ins_id:             1
    """

    new_instances = {}
    ins_prediction = torch.zeros_like(predicted, dtype=torch.int64)
    ins_id = next_ins_id

    thing_classes = torch.unique(predicted)[torch.unique(predicted) < 9]
    thing_classes = thing_classes[thing_classes != 0]

    threshold = {'1': 0.9,  # car
                 '2': 0.8,  # bicycle
                 '3': 0.8,  # motorcycle
                 '4': 0.8,  # truck
                 '5': 0.5,  # other-vehicle
                 '6': 0.7,  # person
                 '7': 0.8,  # bicyclist
                 '8': 0.8}  # motorcyclist

    nbr_clusters = get_nbr_clusters(predicted, centers_output, var_output, embedding, next_ins_id,
                                    points=points, times=times)

    for selected_class in thing_classes:
        embeddings_class = embedding[predicted == selected_class, :]
        points_class = points[0][predicted == selected_class, :]
        embeddings_class = torch.cat((embeddings_class, points_class), axis=1)  # concatenate xyz to embeddings
        variances_class = var_output[predicted == selected_class, :]
        centers_class = centers_output[predicted == selected_class, :]

        # nbr_cluster = 0
        # range_cluster = range(1, 20)
        # sum_of_squared_distances = []
        # if embeddings_class.shape[0] < 5:
        #     continue
        # elif embeddings_class.shape[0] < 15:
        #     nbr_cluster = 1
        # else:
        #     for k in range_cluster:
        #         km = KMeans(n_clusters=k)
        #         km = km.fit(embeddings_class)
        #         sum_of_squared_distances.append(km.inertia_)
        #         nbr_cluster = k
        #         if k == 1:
        #             pass
        #         elif (sum_of_squared_distances[k-1]/sum_of_squared_distances[k-2]) > threshold[str(selected_class.item())]:
        #             break

        km = KMeans(n_clusters=nbr_clusters[str(selected_class.item())])
        km = km.fit(embeddings_class)
        ins_prediction_class = km.predict(embeddings_class) + ins_id
        ins_id += nbr_clusters[str(selected_class.item())]

        ins_colors = np.zeros_like(points_class)
        ins_colors = color_instances(ins_colors, ins_prediction_class)
        display_instances(points_class, ins_colors)

        # ids = (predicted == selected_class).nonzero(as_tuple=True)
        indicies = torch.where(predicted == selected_class)
        ins_prediction[indicies] = torch.from_numpy(ins_prediction_class).type(torch.int64)

    nbr_points_of_instance = torch.bincount(ins_prediction)
    for i in torch.unique(ins_prediction):
        ids = torch.where(ins_prediction == i)
        if nbr_points_of_instance[i] > 25:  # add to instance history
            mean = torch.mean(embedding[ids], 0, True)
            bbox, kalman_bbox = get_bbox_from_points(points[0][ids])
            tracker = KalmanBoxTracker(kalman_bbox, i)
            bbox_proj = None
            var = torch.mean(var_output[ids], 0, True)
            new_instances[i] = {'mean': mean, 'var': var, 'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj,
                                'tracker': tracker, 'kalman_bbox': kalman_bbox}

    # associate instances by hungarian alg. & bbox prediction via kalman filter
    if len(prev_instances.keys()) > 0:

        # association_costs, associations = self.associate_instances(config, prev_instances, new_instances, pose)
        associations = []
        for prev_id, new_id in associations:
            ins_points = torch.where((ins_prediction == new_id))
            ins_prediction[ins_points[0]] = prev_id
            prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
            prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']

            prev_instances[prev_id]['life'] += 1
            prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
            prev_instances[prev_id]['kalman_bbox'] = prev_instances[prev_id]['tracker'].get_state()
            prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

            del new_instances[new_id]

    # should be
    # ins_preds.shape = 82781
    # ins_preds.dtype = torch.int64
    # type(new_instaces) = dict
    # len(new_instaces) = 9
    # type(new_instaces["1"]) = dict   # containing mean, var, life, bbox, bbox_proj, tracker, kalman_bbox
    # type(ins_id) = int

    return ins_prediction, new_instances, ins_id


def get_nbr_clusters(predicted, centers_output, var_output, embedding, next_ins_id, points=None, times=None):
    """
    Calculate instance probabilities for each point with considering old predictions also
    :param predicted: class labels for each point
    :param centers_output: center predictions
    :param var_output : variance predictions
    :param embedding : embeddings for all points
    :param points: xyz location of points
    :return: instance ids for all points, and new instances and next available ins_id
    """
    ins_prediction = torch.zeros_like(predicted)

    if var_output.shape[1] - embedding.shape[1] > 4:
        global_emb, _ = torch.max(embedding, 0, keepdim=True)
        embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

    if var_output.shape[1] - embedding.shape[1] == 3:
        embedding = torch.cat((embedding, points[0]), 1)
    if var_output.shape[1] - embedding.shape[1] == 4:
        embedding = torch.cat((embedding, points[0], times), 1)

    counter = 0
    ins_id = next_ins_id

    while True:
        ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
        # ins_idxs: 1x1 tuple containing ?x1 vector containing values from 0 to N
        # example ?=4308
        if len(ins_idxs[0]) == 0:
            break
        ins_centers = centers_output[ins_idxs]  # ?x1 containing values from 0 to 1
        ins_embeddings = embedding[ins_idxs]  # ?xD containing random emb values
        ins_variances = var_output[ins_idxs]  # ?xD containing random var values
        ins_points = points[0][ins_idxs]  # ?x3 containing random x,y,z values
        if counter == 0:
            sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            # sorted: ?x1 containing values from 0 to 1 in descending order
            # indices: ?x1 containing values from 0 to ?
        if sorted[0 + counter] < 0.1 or (sorted[0] < 0.7):
            break
        idx = indices[0 + counter]  # 1x1 integer giving index of center point with high objectness score
        mean = ins_embeddings[idx]  # 1xD embedding of center
        var = ins_variances[idx]  # 1xD variance of center

        center = points[0][ins_idxs][idx]  # 1x3 containing x,y,z of center
        distances = torch.sum((ins_points - center) ** 2, 1)  # distance of other points to center
        if torch.cuda.device_count() > 1:
            new_device = torch.device("cuda:1")
            probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            # probs: ?x1 containing probabilites of points belonging to center
        else:
            probs = new_pdf_normal(ins_embeddings, mean, var)

        probs[distances > 20] = 0
        ins_points = torch.where(probs >= 0.5)
        # ins_points: 1x1 tuple containing ??x1 containing values from 0 to ? indicating where points belong to center
        # example ??=31
        if ins_points[0].size()[0] < 2:
            counter += 1
            if counter == sorted.shape[0]:
                break
            continue

        ids = ins_idxs[0][ins_points[0]]
        ins_prediction[ids] = ins_id
        counter = 0
        ins_id += 1

    nbr_clusters = {}
    for class_label in range(1, 9):
        class_idx = torch.where(predicted == class_label)
        class_instances = torch.unique(ins_prediction[class_idx])
        nbr_clusters[str(class_label)] = len(class_instances)
    return nbr_clusters
