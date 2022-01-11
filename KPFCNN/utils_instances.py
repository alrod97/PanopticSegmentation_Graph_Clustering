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
