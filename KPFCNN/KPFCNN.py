from blocks import *
from losses import *
import numpy as np
import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from sklearn.cluster import KMeans
from utils_instances import color_instances, display_instances, get_nbr_clusters

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:,
        7:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info associated

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))


def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config):

    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))

class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.batch_norm(x))

class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)

class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])

class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)

class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_var = UnaryBlock(config.first_features_dim, out_dim + config.free_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(config.first_features_dim, 1, False, 0, False)

        self.pre_train = config.pre_train
        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.center_loss = 0
        self.instance_loss = torch.tensor(0)
        self.variance_loss = torch.tensor(0)
        self.instance_half_loss = torch.tensor(0)
        self.reg_loss = 0
        self.variance_l2 = torch.tensor(0)
        self.l1 = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        f = self.head_mlp(x, batch)
        c = self.head_center(f, batch)
        c = self.sigmoid(c)
        v = self.head_var(f, batch)
        v = F.relu(v)
        x = self.head_softmax(f, batch)

        return x, c, v, f

    def loss(self, outputs, centers_p, variances, embeddings, labels, ins_labels, centers_gt, points=None, times=None):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)
        centers_p = centers_p.squeeze()
        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
        self.center_loss = weighted_mse_loss(centers_p, centers_gt[:, 0], weights)

        if not self.pre_train:
            self.instance_half_loss = instance_half_loss(embeddings, ins_labels)
            self.instance_loss = iou_instance_loss(centers_p, embeddings, variances, ins_labels, points, times)
            self.variance_loss = variance_smoothness_loss(variances, ins_labels)
            self.variance_l2 = variance_l2_loss(variances, ins_labels)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        # return self.instance_loss + self.variance_loss
        return self.output_loss + self.reg_loss + self.center_loss + self.instance_loss * 0.1 + self.variance_loss * 0.01

    def ins_pre_2_pass(self, predicted, centers_output, var_output, embedding, prev_instances, next_ins_id,
                       points=None, times=None, pose=None):
        """
        Calculate instance probabilities for each point with considering old predictions also
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param points: xyz location of points
        :return: instance ids for all points, and new instances and next available ins_id
        """
        new_instances = {}
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

        # ins_idxs = torch.where((predicted < 9) & (predicted != 0))
        # ins_embeddings = embedding[ins_idxs]  # ?xD containing random emb values
        # ins_variances = var_output[ins_idxs]  # ?xD containing random var values
        # probs_for_centers = torch.empty((ins_embeddings.shape[0], len(centers)))
        # column = 0
        # for center in centers:
        #     mean = ins_embeddings[center]  # 1xD embedding of center
        #     var = ins_variances[center]
        #     if torch.cuda.device_count() > 1:
        #         new_device = torch.device("cuda:1")
        #         probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
        #         # probs: ?x1 containing probabilites of points belonging to center
        #     else:
        #         probs = new_pdf_normal(ins_embeddings, mean, var)
        #     center_point = points[0][ins_idxs][center]
        #     distances = torch.sum((points[0][ins_idxs] - center_point) ** 2, 1)
        #     probs[distances > 20] = 0
        #     probs_for_centers[:, column] = probs
        #     column += 1
        #
        # for j in range(probs_for_centers.shape[1]):
        #     ins_idxs = torch.where((predicted < 9) & (predicted != 0))[0]
        #     thing_predicted = predicted[ins_idxs]
        #     class_of_center = thing_predicted[centers[j]]
        #     idxs_diff_class = torch.where((thing_predicted != class_of_center) & (thing_predicted < 9) & (thing_predicted != 0))[0]
        #     probs_for_centers[idxs_diff_class, j] = 0
        # _, ins_prediction = torch.max(probs_for_centers, 1)

        # visualize instances
        thing_classes = torch.unique(predicted)[torch.unique(predicted) < 9]
        thing_classes = thing_classes[thing_classes != 0]
        for selected_class in thing_classes:
            points_class = points[0][predicted == selected_class, :]
            ins_prediction_class = ins_prediction[predicted == selected_class]
            ins_colors = np.zeros_like(points_class)
            ins_colors = color_instances(ins_colors, ins_prediction_class)
            display_instances(points_class, ins_colors)

        for instance in torch.unique(ins_prediction):
            ids = torch.where(ins_prediction == instance)
            if ins_points[0].size()[0] > 25:  # add to instance history
                mean = torch.mean(embedding[ids], 0, True)
                bbox, kalman_bbox = get_bbox_from_points(points[0][ids])
                tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                bbox_proj = None
                # var = torch.mean(var_output[ids], 0, True)
                new_instances[ins_id] = {'mean': mean, 'var': var, 'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj,
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

        return ins_prediction, new_instances, ins_id

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

    def ins_pred(self, predicted, centers_output, var_output, embedding, points=None, times=None):
        """
        Calculate instance probabilities for each point on current frame
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param points: xyz location of points
        :return: instance ids for all points
        """
        # predicted = torch.argmax(outputs.data, dim=1)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if var_output.shape[1] - embedding.shape[1] == 4:
            embedding = torch.cat((embedding, points[0], times), 1)

        if var_output.shape[1] == 3:
            embedding = points[0]
        if var_output.shape[1] == 4:
            embedding = torch.cat((points[0], times), 1)

        ins_prediction = torch.zeros_like(predicted)

        counter = 0
        ins_id = 1
        while True:
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            if len(ins_idxs[0]) == 0:
                break
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            if counter == 0:
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0 + counter] < 0.1 or (ins_id == 1 and sorted[0] < 0.7):
                break
            idx = indices[0 + counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]
            # probs = pdf_normal(ins_embeddings, mean, var)
            probs = new_pdf_normal(ins_embeddings, mean, var)

            ins_points = torch.where(probs >= 0.5)
            if ins_points[0].size()[0] < 2:
                counter += 1
                if counter == sorted.shape[0]:
                    break
                continue
            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            counter = 0
            ins_id += 1
        return ins_prediction

    def ins_pred_in_time(self, config, predicted, centers_output, var_output, embedding, prev_instances, next_ins_id,
                         points=None, times=None, pose=None):
        """
        Calculate instance probabilities for each point with considering old predictions also
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param prev_instances : instances which detected in previous frames
        :param next_ins_id : next avaliable ins id
        :param points: xyz location of points
        :return: instance ids for all points, and new instances and next available ins_id
        """
        new_instances = {}
        ins_prediction = torch.zeros_like(predicted)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if var_output.shape[1] - embedding.shape[1] == 4:
            embedding = torch.cat((embedding, points[0], times), 1)

        pose = torch.from_numpy(pose)
        pose = pose.to(embedding.device)

        counter = 0
        ins_id = next_ins_id

        while True:
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            if len(ins_idxs[0]) == 0:
                break
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            ins_points = points[0][ins_idxs]
            if counter == 0:
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0 + counter] < 0.1 or (sorted[0] < 0.7):
                break
            idx = indices[0 + counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]

            center = points[0][ins_idxs][idx]
            distances = torch.sum((ins_points - center) ** 2, 1)
            if torch.cuda.device_count() > 1:
                new_device = torch.device("cuda:1")
                probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            else:
                probs = new_pdf_normal(ins_embeddings, mean, var)

            # from scipy.stats import multivariate_normal
            # if torch.cuda.device_count() > 1:
            #     new_device = torch.device("cuda:1")
            #     probs = multivariate_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            # else:
            #     probs = multivariate_normal(ins_embeddings, mean, var)

            probs[distances > 20] = 0
            ins_points = torch.where(probs >= 0.5)
            if ins_points[0].size()[0] < 2:
                counter += 1
                if counter == sorted.shape[0]:
                    break
                continue

            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            if ins_points[0].size()[0] > 25:  # add to instance history
                ins_prediction[ids] = ins_id
                mean = torch.mean(embedding[ids], 0, True)
                bbox, kalman_bbox = get_bbox_from_points(points[0][ids])
                tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                bbox_proj = None
                # var = torch.mean(var_output[ids], 0, True)
                new_instances[ins_id] = {'mean': mean, 'var': var, 'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj,
                                         'tracker': tracker, 'kalman_bbox': kalman_bbox}

            counter = 0
            ins_id += 1

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

        return ins_prediction, new_instances, ins_id

    def associate_instances(self, config, previous_instances, current_instances, pose):
        pose = pose.cpu()
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        p_n = len(previous_instances.keys())
        c_n = len(current_instances.keys())

        association_costs = torch.zeros(p_n, c_n)
        prev_ids = []
        current_ids = []

        for i, (k, v) in enumerate(previous_instances.items()):
            prev_ids.append(k)
            for j, (k1, v1) in enumerate(current_instances.items()):
                cost_3d = 1 - IoU(v1['bbox'], v['bbox'])
                if cost_3d > 0.75:
                    cost_3d = 1e8
                if v1['bbox_proj'] is not None:
                    cost_2d = 1 - IoU(v1['bbox_proj'], v['bbox_proj'])
                    if cost_2d > 0.5:
                        cost_2d = 1e8
                else:
                    cost_2d = 0

                cost_center = euclidean_dist(v1['kalman_bbox'], v['kalman_bbox'])
                if cost_center > 1:
                    cost_center = 1e8

                feature_cost = 1 - cos(v1['mean'], v['mean'])
                if feature_cost > 0.05:
                    feature_cost = 1e8
                costs = torch.tensor([cost_3d, cost_2d, cost_center, feature_cost])
                for idx, a_w in enumerate(config.association_weights):
                    association_costs[i, j] += a_w * costs[idx]

                if i == 0:
                    current_ids.append(k1)

        idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())

        associations = []

        for i1, i2 in zip(idxes_1, idxes_2):
            # max_cost = torch.sum((previous_instances[prev_ids[i1]]['var'][0,-3:]/2)**2)
            if association_costs[i1][i2] < 1e8:
                associations.append((prev_ids[i1], current_ids[i2]))

        return association_costs, associations

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
