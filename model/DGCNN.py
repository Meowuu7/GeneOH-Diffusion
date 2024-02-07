# Adapted from https://github.com/SimingYan/HPNet/blob/main/models/dgcnn.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


def knn(x, k1, k2):
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx

### 
def knn_points_normals(x: object, k1: object, k2: object) -> object:
    """
    The idea is to design the distance metric for computing
    nearest neighbors such that the normals are not given
    too much importance while computing the distances.
    Note that this is only used in the first layer.
    """
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            # n = x[b: b + 1, 3:6]
            n = x[b: b + 1, 3:]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            nn = torch.sum(n ** 2, dim=1, keepdim=True)
            # n_pairwise_distance = 2 - inner
            n_pairwise_distance = nn - inner + nn.transpose(2, 1)

            # This pays less attention to normals
            if x.size(-1) > 6:
                pairwise_distance = p_pairwise_distance + n_pairwise_distance
            else:
                pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def get_graph_feature(x, k1=20, k2=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def get_graph_feature_with_normals(x: object, k1: object = 20, k2: object = 20, idx: object = None) -> object:
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        # get knn points
        idx = knn_points_normals(x[:, :6], k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNNEncoderGn(nn.Module):
    def __init__(self, mode=0, input_channels=3, nn_nb=80, n_layers=3):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.n_layers = n_layers
        self.dilation_factor = 1
        self.mode = mode
        self.drop = 0.0
        if self.mode == 0 or self.mode == 5:
            self.bn1 = nn.GroupNorm(2, 64)
            self.bn2 = nn.GroupNorm(2, 64)
            self.bn3 = nn.GroupNorm(2, 128)
            self.bn4 = nn.GroupNorm(4, 256)
            self.bn5 = nn.GroupNorm(8, 1024)

            # nn.sequential for conv2d modules # input_
            # input channel, output channel, kernel size, bias; group norm
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            # 1024 dim?
            if self.n_layers == 3:
                mlp_in_dim = 256
            elif self.n_layers == 2:
                mlp_in_dim = 128
            elif self.n_layers == 1:
                mlp_in_dim = 64
            else:
                raise ValueError(f"Unrecognized n_layers for DGCNN: {self.n_layers}.")
            self.mlp1 = nn.Conv1d(mlp_in_dim, 1024, 1)
            # number of groups, number of channels
            self.bnmlp1 = nn.GroupNorm(8, 1024)
            # self.mlp1 = nn.Conv1d(256, 1024, 1)
            # self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.mode == 0 or self.mode == 1:
            # First edge conv
            # self.k self.k

            first_layer_xs = []
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]
            first_layer_xs.append(x1)

            if self.n_layers >= 2:
                # Second edge conv
                x = get_graph_feature(x1, k1=self.k, k2=self.k)
                x = self.conv2(x)
                x2 = x.max(dim=-1, keepdim=False)[0]
                first_layer_xs.append(x2)

                if self.n_layers == 3:
                    # Third edge conv
                    x = get_graph_feature(x2, k1=self.k, k2=self.k)
                    x = self.conv3(x)
                    x3 = x.max(dim=-1, keepdim=False)[0]
                    first_layer_xs.append(x3)

            # x_features = torch.cat((x1, x2, x3), dim=1)
            x_features = torch.cat(first_layer_xs, dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))

            x4 = x.max(dim=2)[0]

            return x4, x_features

        if self.mode == 5: # mode == 5: with normal vecotr
            first_layer_xs = []

            # First edge conv
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            first_layer_xs.append(x1)

            if self.n_layers >= 2:
                # Second edge conv
                # GET graph feature for each point
                x = get_graph_feature(x1, k1=self.k, k2=self.k)
                x = self.conv2(x)
                x2 = x.max(dim=-1, keepdim=False)[0]
                first_layer_xs.append(x2)

                if self.n_layers == 3:
                    # Third edge conv
                    x = get_graph_feature(x2, k1=self.k, k2=self.k)
                    x = self.conv3(x)
                    x3 = x.max(dim=-1, keepdim=False)[0]
                    first_layer_xs.append(x3)

            # Cat features at different convolution level
            # x_features = torch.cat((x1, x2, x3), dim=1)
            x_features = torch.cat(first_layer_xs, dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]

            return x4, x_features


class PrimitivesEmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, opt, emb_size=50, mode=0, num_channels=3, nn_nb=80):
        super(PrimitivesEmbeddingDGCNGn, self).__init__()
        self.opt = opt
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb, n_layers=opt.dgcnn_layers)
        self.drop = 0.0

        self.n_layers = self.opt.dgcnn_layers

        # convolutional layer 1
        if self.n_layers == 3:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        elif self.n_layers == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 128, 512, 1)
        elif self.n_layers == 1:
            self.conv1 = torch.nn.Conv1d(1024 + 64, 512, 1)
        else:
            raise ValueError(f"Unrecognized n_layers for DGCNN: {self.n_layers}.")

        # batch norm 1
        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size

        # ### MLP layers for per-point embedding ####
        self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)  # mlp_layer 1 for embedding
        self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)  # mlp layer 2 for embedding
        self.bn_seg_prob1 = nn.GroupNorm(4, 256)  # how to use GroupNorm?

        # # ### MLP layers for per-point primitive type prediction ####
        # num_primitives = 8
        # self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
        # self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)  # clustering for primitives
        # self.bn_prim_prob1 = nn.GroupNorm(4, 256)  # group norm?

        # # ### MLP layers for per-point normal vector prediction ####
        # self.mlp_normal_prob1 = torch.nn.Conv1d(256, 256, 1)  # normal parameters
        # self.mlp_normal_prob2 = torch.nn.Conv1d(256, 3, 1)
        # self.bn_normal_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points, normals, end_points=None, inds=None, postprocess=False):

        # point.size = bz x N x 6
        # batch_size, N, _ = points.shape

        batch_size = points.shape[0]
        num_points = points.shape[2]
        # get encoder features
        x, first_layer_features = self.encoder(points)

        # num_points
        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        # global features | point features
        x = torch.cat([x, first_layer_features], 1)

        # GET transformed point embeddings
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)
        # GET embedding
        # mlp layers for embedding
        x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
        embedding = self.mlp_seg_prob2(x).permute(0, 2, 1)

        # primitive classification
        # x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
        # type_per_point = self.mlp_prim_prob2(x)

        # type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)

        # x = F.dropout(F.relu(self.bn_normal_prob1(self.mlp_normal_prob1(x_all))), self.drop)
        # # normal per point
        # normal_per_point = self.mlp_normal_prob2(x).permute(0, 2, 1)  # mlp normal prob?
        # normal_norm = torch.norm(normal_per_point, dim=-1,  #
        #                          keepdim=True).repeat(1, 1, 3) + 1e-12
        # normal_per_point = normal_per_point / normal_norm

        return embedding # , type_per_point, normal_per_point


class PrimitiveNet(nn.Module):
    def __init__(self, opt):
        super(PrimitiveNet, self).__init__()
        self.opt = opt

        # input_feature_dim = 3 if self.opt.input_normal else 0

        if self.opt.backbone == 'DGCNN':
            self.affinitynet = PrimitivesEmbeddingDGCNGn(
                opt=opt,
                emb_size=self.opt.dgcnn_out_dim,
                mode=5,
                num_channels=opt.dgcnn_in_feat_dim,  # number of input channel [pos, normal]
            )

    def forward(self, xyz, normal, inds=None, postprocess=False):
        # xyz 
        # Type prediction --- type classification prediction;
        # Type parameter prediction --- related parameter prediction
        feat_spec_embedding = self.affinitynet(
            xyz.transpose(1, 2).contiguous(),
            normal.transpose(1, 2).contiguous(),
            inds=inds,
            postprocess=postprocess)

        return feat_spec_embedding 

