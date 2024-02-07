# Adapted from SPFN

import torch
import torch.nn as nn
try:
    from torch_cluster import fps
except:
    pass
# from .point_convolution_universal import TransitionDown, TransitionUp
# from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet, set_bn_not_training, set_grad_to_none
# from .utils import farthest_point_sampling, get_knn_idx, batched_index_select


def set_bn_not_training(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_bn_not_training(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            if isinstance(block, nn.BatchNorm1d) or isinstance(block, nn.BatchNorm2d):
                block.is_training = False
    else:
        raise ValueError("Not recognized module to set not training!")

def set_grad_to_none(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_grad_to_none(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            for param in block.parameters():
                param.grad = None
    else:
        raise ValueError("Not recognized module to set not training!")


def apply_module_with_conv2d_bn(x, module): # bsz x npts x feats -> bsz x feats x npts -> 
    x = x.transpose(2, 3).contiguous().transpose(1, 2).contiguous()
    # print(x.size())
    for layer in module:
        for sublayer in layer:
            x = sublayer(x.contiguous())
        x = x.float()
    x = torch.transpose(x, 1, 2).transpose(2, 3)
    return x

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def init_weight(blocks):
    for module in blocks:
        if isinstance(module, nn.Sequential):
            for subm in module:
                if isinstance(subm, nn.Linear):
                    nn.init.xavier_uniform_(subm.weight)
                    nn.init.zeros_(subm.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def construct_conv_modules(mlp_dims, n_in, last_act=True, bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act)):
            blk = nn.Sequential(
                    nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                    nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list



def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx



class PointnetPP(nn.Module):
    def __init__(self, in_feat_dim: int):
        super(PointnetPP, self).__init__()

        # if args is not None: # 
        #     self.skip_global = args.skip_global
        # else:
        self.skip_global = False

        # self.n_samples = [512, 128, 1] # if "motion" not in args.task else [256, 128, 1]
        self.n_samples = [256, 128, 1]
        # self.n_samples = [1024, 512, 1]
        mlps = [[64,64,128], [128,128,256], [256,512,1024]]
        mlps_in = [[in_feat_dim,64,64], [128+3,128,128], [256+3,256,512]]

        # up_mlps = [[256, 256], [256, 128], [128, 128, 128]]
        up_mlps = [[512, 512], [512, 512], [512, 512, 512]]
        # up_mlps_in = [1024+256, 256+128, 128+3+3]
        up_mlps_in = [1024 + 256, 512 + 128, 512 + in_feat_dim]

        self.in_feat_dim = in_feat_dim
        self.radius = [0.2, 0.4, None]
        
        self.radius = [None, None, None]

        # if args is not None: # radius? #
        #     n_layers = args.pnpp_n_layers
        #     self.n_samples = self.n_samples[:n_layers]
        #     mlps, mlps_in = mlps[:n_layers], mlps_in[:n_layers]
        #     self.radius = self.radius[:n_layers]

        #     up_mlps = up_mlps[-n_layers:]
        #     up_mlps_in = up_mlps_in[-n_layers:]

        self.mlp_layers = nn.ModuleList()

        for i, (dims_in, dims_out) in enumerate(zip(mlps_in, mlps)):
            # if self.skip_global and i == len(mlps_in) - 1:
            #     break
            conv_layers = construct_conv_modules(
                mlp_dims=dims_out, n_in=dims_in[0],
                last_act=True,
                bn=True
            )
            self.mlp_layers.append(conv_layers)

        self.up_mlp_layers = nn.ModuleList()

        for i, (dim_in, dims_out) in enumerate(zip(up_mlps_in, up_mlps)):
            # if self.skip_global and i == 0:
            #     continue
            conv_layers = construct_conv_modules(
                mlp_dims=dims_out, n_in=dim_in,
                # last_act=False,
                last_act=True,
                bn=True
            )
            self.up_mlp_layers.append(conv_layers)
            
    def eval(self):
        super().eval()
        self.set_bn_no_training()
        # return super().eval()

    def set_bn_no_training(self):
        for sub_module in self.mlp_layers:
            set_bn_not_training(sub_module)
        for sub_module in self.up_mlp_layers:
            set_bn_not_training(sub_module)

    def set_grad_to_none(self):
        for sub_module in self.mlp_layers:
            set_grad_to_none(sub_module)
        for sub_module in self.up_mlp_layers:
            set_grad_to_none(sub_module)

    def sample_and_group(self, feat, pos, n_samples, use_pos=True, k=64):
        bz, N = pos.size(0), pos.size(1)
        fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=n_samples)
        # bz x n_samples x pos_dim
        # sampled_pos = batched_index_select(values=pos, indices=fps_idx, dim=1)
        sampled_pos = pos.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples, -1)
        ppdist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        ppdist = torch.sqrt(ppdist)
        topk_dist, topk_idx = torch.topk(ppdist, k=k, dim=2, largest=False)

        # if n_samples == 1:
        #
        grouped_pos = batched_index_select(values=pos, indices=topk_idx, dim=1)
        grouped_pos = grouped_pos - sampled_pos.unsqueeze(2)
        if feat is not None:
            grouped_feat = batched_index_select(values=feat, indices=topk_idx, dim=1)
            if use_pos:
                grouped_feat = torch.cat([grouped_pos, grouped_feat], dim=-1)
        else:
            grouped_feat = grouped_pos
        return grouped_feat, topk_dist, sampled_pos

    def max_pooling_with_r(self, grouped_feat, ppdist, r=None):
        if r is None:
            res, _ = torch.max(grouped_feat, dim=2)
        else:
            # bz x N x k
            indicators = (ppdist <= r).float()
            indicators_expand = indicators.unsqueeze(-1).repeat(1, 1, 1, grouped_feat.size(-1))
            indicators_expand[indicators_expand < 0.5] = -1e8
            indicators_expand[indicators_expand > 0.5] = 0.
            # grouped_feat[indicators_expand < 0.5] = -1e8
            # res, _ = torch.max(grouped_feat, dim=2)
            res, _ = torch.max(grouped_feat +  indicators_expand, dim=2)
        return res

    def interpolate_features(self, feat, p1, p2, ):
        dist = p2[:, :, None, :] - p1[:, None, :, :]
        dist = torch.norm(dist, dim=-1, p=2, keepdim=False)
        topkk = min(3, dist.size(-1))
        dist, idx = dist.topk(topkk, dim=-1, largest=False)

        # bz x N2 x 3
        # print(dist.size(), idx.size())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # weight.size() = bz x N2 x 3; idx.size() = bz x N2 x 3
        three_nearest_features = batched_index_select(feat, idx, dim=1)  # 1 is the idx dimension
        interpolated_feats = torch.sum(three_nearest_features * weight[:, :, :, None], dim=2, keepdim=False)
        return interpolated_feats

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, return_global=False, 
                ):

        # x = x[:, :, 3:] # bsz x nnf x nnbasepts x nnbaseptsfeats #
        bz = pos.size(0)

        cache = []
        cache.append((None if x is None else x.clone(), pos.clone()))

        n_samples = self.n_samples
        for i, n_samples in enumerate(n_samples): # point view ---> how to look joints from the base pts here --> and for the point convs #
            if n_samples == 1:
                grouped_feat = x.unsqueeze(1)
                grouped_feat = torch.cat(
                    [pos.unsqueeze(1), grouped_feat], dim=-1
                )
                grouped_feat = apply_module_with_conv2d_bn(
                    grouped_feat, self.mlp_layers[i]
                ).squeeze(1)
                x, _ = torch.max(grouped_feat, dim=1, keepdim=True)
                sampled_pos = torch.zeros((bz, 1, 3), dtype=torch.float, device=pos.device)
                pos = sampled_pos
            else:
                grouped_feat, topk_dist, pos = self.sample_and_group(x, pos, n_samples, use_pos=True, k=64)
                # print(f"x: {x.size()}, pos: {pos.size()}, grouped_feat: {grouped_feat.size()}")
                grouped_feat = apply_module_with_conv2d_bn(
                    grouped_feat, self.mlp_layers[i]
                )
                cur_radius = self.radius[i]
                x = self.max_pooling_with_r(grouped_feat, topk_dist, r=cur_radius)
            cache.append((x.clone(), pos.clone()))

        up_mlp_layers = self.up_mlp_layers

        # global_x = x
        for i, up_conv_layers in enumerate(up_mlp_layers):
            prev_x, prev_pos = cache[-i-2][0], cache[-i-2][1]
            # print(prev_pos.size(), x.size(), pos.size())
            # interpolate x via pos & prev_pos # interpolate features 
            interpolated_feats = self.interpolate_features(x, pos, prev_pos)

            if prev_x is None:
                prev_x = prev_pos
            elif i == len(self.up_mlp_layers) - 1:
                prev_x = torch.cat([prev_x, prev_pos], dim=-1)
            # if without previous x, we only have the interpolated feature
            cur_up_feats = torch.cat([interpolated_feats, prev_x], dim=-1)
            x = apply_module_with_conv2d_bn(
                cur_up_feats.unsqueeze(2), up_conv_layers
            ).squeeze(2)
            pos = prev_pos

        # # bsz x nnf x nnbasepts x nnbaseptsfeats #
        # if return_global:
        #     return x, global_x, pos # pos, base_pts_feats #
        # else:
        return x, pos
