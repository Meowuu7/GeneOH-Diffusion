import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

try:
    from torch_cluster import fps
except:
    pass
from collections import OrderedDict
import os, argparse, copy, json
import math

def sample_pcd_from_mesh(vertices, triangles, npoints=512):
    arears = []
    for i in range(triangles.shape[0]):
        v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(triangles[i, 2].item())
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        ab, ac = v_b - v_a, v_c - v_a
        cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9, a_max=9999999.0)).item()
        sin_ab_ac = math.sqrt(1. - cos_ab_ac ** 2)
        cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()
        arears.append(cur_area)
    tot_area = sum(arears)

    sampled_pcts = []
    tot_indices = []
    tot_factors = []
    for i in range(triangles.shape[0]):

        v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
            triangles[i, 2].item())
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        # ab, ac = v_b - v_a, v_c - v_a
        # cur_sampled_pts = int(npoints * (arears[i] / tot_area))
        cur_sampled_pts = math.ceil(npoints * (arears[i] / tot_area))
        # if cur_sampled_pts == 0:

        cur_sampled_pts = int(arears[i] * npoints)
        cur_sampled_pts = 1 if cur_sampled_pts == 0 else cur_sampled_pts

        tmp_x, tmp_y = np.random.uniform(0, 1., (cur_sampled_pts,)).tolist(), np.random.uniform(0., 1., (cur_sampled_pts,)).tolist()

        for xx, yy in zip(tmp_x, tmp_y):
            sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
            aa = 1. - sqrt_xx
            bb = sqrt_xx * (1. - yy)
            cc = yy * sqrt_xx
            cur_pos = v_a * aa + v_b * bb + v_c * cc
            sampled_pcts.append(cur_pos)

            tot_indices.append(triangles[i]) # tot_indices for triangles # # vertices indices
            tot_factors.append([aa, bb, cc])

    tot_indices = np.array(tot_indices, dtype=np.long)
    tot_factors = np.array(tot_factors, dtype=np.float32)

    sampled_ptcs = np.array(sampled_pcts)
    print("sampled points  from surface:", sampled_ptcs.shape)
    # sampled_pcts = np.concatenate([sampled_pcts, vertices], axis=0)
    return sampled_ptcs, tot_indices, tot_factors


def read_obj_file_ours(obj_fn, sub_one=False):
  vertices = []
  faces = []
  with open(obj_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if items[0] == 'v':
        cur_verts = items[1:]
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif items[0] == 'f':
        cur_faces = items[1:] # faces
        cur_face_idxes = []
        for cur_f in cur_faces:
          try:
            cur_f_idx = int(cur_f.split("/")[0])
          except:
            cur_f_idx = int(cur_f.split("//")[0])
          cur_face_idxes.append(cur_f_idx if not sub_one else cur_f_idx - 1)
        faces.append(cur_face_idxes)
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  return vertices, faces

def clamp_gradient(model, clip):
    for p in model.parameters():
        torch.nn.utils.clip_grad_value_(p, clip)

def clamp_gradient_norm(model, max_norm, norm_type=2):
    for p in model.parameters():
        torch.nn.utils.clip_grad_norm_(p, max_norm, norm_type=2)


def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step
    """
    # warnings.DeprecationWarning("load_network is deprecated. Use module.load_state_dict(strict=False) instead.")
    if isinstance(path, str):
        logger.info("loading network from {}".format(path))
        if path[-3:] == "pth":
            loaded_state = torch.load(path)
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
        else:
            loaded_state = np.load(path).item()
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if len(missingkeys)>0:
        logger.warn("load_network {} missing keys".format(len(missingkeys)), "\n".join(missingkeys))
    if len(unexpectedkeys)>0:
        logger.warn("load_network {} unexpected keys".format(len(unexpectedkeys)), "\n".join(unexpectedkeys))



def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

def seal(mesh_to_seal):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.v[circle_v_id, :]).mean(0)

    sealed_mesh = copy.copy(mesh_to_seal)
    sealed_mesh.v = np.vstack([mesh_to_seal.v, center])
    center_v_id = sealed_mesh.v.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        sealed_mesh.f = np.vstack([sealed_mesh.f, new_faces])
    return sealed_mesh

def read_pos_fr_txt(txt_fn):
    pos_data = []
    with open(txt_fn, "r") as rf:
        for line in rf:
            cur_pos = line.strip().split(" ")
            cur_pos = [float(p) for p in cur_pos]
            pos_data.append(cur_pos)
        rf.close()
    pos_data = np.array(pos_data, dtype=np.float32)
    print(f"pos_data: {pos_data.shape}")
    return pos_data

def read_field_data_fr_txt(field_fn):
    field_data = []
    with open(field_fn, "r") as rf:
        for line in rf:
            cur_field = line.strip().split(" ")
            cur_field = [float(p) for p in cur_field]
            field_data.append(cur_field)
        rf.close()
    field_data = np.array(field_data, dtype=np.float32)
    print(f"filed_data: {field_data.shape}")
    return field_data

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
  
  
def batched_index_select_ours(values, indices, dim = 1):
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

def compute_nearest(query, verts):
    # query: bsz x nn_q x 3
    # verts: bsz x nn_q x 3
    dists = torch.sum((query.unsqueeze(2) - verts.unsqueeze(1)) ** 2, dim=-1)
    minn_dists, minn_dists_idx = torch.min(dists, dim=-1) # bsz x nn_q
    minn_pts_pos = batched_index_select_ours(values=verts, indices=minn_dists_idx, dim=1)
    minn_pts_pos = minn_pts_pos.unsqueeze(2)
    minn_dists_idx = minn_dists_idx.unsqueeze(2)
    return minn_dists, minn_dists_idx, minn_pts_pos
    

def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def batched_get_rot_mtx_fr_vecs(normal_vecs):
    # normal_vecs: nn_pts x 3 #
    # 
    normal_vecs = normal_vecs / torch.clamp(torch.norm(normal_vecs, p=2, dim=-1, keepdim=True), min=1e-5)
    sin_theta = normal_vecs[..., 0]
    cos_theta = torch.sqrt(1. - sin_theta ** 2)
    sin_phi = normal_vecs[..., 1] / torch.clamp(cos_theta, min=1e-5)
    # cos_phi = torch.sqrt(1. - sin_phi ** 2)
    cos_phi = normal_vecs[..., 2] / torch.clamp(cos_theta, min=1e-5)
    
    sin_phi[cos_theta < 1e-5] = 1.
    cos_phi[cos_theta < 1e-5] = 0.
    
    # 
    y_rot_mtx = torch.stack(
        [
            torch.stack([cos_theta, torch.zeros_like(cos_theta), -sin_theta], dim=-1),
            torch.stack([torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([sin_theta, torch.zeros_like(cos_theta), cos_theta], dim=-1)
        ], dim=-1
    )
    x_rot_mtx = torch.stack(
        [
            torch.stack([torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta)], dim=-1),
            torch.stack([torch.zeros_like(cos_phi), cos_phi, -sin_phi], dim=-1), 
            torch.stack([torch.zeros_like(cos_phi), sin_phi, cos_phi], dim=-1)
        ], dim=-1
    )
    rot_mtx = torch.matmul(x_rot_mtx, y_rot_mtx)
    return rot_mtx


def batched_get_rot_mtx_fr_vecs_v2(normal_vecs):
    # normal_vecs: nn_pts x 3 #
    # 
    normal_vecs = normal_vecs / torch.clamp(torch.norm(normal_vecs, p=2, dim=-1, keepdim=True), min=1e-5)
    sin_theta = normal_vecs[..., 0]
    cos_theta = torch.sqrt(1. - sin_theta ** 2)
    sin_phi = normal_vecs[..., 1] / torch.clamp(cos_theta, min=1e-5)
    # cos_phi = torch.sqrt(1. - sin_phi ** 2)
    cos_phi = normal_vecs[..., 2] / torch.clamp(cos_theta, min=1e-5)
    
    sin_phi[cos_theta < 1e-5] = 1.
    cos_phi[cos_theta < 1e-5] = 0.
    
    # o: nn_pts x 3 #
    o = torch.stack(
        [torch.zeros_like(cos_phi), cos_phi, -sin_phi], dim=-1
    )
    nxo = torch.cross(o, normal_vecs)
    # rot_mtx: nn_pts x 3 x 3 #
    rot_mtx = torch.stack(
        [nxo, o, normal_vecs], dim=-1
    )
    return rot_mtx
    

def batched_get_orientation_matrices(rot_vec):
    rot_matrices = []
    for i_w in range(rot_vec.shape[0]):
        cur_rot_vec = rot_vec[i_w]
        cur_rot_mtx = R.from_rotvec(cur_rot_vec).as_matrix()
        rot_matrices.append(cur_rot_mtx)
    rot_matrices = np.stack(rot_matrices, axis=0)
    return rot_matrices

def batched_get_minn_dist_corresponding_pts(tips, obj_pcs):
    dist_tips_to_obj_pc_minn_idx = np.argmin(
        ((tips.reshape(tips.shape[0], tips.shape[1], 1, 3) - obj_pcs.reshape(obj_pcs.shape[0], 1, obj_pcs.shape[1], 3)) ** 2).sum(axis=-1), axis=-1
    )
    obj_pcs_th = torch.from_numpy(obj_pcs).float()
    dist_tips_to_obj_pc_minn_idx_th = torch.from_numpy(dist_tips_to_obj_pc_minn_idx).long()
    nearest_pc_th = batched_index_select(obj_pcs_th, 1, dist_tips_to_obj_pc_minn_idx_th)
    return nearest_pc_th, dist_tips_to_obj_pc_minn_idx_th

def get_affinity_fr_dist(dist, s=0.02):
    ### affinity scores ###
    k = 0.5 * torch.cos(torch.pi / s * torch.abs(dist)) + 0.5
    return k

def batched_reverse_transform(rot, transl, t_pc, trans=True):
    # t_pc: ws x nn_obj x 3 
    # rot; ws x 3 x 3 
    # transl: ws x 1 x 3
    if trans:
        reverse_trans_pc = t_pc - transl
    else:
        reverse_trans_pc = t_pc
    reverse_trans_pc = np.matmul(np.transpose(rot, (0, 2, 1)), np.transpose(reverse_trans_pc, (0, 2, 1)))
    reverse_trans_pc = np.transpose(reverse_trans_pc, (0, 2, 1))
    return reverse_trans_pc
    

def capsule_sdf(mesh_verts, mesh_normals, query_points, query_normals, caps_rad, caps_top, caps_bot, foreach_on_mesh):
    # if caps on hand: mesh_verts = hand vert
    """
    Find the SDF of query points to mesh verts
    Capsule SDF formulation from https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    :param mesh_verts: (batch, V, 3)
    :param mesh_normals: (batch, V, 3)
    :param query_points: (batch, Q, 3)
    :param caps_rad: scalar, radius of capsules
    :param caps_top: scalar, distance from mesh to top of capsule
    :param caps_bot: scalar, distance from mesh to bottom of capsule
    :param foreach_on_mesh: boolean, foreach point on mesh find closest query (V), or foreach query find closest mesh (Q)
    :return: normalized sdsf + 1 (batch, V or Q)
    """
    # TODO implement normal check?
    if foreach_on_mesh:     # Foreach mesh vert, find closest query point
        # knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(mesh_verts, query_points, K=1, return_nn=True)   # TODO should attract capsule middle?
        # knn_dists, nearest_idx, nearest_pos =  compute_nearest(query_points, mesh_verts)
        knn_dists, nearest_idx, nearest_pos =  compute_nearest(mesh_verts, query_points)

        capsule_tops = mesh_verts + mesh_normals * caps_top
        capsule_bots = mesh_verts + mesh_normals * caps_bot
        delta_top = nearest_pos[:, :, 0, :] - capsule_tops
        normal_dot = torch.sum(mesh_normals * batched_index_select(query_normals, 1, nearest_idx.squeeze(2)), dim=2)

        rt_nearest_verts = mesh_verts
        rt_nearest_normals = mesh_normals
        
    else:   # Foreach query vert, find closest mesh point
        # knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(query_points, mesh_verts, K=1, return_nn=True)   # TODO should attract capsule middle?
        st_time = time.time()
        knn_dists, nearest_idx, nearest_pos =  compute_nearest(query_points, mesh_verts)
        ed_time = time.time()
        # print(f"Time for computing nearest: {ed_time - st_time}")
        
        closest_mesh_verts = batched_index_select(mesh_verts, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)
        closest_mesh_normals = batched_index_select(mesh_normals, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)

        capsule_tops = closest_mesh_verts + closest_mesh_normals * caps_top  # Coordinates of the top focii of the capsules (batch, V, 3)
        capsule_bots = closest_mesh_verts + closest_mesh_normals * caps_bot
        delta_top = query_points - capsule_tops
        # normal_dot = torch.sum(query_normals * closest_mesh_normals, dim=2)
        normal_dot = None
        
        rt_nearest_verts = closest_mesh_verts
        rt_nearest_normals = closest_mesh_normals

    # (top -> bot) #!!#
    bot_to_top = capsule_bots - capsule_tops  # Vector from capsule bottom to top
    along_axis = torch.sum(delta_top * bot_to_top, dim=2)   # Dot product
    top_to_bot_square = torch.sum(bot_to_top * bot_to_top, dim=2)
    
    # print(f"top_to_bot_square: {top_to_bot_square[..., :10]}")
    h = torch.clamp(along_axis / top_to_bot_square, 0, 1)   # Could avoid NaNs with offset in division here
    dist_to_axis = torch.norm(delta_top - bot_to_top * h.unsqueeze(2), dim=2)   # Distance to capsule centerline
    
    # two endpoints;  edge of the capsule #
    return dist_to_axis / caps_rad, normal_dot, rt_nearest_verts, rt_nearest_normals  # (Normalized SDF)+1 0 on endpoint, 1 on edge of capsule



def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)  ### std and eps --> 
    eps = torch.randn(std.size()).to(mean.device)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z): # feature dim
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out


def random_rotate_np(x):
    aa = np.random.randn(3)
    theta = np.sqrt(np.sum(aa**2))
    k = aa / np.maximum(theta, 1e-6)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K, K)
    R = R.astype(np.float32)
    return np.matmul(x, R), R


def rotate_x(x, rad):
    rad = -rad
    rotmat = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])
    return np.dot(x, rotmat)

def rotate_y(x, rad):
    rad = -rad
    rotmat = np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])
    return np.dot(x, rotmat)

def rotate_z(x, rad):
    rad = -rad
    rotmat = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])
    return np.dot(x, rotmat)

        