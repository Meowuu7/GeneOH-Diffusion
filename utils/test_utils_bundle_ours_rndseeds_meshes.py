# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

import numpy as np
import torch.nn as nn
# import layer_utils
import torch
# import data_utils_torch as data_utils
import math ## 
import os
# from options.options import opt

# import model_util
# 

### smoothness 
### whether in the object -> using vertices and using joints ###
### 
import trimesh



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



def calculate_joint_smoothness(joint_seq):
  # joint_seq: nf x nnjoints x 3
  disp_seq = joint_seq[1:] - joint_seq[:-1] # (nf - 1) x nnjoints x 3 #
  disp_seq = np.sum(disp_seq ** 2, axis=-1)
  disp_seq = np.mean(disp_seq)
  # disp_seq = np.
  disp_seq = disp_seq.item()
  return disp_seq




def calculate_penetration_depth(subj_seq, obj_verts, obj_faces):
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
            )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  for i_f in range(subj_seq.shape[0]): ## total sequence length ##
  # for i_f in range(10):
    cur_subj_seq = subj_seq[i_f]
    cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #
    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts #
      (np.reshape(cur_subj_seq, (cur_subj_seq.shape[0], 1, 3)) - np.reshape(obj_verts, (1, obj_verts.shape[0], 3))) ** 2, axis=-1
    )
    # dist_cur_subj_to_obj_verts 
    nearest_obj_dist = np.min(dist_cur_subj_to_obj_verts, axis=-1) # nn_subj_pts
    nearest_obj_dist = np.sqrt(nearest_obj_dist)
    cur_pene_depth = np.zeros_like(nearest_obj_dist)
    cur_pene_depth[cur_subj_seq_in_obj] = nearest_obj_dist[cur_subj_seq_in_obj]
    tot_penetration_depth.append(cur_pene_depth)
#   tot_penetration_depth = max(tot_penetration_depth)
  tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
  tot_penetration_depth = np.mean(tot_penetration_depth).item()
  return tot_penetration_depth


def calculate_max_penetration_depth(subj_seq, obj_verts, obj_faces):
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  print(f"")
  obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
            )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  for i_f in range(subj_seq.shape[0]): ## total sequence length ##
  # for i_f in range(10):
    cur_subj_seq = subj_seq[i_f]
    # print(f"cur_subj_seq: {cur_subj_seq.shape}")
    cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #
    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts #
      (np.reshape(cur_subj_seq, (cur_subj_seq.shape[0], 1, 3)) - np.reshape(obj_verts, (1, obj_verts.shape[0], 3))) ** 2, axis=-1
    )
    # dist_cur_subj_to_obj_verts 
    nearest_obj_dist = np.min(dist_cur_subj_to_obj_verts, axis=-1) # nn_subj_pts
    nearest_obj_dist = np.sqrt(nearest_obj_dist)
    cur_pene_depth = np.zeros_like(nearest_obj_dist)
    cur_pene_depth[cur_subj_seq_in_obj] = nearest_obj_dist[cur_subj_seq_in_obj]

    cur_pene_depth = float(np.max(cur_pene_depth))
    tot_penetration_depth.append(cur_pene_depth)
  tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
#   tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
#   tot_penetration_depth = np.mean(tot_penetration_depth).item()
  return tot_penetration_depth


def calculate_proximity_dist(subj_seq, subj_seq_gt, obj_verts, obj_faces):
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
            )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  
  # nf x nn_subj_pts x 3 # # nf x nn_subj_pts x nn_obj_pts
  dist_subj_seq_to_obj_verts_gt = np.sum(
    (np.reshape(subj_seq_gt, (subj_seq_gt.shape[0], subj_seq_gt.shape[1], 1, 3)) - np.reshape(obj_verts, (1, 1, obj_verts.shape[0], 3))) ** 2, axis=-1
  )
  minn_dist_subj_seq_to_obj_verts_gt = np.min(dist_subj_seq_to_obj_verts_gt, axis=-1) # nf x nn_subj_pts 
  
  
  # nf x nn_subj_pts x 3 # # nf x nn_subj_pts x nn_obj_pts
  dist_subj_seq_to_obj_verts = np.sum(
    (np.reshape(subj_seq, (subj_seq.shape[0], subj_seq.shape[1], 1, 3)) - np.reshape(obj_verts, (1, 1, obj_verts.shape[0], 3))) ** 2, axis=-1
  )
  minn_dist_subj_seq_to_obj_verts = np.min(dist_subj_seq_to_obj_verts, axis=-1) # nf x nn_subj_pts 
  
  dist_minn_dist = np.mean(
    (minn_dist_subj_seq_to_obj_verts_gt[46:][..., -5:-3] - minn_dist_subj_seq_to_obj_verts[46:][..., -5:-3]) ** 2
  ).item()
  
  # dist_minn_dist = np.sum(
  #   (minn_dist_subj_seq_to_obj_verts_gt[46:][..., -5:-3] - minn_dist_subj_seq_to_obj_verts[46:][..., -5:-3]) ** 2, axis=-1
  # )
  
  # dist_minn_dist = np.mean(
  #   (minn_dist_subj_seq_to_obj_verts_gt - minn_dist_subj_seq_to_obj_verts) ** 2
  # ).item()
  
  return dist_minn_dist
  
  for i_f in range(subj_seq.shape[0]):
    cur_subj_seq = subj_seq[i_f]
    cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #
    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts #
      (np.reshape(cur_subj_seq, (cur_subj_seq.shape[0], 1, 3)) - np.reshape(obj_verts, (1, obj_verts.shape[0], 3))) ** 2, axis=-1
    )
    # dist_cur_subj_to_obj_verts 
    nearest_obj_dist = np.min(dist_cur_subj_to_obj_verts, axis=-1) # nn_subj_pts
    nearest_obj_dist = np.sqrt(nearest_obj_dist)
    cur_pene_depth = np.zeros_like(nearest_obj_dist)
    cur_pene_depth[cur_subj_seq_in_obj] = nearest_obj_dist[cur_subj_seq_in_obj]
    tot_penetration_depth.append(cur_pene_depth)
  tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
  tot_penetration_depth = np.mean(tot_penetration_depth).item()
  return tot_penetration_depth




def calculate_moving_consistency(base_pts_trans, joints_trans):
  # base_pts_trans: nf x nn_base_pts x 3 #
  # joints_trans: nf x nn_jts x 3 #
  base_pts_trans = torch.from_numpy(base_pts_trans).float()
  joints_trans = torch.from_numpy(joints_trans).float()
  # dist_joints_to_base_pts = np.sum
  dist_joints_to_base_pts = torch.sum(
    (joints_trans.unsqueeze(2) - base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts x nn_base_pts #
  )
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts)
  dist_joints_to_base_pts, joints_to_base_pts_minn_idxes = torch.min(dist_joints_to_base_pts, dim=-1) 
  
  minn_dist_joints_to_base_pts_across_joints, minn_dist_joints_to_base_pts_idxes = torch.min(dist_joints_to_base_pts, dim=-1) # (nf - 1)
  minn_dist_joints_to_base_pts_idxes = minn_dist_joints_to_base_pts_idxes[:-1]
  
  disp_joints_to_base_pts_minn_idxes = joints_to_base_pts_minn_idxes[:-1]
  disp_base_pts = base_pts_trans[1:] - base_pts_trans[:-1]
  disp_joints = joints_trans[1:] - joints_trans[:-1] # (nf - 1) x nn_jts x 3 
  dist_joints_to_base_pts = dist_joints_to_base_pts[:-1]

  k_f = 100.
  k = torch.exp(
    -k_f * dist_joints_to_base_pts
  )

  disp_joints_base_pts = batched_index_select_ours(disp_base_pts, indices=disp_joints_to_base_pts_minn_idxes, dim=1) # (nf - 1) x nn_jts x 3 
  
  nearest_joints_disp = batched_index_select_ours(disp_joints_base_pts, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1) # (nf - 1) x 1
  nearest_joints_disp = nearest_joints_disp.squeeze(1) # (nf - 1) x 3 #
  
  disp_joints = batched_index_select_ours(disp_joints, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1).squeeze(1) # (nf - 1) x 3 
  
  nearest_k = batched_index_select_ours(k, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1).squeeze(1) # (nf - 1)
  
  ##### use k for weighting disp ##### # disp_joints for the joints  # the delta distance #
  disp_joints_to_nearest_base_pts = disp_joints *  nearest_k.unsqueeze(-1) # ### (nf - 1 ) x 3 
  diff_disp_joints_to_nearest_base_pts_disp = torch.sum(
    (disp_joints_to_nearest_base_pts - nearest_joints_disp) ** 2, dim=-1 # squared joint jdistance pairs #
  )
  diff_disp_joints_base_pts = diff_disp_joints_to_nearest_base_pts_disp.mean()
  ##### use k for weighting disp #####
  
  # diff_disp_joints_base_pts = torch.sum(
  # )
  
  ##### use k for weighting diff #####
  # diff_disp_joints_base_pts = torch.sum(
  #   (disp_joints - disp_joints_base_pts) ** 2, dim=-1 # (nf - 1) x nn_jts
  # )
  # diff_disp_joints_base_pts = torch.sqrt(diff_disp_joints_base_pts) * k
  
  
  
  # diff_disp_joints_base_pts = diff_disp_joints_base_pts.mean() # # mean of the base_pts # 
  ##### use k for weighting diff #####
  
  return diff_disp_joints_base_pts.item()


# static grasping stability -> the gravity direction and the linear combination of contact directions
# dynamic grasping stability -> 1) rotation and acceleration dynamics of the object 2) forces added by contact points

def calculate_grasping_stability(hand_verts, obj_verts, obj_normals, obj_grav_dirs=None): # obj_grav_dir: nf x 3 --> negative to the object gravity dir here #
  # hand_verts: nf x nn_verts x 3 #
  # obj_verts: nf x nn_obj_verts x 3 #
  # obj_normals: nf x nn_obj_normals x 3 #
  # obj_grav_dirs: nf x 3 # --> for object gravity directions #
  contact_thres = 0.002 # 2mm #
  # gravity_dir = np.zeros((3,),dtype=np.float32)
  if obj_grav_dirs is None:
    gravity_dir = np.array([0., 1., 0.], dtype=np.float32) # (3,) for the gravity direction # # negative to the gravity direction #
  # concert to 
  # if not isinstance(hand_verts, torch.Tensor):
  #   hand_verts = torch.from_numpy(hand_verts).float().cuda()
  #   # dir_dim x nn_candidate_dirs xxxx nn_candidate_dirs x 1 --> dir_dim x 1 # a leasts square problem?
  nn_hand_verts = hand_verts.shape[1] # nf x nn_hand_verts x 3 #
  nn_obj_verts = obj_verts.shape[1] # nf x nn_obj_verts x 3 #
  nn_frames = hand_verts.shape[0]
  dist_hand_verts_to_obj_verts = np.sum(
    (hand_verts.reshape(nn_frames, nn_hand_verts, 1, 3) - obj_verts.reshape(nn_frames, 1, nn_obj_verts, 3)) ** 2, axis=-1 # nf x nn_hand_verts x nn_obj_verts
  )
  minn_dist_hand_verts_to_obj_verts_idxes = np.argmin(dist_hand_verts_to_obj_verts, axis=-1) # nf x nn_hand_verts
  minn_dist_hand_verts_to_obj_verts = np.min(dist_hand_verts_to_obj_verts, axis=-1) # nf x nn_hand_verts #
  # minn_dist_hand_vert
  hand_verts_in_contact_mask = minn_dist_hand_verts_to_obj_verts <= contact_thres ## nf x nn_hand_verts #
  
  # nf x nn_hand_verts # 
  
  obj_normals_th = torch.from_numpy(obj_normals).float()
  minn_dist_hand_verts_to_obj_verts_idxes_th = torch.from_numpy(minn_dist_hand_verts_to_obj_verts_idxes).long()
  hand_verts_in_contact_obj_normals_th = batched_index_select_ours(obj_normals_th, minn_dist_hand_verts_to_obj_verts_idxes_th, dim=1) # nf x nn_hand_verts x 3 
  hand_verts_in_contact_obj_normals = hand_verts_in_contact_obj_normals_th.numpy()
  # hand_verts_in_contact_obj_normals = obj_normals[ minn_dist_hand_verts_to_obj_verts_idxes] # nf x nn_hand_verts x 3 # for obj normals in contact with hand verts #
  # print(f"Selected hand_verts_in_contact_obj_normals: {hand_verts_in_contact_obj_normals.shape}")
  hand_verts_in_contact_obj_normals = hand_verts_in_contact_obj_normals * -1.0
  diff_cloest_dir_to_gravity_dir = []
  for i_f in range(nn_frames):
    if obj_grav_dirs is not None:
      gravity_dir = obj_grav_dirs[i_f]
    
    cur_fr_hand_verts_in_contact_mask = hand_verts_in_contact_mask[i_f] # nn_hnad_verts #
    cur_fr_hand_verts_in_contact_obj_normals = hand_verts_in_contact_obj_normals[i_f] # nn_hand_verts x 3 # 
    if np.sum(cur_fr_hand_verts_in_contact_mask.astype(np.float32)).item() == 0:
      cur_diff_cloest_dir_to_gravity_dir = np.sqrt(np.sum(gravity_dir ** 2, axis=0)).item()
      diff_cloest_dir_to_gravity_dir.append(cur_diff_cloest_dir_to_gravity_dir)
      continue
    # print(f"cur_fr_hand_verts_in_contact_obj_normals: {cur_fr_hand_verts_in_contact_obj_normals.shape}, cur_fr_hand_verts_in_contact_mask: {cur_fr_hand_verts_in_contact_mask.shape}")
    cur_fr_hand_verts_in_contact_obj_normals = cur_fr_hand_verts_in_contact_obj_normals[cur_fr_hand_verts_in_contact_mask] ## nn_in_contact_pts x 3 #
    in_contact_coeff, res, _, _ = np.linalg.lstsq(cur_fr_hand_verts_in_contact_obj_normals.T, gravity_dir.reshape(3, 1)) # nn_in_contact_pts x 1 as the combination coefficients #
    # print(f"in_contact_coeff: {in_contact_coeff.shape}")
    # print()
    combined_in_contact_dir = np.matmul(
      cur_fr_hand_verts_in_contact_obj_normals.T, in_contact_coeff
    )
    combined_in_contact_dir = combined_in_contact_dir.reshape(3,)
    diff_combined_in_contact_dir_to_gravity_dir = np.sum(
      (combined_in_contact_dir - gravity_dir) ** 2, axis=-1
    ).item() # for the in_contact direction 
    diff_cloest_dir_to_gravity_dir.append(diff_combined_in_contact_dir_to_gravity_dir)
  diff_cloest_dir_to_gravity_dir = sum(diff_cloest_dir_to_gravity_dir) / float(len(diff_cloest_dir_to_gravity_dir))
  return diff_cloest_dir_to_gravity_dir

### metrics ### # avg acc metrics ##
def get_acc_metrics(outputs, gt_joints):
  # 
  # outputs: ws x nn_jts x 3 #
  # gt_joints: ws x nn_jts x 3 #
  dist_outputs_gt_joints = np.sqrt(np.sum((outputs - gt_joints) ** 2, axis=-1)) # ws x nn_jts #
  avg_dist_outputs_gt_joints = np.mean(dist_outputs_gt_joints).item()
  return avg_dist_outputs_gt_joints #### m -> the average ###




# smoothness: 4.6728710003662854e-05, average penetration depth: 6.894875681965049e-05, minn_dist_dist: 4.546074229087353e-07
#  smoothness: 4.832542617805302e-05, average penetration depth: 4.242679380969064e-05, minn_dist_dist: 3.059734870003453e-07

# jts and rel: smoothness: 4.6728710003662854e-05, average penetration depth: 6.894875681965049e-05, minn_dist_dist: 4.624256727994665e-07
# jts: smoothness: 4.832542617805302e-05, average penetration depth: 4.242679380969064e-05, minn_dist_dist: 4.4718556466597084e-07
# 

# T = 400
# smoothness: 3.935288259526715e-05, average penetration depth: 0.000366439988587091, minn_dist_dist: 8.311451183183466e-07
# smoothness: 0.00011870301386807114, average penetration depth: 0.0002747326595483027, minn_dist_dist: 8.44164813606991e-07

# smoothness: 6.404393207048997e-05, average penetration depth: 0.00037923554579558723, minn_dist_dist: 1.3080925943256124e-06 # 
# smoothness: 3.935288259526715e-05, average penetration depth: 0.000366439988587091, minn_dist_dist: 8.311451183183466e-07 # 3
# # 3.0236743775145877e-07, 5.490092818343169e-07

# the examples on different noise scales #

# T = 300
# smoothness: 4.595715654431842e-05, average penetration depth: 0.00021960893150693885, minn_dist_dist: 5.6245558029369e-07 -- jts only
# smoothness: 0.00011870301386807114, average penetration depth: 0.0002747326595483027, minn_dist_dist: 8.44164813606991e-07
# 


from manopth.manolayer import ManoLayer
def get_mano_model():
  mano_path =  "/data1/sim/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  )
  return mano_layer



def load_grab_clip_data_clean_subj(clip_seq_idx, split = "train", pert=False, more_pert=False, other_noise=False):
  mano_model = get_mano_model()
  grab_path = "/data1/sim/GRAB_extracted" # extracted 
  # split = "test"
  window_size = 60
  singe_seq_path = f"/data1/sim/GRAB_processed/{split}/{clip_seq_idx}.npy"
  clip_clean = np.load(singe_seq_path)
  subj_root_path = '/data1/sim/GRAB_processed_wsubj'
  subj_seq_path = f"{clip_seq_idx}_subj.npy"  
  
  subj_params_fn = os.path.join(subj_root_path, split, subj_seq_path)

  subj_params = np.load(subj_params_fn, allow_pickle=True).item()

  window_size = min(window_size, subj_params["rhand_transl"].shape[0])

  rhand_transl = subj_params["rhand_transl"][:window_size].astype(np.float32)
  rhand_betas = subj_params["rhand_betas"].astype(np.float32)
  rhand_pose = clip_clean['f2'][:window_size].astype(np.float32) ## rhand pose ## # 
  rhand_global_orient = clip_clean['f1'][:window_size].astype(np.float32)
  # rhand_pose = clip_clean['f2'][:window_size].astype(np.float32)

  rhand_global_orient_var = torch.from_numpy(rhand_global_orient).float()
  rhand_pose_var = torch.from_numpy(rhand_pose).float()
  rhand_beta_var = torch.from_numpy(rhand_betas).float()
  rhand_transl_var = torch.from_numpy(rhand_transl).float() 

  aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
  # aug_trans, aug_rot, aug_pose = 0.01, 0.1, 0.5
  # aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
  # aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3

  if pert:

    if more_pert:
      aug_trans, aug_rot, aug_pose = 0.04, 0.2, 0.8

    if other_noise:
      dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
      print(f"here!")
      aug_pose_var = dist_beta.sample(rhand_pose_var.size()).squeeze(-1) * aug_pose
      aug_global_orient_var = dist_beta.sample(rhand_global_orient_var.size()).squeeze(-1) * aug_rot
      print(f"aug_pose_var: {aug_pose_var.size()}, aug_global_orient_var: {aug_global_orient_var.size()}")
      aug_transl_var = dist_beta.sample(rhand_transl_var.size()).squeeze(-1) * aug_trans
    else:
      aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
      aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
      aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans

    rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
    rnd_aug_pose_var = rhand_pose_var + aug_pose_var
    rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl ### 
  else:
    rnd_aug_global_orient_var = rhand_global_orient_var
    rnd_aug_pose_var = rhand_pose_var
    rnd_aug_transl_var = rhand_transl_var ### aug transl 


  
  rhand_verts, rhand_joints = mano_model(
      torch.cat([rnd_aug_global_orient_var, rnd_aug_pose_var], dim=-1),
      rhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), rnd_aug_transl_var
  )
  ### rhand_joints: for joints ###
  rhand_verts = rhand_verts * 0.001
  rhand_joints = rhand_joints * 0.001
        
#   return rhand_verts
#   return rhand_joints.detach().cpu().numpy()
  return rhand_verts.detach().cpu().numpy()


def get_contact_ious(hand_contact_maps_a, hand_contact_maps_b):
  intersection_a = ((hand_contact_maps_a + hand_contact_maps_b) > 1.5).astype(np.float32) # nf x nn_obj_verts #
  intersection_b = ((hand_contact_maps_a + hand_contact_maps_b) < 0.5).astype(np.float32) # nf x nn_obj_verts #
#   intersection = ((intersection_a + intersection_b) > 0.5).astype(np.float32) # larger than the intersection #

  intersection = intersection_a
#   intersection = np.sum(intersection, axis=-1)
#   intersection = intersection / float(hand_contact_maps_a.shape[1])
#   intersection = np.mean(intersection).item()
#   return intersection
  union = ((hand_contact_maps_a + hand_contact_maps_b) > 0.5).astype(np.float32) # nf x nn_obj_verts #
  intersection = np.sum(intersection, axis=-1)
  union = np.sum(union, axis=-1)

  intersection = np.sum(intersection).item()
  union = np.sum(union).item()
  intersection_over_union = float(intersection) / float(union)
  return intersection_over_union

#   intersection_over_union = intersection / np.clip(union, a_min=1e-9, a_max=None)
#   intersection_over_union = np.mean(intersection_over_union[union > 0.5]).item()
#   return intersection_over_union



# calcualte_contacts(obj_verts, hand_verts)
# get_contact_ious(hand_contact_maps_a, hand_contact_maps_b)
def calcualte_contacts(obj_verts, hand_verts):
  # obj_verts: nn_obj_verts x 3 # 
  # hand_verts: nf x nn_hand_verts x 3 # ---> nf x nn_obj_verts x nn_hand_verts
  dist_obj_verts_to_hand_verts = np.sum(
    (np.reshape(obj_verts, (1, obj_verts.shape[0], 1, 3)) -  np.reshape(hand_verts, (hand_verts.shape[0], 1, hand_verts.shape[1], 3))) ** 2, axis=-1
  )
  minn_dist_obj_to_hand = np.min(dist_obj_verts_to_hand_verts, axis=-1) # nf x nn_obj_verts #
  minn_dist_obj_to_hand = np.sqrt(minn_dist_obj_to_hand)
  hand_contact_maps = np.zeros_like(minn_dist_obj_to_hand)
  hand_contact_maps[minn_dist_obj_to_hand < 0.002] = 1.
  return hand_contact_maps

def get_resplit_test_idxes():
    test_split_mesh_nm_to_seq_idxes = "/home/xueyi/sim/motion-diffusion-model/test_mesh_nm_to_test_seqs.npy"
    test_split_mesh_nm_to_seq_idxes = np.load(test_split_mesh_nm_to_seq_idxes, allow_pickle=True).item()
    tot_test_seq_idxes = []
    for tst_nm in test_split_mesh_nm_to_seq_idxes:
        tot_test_seq_idxes = tot_test_seq_idxes + test_split_mesh_nm_to_seq_idxes[tst_nm]
    return tot_test_seq_idxes

   

# 8.121088892826279e-07
# 2.951481956519773e-07
# smoothness: 6.213585584191605e-05, average penetration depth: 4.278580710096256e-05, minn_dist_dist: 7.753524560089469e-07
# smoothness: 5.4274405556498095e-05, average penetration depth: 3.388783391933507e-05, minn_dist_dist: 5.682716970124003e-07
# /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_ours_rndseeds_meshes.py
if __name__=='__main__':
  # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy"
  # predicted_infos_jtsonly.npy # predicted infos jts ##
  
  tot_APD = []
  tot_smoothness = []
  tot_proximity_error = []
  tot_consistency_value = []
  
  tot_stability = []
   
  use_toch = False
  # use_toch = True

  resplit = False
  resplit = True


  st_idx = 0
  ed_idx = 246 #
  # ed_idx = 160
  tot_seq_nn = 0
  maxx_seq_nn = 246

  all_seq_tot_metrics = []

  all_seqs_jts_acc = []

  all_seqs_sv_data = []
  avg_max_penetration_depth = []

  tot_test_seq_idxes = range(st_idx, ed_idx, 1)

  if resplit:
    tot_test_seq_idxes = get_resplit_test_idxes()
    seq_root = "/data1/sim/GRAB_processed/train"


  ## cotnact map iou ##
  # st_idx = 58
  # ed_idx = 111
  # st_idx = 0
  # ed_idx = 246
  # for test_seq_idx in range(1, 102, 10):
  # for test_seq_idx in range(1, 102, 1):
  # for test_seq_idx in range(1, 11, 1):
  for test_seq_idx in tot_test_seq_idxes:
    if tot_seq_nn >= maxx_seq_nn:
        break
    tot_jts_acc = []
    seed_to_metrics = {}
    tot_metrics = []

    if use_toch: ## load grab clip ##
        # cur_seq_gt_joints = load_grab_clip_data_clean_subj(test_seq_idx, split = "test", pert=False, more_pert=False, other_noise=False)
        cur_seq_gt_verts = load_grab_clip_data_clean_subj(test_seq_idx, split = "test", pert=False, more_pert=False, other_noise=False)
    

    if use_toch:
      seed_rng = range(0, 12, 11)
      seed_rng = range(0, 122, 11)
    else:
      seed_rng = range(0, 122, 11)

    # test_seq_idx #


    # for i_seed in range(0, 122, 11):
    # for i_seed in range(0, 12, 11):
    for i_seed in seed_rng:
        seed = 31
        seed = 77
        test_tag = "cond_jtsobj"
        # test_tag = "jts_only"
        # test_tag = "rep_only" # 
        test_tag = "rep_only_real"
        test_tag = "rep_only_real_sel_base_0"
        test_tag = "jts_only"
        test_tag = "jts_rep_28_cbd"
        test_tag = "rep_only_real_mean_"
        # /data1/sim/mdm/eval_save/predicted_infos_seq_37_seed_77_tag_jts_only_gaussian_hoi4d_t_300_.npy
        test_tag = "jts_only_gaussian_hoi4d_t_300_"
        # sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        # sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_rep_only_real_sel_base_0.npy"
        # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_101_seed_31_tag_rep_only_real_sel_base_mean.npy
        # sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_rep_only_real_sel_base_mean.npy"
        sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_101_seed_77_tag_rep_only_real_sel_base_mean.npy
        # predicted_infos_seq_101_seed_77_tag_jts_rep_28_cbd.npy
        # sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_jts_only.npy"
        test_tag = "rep_only_real_sel_base_mean"
        seed = 77
        
        test_tag = "jts_only"
        seed = 77
        sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        
        
        test_tag = "jts_rep_28_cbd"
        seed = 77
        
        # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_101_seed_77_tag_jts_rep_55_cbd.npy
        test_tag = "jts_rep_55_cbd"
        seed = 77
        
        test_tag = "jts_rep_28_cbd_t_400"
        seed = 77
        
        test_tag = "jts_only_t_400"
        
        test_tag = "jts_rep_28_cbd_t_400_real"
        test_tag = "jts_rep_19_cbd_t_400_real"
        test_tag = "rep_only_real_sel_base_0_t_400"
        test_tag = "jts_rep_19_cbd_t_300_real"
        
        test_tag = "jts_rep_19_cbd_t_300_real"
        
        test_tag = "jts_only_gaussian_hoi4d_t_300_"
        sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        
        # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_101_seed_77_tag_jts_rep_55_cbd.npy
        
        sv_dir = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
        
        
        
        # /data1/sim/mdm/eval_save/predicted_infos_seq_2_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy
        # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_.npy
        sv_dir = "/data1/sim/mdm/eval_save"
        sv_dir_ours = "/data1/sim/mdm/eval_save"

        # /data2/sim/eval_save/GRAB
        sv_dir = "/data2/sim/eval_save/GRAB"
        sv_dir_ours = "/data2/sim/eval_save/GRAB"

        # sv_dir = "/data1/sim/mdm/eval_save"
        # sv_dir_ours =  "/data1/sim/mdm/eval_save"
        
        # test_tag = "jts_rep_19_cbd_t_300_real" # all noise ##
        test_tag = "jts_only"
        test_tag = "rep_only_real_mean_"
        test_tag = "rep_only_real_mean_same_noise_"
        test_tag = "rep_only_real_mean_same_noise_t_400_"
        test_tag = "rep_only_real_mean_t_400_"
        test_tag = "rep_only_real_mean_t_200_"
        test_tag = "rep_only_real_mean_t_400_nores_"
        # f"/data1/sim/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only_uniform_t_300_.npy"
        test_tag = "jts_only_uniform_t_300_"
        # /data1/sim/mdm/eval_save/predicted_infos_seq_245_seed_77_tag_jts_repmean_only_uniform_t_200_.npy
        test_tag = "jts_repmean_only_uniform_t_200_"
        # test_tag = "rep_only_real_mean_same_noise_"
        # test_tag = "rep_only_real_mean_"
        # test_tag = "rep_only_real_sel_base_mean_all_noise_"
        
        test_tag = "jts_only_gaussian_hoi4d_t_300_"
        # /data1/sim/mdm/eval_save/predicted_infos_seq_37_seed_77_tag_rep_only_mean_shape_hoi4d_t_200_res_jts_.npy
        test_tag = "rep_only_mean_shape_hoi4d_t_200_res_jts_"

        test_tag = "rep_res_jts_grab_t_200_scale_1_"
        test_tag = "rep_res_jts_grab_t_200_scale_2_" # 
        test_tag = "jts_grab_t_400_scale_1_"
        test_tag = "jts_grab_t_400_scale_2_"
        test_tag = "jts_grab_t_400_scale_3_"
        # rep_only_mean_shape_hoi4d_t_200_res_jts_
        # # rep_only_mean_shape_hoi4d_t_400_
        # # test_tag = "rep_only_mean_shape_hoi4d_t_400_"
        
        
        # test_tag = "jts_only_beta_t_300_"
        # /data1/sim/mdm/eval_save_toch/14.npy
        
        seed = 77

        seed = 11
        seed = 22
        ws = 60
        
        ### seed, i_seed ###
        seed = i_seed
        test_tag = "rep_res_jts_grab_t_200_"
        test_tag = "rep_jts_grab_t_200_resplit_res_jts_" # rep res jts ##
        # test_tag = "jts_grab_t_400_"
        # test_tag = "repmean_only_beta_t_200_res_jts_"
        # test_tag = "jts_only_beta_t_300_"
        # # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_jts_only.npy
        # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_.npy
        # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_same_noise_.npy
        sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy" # sv predicted info fn ##
        
        # predicted_infos_seq_99_seed_77_tag_jts_only_beta_t_300_.npy ### seed and seed and seed ###
        sv_predicted_info_fn_ours = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"

        # predicted_infos_seq_99_seed_77_tag_jts_only_beta_t_300_.npy ### seed and seed and seed ###
        sv_predicted_info_fn_ours = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        
        ### for toch ###
        if use_toch:
            # sv_dir = "/data1/sim/mdm/eval_save_toch"
            # # other_noise_grab
            # sv_predicted_info_fn = f"{test_seq_idx}.npy"
            # sv_dir = "/data1/sim/mdm/eval_save_toch"
            # # other_noise_grab
            # sv_predicted_info_fn = f"{test_seq_idx}.npy"
            sv_dir = "/data2/sim/eval_save/GRAB_TOCH"
            # sv_toch_fn = f'hand_verts_seq_{test_seq_idx}.npy' ## sv_toch_fn ##
            # sv_toch_fn = f'hand_verts_seq_aug_{test_seq_idx}.npy' ### with augmentations ##
            sv_predicted_info_fn = f'hand_verts_seq_mixstyle_{test_seq_idx}.npy' 
            # sv_predicted_info_fn = f"other_noise_grab_{test_seq_idx}.npy"
        # # # sv_predicted_info_fn = f"other_noise_grab_{test_seq_idx}.npy"
        ### for toch ###

        pred_infos_sv_folder = f"/data2/sim/eval_save/GRAB"
        optimized_info_fn = f"optimized_infos_sv_dict_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{0.01}_with_proj_{False}.npy"
        
        print(f"sv_predicted_info_fn: {sv_predicted_info_fn}")
        sv_predicted_info_fn = os.path.join(sv_dir, sv_predicted_info_fn)
        predicted_ours_info_fn = os.path.join(sv_dir_ours, sv_predicted_info_fn_ours)
        optimized_info_fn = os.path.join(pred_infos_sv_folder, optimized_info_fn)

        # predicted_infos_seq_1354_seed_44_tag_rep_jts_grab_t_200_resplit_res_jts_.npy
        # predicted_infos_seq_1347_seed_44_tag_rep_jts_grab_t_200_resplit_res_jts_.npy

        # if not os.path.exists(optimized_info_fn) or not os.path.exists(predicted_ours_info_fn):
        #   continue
        if not os.path.exists(predicted_ours_info_fn):
          continue

        ### load optimized infos ###
        # optimized_infos = np.load(optimized_info_fn, allow_pickle=True).item()
        # hand_verts = optimized_infos["hand_verts"]

        # /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_ours_rndseeds_meshes.py
        ### load predicted data ###
        data = np.load(sv_predicted_info_fn, allow_pickle=True).item()

        # targets = data['targets']
        print(f"data: {data.keys()}")
        outputs = data['outputs'][:ws]
        obj_verts = data['obj_verts'][0]
        obj_faces = data['obj_faces'] # groudn-truth joints ##
        # tot_base_pts = data["tot_base_pts"][0] # total base points # # bsz x nnbasepts x 3 #
        # print(f"data: {data.keys()}", len(data['tot_rhand_joints']))
        # gt_joints = data['tot_gt_rhand_joints'][0] # total gt rhand joints #

        # outputs = data['tot_gt_rhand_joints'][:ws]
        tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
        tot_obj_transl = data['tot_obj_transl'][0] # nf x 3

        cur_split = "test" if not resplit else "train"
        cur_seq_gt_verts = load_grab_clip_data_clean_subj(test_seq_idx, split =cur_split, pert=False, more_pert=False, other_noise=False)

        # hand_verts = np.matmul(
        #     hand_verts - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1)) # nf x nn_joints x 3 --> as the transformed output joints ##
        # )
        # hand_verts_gt = np.matmul(
        #     cur_seq_gt_verts - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1)) # nf x nn_joints x 3 --> as the transformed output joints ##
        # )


        # gt_contact_maps = calcualte_contacts(obj_verts, hand_verts_gt)
        # pred_contact_maps = calcualte_contacts(obj_verts, hand_verts)
        # cur_cotact_iou = get_contact_ious(gt_contact_maps, pred_contact_maps)
        
        cur_cotact_iou = 0.
        print(f"cur_contact_iou: {cur_cotact_iou}")
        

        
        # if not os.path.exists() 
        # /data1/sim/mdm/eval_save/optimized_infos_sv_dict_seq_37_seed_77_tag_rep_only_mean_shape_hoi4d_t_200_res_jts_.npy
        # sv_optimized_info_fn = f"optimized_infos_sv_dict_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
        # sv_optimized_info_fn = os.path.join(sv_dir_ours, sv_optimized_info_fn) # 
        
        #### only for predicted info fn ####
        # if os.path.exists(sv_predicted_info_fn):
        #   data = np.load(sv_predicted_info_fn, allow_pickle=True).item()
        #   # targets = data['targets']
        #   outputs = data['outputs']
        #   obj_verts = data['obj_verts'][0]
        #   obj_faces = data['obj_faces']
        #   tot_base_pts = data["tot_base_pts"][0] # total base points # # bsz x nnbasepts x 3 #
        #   gt_joints = data['tot_gt_rhand_joints'][0] # gt rhand joints? # object model; object verts here #
        #   # tot_rhand_joints = data["tot_rhand_joints"][0]
        #   print(f"gt_joints: {gt_joints.shape}")
        #   smoothness = calculate_joint_smoothness(outputs)
        #   APD = calculate_penetration_depth(outputs, obj_verts, obj_faces)
        #   dist_dist_avg = calculate_proximity_dist(outputs, gt_joints, obj_verts, obj_faces)
        #   print(f"smoothness: {smoothness}, average penetration depth: {APD}, minn_dist_dist: {dist_dist_avg}")
        #   tot_APD.append(APD)
        #   tot_smoothness.append(smoothness)
        #   tot_proximity_error.append(dist_dist_avg)
        #   print(f"for sequence {test_seq_idx}, APD: {APD}, smoothness: {smoothness}")
        
        #### only for predicted info fn ####
        # if os.path.exists(sv_predicted_info_fn) and os.path.exists(sv_optimized_info_fn):
        ### sv optimized info fn ###
        print(f"predicted info fn: {sv_predicted_info_fn}")
        ws = 60
        
        if os.path.exists(sv_predicted_info_fn) and os.path.exists(predicted_ours_info_fn):
            print(f"predicted info fn: {sv_predicted_info_fn}")
            tot_seq_nn += 1
            # if os.path.exists(sv_predicted_info_fn) and os.path.exists(predicted_ours_info_fn) and os.path.exists(sv_optimized_info_fn):
            data = np.load(sv_predicted_info_fn, allow_pickle=True).item()
            # targets = data['targets']
            print(f"data: {data.keys()}")
            outputs = data['outputs'][:ws]
            obj_verts = data['obj_verts'][0]
            obj_faces = data['obj_faces'] # groudn-truth joints ##
            tot_base_pts = data["tot_base_pts"][0] # total base points # # bsz x nnbasepts x 3 #
            # print(f"data: {data.keys()}", len(data['tot_rhand_joints']))
            gt_joints = data['tot_gt_rhand_joints'][0] # total gt rhand joints #

            # outputs = data['tot_gt_rhand_joints'][:ws]

            print(f"obj_verts: {obj_verts.shape}, obj_faces: {obj_faces.shape}")

            if use_toch:
            #   gt_joints = cur_seq_gt_joints
              gt_joints = cur_seq_gt_verts

            # outputs = data['tot_rhand_joints'][0] # total gt rhand joints #
            
            # outputs = gt_joints.copy() # for the 
            
            ### for toch ####
            if use_toch:
                ours_data = np.load(predicted_ours_info_fn, allow_pickle=True).item()
                obj_verts = ours_data['obj_verts'][0]
                obj_faces = ours_data['obj_faces']
            # gt_joints = ours_data['tot_gt_rhand_joints'][0] # [0]
            # print(f"gt_joints: {gt_joints.shape}")
            ### for toch ###
            

            ### for toch ###
            ### optimized sv dict ###
            ### optimized sv dict ###
            #   optimized_sv_dict = np.load(sv_optimized_info_fn, allow_pickle=True).item()
            #   outputs = optimized_sv_dict['optimized_out_hand_joints'] # nf x nn_joints x 3 #
            #   # outputs_vert = 
            # tot_obj_rot = ours_data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
            # tot_obj_transl = ours_data['tot_obj_transl'][0] # nf x 3
            # outputs = np.matmul(
            # outputs - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1)) # nf x nn_joints x 3 --> as the transformed output joints ##
            # )
            # outputs = np.matmul(
            # outputs - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 1, 2)) # nf x nn_joints x 3 --> as the transformed output joints ##
            # )
            # gt_joints = np.matmul(
            # gt_joints , np.transpose(tot_obj_rot, (0, 2, 1)) # nf x nn_joints x 3 --> as the transformed output joints ##
            # ) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
            print(f"gt_joints: {gt_joints.shape}")
            ### optimized sv dict ###

            all_seqs_sv_data.append(
               {
                  "outputs": outputs, 
                  "gt_joints": gt_joints,
               }
            )
            
            
            # jts_acc = calculate
            jts_acc =  get_acc_metrics(outputs, gt_joints)
            tot_jts_acc.append(jts_acc)


            # # tot_rhand_joints = data["tot_rhand_joints"][0]
            # print(f"gt_joints: {gt_joints.shape}")


            # smoothness = calculate_joint_smoothness(outputs)
            # APD = calculate_penetration_depth(outputs, obj_verts, obj_faces)
            # dist_dist_avg = calculate_proximity_dist(outputs, gt_joints, obj_verts, obj_faces)
            # calculate_max_penetration_depth
            APD = calculate_max_penetration_depth(outputs, obj_verts, obj_faces)
            # APD = 0.
            
            # tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
            # tot_obj_transl = data['tot_obj_transl'][0] # nf x 3
            
            # # outputs = targets
            # print(f"outputs: {outputs.shape}")
            
            ### for toch ####
            # tot_obj_rot = ours_data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
            # tot_obj_transl = ours_data['tot_obj_transl'][0] # nf x 3
            # tot_base_pts = ours_data["tot_base_pts"][0] 
            ### for toch ###
            
            # calculate_grasping_stability(hand_verts, obj_verts, obj_normals, obj_grav_dirs=None): # obj_grav_dir: nf x 3 --> negative to the object gravity dir here #
            #### optimized_sv_dict ####
            #   outputs_verts = optimized_sv_dict["optimized_out_hand_verts"] # nf x nn_verts x 3 #
            #   outputs_verts = np.matmul(
            #     outputs_verts - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1))
            #   )
            ### for toch 3##
            # outputs_verts = data["obj_verts"]
            # ### for toch ##
            # # outputs_verts 
            #   tot_base_pts_exp = np.repeat(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), repeats=outputs_verts.shape[0], axis=0) # nf x nn_obj_verts x 3 
            
            # tot_base_normals = data['tot_base_normals'][0]
            
            # ### for toch ###
            # tot_base_normals = ours_data['tot_base_normals'][0]
            # # ### for toch ###
            
            #   tot_base_normals_exp = np.repeat(tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), repeats=outputs_verts.shape[0], axis=0) 
            #   obj_grav_dirs = np.zeros((outputs_verts.shape[0], 3), dtype=np.float32) # obj_grav_dirs #
            #   obj_grav_dirs[:, 1] = 1.
            #   cur_avg_stability = calculate_grasping_stability(outputs_verts, tot_base_pts_exp, tot_base_normals_exp, obj_grav_dirs=obj_grav_dirs)
            #   if cur_avg_stability < 100.:
            #     tot_stability.append(cur_avg_stability) # for the current stability values # 
            #   print(f"cur_avg_stability: {cur_avg_stability}")
            #### optimized_sv_dict ####
            
            
            # ### get joints_trans and base_pts_trans ###
            # joints_trans = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) #
            # base_pts_trans = np.matmul(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
            # ### get joints_trans and base_pts_trans ###

            # ### calculate moving consistency ### # moving  # 
            # consistency_value = calculate_moving_consistency(base_pts_trans, joints_trans)
            
            # print(f"smoothness: {smoothness}, average penetration depth: {APD}, minn_dist_dist: {dist_dist_avg}, consistency_value: {consistency_value}")
            tot_APD.append(APD)
            print(f"APD: {APD}")
            tot_metrics.append((jts_acc, APD))
            # tot_smoothness.append(smoothness)
            # tot_proximity_error.append(dist_dist_avg)
            # tot_consistency_value.append(consistency_value)
            if use_toch:
              break
        # print(f"for sequence {test_seq_idx}, APD: {APD}, smoothness: {smoothness}")
    
    # tot_metricstot_metrics #
    tot_metrics = sorted(tot_metrics, key=lambda ii: ii[0], reverse=False)
    if len(tot_jts_acc) == 0:
        continue
    minn_acc = tot_metrics[0][0]
    # minn_acc = min(tot_jts_acc)

    all_seq_tot_metrics.append(tot_metrics)

    # minn_acc = sum(tot_jts_acc) / float(len(tot_jts_acc))
    all_seqs_jts_acc.append(minn_acc)
    avg_max_penetration_depth.append(tot_metrics[0][1])

    avg_acc = sum(all_seqs_jts_acc) / float(len(all_seqs_jts_acc))
    avg_avg_max_penetration_depth = sum(avg_max_penetration_depth) / float(len(avg_max_penetration_depth))
    print(f"cur_pred_value, cur_seq_acc: {minn_acc}, tot_seq_acc: {avg_acc}, avg_avg_max_penetration_depth: {avg_avg_max_penetration_depth}")

  avg_acc = sum(all_seqs_jts_acc) / float(len(all_seqs_jts_acc))
  avg_avg_max_penetration_depth = sum(avg_max_penetration_depth) / float(len(avg_max_penetration_depth))
  print(f"avg_acc: {avg_acc}, avg_avg_max_penetration_depth: {avg_avg_max_penetration_depth}")

#   all_seq_tot_metrics # all seq tot metrics ##
# /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_ours_rndseeds.py
  np.save(f"all_seq_tot_metrics.npy", all_seq_tot_metrics) #
  print(f"tot metrics saved to all_seq_tot_metrics.npy") # # seq_tot_metrics ###

#   np.save("all_seqs_sv_data.npy", all_seqs_sv_data)

    # avg_APD = sum(tot_APD) / float(len(tot_APD)) ## total penetration depth 
    # avg_smoothness = sum(tot_smoothness) / float(len(tot_smoothness))
    # avg_proximity_error = sum(tot_proximity_error) / float(len(tot_proximity_error))
    # avg_consistency_value = sum(tot_consistency_value) / float(len(tot_consistency_value))
    
    # if len(tot_stability) > 0:
    #     avg_stability = sum(tot_stability) / float(len(tot_stability))
    #     print(f"avg_Stability: {avg_stability}")
    
    # print(f"avg_APD: {avg_APD}, avg_smoothness: {avg_smoothness}, avg_proximity_error: {avg_proximity_error}, avg_consistency_value: {avg_consistency_value}")
    # print(tot_APD)
    # print(tot_smoothness)
    # print(tot_proximity_error)
    # print(tot_consistency_value)
    # print(f"avg_APD: {avg_APD}, avg_smoothness: {avg_smoothness}, avg_proximity_error: {avg_proximity_error}, avg_consistency_value: {avg_consistency_value}")
    
    
    # /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_ours_rndseeds.py
    # /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_only_jts.py
    # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_jts.npy"
    # data = np.load(predicted_info_fn, allow_pickle=True).item()
    # # targets = data['targets']
    # outputs = data['outputs']
    # obj_verts = data['obj_verts'][0]
    # obj_faces = data['obj_faces']
    # tot_base_pts = data["tot_base_pts"][0] # total base points # bsz x nnbasepts x 3 #
    # gt_joints = data['tot_gt_rhand_joints'][0]
    # # tot_rhand_joints = data["tot_rhand_joints"][0]
    # print(f"gt_joints: {gt_joints.shape}")
    # smoothness = calculate_joint_smoothness(outputs)
    # APD = calculate_penetration_depth(outputs, obj_verts, obj_faces)
    # dist_dist_avg = calculate_proximity_dist(outputs, gt_joints, obj_verts, obj_faces)
    # print(f"smoothness: {smoothness}, average penetration depth: {APD}, minn_dist_dist: {dist_dist_avg}")
    




