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



def calculate_max_penetration_depth(subj_seq, obj_verts, obj_faces):
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

    cur_pene_depth = float(np.max(cur_pene_depth))
    tot_penetration_depth.append(cur_pene_depth)
  tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
#   tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
#   tot_penetration_depth = np.mean(tot_penetration_depth).item()
  return tot_penetration_depth



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
  tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
  tot_penetration_depth = np.mean(tot_penetration_depth).item()
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
  
  ##### use k for weighting disp #####
  disp_joints_to_nearest_base_pts = disp_joints *  nearest_k.unsqueeze(-1) # ### (nf - 1 ) x 3 
  diff_disp_joints_to_nearest_base_pts_disp = torch.sum(
    (disp_joints_to_nearest_base_pts - nearest_joints_disp) ** 2, dim=-1
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


# T = 300
# smoothness: 4.595715654431842e-05, average penetration depth: 0.00021960893150693885, minn_dist_dist: 5.6245558029369e-07 -- jts only
# smoothness: 0.00011870301386807114, average penetration depth: 0.0002747326595483027, minn_dist_dist: 8.44164813606991e-07
# 

# 8.121088892826279e-07
# 2.951481956519773e-07
# smoothness: 6.213585584191605e-05, average penetration depth: 4.278580710096256e-05, minn_dist_dist: 7.753524560089469e-07
# smoothness: 5.4274405556498095e-05, average penetration depth: 3.388783391933507e-05, minn_dist_dist: 5.682716970124003e-07
if __name__=='__main__':
  # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy"
  # predicted_infos_jtsonly.npy # predicted infos jts ##
  
  tot_APD = []
  tot_smoothness = []
  tot_proximity_error = []
  tot_consistency_value = []
  
  tot_stability = []
   
  st_idx = 0
  ed_idx = 38
  
  st_idx = 58
  ed_idx = 111 #
  
  st_idx = 0
  ed_idx = 20 #
  
  st_idx = 139
  ed_idx = 158 #
  
  st_idx = 139
  ed_idx = 190 #

  st_idx = 0
  ed_idx = 255 # st_idx and ed_idx 
  
  # st_idx = 58
  # ed_idx = 111
  # st_idx = 0
  # ed_idx = 246
  # for test_seq_idx in range(1, 102, 10):
  # for test_seq_idx in range(1, 102, 1):
  # for test_seq_idx in range(1, 11, 1):
  for test_seq_idx in range(st_idx, ed_idx, 1):
  # for test_seq_idx in range(36, 37, 1):
    seed = 77
    
    test_tag = "jts_rep_19_cbd_t_300_real"
    
    test_tag = "jts_only_gaussian_hoi4d_t_300_"
    sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
    
    # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_101_seed_77_tag_jts_rep_55_cbd.npy
    
    sv_dir = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
    
    
    
    # /data1/sim/mdm/eval_save/predicted_infos_seq_2_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy
    # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_.npy
    sv_dir = "/data1/sim/mdm/eval_save"
    sv_dir_ours = "/data1/sim/mdm/eval_save"
    
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
    test_tag = "rep_only_real_mean_"
    # test_tag = "rep_only_real_sel_base_mean_all_noise_"
    
    # test_tag = "jts_only_gaussian_hoi4d_t_300_"
    # # /data1/sim/mdm/eval_save/predicted_infos_seq_37_seed_77_tag_rep_only_mean_shape_hoi4d_t_200_res_jts_.npy
    # test_tag = "rep_only_mean_shape_hoi4d_t_200_res_jts_"
    # # rep_only_mean_shape_hoi4d_t_200_res_jts_
    # # rep_only_mean_shape_hoi4d_t_400_
    # # test_tag = "rep_only_mean_shape_hoi4d_t_400_"
    
    
    # test_tag = "jts_only_beta_t_300_"
    # /data1/sim/mdm/eval_save_toch/14.npy
    
    seed = 77
    # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_jts_only.npy
    # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_.npy
    # /data1/sim/mdm/eval_save/predicted_infos_seq_7_seed_77_tag_rep_only_real_mean_same_noise_.npy
    sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
    sv_predicted_info_fn = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
    # predicted_infos_seq_99_seed_77_tag_jts_only_beta_t_300_.npy
    sv_predicted_info_fn_ours = f"predicted_infos_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
    
    ### for toch ###
    # sv_dir = "/data1/sim/mdm/eval_save_toch"
    # # other_noise_grab
    # sv_predicted_info_fn = f"{test_seq_idx}.npy"
    # # sv_predicted_info_fn = f"other_noise_grab_{test_seq_idx}.npy"
    ### for toch ###

    sv_toch_dir = "/data2/sim/eval_save/GRAB_TOCH"
    # sv_toch_fn = f'hand_verts_seq_{test_seq_idx}.npy' ## sv_toch_fn ##
    # sv_toch_fn = f'hand_verts_seq_aug_{test_seq_idx}.npy' ### with augmentations ##
    sv_toch_fn = f'hand_verts_seq_mixstyle_{test_seq_idx}.npy' 
    sv_toch_fn = os.path.join(sv_toch_dir, sv_toch_fn)
    sv_predicted_info_fn = sv_toch_fn

    print(f"sv_toch_fn: {sv_toch_fn}")
    sv_predicted_info_fn = os.path.join(sv_dir, sv_predicted_info_fn)
    predicted_ours_info_fn = os.path.join(sv_dir_ours, sv_predicted_info_fn_ours)
    
    
    # # /data1/sim/mdm/eval_save/optimized_infos_sv_dict_seq_37_seed_77_tag_rep_only_mean_shape_hoi4d_t_200_res_jts_.npy
    # sv_optimized_info_fn = f"optimized_infos_sv_dict_seq_{test_seq_idx}_seed_{seed}_tag_{test_tag}.npy"
    # sv_optimized_info_fn = os.path.join(sv_dir_ours, sv_optimized_info_fn) # 
    
    #### only for predicted info fn #### # and for the bowls and scissors #
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
    ws = 60
    if os.path.exists(sv_predicted_info_fn) and os.path.exists(predicted_ours_info_fn): #  and os.path.exists(sv_optimized_info_fn):
      data = np.load(sv_predicted_info_fn, allow_pickle=True).item()
      # targets = data['targets']
    #   outputs = data['outputs'][:ws]
    #   obj_verts = data['obj_verts'][0]
    #   obj_faces = data['obj_faces']
    #   tot_base_pts = data["tot_base_pts"][0] # total base points # # bsz x nnbasepts x 3 #
    #   gt_joints = data['tot_gt_rhand_joints'][0] # total gt rhand joints #

      print(f"data: {data.keys()}")
      outputs = data["tot_hand_joints"][:ws]


      print(f"outputs: {outputs.shape}")




      #   tot_hand_joints = 
      
      
      ### for toch ####
      ours_data = np.load(predicted_ours_info_fn, allow_pickle=True).item()
      obj_verts = ours_data['obj_verts'][0]
      obj_faces = ours_data['obj_faces']
      gt_joints = ours_data['tot_gt_rhand_joints'][0][:ws] # [0]
      ### for toch ###

      ### for toch ####
      tot_obj_rot = ours_data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
      tot_obj_transl = ours_data['tot_obj_transl'][0] # nf x 3
      tot_base_pts = ours_data["tot_base_pts"][0] 
      ### for toch ###

    #   # outputs: ws x nn_joints x 3 #
      outputs = np.matmul(  # outputs for the tot_obj_transl; tot_obj_rot; #
        outputs - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1))
      )
      
      
      ### optimized sv dict ###
    #   optimized_sv_dict = np.load(sv_optimized_info_fn, allow_pickle=True).item()
    #   outputs = optimized_sv_dict['optimized_out_hand_joints'] # nf x nn_joints x 3 #
    #   # outputs_vert = 
    #   tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
    #   tot_obj_transl = data['tot_obj_transl'][0] # nf x 3
    #   outputs = np.matmul(
    #     outputs - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1)) # nf x nn_joints x 3 --> as the transformed output joints ##
    #   )
      ### optimized sv dict ###

      # avg_APD: 8.946275378162125e-05, avg_smoothness: 5.883042995608691e-05, avg_proximity_error: 7.648900803354692e-08, avg_consistency_value: 1.7371066019751555e-05
      
      # tot_rhand_joints = data["tot_rhand_joints"][0]
      print(f"gt_joints: {gt_joints.shape}")
      print(f"predicted joints: {outputs.shape}")

      # outputs = gt_joints.copy()

      smoothness = calculate_joint_smoothness(outputs)
      # APD = calculate_penetration_depth(outputs, obj_verts, obj_faces)
      APD = calculate_max_penetration_depth(outputs, obj_verts, obj_faces)
      dist_dist_avg = calculate_proximity_dist(outputs, gt_joints, obj_verts, obj_faces)
      
      
      
    #   tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
    #   tot_obj_transl = data['tot_obj_transl'][0] # nf x 3
      
      # outputs = targets
      print(f"outputs: {outputs.shape}")
      
      
      
      # calculate_grasping_stability(hand_verts, obj_verts, obj_normals, obj_grav_dirs=None): # obj_grav_dir: nf x 3 --> negative to the object gravity dir here #
      #### optimized_sv_dict ####
    #   outputs_verts = optimized_sv_dict["optimized_out_hand_verts"] # nf x nn_verts x 3 #
    #   outputs_verts = np.matmul(
    #     outputs_verts - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]), np.transpose(tot_obj_rot, (0, 2, 1))
    #   )
    #   ### for toch 3##
    #   # outputs_verts = data["obj_verts"]
    #   # ### for toch ##
    #   # # outputs_verts 
    #   tot_base_pts_exp = np.repeat(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), repeats=outputs_verts.shape[0], axis=0) # nf x nn_obj_verts x 3 
      
    #   tot_base_normals = data['tot_base_normals'][0]
      
    #   # ### for toch ###
    #   # tot_base_normals = ours_data['tot_base_normals'][0]
    #   # # ### for toch ###
      
    #   tot_base_normals_exp = np.repeat(tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), repeats=outputs_verts.shape[0], axis=0) 
    #   obj_grav_dirs = np.zeros((outputs_verts.shape[0], 3), dtype=np.float32) # obj_grav_dirs #
    #   obj_grav_dirs[:, 1] = 1.
    #   cur_avg_stability = calculate_grasping_stability(outputs_verts, tot_base_pts_exp, tot_base_normals_exp, obj_grav_dirs=obj_grav_dirs)
    #   if cur_avg_stability < 100.:
    #     tot_stability.append(cur_avg_stability) # for the current stability values # 
    #   print(f"cur_avg_stability: {cur_avg_stability}")
      #### optimized_sv_dict ####
      
      
      ### get joints_trans and base_pts_trans ###
      joints_trans = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) #
      base_pts_trans = np.matmul(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
      ### get joints_trans and base_pts_trans ###

      ### calculate moving consistency ### # moving 
      consistency_value = calculate_moving_consistency(base_pts_trans, joints_trans)
      
      print(f"smoothness: {smoothness}, average penetration depth: {APD}, minn_dist_dist: {dist_dist_avg}, consistency_value: {consistency_value}")
      tot_APD.append(APD)
      tot_smoothness.append(smoothness)
      tot_proximity_error.append(dist_dist_avg)
      tot_consistency_value.append(consistency_value)
      # print(f"for sequence {test_seq_idx}, APD: {APD}, smoothness: {smoothness}")

      ### avg apd ###
      avg_APD = sum(tot_APD) / float(len(tot_APD)) ## total penetration depth 
      avg_smoothness = sum(tot_smoothness) / float(len(tot_smoothness))
      avg_proximity_error = sum(tot_proximity_error) / float(len(tot_proximity_error))
      avg_consistency_value = sum(tot_consistency_value) / float(len(tot_consistency_value))
      
      if len(tot_stability) > 0:
        avg_stability = sum(tot_stability) / float(len(tot_stability))
        print(f"avg_Stability: {avg_stability}")
      
      print(f"avg_APD: {avg_APD}, avg_smoothness: {avg_smoothness}, avg_proximity_error: {avg_proximity_error}, avgtb_consistency_value: {avg_consistency_value}")

  print(tot_APD)
  print(tot_smoothness)
  print(tot_proximity_error)
  print(tot_consistency_value)
  

  # /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_toch.py
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
  




