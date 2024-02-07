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
import utils.model_util as model_util
# from options.options import opt

### smoothness 
### whether in the object -> using vertices and using joints ###
### 
import trimesh

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
  for i_f in range(subj_seq.shape[0]):
  # for i_f in range(subj_seq.shape[0] - 5, subj_seq.shape[0]):
  # for i_f in range(10):
    cur_subj_seq = subj_seq[i_f]
    cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #
    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts # # to obj verts #
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

  disp_joints_base_pts = model_util.batched_index_select_ours(disp_base_pts, indices=disp_joints_to_base_pts_minn_idxes, dim=1) # (nf - 1) x nn_jts x 3 
  
  nearest_joints_disp = model_util.batched_index_select_ours(disp_joints_base_pts, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1) # (nf - 1) x 1
  nearest_joints_disp = nearest_joints_disp.squeeze(1) # (nf - 1) x 3 #
  
  disp_joints = model_util.batched_index_select_ours(disp_joints, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1).squeeze(1) # (nf - 1) x 3 
  
  nearest_k = model_util.batched_index_select_ours(k, indices=minn_dist_joints_to_base_pts_idxes.unsqueeze(-1), dim=1).squeeze(1) # (nf - 1)
  
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
  
  # data_sv_fn = "/data3/hlyang/results/test_data/20231105/20231105_001.pkl"
  
  
  # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy"
  # predicted_infos_jtsonly.npy # predicted infos jts ##
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_jts.npy"
  # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy"
  # predicted_infos_seq_67_seed_31_tag_jts_only.npy
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_77_seed_31_tag_jts_only.npy"
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_36_seed_77_tag_jts_only.npy"
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_36_seed_77_tag_jts_only.npy"
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_80_wtrans.npy" 
  # predicted_infos_80_wtrans_rep
  predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_80_wtrans_rep.npy" 
  # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_80_seed_77_tag_jts_only.npy"
  # predicted_info_fn = "/data1/sim/mdm/eval_save/predicted_infos_seq_36_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # predicted_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_77.npy"
  data = np.load(predicted_info_fn, allow_pickle=True).item()
  targets = data['targets']
  outputs = data['outputs']
  obj_verts = data['obj_verts'][0]
  obj_faces = data['obj_faces']
  tot_base_pts = data["tot_base_pts"][0] # total base points # nnbasepts x 3 #
  gt_joints = data['tot_gt_rhand_joints'][0]
  
  # outputs = optimized_data['optimized_out_hand_joints']
  tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
  tot_obj_transl = data['tot_obj_transl'][0] # nf x 3
  
  # outputs = targets
  outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) #
  
  
  # predicted info fn #
  optimized_info_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/optimized_infos_sv_dict.npy"
  optimized_data = np.load(optimized_info_fn, allow_pickle=True).item()
  
  
  
  # outputs = optimized_data['optimized_out_hand_joints_ne']
  # outputs = optimized_data['optimized_out_hand_joints']
  
  
  
  tot_base_pts_trans = np.matmul( # nf x nn_base_pts x 3 # 
    tot_base_pts.reshape(1, tot_base_pts.shape[0], tot_base_pts.shape[1]), tot_obj_rot
  ) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # 
  
  diff_disp_joints_base_pts = calculate_moving_consistency(tot_base_pts_trans, outputs)
  
  outputs = np.matmul(outputs - tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, 3), np.transpose(tot_obj_rot, (0, 2, 1))) # ws x nn_joints x 3 #  
  
  # tot_rhand_joints = data["tot_rhand_joints"][0] # 
  print(f"gt_joints: {gt_joints.shape}")
  smoothness = calculate_joint_smoothness(outputs)
  APD = calculate_penetration_depth(outputs, obj_verts, obj_faces)
  dist_dist_avg = calculate_proximity_dist(outputs, gt_joints, obj_verts, obj_faces)
  print(f"smoothness: {smoothness}, average penetration depth: {APD}, minn_dist_dist: {dist_dist_avg}, consistency: {diff_disp_joints_base_pts}")
  




