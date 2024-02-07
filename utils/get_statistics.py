import torch
import torch.nn.functional as F
import torch.utils.data
from manopth.manolayer import ManoLayer
import numpy as np
import os
import data_loaders.humanml.data.utils as utils



def get_mano_model():
  mano_path = "/data1/sim/mano_models/mano/models" ### mano_path
  mano_model = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  )
  
  return mano_model

def get_rhand_joints_stats(split, mano_model):
  subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
  data_folder = "/data1/sim/GRAB_processed"
  subj_data_folder = os.path.join(subj_data_folder, split)
  tot_subj_params_fns = os.listdir(subj_data_folder)
  tot_subj_params_fns = [fn for fn in tot_subj_params_fns if fn.endswith("_subj.npy")]
  tot_n_frames = 0
  tot_joints = []
  for i_subj_fn, subj_fn in enumerate(tot_subj_params_fns):
    cur_idx = subj_fn.split("_")[0]
    # cur_clean_clip_fn = os.path.join(data_folder, split, f"{cur_idx}.npy")
    # clip_clean = np.load(cur_clean_clip_fn)
    
    # nn_frames = clip_clean['f3'].shape[0]
    
    # ### Relative positions from base points to rhand joints ###
    # object_pc = clip_clean['f3'].reshape(nn_frames, -1, 3).astype(np.float32)
    # object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
    # object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
    # # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
    # object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
    # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
    
    
    
    if i_subj_fn % 10 == 0:
      print(f"Processing {i_subj_fn} / {len(tot_subj_params_fns)}")
    cur_full_subj_fn = os.path.join(subj_data_folder, subj_fn)
    subj_params = np.load(cur_full_subj_fn, allow_pickle=True).item()
    rhand_transl = subj_params["rhand_transl"]
    rhand_betas = subj_params["rhand_betas"] ## rhand betas ##
    rhand_hand_pose = subj_params['rhand_hand_pose']
    rhand_global_orient = subj_params['rhand_global_orient']
    
    rhand_transl = torch.from_numpy(rhand_transl).float()
    rhand_betas = torch.from_numpy(rhand_betas).float()
    rhand_hand_pose = torch.from_numpy(rhand_hand_pose).float()
    rhand_global_orient = torch.from_numpy(rhand_global_orient).float()
    
    rhand_verts, rhand_joints = mano_model(
        torch.cat([rhand_global_orient, rhand_hand_pose], dim=-1),
        rhand_betas.unsqueeze(0).repeat(rhand_transl.size(0), 1).view(-1, 10), rhand_transl
    )
    rhand_joints = rhand_joints * 0.001
    
    tot_joints.append(rhand_joints)
  tot_joints = torch.cat(tot_joints, dim=0)
  print(f"tot_joints: {tot_joints.size()}")
  avg_joints = torch.mean(tot_joints, dim=0, keepdim=True)
  std_joints = torch.std(tot_joints, dim=0, keepdim=True)
  np.save(f"avg_joints_motion_ours.npy", avg_joints.numpy())
  np.save(f"std_joints_motion_ours.npy", std_joints.numpy())
    

def get_rhand_joints_base_pts_rel_stats(split, mano_model):
  subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
  data_folder = "/data1/sim/GRAB_processed"
  subj_data_folder = os.path.join(subj_data_folder, split)
  tot_subj_params_fns = os.listdir(subj_data_folder)
  tot_subj_params_fns = [fn for fn in tot_subj_params_fns if fn.endswith("_subj.npy")]
  tot_n_frames = 0
  tot_joints = []
  tot_joints_dists = []
  
  
  for i_subj_fn, subj_fn in enumerate(tot_subj_params_fns):
    if i_subj_fn % 10 == 0:
      print(f"Processing {i_subj_fn} / {len(tot_subj_params_fns)}")
    
    cur_idx = subj_fn.split("_")[0]
    cur_clean_clip_fn = os.path.join(data_folder, split, f"{cur_idx}.npy")
    clip_clean = np.load(cur_clean_clip_fn)
    
    nn_frames = clip_clean['f3'].shape[0]
    
    ''' Object information '''
    # ### Relative positions from base points to rhand joints ###
    object_pc = clip_clean['f3'].reshape(nn_frames, -1, 3).astype(np.float32)
    object_normal = clip_clean['f4'].reshape(nn_frames, -1, 3).astype(np.float32)
    object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
    # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
    object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
    # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
    object_global_orient = clip_clean['f5'] ### get object global orientations ###
    object_trcansl = clip_clean['f6']
    
    object_global_orient = object_global_orient.reshape(nn_frames, -1).astype(np.float32)
    object_trcansl = object_trcansl.reshape(nn_frames, -1).astype(np.float32)
    object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
    object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
    object_trcansl_th = torch.from_numpy(object_trcansl).float()
    ''' Object information '''
    
  
    ''' Subject information '''
    cur_full_subj_fn = os.path.join(subj_data_folder, subj_fn)
    subj_params = np.load(cur_full_subj_fn, allow_pickle=True).item()
    rhand_transl = subj_params["rhand_transl"]
    rhand_betas = subj_params["rhand_betas"] ## rhand betas ##
    rhand_hand_pose = subj_params['rhand_hand_pose']
    rhand_global_orient = subj_params['rhand_global_orient']
    
    rhand_transl = torch.from_numpy(rhand_transl).float()
    rhand_betas = torch.from_numpy(rhand_betas).float()
    rhand_hand_pose = torch.from_numpy(rhand_hand_pose).float()
    rhand_global_orient = torch.from_numpy(rhand_global_orient).float()
    
    rhand_verts, rhand_joints = mano_model(
        torch.cat([rhand_global_orient, rhand_hand_pose], dim=-1),
        rhand_betas.unsqueeze(0).repeat(rhand_transl.size(0), 1).view(-1, 10), rhand_transl
    )
    rhand_joints = rhand_joints * 0.001
    ''' Subject information '''
    
    
    dist_rhand_joints_to_obj_pc = torch.sum(
        (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
    )
    # dist_pert_rhand_joints_obj_pc = torch.sum(
    #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
    # )
    _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
    # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
    
    object_pc_th = object_pc_th[0].unsqueeze(0).repeat(nn_frames, 1, 1).contiguous()
    nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
    # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
    dist_object_pc_nearest_pcs = torch.sum(
        (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
    )
    dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
    dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
    # # dist_threshold = 0.01
    dist_threshold = 0.005
    # # dist_threshold for pc_nearest_pcs #
    dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
    
    # # base_pts_mask: nn_obj_pcs #
    base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
    # # nn_base_pts x 3 -> torch tensor #
    base_pts = object_pc_th[0][base_pts_mask]
    # # base_pts_bf_sampling = base_pts.clone()
    base_normals = object_normal_th[0][base_pts_mask]
    
    nn_base_pts = 700
    base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
    base_pts_idxes = base_pts_idxes[:nn_base_pts]
    # if self.debug:
    #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
    
    # ### get base points ### # base_pts and base_normals #
    base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
    base_normals = base_normals[base_pts_idxes]
    
    
    # # object_global_orient_mtx # nn_ws x 3 x 3 #
    base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
    base_pts_transl = object_trcansl_th[0] # 3
    
    
    base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
        ) # .transpose(0, 1)
    base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
        ) # .transpose(0, 1)
    
    
    rhand_joints = torch.matmul(
        rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
    )
    
    
    # rhand_joints = rhand_joints * 5.
    # base_pts = base_pts * 5.

    # nf x nnj x nnb x 3 # 
    rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
    # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
    dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
    
    
    # tot_joints.append(rel_base_pts_to_rhand_joints.mean(dim=-2))
    # tot_joints_dists.append(dist_base_pts_to_rhand_joints.mean(dim=-1))
    
    tot_joints.append(rel_base_pts_to_rhand_joints) ## use rel and distances 
    tot_joints_dists.append(dist_base_pts_to_rhand_joints)
    
  tot_joints = torch.cat(tot_joints, dim=0)
  tot_joints_dists = torch.cat(tot_joints_dists,dim=0)
  print(f"tot_joints: {tot_joints.size()}, tot_joints_dists: {tot_joints_dists.size()}")
  
  ## nf x nnj x nnb x 3 ## -> for all the max and min valeus 
  tot_joints_exp = tot_joints.view(tot_joints.size(0) * tot_joints.size(1) * tot_joints.size(2), 3).contiguous()
  tot_joints_dists_exp = tot_joints_dists.view(tot_joints_dists.size(0) * tot_joints_dists.size(1) * tot_joints_dists.size(2), 1).contiguous()
  maxx_tot_joints_exp, _ = torch.max(tot_joints_exp, dim=0)
  minn_tot_joints_exp, _ = torch.min(tot_joints_exp, dim=0)
  
  maxx_joints_dists, _ = torch.max(tot_joints_dists_exp, dim=0)
  minn_joints_dists, _ = torch.min(tot_joints_dists_exp, dim=0)
  print(f"maxx_rel: {maxx_tot_joints_exp}, minn_rel: {minn_tot_joints_exp}")
  print(f"maxx_joints_dists: {maxx_joints_dists}, minn_joints_dists: {minn_joints_dists}")
  
  sv_stats_dict = {
    'maxx_rel': maxx_tot_joints_exp.numpy(), 
    'minn_rel': minn_tot_joints_exp.numpy(),
    'maxx_dists': maxx_joints_dists.numpy(),
    'minn_dists': minn_joints_dists.numpy(),
  }
  sv_stats_dict_fn = "base_pts_rel_dists_stats.npy"
  np.save(sv_stats_dict_fn, sv_stats_dict)
  print()
  
  
  
  ''' V1 '''
  # avg_joints = torch.mean(tot_joints, dim=0, keepdim=True)
  # std_joints = torch.std(tot_joints, dim=0, keepdim=True)
  # np.save(f"avg_joints_motion_ours_nb_{700}_nth_{0.005}.npy", avg_joints.numpy())
  # np.save(f"std_joints_motion_ours_nb_{700}_nth_{0.005}.npy", std_joints.numpy())
  
  # avg_joints_dists = torch.mean(tot_joints_dists, dim=0, keepdim=True)
  # std_joints_dists = torch.std(tot_joints_dists, dim=0, keepdim=True)
  # np.save(f"avg_joints_dist_motion_ours_nb_{700}_nth_{0.005}.npy", avg_joints_dists.numpy())
  # np.save(f"std_joints_dist_motion_ours_nb_{700}_nth_{0.005}.npy", std_joints_dists.numpy())
    
    

def get_rhand_joints_base_pts_rel_stats_jts_stats(split, mano_model):
  subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
  data_folder = "/data1/sim/GRAB_processed"
  subj_data_folder = os.path.join(subj_data_folder, split)
  tot_subj_params_fns = os.listdir(subj_data_folder)
  tot_subj_params_fns = [fn for fn in tot_subj_params_fns if fn.endswith("_subj.npy")]
  tot_n_frames = 0
  tot_joints = []
  tot_joints_dists = []
  tot_rhand_joints = []
  
  
  ws = 30
  ### Processing xxx ###
  for i_subj_fn, subj_fn in enumerate(tot_subj_params_fns):
    if i_subj_fn % 10 == 0:
      print(f"Processing {i_subj_fn} / {len(tot_subj_params_fns)}")
    
    cur_idx = subj_fn.split("_")[0]
    cur_clean_clip_fn = os.path.join(data_folder, split, f"{cur_idx}.npy")
    clip_clean = np.load(cur_clean_clip_fn)
    
    nn_frames = clip_clean['f3'].shape[0]
    
    ''' Object information '''
    # # ### Relative positions from base points to rhand joints ###
    # object_pc = clip_clean['f3'].reshape(nn_frames, -1, 3).astype(np.float32)
    # object_normal = clip_clean['f4'].reshape(nn_frames, -1, 3).astype(np.float32)
    # object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
    # # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
    # object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
    # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
    object_global_orient = clip_clean['f5'] ### get object global orientations ###
    object_trcansl = clip_clean['f6']
    
    object_global_orient = object_global_orient.reshape(nn_frames, -1).astype(np.float32)
    object_trcansl = object_trcansl.reshape(nn_frames, -1).astype(np.float32)
    object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
    object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
    object_trcansl_th = torch.from_numpy(object_trcansl).float()
    ''' Object information '''
    
  
    ''' Subject information '''
    cur_full_subj_fn = os.path.join(subj_data_folder, subj_fn)
    subj_params = np.load(cur_full_subj_fn, allow_pickle=True).item()
    rhand_transl = subj_params["rhand_transl"]
    rhand_betas = subj_params["rhand_betas"] ## rhand betas ##
    rhand_hand_pose = subj_params['rhand_hand_pose']
    rhand_global_orient = subj_params['rhand_global_orient']
    
    rhand_transl = torch.from_numpy(rhand_transl).float()
    rhand_betas = torch.from_numpy(rhand_betas).float()
    rhand_hand_pose = torch.from_numpy(rhand_hand_pose).float()
    rhand_global_orient = torch.from_numpy(rhand_global_orient).float()
    
    rhand_verts, rhand_joints = mano_model(
        torch.cat([rhand_global_orient, rhand_hand_pose], dim=-1),
        rhand_betas.unsqueeze(0).repeat(rhand_transl.size(0), 1).view(-1, 10), rhand_transl
    )
    rhand_joints = rhand_joints * 0.001
    ''' Subject information '''
    
    
    # dist_rhand_joints_to_obj_pc = torch.sum(
    #     (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
    # )
    # # dist_pert_rhand_joints_obj_pc = torch.sum(
    # #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
    # # )
    # _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
    # # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
    
    # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(nn_frames, 1, 1).contiguous()
    # nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
    # # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
    # dist_object_pc_nearest_pcs = torch.sum(
    #     (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
    # )
    # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
    # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
    # # # dist_threshold = 0.01
    # dist_threshold = 0.005
    # # # dist_threshold for pc_nearest_pcs #
    # dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
    
    # # # base_pts_mask: nn_obj_pcs #
    # base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
    # # # nn_base_pts x 3 -> torch tensor #
    # base_pts = object_pc_th[0][base_pts_mask]
    # # # base_pts_bf_sampling = base_pts.clone()
    # base_normals = object_normal_th[0][base_pts_mask]
    
    # nn_base_pts = 700
    # base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
    # base_pts_idxes = base_pts_idxes[:nn_base_pts]
    # # if self.debug:
    # #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
    
    # # ### get base points ### # base_pts and base_normals #
    # base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
    # base_normals = base_normals[base_pts_idxes]
    
    # # # object_global_orient_mtx # nn_ws x 3 x 3 #
    # base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
    # base_pts_transl = object_trcansl_th[0] # 3
    
    # base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
    #     ) # .transpose(0, 1)
    # base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
    #     ) # .transpose(0, 1)
    
    rhand_joints = torch.matmul(
        rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
    )
    
    if ws is not None:
      step = ws // 2
      for st_idx in range(0, nn_frames - ws, step):
        cur_joints_exp = rhand_joints[st_idx: st_idx + ws].view(ws * rhand_joints.size(1), 3).contiguous()
        maxx_rhand_joints, _ = torch.max(cur_joints_exp, dim=0)
        minn_rhand_joints, _ = torch.min(cur_joints_exp, dim=0)
        avg_rhand_joints = (maxx_rhand_joints + minn_rhand_joints) / 2.
        cur_clip_rhand_joints = rhand_joints[st_idx: st_idx + ws, :, :]
        cur_clip_rhand_joints = cur_clip_rhand_joints - avg_rhand_joints.unsqueeze(0).unsqueeze(0)
        cur_clip_rhand_joints_exp = cur_clip_rhand_joints.view(cur_clip_rhand_joints.size(0) * cur_clip_rhand_joints.size(1), 3).contiguous()
        tot_rhand_joints.append(cur_clip_rhand_joints_exp)
    else:    
      # rhand_joints: nf x nnj x 3 
      rhand_joints_exp = rhand_joints.view(rhand_joints.size(0) * rhand_joints.size(1), 3).contiguous()
      maxx_rhand_joints, _ = torch.max(rhand_joints_exp, dim=0)
      minn_rhand_joints, _ = torch.min(rhand_joints_exp, dim=0)
      avg_rhand_joints = (maxx_rhand_joints + minn_rhand_joints) / 2.
      
      rhand_joints_exp = rhand_joints_exp - avg_rhand_joints.unsqueeze(0)
      maxx_rhand_joints, _ = torch.max(rhand_joints_exp, dim=0)
      minn_rhand_joints, _ = torch.min(rhand_joints_exp, dim=0)
      print(f"maxx_rhand_joints: {maxx_rhand_joints}, minn_rhand_joints: {minn_rhand_joints}")
      
      tot_rhand_joints.append(rhand_joints_exp)
  
  tot_rhand_joints = torch.cat(tot_rhand_joints, dim=0)
  maxx_rhand_joints, _ = torch.max(tot_rhand_joints, dim=0)
  minn_rhand_joints, _ = torch.min(tot_rhand_joints, dim=0)
  print(f"tot_maxx_rhand_joints: {maxx_rhand_joints}, tot_minn_rhand_joints: {minn_rhand_joints}")
  
  
def test_subj_file(subj_fn):
  subj_data = np.load(subj_fn, allow_pickle=True).item()
  for k in subj_data:
    print(f"k: {k}, v: {subj_data[k].shape}")
    # k, v #
    # with normalization for bse pts features here #

    
def test_joints_statistics():
  avg_jts_fn = "/home/xueyi/sim/motion-diffusion-model/avg_joints_motion_ours.npy"
  std_jts_fn = "/home/xueyi/sim/motion-diffusion-model/std_joints_motion_ours.npy"
  avg_jts = np.load(avg_jts_fn, allow_pickle=True)
  std_jts = np.load(std_jts_fn, allow_pickle=True)
  print(avg_jts.shape)
  print(std_jts.shape)
  
if __name__=='__main__':
  # subj_fn = '/data1/sim/GRAB_processed_wsubj/train/1_subj.npy'
  # test_subj_file(subj_fn)
  
  mano_model = get_mano_model()
  split = "train"
  # get_rhand_joints_stats(split, mano_model)
  ##### === rel base_pts to rhand_joints === #####
  # get_rhand_joints_base_pts_rel_stats(split, mano_model)
  ##### === rel base_pts to rhand_joints joints... === #####
  get_rhand_joints_base_pts_rel_stats_jts_stats(split, mano_model)
  # test_joints_statistics()
  
  
  