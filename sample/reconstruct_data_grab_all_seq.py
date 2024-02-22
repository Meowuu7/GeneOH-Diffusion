import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from torch import optim, nn
import numpy as np
import os, argparse, copy, json
import pickle as pkl
from scipy.spatial.transform import Rotation as R
# from psbody.mesh import Mesh
from manopth.manolayer import ManoLayer
# from dataloading import GRAB_Single_Frame, GRAB_Single_Frame_V6, GRAB_Single_Frame_V7, GRAB_Single_Frame_V8, GRAB_Single_Frame_V9, GRAB_Single_Frame_V9_Ours, GRAB_Single_Frame_V10
# use_trans_encoders
# from model import TemporalPointAE, TemporalPointAEV2, TemporalPointAEV5, TemporalPointAEV6, TemporalPointAEV7, TemporalPointAEV8, TemporalPointAEV9, TemporalPointAEV10, TemporalPointAEV4, TemporalPointAEV3_Real, TemporalPointAEV11, TemporalPointAEV12, TemporalPointAEV13, TemporalPointAEV14, TemporalPointAEV17, TemporalPointAEV19, TemporalPointAEV20, TemporalPointAEV21, TemporalPointAEV22, TemporalPointAEV23, TemporalPointAEV24, TemporalPointAEV25, TemporalPointAEV26
import trimesh
from utils import *
# import utils
import utils.model_util as model_util
from utils.anchor_utils import masking_load_driver, anchor_load_driver, recover_anchor_batch
# from anchorutils import anchor_load_driver, recover_anchor, recover_anchor_batch

# K x (1 + 1)
# minimum distance; disp - k * disp_o(along_disp_dir)  (l2 norm); k * disp_o(vertical_disp_dir) (l2 norm) -> how those 
# # object moving and the contact information ? #
# only textures on the hand vertices # themselves # 

## the effectiveness of those values themselves --> 
# torch, not_batched #
def calculate_disp_quants(joints, base_pts_trans, minn_base_pts_idxes=None):
  # joints: nf x nn_joints x 3; 
  # base_pts_trans: nf x nn_base_pts x 3; # base pts trans #
  # nf - 1 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_joints x nn_base_pts x 3 --> nf x nnjoints x nnbasepts 
  )
  cur_dist_joints_to_base_pts, cur_minn_base_pts_idxes = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints
  if minn_base_pts_idxes is None:
    minn_base_pts_idxes = cur_minn_base_pts_idxes
  # dist_joints_to_base_pts: nf  nn_joints #
  dist_joints_to_base_pts = model_util.batched_index_select_ours(dist_joints_to_base_pts, minn_base_pts_idxes.unsqueeze(-1), dim=2).squeeze(-1)
  dist_joints_to_base_pts  = torch.sqrt(dist_joints_to_base_pts) # nf x nn_joints #
  k_f = 1. # dist_joints_to_base_pts --> nf x nn_joints x nn_base_pts #
  k = torch.exp(-1. * k_f * (dist_joints_to_base_pts.detach())) # 0 -> 1 value # # nf x nn_joints # nf x nn_joints #
  disp_base_pts = base_pts_trans[1:] -base_pts_trans[:-1] # basepts trans # 
  disp_joints = joints[1:] - joints[:-1] # (nf - 1) x nn_joints x 3 --> for joints displacement here #
  
  minn_base_pts_idxes = minn_base_pts_idxes[:-1]
  k = k[:-1]
  
  dir_disp_base_pts = disp_base_pts / torch.clamp(torch.norm(disp_base_pts, p=2, keepdim=True, dim=-1), min=1e-9) # (nf - 1) x nn_base_pts x 3 
  dir_disp_base_pts = model_util.batched_index_select_ours(dir_disp_base_pts, minn_base_pts_idxes.detach(), dim=1) # (nf - 1) x nf x 3
  disp_base_pts = model_util.batched_index_select_ours(disp_base_pts, minn_base_pts_idxes.detach(), dim=1) 
  # disp along base disp dir #
  disp_along_base_disp_dir = disp_joints * dir_disp_base_pts # (nf - 1) x nn_joints x 3 # along disp dir 
  disp_vt_base_disp_dir = disp_joints - disp_along_base_disp_dir # (nf - 1) x nn_joints x 3  # vt disp dir 
  
  # disp; disp optimziation; and the distances between disps # # moving consistency correction --> but not the optimization? #
  dist_disp_along_dir = disp_base_pts - k.unsqueeze(-1) * disp_along_base_disp_dir
  dist_disp_along_dir = torch.norm(dist_disp_along_dir, dim=-1, p=2) # (nf - 1) x nn_joints # dist_disp_along_dir
  dist_disp_vt_dir = torch.norm(disp_vt_base_disp_dir, dim=-1, p=2) # (nf - 1) x nn_joints # 
  dist_joints_to_base_pts_disp = dist_joints_to_base_pts[:-1] # (nf - 1) x nn_joints # 
  return dist_joints_to_base_pts_disp, dist_disp_along_dir, dist_disp_vt_dir


## batched get quantities #
# batched get quantities here #
# torch, not_batched #
# dist_joints_to_base_pts_disp, dist_disp_along_dir, dist_disp_vt_dir = calculate_disp_quants_batched(joints, base_pts_trans)
def calculate_disp_quants_batched(joints, base_pts_trans):
  # joints: nf x nn_joints x 3; 
  # base_pts_trans: nf x nn_base_pts x 3;
  # nf - 1 # nf x nn_joints x nn_base_pts x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts_trans.unsqueeze(-3)) ** 2, dim=-1 # nf x nn_joints x nn_base_pts x 3 --> nf x nnjoints x nnbasepts 
  )
  dist_joints_to_base_pts, minn_base_pts_idxes = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints
  dist_joints_to_base_pts  = torch.sqrt(dist_joints_to_base_pts) # nf x nn_joints #
  k_f = 1.
  k = torch.exp(-1. * k_f * (dist_joints_to_base_pts)) # 0 -> 1 value # # nf x nn_joints # nf x nn_joints #
  disp_base_pts = base_pts_trans[:, 1:] - base_pts_trans[:, :-1] # basepts trans # 
  disp_joints = joints[:, 1:] - joints[:, :-1] # (nf - 1) x nn_joints x 3 --> for joints displacement here #
  
  minn_base_pts_idxes = minn_base_pts_idxes[:, :-1] # bsz x (nf - 1) # 
  k = k[:, :-1]
  
  dir_disp_base_pts = disp_base_pts / torch.clamp(torch.norm(disp_base_pts, p=2, keepdim=True, dim=-1), min=1e-23) # (nf - 1) x nn_base_pts x 3 
  dir_disp_base_pts = model_util.batched_index_select_ours(dir_disp_base_pts, minn_base_pts_idxes, dim=2) # (nf - 1) x nnjoints x 3
  # disp_base_pts, minn_base_pts_idxes --> bsz x (nf - 1) x nnjoints
  disp_base_pts = model_util.batched_index_select_ours(disp_base_pts, minn_base_pts_idxes, dim=2) 
  
  disp_along_base_disp_dir = disp_joints * dir_disp_base_pts # bsz x (nf - 1) x nn_joints x 3 # along disp dir 
  disp_vt_base_disp_dir = disp_joints - disp_along_base_disp_dir # bsz x (nf - 1) x nn_joints x 3  # vt disp dir 
  
  # disp_base_pts -> bsz x (nf - 1) x njoints x 3 # dist_disp_along_dir 
  dist_disp_along_dir = disp_base_pts - k.unsqueeze(-1) * disp_along_base_disp_dir
  dist_disp_along_dir = torch.norm(dist_disp_along_dir, dim=-1, p=2) # bsz x (nf - 1) x nn_joints # dist_disp_along_dir
  dist_disp_vt_dir = torch.norm(disp_vt_base_disp_dir, dim=-1, p=2) # bsz x  (nf - 1) x nn_joints # 
  dist_joints_to_base_pts_disp = dist_joints_to_base_pts[:, :-1] # bsz x (nf - 1) x nn_joints # 
  return dist_joints_to_base_pts_disp, dist_disp_along_dir, dist_disp_vt_dir  
  
  

# batched get quantities here #
# torch, not_batched #
# dist_joints_to_base_pts_disp, dist_disp_along_dir, dist_disp_vt_dir = calculate_disp_quants_batched(joints, base_pts_trans)
def calculate_disp_quants_v2(joints, base_pts_trans,  canon_joints, canon_base_normals):
  # joints: nf x nn_joints x 3; 
  # base_pts_trans: nf x nn_base_pts x 3;
  # nf - 1 # nf x nn_joints x nn_base_pts x 3 #
  # joints: nf x nn_joints x 3 # --> nf x nn_joint x 1 x 3 - nf x 1 x nn_base_pts x 3 --> nf x nn_jts x nn_base_pts x 3 #
  # base_pts_trans: nf x nn_base_pt x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts_trans.unsqueeze(-3)) ** 2, dim=-1 # nf x nn_joints x nn_base_pts x 3 --> nf x nnjoints x nnbasepts 
  )
  dist_joints_to_base_pts, minn_base_pts_idxes = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints
  dist_joints_to_base_pts  = torch.sqrt(dist_joints_to_base_pts) # nf x nn_joints #
  k_f = 1.
  k = torch.exp(-1. * k_f * (dist_joints_to_base_pts)) # 0 -> 1 value # # nf x nn_joints # nf x nn_joints #
  ### 
  
  ### base pts velocity ###
  disp_base_pts = base_pts_trans[1:] - base_pts_trans[:-1] # basepts trans # 
  ### joints velocity ###
  disp_joints = joints[1:] - joints[:-1] # (nf - 1) x nn_joints x 3 --> for joints displacement here #
  
  minn_base_pts_idxes = minn_base_pts_idxes[:-1] # bsz x (nf - 1) # 
  k = k[:-1]
  
  ### joints velocity in the canonicalized space ###
  disp_canon_joints = canon_joints[1:] - canon_joints[:-1]
  
  ### baes points normals information ###
  disp_canon_base_normals = canon_base_normals[:-1] # bsz x (nf - 1) x  3 --> normals of base points ##
  
  # bsz x (nf - 1) x nn_joints x 3 ##
  disp_canon_base_normals = model_util.batched_index_select_ours(values=disp_canon_base_normals, indices=minn_base_pts_idxes, dim=1) 
  ### joint velocity along normals ###
  disp_joints_along_normals = disp_canon_base_normals * disp_canon_joints
  dotprod_disp_joints_along_normals = disp_joints_along_normals.sum(dim=-1) # bsz x (nf - 1) x nn_joints 
  
  disp_joints_vt_normals = disp_canon_joints - dotprod_disp_joints_along_normals.unsqueeze(-1) * disp_canon_base_normals
  l2_disp_joints_vt_normals = torch.norm(disp_joints_vt_normals, p=2, keepdim=False, dim=-1) # bsz x (nf - 1) x nn_joints # --> for l2 norm vt normals  # l2 normal of the disp_joints ###
  
  
  
  # dir_disp_base_pts = disp_base_pts / torch.clamp(torch.norm(disp_base_pts, p=2, keepdim=True, dim=-1), min=1e-23) # (nf - 1) x nn_base_pts x 3 
  # dir_disp_base_pts = model_util.batched_index_select_ours(dir_disp_base_pts, minn_base_pts_idxes, dim=2) # (nf - 1) x nnjoints x 3
  # # disp_base_pts, minn_base_pts_idxes --> bsz x (nf - 1) x nnjoints
  disp_base_pts = model_util.batched_index_select_ours(disp_base_pts, minn_base_pts_idxes, dim=1) 
  
  # disp_along_base_disp_dir = disp_joints * dir_disp_base_pts # bsz x (nf - 1) x nn_joints x 3 # along disp dir 
  # disp_vt_base_disp_dir = disp_joints - disp_along_base_disp_dir # bsz x (nf - 1) x nn_joints x 3  # vt disp dir 
  
  # # disp_base_pts -> bsz x (nf - 1) x njoints x 3 # dist_disp_along_dir 
  # dist_disp_along_dir = disp_base_pts - k.unsqueeze(-1) * disp_along_base_disp_dir
  # dist_disp_along_dir = torch.norm(dist_disp_along_dir, dim=-1, p=2) # bsz x (nf - 1) x nn_joints # dist_disp_along_dir
  # dist_disp_vt_dir = torch.norm(disp_vt_base_disp_dir, dim=-1, p=2) # bsz x  (nf - 1) x nn_joints # 
  dist_joints_to_base_pts_disp = dist_joints_to_base_pts[:-1] # bsz x (nf - 1) x nn_joints # 
  # disp_base_pts: (nf - 1) x nn_joints x 3 # -> disp of base pts for each joint # 
  # dist_joints_to_base_pts_disp: (nf - 1) x nn_joints #
  # dotprod_disp_joints_along_normals: (nf - 1) x nn_joints #
  # l2_disp_joints_vt_normals: (nf - 1) x nn_joints #
  # disp_base_pts, dist_joints_to_base_pts_disp, dotprod_disp_joints_along_normals, l2_disp_joints_vt_normals # 
  return disp_base_pts, dist_joints_to_base_pts_disp, dotprod_disp_joints_along_normals, l2_disp_joints_vt_normals  




# batched get quantities here #
# torch, not_batched #
# dist_joints_to_base_pts_disp, dist_disp_along_dir, dist_disp_vt_dir = calculate_disp_quants_batched(joints, base_pts_trans)
def calculate_disp_quants_batched_v2(joints, base_pts_trans,  canon_joints, canon_base_normals):
  # joints: nf x nn_joints x 3; 
  # base_pts_trans: nf x nn_base_pts x 3;
  # nf - 1 # nf x nn_joints x nn_base_pts x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts_trans.unsqueeze(-3)) ** 2, dim=-1 # nf x nn_joints x nn_base_pts x 3 --> nf x nnjoints x nnbasepts 
  )
  dist_joints_to_base_pts, minn_base_pts_idxes = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints
  dist_joints_to_base_pts  = torch.sqrt(dist_joints_to_base_pts) # nf x nn_joints #
  k_f = 1.
  k = torch.exp(-1. * k_f * (dist_joints_to_base_pts)) # 0 -> 1 value # # nf x nn_joints # nf x nn_joints #
  ### 
  
  ### base pts velocity ###
  disp_base_pts = base_pts_trans[:, 1:] - base_pts_trans[:, :-1] # basepts trans # 
  ### joints velocity ###
  disp_joints = joints[:, 1:] - joints[:, :-1] # (nf - 1) x nn_joints x 3 --> for joints displacement here #
  
  minn_base_pts_idxes = minn_base_pts_idxes[:, :-1] # bsz x (nf - 1) # 
  k = k[:, :-1]
  
  ### joints velocity in the canonicalized space ###
  disp_canon_joints = canon_joints[:, 1:] - canon_joints[:, :-1]
  
  ### baes points normals information ###
  disp_canon_base_normals = canon_base_normals[:, :-1] # bsz x (nf - 1) x  3 --> normals of base points ##
  
  # bsz x (nf - 1) x nn_joints x 3 ##
  disp_canon_base_normals = model_util.batched_index_select_ours(values=disp_canon_base_normals, indices=minn_base_pts_idxes, dim=2) 
  ### joint velocity along normals ###
  disp_joints_along_normals = disp_canon_base_normals * disp_canon_joints
  dotprod_disp_joints_along_normals = disp_joints_along_normals.sum(dim=-1) # bsz x (nf - 1) x nn_joints 
  
  disp_joints_vt_normals = disp_canon_joints - dotprod_disp_joints_along_normals.unsqueeze(-1) * disp_canon_base_normals
  l2_disp_joints_vt_normals = torch.norm(disp_joints_vt_normals, p=2, keepdim=False, dim=-1) # bsz x (nf - 1) x nn_joints # --> for l2 norm vt normals 
  
  
  
  
  # dir_disp_base_pts = disp_base_pts / torch.clamp(torch.norm(disp_base_pts, p=2, keepdim=True, dim=-1), min=1e-23) # (nf - 1) x nn_base_pts x 3 
  # dir_disp_base_pts = model_util.batched_index_select_ours(dir_disp_base_pts, minn_base_pts_idxes, dim=2) # (nf - 1) x nnjoints x 3
  # # disp_base_pts, minn_base_pts_idxes --> bsz x (nf - 1) x nnjoints
  disp_base_pts = model_util.batched_index_select_ours(disp_base_pts, minn_base_pts_idxes, dim=2) 
  
  # disp_along_base_disp_dir = disp_joints * dir_disp_base_pts # bsz x (nf - 1) x nn_joints x 3 # along disp dir 
  # disp_vt_base_disp_dir = disp_joints - disp_along_base_disp_dir # bsz x (nf - 1) x nn_joints x 3  # vt disp dir 
  
  # # disp_base_pts -> bsz x (nf - 1) x njoints x 3 # dist_disp_along_dir 
  # dist_disp_along_dir = disp_base_pts - k.unsqueeze(-1) * disp_along_base_disp_dir
  # dist_disp_along_dir = torch.norm(dist_disp_along_dir, dim=-1, p=2) # bsz x (nf - 1) x nn_joints # dist_disp_along_dir
  # dist_disp_vt_dir = torch.norm(disp_vt_base_disp_dir, dim=-1, p=2) # bsz x  (nf - 1) x nn_joints # 
  dist_joints_to_base_pts_disp = dist_joints_to_base_pts[:, :-1] # bsz x (nf - 1) x nn_joints # 
  return disp_base_pts, dist_joints_to_base_pts_disp, dotprod_disp_joints_along_normals, l2_disp_joints_vt_normals  
 




def get_optimized_hand_fr_joints(joints):
  joints = torch.from_numpy(joints).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
  
  
  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      
  num_iters = 2000
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()





def get_affinity_fr_dist(dist, s=0.02):
    ### affinity scores ###
    k = 0.5 * torch.cos(torch.pi / s * torch.abs(dist)) + 0.5
    return k
  
def get_optimized_hand_fr_joints_v2(joints, base_pts):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  minn_dist_mask = minn_dist_mask.float()
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)
        
  
  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
  
  
  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()



  
def get_optimized_hand_fr_joints_v3(joints, base_pts, tot_base_pts_trans):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  minn_dist_mask = minn_dist_mask.float()
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)
        
  
  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
  
  
  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()


  
def get_optimized_hand_fr_joints_v4(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  # the direction of the normal vector and the moving direction of the object point -> whether the point should be selected
  # the contact maps of the object should be like? #
  # the direction of the normal vector and the moving direction 
  # define the attraction loss's weight; and attract points to the object surface #
  # 
  # 
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_ct_verts = hand_verts.detach().cpu().numpy()
  bf_ct_joints = hand_joints.detach().cpu().numpy()
  
  if with_contact_opt:
    num_iters = 2000
    num_iters = 1000 # seq 77 # if with contact opt #
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    # tot_base_pts_trans # nf x nn_base_pts x 3
    disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
    disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
      [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
    )
    # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
    dist_joints_to_base_pts_trans = torch.sum(
      (joints.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
    )
    minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
    nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
    nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
    dot_nearest_base_normals_trans = torch.sum(
      nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
    )
    trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
    nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
    # nearest_dist_mask = nearest_dist < 0.01 # hoi seq
    nearest_dist_mask = nearest_dist < 0.1
    k_attr = 100.
    joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
    attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
    
    
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        dist_joints_to_base_pts_sqr = torch.sum(
            (hand_joints.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        )
        # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # attaction_loss = attaction_loss
        # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 80
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 70
        # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # new version relying on new mask #
        # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        ### original version ###
        # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        # attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        ### general ###
        # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        # tune for seq 140
        loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tAttraction Loss: {}'.format(attaction_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
  
  
  # bf_ct_verts, bf_ct_joints #
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints


  
def get_optimized_hand_fr_joints_v4_verts(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  # the direction of the normal vector and the moving direction of the object point -> whether the point should be selected
  # the contact maps of the object should be like? #
  # the direction of the normal vector and the moving direction 
  # define the attraction loss's weight; and attract points to the object surface #
  # 
  # 
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_ct_verts = hand_verts.detach().cpu().numpy()
  bf_ct_joints = hand_joints.detach().cpu().numpy()
  
  
  
  dist_joints_to_base_pts = torch.sum(
    (hand_verts.detach().unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  # minn_dist[:, :-5] = 1e9
  # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  # minn_topk_mask = torch.zeros_like(minn_dist)
  # # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  # minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  
  
  if with_contact_opt:
    num_iters = 2000
    num_iters = 1000 # seq 77
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    # tot_base_pts_trans # nf x nn_base_pts x 3
    disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
    disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
      [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
    )
    # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
    dist_joints_to_base_pts_trans = torch.sum(
      (hand_verts.detach().unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
    )
    minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
    nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
    nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
    dot_nearest_base_normals_trans = torch.sum(
      nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
    )
    trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
    nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
    # nearest_dist_mask = nearest_dist < 0.01 # hoi seq
    nearest_dist_mask = nearest_dist < 0.01
    k_attr = 1000.
    joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
    attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
    
    
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        dist_joints_to_base_pts_sqr = torch.sum(
            (hand_verts.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        )
        # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # attaction_loss = attaction_loss
        # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 80
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 70
        # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # new version relying on new mask #
        # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        ### original version ###
        # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        attaction_loss = torch.mean(attaction_loss * attraction_mask_new_new * joint_attraction_k.unsqueeze(-1))
        
        
        # attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tAttraction Loss: {}'.format(attaction_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
  
  
  # bf_ct_verts, bf_ct_joints #
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints



# get_optimized_hand_fr_joints_v5 ---> get optimized i
def get_optimized_hand_fr_joints_v5(joints, tot_gt_rhand_joints, base_pts, tot_base_pts_trans, predicted_joint_quants=None):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  tot_gt_rhand_joints = torch.from_numpy(tot_gt_rhand_joints).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  # nf x nn_joitns for each quantity # # 
  
  if predicted_joint_quants is None:
    gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
  else:
    predicted_joint_quants = torch.from_numpy(predicted_joint_quants).float().cuda()
    gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = predicted_joint_quants[..., 0], predicted_joint_quants[..., 1], predicted_joint_quants[..., 2] # 
    print(f"gt_dist_joints_to_base_pts_disp: {gt_dist_joints_to_base_pts_disp.size()}, gt_dist_disp_along_dir: {gt_dist_disp_along_dir.size()}, gt_dist_disp_vt_dir: {gt_dist_disp_vt_dir.size()}")
    
    
    gt_dist_joints_to_base_pts_disp_real, gt_dist_disp_along_dir_real, gt_dist_disp_vt_dir_real = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
    diff_gt_dist_joints = torch.mean((gt_dist_joints_to_base_pts_disp_real - gt_dist_joints_to_base_pts_disp) ** 2)
    diff_gt_disp_along_normals = torch.mean((gt_dist_disp_along_dir_real - gt_dist_disp_along_dir) ** 2)
    diff_gt_disp_vt_normals = torch.mean((gt_dist_disp_vt_dir_real - gt_dist_disp_vt_dir) ** 2)
    print(f"diff_gt_dist_joints: {diff_gt_dist_joints.item()}, diff_gt_disp_along_normals: {diff_gt_disp_along_normals.item()}, diff_gt_disp_vt_normals: {diff_gt_disp_vt_normals.item()}")
    
    disp_info_pred_gt_sv_dict = {
      'gt_dist_joints_to_base_pts_disp_real': gt_dist_joints_to_base_pts_disp_real.detach().cpu().numpy(),
      'gt_dist_disp_along_dir_real': gt_dist_disp_along_dir_real.detach().cpu().numpy(),
      'gt_dist_disp_vt_dir_real': gt_dist_disp_vt_dir_real.detach().cpu().numpy(),
      'gt_dist_joints_to_base_pts_disp': gt_dist_joints_to_base_pts_disp.detach().cpu().numpy(),
      'gt_dist_disp_along_dir': gt_dist_disp_along_dir.detach().cpu().numpy(),
      'gt_dist_disp_vt_dir': gt_dist_disp_vt_dir.detach().cpu().numpy(),
    }
    # 
    disp_info_pred_gt_sv_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
    disp_info_pred_gt_sv_fn = os.path.join(disp_info_pred_gt_sv_fn, f"pred_gt_disp_info.npy")
    np.save(disp_info_pred_gt_sv_fn, disp_info_pred_gt_sv_dict)
    print(f"saved to {disp_info_pred_gt_sv_fn}")
    
  
  # tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()
  ### nn_frames ###
  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  minn_dist_mask = minn_dist_mask.float()
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)
        
  
  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
  
  
  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (hand_joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  # nn_base_pts = dist_joints_to_base_pts.size(-1)
  # nn_joints = dist_joints_to_base_pts.size(1)
  
  # dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  
  num_iters = 2000
  # num_iters = 1000
  ori_theta_var = theta_var.detach().clone()
  # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
  opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      
      # gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
      # nf x nn_joints here # # cur_dist_disp_vt_dir
      cur_dist_joints_to_base_pts_disp, cur_dist_disp_along_dir, cur_dist_disp_vt_dir = calculate_disp_quants(hand_joints, tot_base_pts_trans, minn_base_pts_idxes=minn_dist_idx)
      
      dist_joints_to_base_pts_disp_loss =  ((cur_dist_joints_to_base_pts_disp - gt_dist_joints_to_base_pts_disp) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      dist_disp_along_dir_loss = ((cur_dist_disp_along_dir - gt_dist_disp_along_dir) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      dist_disp_vt_dir_loss = ((cur_dist_disp_vt_dir - gt_dist_disp_vt_dir) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      # dist joints to baes pts 
      dist_joints_to_base_pts_disp_loss = ((cur_dist_joints_to_base_pts_disp - gt_dist_joints_to_base_pts_disp) ** 2).mean(dim=-1).mean(dim=-1)
      dist_disp_along_dir_loss = ((cur_dist_disp_along_dir - gt_dist_disp_along_dir) ** 2).mean(dim=-1).mean(dim=-1)
      dist_disp_vt_dir_loss = ((cur_dist_disp_vt_dir - gt_dist_disp_vt_dir) ** 2).mean(dim=-1).mean(dim=-1)
      # dist dip losses for the joint to base pts disp loss #
      # dist_disp_losses -> dist disp losses #
      dist_disp_losses = dist_joints_to_base_pts_disp_loss + dist_disp_along_dir_loss + dist_disp_vt_dir_loss
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
      # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
      loss = dist_disp_losses * 1000000. + theta_smoothness_loss * 0.0000001
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001 # joints smoothness loss #
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      # theta_smoothness_loss
      print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
      # dist_disp_losses
      print('\tDist Disp Loss: {}'.format(dist_disp_losses.item())) # dist disp loss #
  
  
  
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()




# get_optimized_hand_fr_joints_v5 ---> get optimized i
def get_optimized_hand_fr_joints_v6(joints, tot_gt_rhand_joints, base_pts, tot_base_pts_trans, predicted_joint_quants=None):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  tot_gt_rhand_joints = torch.from_numpy(tot_gt_rhand_joints).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  # nf x nn_joitns for each quantity # # 
  
  if predicted_joint_quants is None:
    gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
  else:
    predicted_joint_quants = torch.from_numpy(predicted_joint_quants).float().cuda()
    gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = predicted_joint_quants[..., 0], predicted_joint_quants[..., 1], predicted_joint_quants[..., 2] # 
    print(f"gt_dist_joints_to_base_pts_disp: {gt_dist_joints_to_base_pts_disp.size()}, gt_dist_disp_along_dir: {gt_dist_disp_along_dir.size()}, gt_dist_disp_vt_dir: {gt_dist_disp_vt_dir.size()}")
    
    
    gt_dist_joints_to_base_pts_disp_real, gt_dist_disp_along_dir_real, gt_dist_disp_vt_dir_real = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
    diff_gt_dist_joints = torch.mean((gt_dist_joints_to_base_pts_disp_real - gt_dist_joints_to_base_pts_disp) ** 2)
    diff_gt_disp_along_normals = torch.mean((gt_dist_disp_along_dir_real - gt_dist_disp_along_dir) ** 2)
    diff_gt_disp_vt_normals = torch.mean((gt_dist_disp_vt_dir_real - gt_dist_disp_vt_dir) ** 2)
    print(f"diff_gt_dist_joints: {diff_gt_dist_joints.item()}, diff_gt_disp_along_normals: {diff_gt_disp_along_normals.item()}, diff_gt_disp_vt_normals: {diff_gt_disp_vt_normals.item()}")
    
    disp_info_pred_gt_sv_dict = {
      'gt_dist_joints_to_base_pts_disp_real': gt_dist_joints_to_base_pts_disp_real.detach().cpu().numpy(),
      'gt_dist_disp_along_dir_real': gt_dist_disp_along_dir_real.detach().cpu().numpy(),
      'gt_dist_disp_vt_dir_real': gt_dist_disp_vt_dir_real.detach().cpu().numpy(),
      'gt_dist_joints_to_base_pts_disp': gt_dist_joints_to_base_pts_disp.detach().cpu().numpy(),
      'gt_dist_disp_along_dir': gt_dist_disp_along_dir.detach().cpu().numpy(),
      'gt_dist_disp_vt_dir': gt_dist_disp_vt_dir.detach().cpu().numpy(),
    }
    # 
    disp_info_pred_gt_sv_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
    disp_info_pred_gt_sv_fn = os.path.join(disp_info_pred_gt_sv_fn, f"pred_gt_disp_info.npy")
    np.save(disp_info_pred_gt_sv_fn, disp_info_pred_gt_sv_dict)
    print(f"saved to {disp_info_pred_gt_sv_fn}")
    
  
  # tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()
  ### nn_frames ###
  nn_frames = joints.size(0)
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  minn_dist_mask = minn_dist_mask.float()
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)
        
  
  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
  
  
  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (hand_joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  # joints: nf x nnjoints x 3 #
  dist_joints_to_base_pts = torch.sum(
    (hand_verts.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  # nn_base_pts = dist_joints_to_base_pts.size(-1)
  # nn_joints = dist_joints_to_base_pts.size(1)
  
  # dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  
  num_iters = 2000
  # num_iters = 1000
  ori_theta_var = theta_var.detach().clone()
  # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
  opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      
      # gt_dist_joints_to_base_pts_disp, gt_dist_disp_along_dir, gt_dist_disp_vt_dir = calculate_disp_quants(tot_gt_rhand_joints, tot_base_pts_trans)
      # nf x nn_joints here # # cur_dist_disp_vt_dir
      cur_dist_joints_to_base_pts_disp, cur_dist_disp_along_dir, cur_dist_disp_vt_dir = calculate_disp_quants(hand_joints, tot_base_pts_trans, minn_base_pts_idxes=minn_dist_idx)
      
      dist_joints_to_base_pts_disp_loss =  ((cur_dist_joints_to_base_pts_disp - gt_dist_joints_to_base_pts_disp) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      dist_disp_along_dir_loss = ((cur_dist_disp_along_dir - gt_dist_disp_along_dir) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      dist_disp_vt_dir_loss = ((cur_dist_disp_vt_dir - gt_dist_disp_vt_dir) ** 2)[:, -5:-3].mean(dim=-1).mean(dim=-1)
      # dist joints to baes pts 
      dist_joints_to_base_pts_disp_loss = ((cur_dist_joints_to_base_pts_disp - gt_dist_joints_to_base_pts_disp) ** 2).mean(dim=-1).mean(dim=-1)
      dist_disp_along_dir_loss = ((cur_dist_disp_along_dir - gt_dist_disp_along_dir) ** 2).mean(dim=-1).mean(dim=-1)
      dist_disp_vt_dir_loss = ((cur_dist_disp_vt_dir - gt_dist_disp_vt_dir) ** 2).mean(dim=-1).mean(dim=-1)
      # dist dip losses for the joint to base pts disp loss #
      # dist_disp_losses -> dist disp losses #
      dist_disp_losses = dist_joints_to_base_pts_disp_loss + dist_disp_along_dir_loss + dist_disp_vt_dir_loss
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      dist_joints_to_base_pts_sqr = torch.sum(
          (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
      # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
      loss = dist_disp_losses * 1000000. + theta_smoothness_loss * 0.0000001
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001 # joints smoothness loss #
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      # theta_smoothness_loss
      print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
      # dist_disp_losses
      print('\tDist Disp Loss: {}'.format(dist_disp_losses.item())) # dist disp loss #
  
  
  
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()



def get_penetration_masks(obj_verts, obj_faces, hand_verts):
  # you should pass numpy arrays for obj_verts and obj_faces here #
  
  tot_penetration_masks = []
  for i_fr in range(hand_verts.size(0)):
    cur_obj_verts = obj_verts[i_fr]
    obj_mesh = trimesh.Trimesh(vertices=cur_obj_verts, faces=obj_faces,
            )
    cur_hand_verts = hand_verts[i_fr].detach().cpu().numpy()
    cur_subj_seq_in_obj = obj_mesh.contains(cur_hand_verts) 
    cur_subj_seq_in_obj_th = torch.from_numpy(cur_subj_seq_in_obj).bool().to(hand_verts.device)
    tot_penetration_masks.append(cur_subj_seq_in_obj_th)
  tot_penetration_masks = torch.stack(tot_penetration_masks, dim=0) ### nn_frames x nn_hand_verts
  return tot_penetration_masks
    


# interpolatio for unsmooth hand parameters? #
def get_optimized_hand_fr_joints_v4_anchors(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False, nn_hand_params=24, rt_vars=False, with_proj=False, obj_verts_trans=None, obj_faces=None, with_params_smoothing=False, dist_thres=0.005, with_ctx_mask=False):
  # obj_verts_trans, obj_faces
  joints = torch.from_numpy(joints).float().cuda() # joints 
  base_pts = torch.from_numpy(base_pts).float().cuda() # base_pts 
  
  if nn_hand_params < 45:
    use_pca = True
  else:
    use_pca = False
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  
  
  # anchor_load_driver, masking_load_driver #
  inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
  palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
  hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
  # self.hand_palm_vertex_mask for hand palm mask #
  hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool().cuda() ## the mask for hand palm to get hand anchors #
      
  
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, nn_hand_params]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # 3 + 45 + 3 = 51 for computing #
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  # 
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  ## tot base pts 
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  ### tot base pts trans disp ###
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  s = 1.0
  # affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  # the direction of the normal vector and the moving direction of the object point -> whether the point should be selected
  # the contact maps of the object should be like? #
  # the direction of the normal vector and the moving direction 
  # define the attraction loss's weight; and attract points to the object surface #
  # define the attraction
  # 
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 100
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean() # theta var #
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # # attaction_loss = attaction_loss
      # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      if not use_pca:
        loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.5 + shape_prior_loss * 0.001 + pose_prior_loss * 0.1 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      # print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  
  
  
  if with_params_smoothing:
    # we want to smooth theta_var here? 
    # theta_var # 
    # if theta_var
    # theta_smooth_thres = 0.01
    theta_smooth_thres = 1.
    theta_smooth_thres = 0.5
    smoothed_theta_var = [theta_var[0]] # theta_var_dim
    for i_fr in range(1, theta_var.size(0)):
      cur_theta_var = theta_var[i_fr] # get the theta var of the current frame #
      prev_theta_var = smoothed_theta_var[-1]
      diff_cur_theta_var_with_prev = torch.mean((cur_theta_var - prev_theta_var) ** 2).item()
      if diff_cur_theta_var_with_prev > theta_smooth_thres:
        print(f"i_fr: {i_fr}, diff_cur_theta_var_with_prev: {diff_cur_theta_var_with_prev}")
        smoothed_theta_var.append(prev_theta_var.detach().clone())
      else:
        smoothed_theta_var.append(cur_theta_var.detach().clone())
    smoothed_theta_var = torch.stack(smoothed_theta_var, dim=0) # smoothed_theta_var: nf x nn_theta_dim
    theta_var = torch.randn_like(smoothed_theta_var).cuda()
    theta_var.data = smoothed_theta_var.data
    # theta_var = smoothed_theta_var.clone()
    theta_var.requires_grad_() # for 
    
    hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
        beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
    hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
    hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
    
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_ct_verts = hand_verts.detach().cpu().numpy()
  bf_ct_joints = hand_joints.detach().cpu().numpy()
  bf_ct_rot_var = rot_var.detach().cpu().numpy()
  bf_ct_theta_var = theta_var.detach().cpu().numpy()
  bf_ct_beta_var = beta_var.detach().cpu().numpy()
  bf_ct_transl_var = transl_var.detach().cpu().numpy()
  
  bf_ct_verts_th = hand_verts.detach().clone()
  bf_ct_joints_th = hand_joints.detach().clone()
  bf_ct_rot_var_th = rot_var.detach().clone()
  bf_ct_theta_var_th = theta_var.detach().clone()
  bf_ct_beta_var_th = beta_var.detach().clone()
  bf_ct_transl_var_th = transl_var.detach().clone()
  
  bf_ct_optimized_dict = {
    'bf_ct_verts': bf_ct_verts,
    'bf_ct_joints': bf_ct_joints,
    'bf_ct_rot_var': bf_ct_rot_var,
    'bf_ct_theta_var': bf_ct_theta_var,
    'bf_ct_beta_var': bf_ct_beta_var,
    'bf_ct_transl_var': bf_ct_transl_var,
  }
  
  if with_ctx_mask:
    # obj_verts_trans, obj_faces
    # tot_penetration_masks_bf_contact_opt: nn_frames x nn_vert here for penetration masks ##
    tot_penetration_masks_bf_contact_opt = get_penetration_masks(obj_verts_trans, obj_faces, hand_verts)
    tot_penetration_masks_bf_contact_opt_frame = tot_penetration_masks_bf_contact_opt.float().sum(dim=-1) > 0.5 ### 
    tot_penetration_masks_bf_contact_opt_frame_nmask = (1. - tot_penetration_masks_bf_contact_opt_frame.float()).bool() ### nn_frames x nn_verts here
  
  # tot_penetration_masks_bf_contact_opt_nmask_frame
  
  
  window_size = hand_verts.size(0)
  if with_contact_opt:
    num_iters = 2000
    num_iters = 1000 # seq 77 # if with contact opt #
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    # tot_base_pts_trans # nf x nn_base_pts x 3
    disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
    disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
      [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
    )
    
    rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

    dist_joints_to_base_pts = torch.sum(
      (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
    )
    
    nn_base_pts = dist_joints_to_base_pts.size(-1)
    nn_joints = dist_joints_to_base_pts.size(1)
    
    dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
    minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
    
    nk_contact_pts = 2
    minn_dist[:, :-5] = 1e9
    # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
    # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
    # minn_topk_mask = torch.zeros_like(minn_dist)
    # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
    # minn_topk_mask[:, -5: -3] = 1.
    basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
    minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
    # for seq 101
    # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
    minn_dist_mask = minn_dist_mask.float()
    
    # 
    # for seq 47
    if cat_nm in ["Scissors"]:
      # minn_dist_mask[:] = minn_dist_mask[11:12, :, :]
      # minn_dist_mask[:11] = False

      # if i_test_seq == 24:
      #   minn_dist_mask[20:] = minn_dist_mask[20:21, :, :]
      # else:

      # minn_dist_mask[:] = minn_dist_mask[11:12, :, :]
      minn_dist_mask[:] = minn_dist_mask[20:21, :, :]
    
    attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
    
    
    
    # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
    dist_joints_to_base_pts_trans = torch.sum(
      (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
    )
    minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
    nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
    nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
    dot_nearest_base_normals_trans = torch.sum(
      nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
    )
    trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
    nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
    
    # dist_thres
    nearest_dist_mask = nearest_dist < dist_thres # hoi seq
    # nearest_dist_mask = nearest_dist < 0.005 # hoi seq
    
    # nearest_dist_mask = nearest_dist < 0.03 # hoi seq
    # nearest_dist_mask = nearest_dist < 0.5 # hoi seq # seq 47
    # nearest_dist_mask = nearest_dist < 0.1 # hoi seq # seq 47
    
    
    
    # nearest_dist_mask = nearest_dist < 0.1
    k_attr = 100.
    joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
    attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
    
    # if cat_nm in ["ToyCar", "Pliers", "Bottle", "Mug", "Scissors"]:
    anchor_masks = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
    anchor_nmasks = [iid for iid in range(attraction_mask_new_new.size(1)) if iid not in anchor_masks]
    anchor_nmasks = torch.tensor(anchor_nmasks, dtype=torch.long).cuda()
    attraction_mask_new_new[:, anchor_nmasks, :] = False # scissors 
    
    # # for seq 47
    # elif cat_nm in ["Scissors"]:
    #   anchor_masks = [2, 3, 4, 15, 16, 17, 22, 23, 24]
    #   anchor_nmasks = [iid for iid in range(attraction_mask_new_new.size(1)) if iid not in anchor_masks]
    #   anchor_nmasks = torch.tensor(anchor_nmasks, dtype=torch.long).cuda()
    #   attraction_mask_new_new[:, anchor_nmasks, :] = False
    
    # anchor_masks = torch.array([2, 3, 4, 15, 16, 17, 22, 23, 24], dtype=torch.long).cuda()
    # anchor_masks = torch.arange(attraction_mask_new_new.size(1)).unsqueeze(0).unsqueeze(-1).cuda() != 
    
    # [2, 3, 4]
    # [9, 10, 11]
    # [15, 16, 17]
    # [22, 23, 24]
    # seq 47: [2, 3, 4, 15, 16, 17, 22, 23, 24]
    # motion planning?  # 
    
    
    transl_var_ori = transl_var.clone().detach()
    # transl_var, theta_var, rot_var, beta_var # 
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    # opt = optim.Adam([theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        # 
        rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

        
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        dist_joints_to_base_pts_sqr = torch.sum(
            (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        )
        # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # attaction_loss = attaction_loss
        # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 80
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 70
        # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # new version relying on new mask #
        # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        ### original version ###
        # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        
        attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)

        transl_smoothness_loss = F.mse_loss(transl_var_ori, transl_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        ### general ###
        # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        # tune for seq 140
        loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        # ToyCar # 
        loss = attaction_loss * 10000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        if cat_nm in ["Scissors"]:
          # scissors and 
          # if dist_thres < 0.05:
          #   loss = transl_smoothness_loss * 0.5 + attaction_loss * 10000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5
          # else:          
          #   # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.05
          #   loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.005
          #   # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.005

          if dist_thres < 0.05:
            # loss = transl_smoothness_loss * 0.5 + attaction_loss * 10000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5
            # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 

            # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + pose_smoothness_loss * 0.0005 + joints_smoothness_loss * 0.5

            nearest_dist_shape, _ = torch.min(nearest_dist, dim=-1)
            nearest_dist_shape_mask = nearest_dist_shape > 0.01
            transl_smoothness_loss_v2 = torch.sum((transl_var_ori - transl_var) ** 2, dim=-1)
            transl_smoothness_loss_v2 = torch.mean(transl_smoothness_loss_v2[nearest_dist_shape_mask])
            loss = transl_smoothness_loss * 0.5 + attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5

          else:          
            # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.05
            loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.005
            
            # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.005
           
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tAttraction Loss: {}'.format(attaction_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
        # transl_smoothness_loss
        print('\tTransl Smoothness Loss: {}'.format(transl_smoothness_loss.item()))
  
  rhand_anchors_np = rhand_anchors.detach().cpu().numpy()
  np.save("out_anchors.npy", rhand_anchors_np)
  ### optimized dict before projection ###
  bf_proj_optimized_dict = {
    'bf_ctx_mask_verts': hand_verts.detach().cpu().numpy(),
    'bf_ctx_mask_joints': hand_joints.detach().cpu().numpy(),
    'bf_ctx_mask_rot_var': rot_var.detach().cpu().numpy(),
    'bf_ctx_mask_theta_var': theta_var.detach().cpu().numpy(),
    'bf_ctx_mask_beta_var': beta_var.detach().cpu().numpy(),
    'bf_ctx_mask_transl_var': transl_var.detach().cpu().numpy(),
  }
  
  
  
  ### 
  if with_ctx_mask:
    tot_penetration_masks_bf_contact_opt_frame_nmask_np = tot_penetration_masks_bf_contact_opt_frame_nmask.detach().cpu().numpy()
    hand_verts = hand_verts.detach()
    hand_joints = hand_joints.detach()
    rot_var = rot_var.detach()
    theta_var = theta_var.detach()
    beta_var = beta_var.detach()
    transl_var = transl_var.detach()
    hand_verts = hand_verts.detach()
    
    # tot_base_pts_trans_disp_mask ### total penetration masks  
    tot_base_pts_trans_disp_mask_n = (1. - tot_base_pts_trans_disp_mask.float()) > 0.5 ### that object do not move
    tot_penetration_masks_bf_contact_opt_frame_nmask = (tot_penetration_masks_bf_contact_opt_frame_nmask.float() + tot_base_pts_trans_disp_mask_n.float()) > 1.5
    hand_verts[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_verts_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    hand_joints[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_joints_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    rot_var[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_rot_var_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    theta_var[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_theta_var_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    # beta_var[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_beta_var_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    transl_var[tot_penetration_masks_bf_contact_opt_frame_nmask] = bf_ct_transl_var_th[tot_penetration_masks_bf_contact_opt_frame_nmask]
    
    rot_var_tmp = torch.randn_like(rot_var)
    theta_var_tmp = torch.randn_like(theta_var)
    transl_var_tmp = torch.randn_like(transl_var)
    
    rot_var_tmp.data = rot_var.data.clone()
    theta_var_tmp.data = theta_var.data.clone()
    transl_var_tmp.data = transl_var.data.clone()
    
    rot_var = torch.randn_like(rot_var)
    theta_var = torch.randn_like(theta_var)
    transl_var = torch.randn_like(transl_var)
    
    rot_var.data = rot_var_tmp.data.clone()
    theta_var.data = theta_var_tmp.data.clone()
    transl_var.data = transl_var_tmp.data.clone()
    
    
    rot_var = rot_var.requires_grad_()
    theta_var = theta_var.requires_grad_()
    beta_var = beta_var.requires_grad_()
    transl_var = transl_var.requires_grad_()
    
    
  
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_proj_verts = hand_verts.detach().cpu().numpy()
  bf_proj_joints = hand_joints.detach().cpu().numpy()
  bf_proj_rot_var = rot_var.detach().cpu().numpy()
  bf_proj_theta_var = theta_var.detach().cpu().numpy()
  bf_proj_beta_var = beta_var.detach().cpu().numpy()
  bf_proj_transl_var = transl_var.detach().cpu().numpy()
  
  bf_proj_optimized_dict.update( {
    'bf_proj_verts': bf_proj_verts,
    'bf_proj_joints': bf_proj_joints,
    'bf_proj_rot_var': bf_proj_rot_var,
    'bf_proj_theta_var': bf_proj_theta_var,
    'bf_proj_beta_var': bf_proj_beta_var,
    'bf_proj_transl_var': bf_proj_transl_var,
  } )

  if with_proj:
    num_iters = 2000
    num_iters = 1000 # seq 77 # if with contact opt #
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    nearest_base_pts=None
    nearest_base_normals=None
    
    # obj_verts_trans, obj_faces
    if cat_nm in ["Mug"]:
      # tot_penetration_masks = None
      tot_penetration_masks = get_penetration_masks(obj_verts_trans, obj_faces, hand_verts)
    else:
      tot_penetration_masks = get_penetration_masks(obj_verts_trans, obj_faces, hand_verts)
    # tot_penetration_masks = None
    
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        # tot_base_pts_trans, tot_base_normals_trans #
        # obj_verts_trans, obj_faces
        proj_loss, nearest_base_pts, nearest_base_normals  = get_proj_losses(hand_verts, tot_base_pts_trans, tot_base_normals_trans, tot_penetration_masks, nearest_base_pts=nearest_base_pts, nearest_base_normals=nearest_base_normals)
        
        # rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))


        # hand_joints = rhand_anchors
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        
        
        
        # # dist_joints_to_base_pts_sqr = torch.sum(
        # #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        # )
        # # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # # attaction_loss = attaction_loss
        # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # # seq 80
        # # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # # seq 70
        # # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # # new version relying on new mask #
        # # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        # ### original version ###
        # # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        # # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        
        # attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        ### general ###
        # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        # tune for seq 140
        # loss = proj_loss * 1. + joints_pred_loss * 0.05 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        loss = proj_loss * 1. + joints_pred_loss * 1.0 + joints_smoothness_loss * 0.5 
        if cat_nm in ["Pliers"]:
          #  loss = proj_loss * 1. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.05 
          loss = proj_loss * 1. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.05 

        elif cat_nm in ["Bottle"]:
          loss = proj_loss * 1. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.05 
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tproj_loss Loss: {}'.format(proj_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
  
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  hand_verts = hand_verts.detach().cpu().numpy()
  hand_joints = hand_joints.detach().cpu().numpy()
  rot_var = rot_var.detach().cpu().numpy()
  theta_var = theta_var.detach().cpu().numpy()
  beta_var = beta_var.detach().cpu().numpy()
  transl_var = transl_var.detach().cpu().numpy()
  
  # bf_ct_optimized_dict, bf_proj_optimized_dict, optimized_dict
  optimized_dict = {
    'hand_verts': hand_verts,
    'hand_joints': hand_joints,
    'rot_var': rot_var,
    'theta_var': theta_var,
    'beta_var': beta_var,
    'transl_var': transl_var,
  }
  
  
  return bf_ct_optimized_dict, bf_proj_optimized_dict, optimized_dict
  # if rt_vars:
  #   return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints, transl_var.detach().cpu().numpy(), theta_var.detach().cpu().numpy(), rot_var.detach().cpu().numpy(), beta_var.detach().cpu().numpy()
  # else:
  #   # rt_vars: transl_var, theta_var, rot_var, beta_var # 
  #   # bf_ct_verts, bf_ct_joints #
  #   return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints


def select_nearest_base_pts_via_normals(nearest_base_pts, nearest_base_normals):
  # nearest_base_pts: bsz x nf x nn_jts x 3 
  # nearest_base_normals: bsz x nf x nn_jts x 3 #
  bsz, nf, nn_jts = nearest_base_pts.size()[:3] # nearest_base_pts # bsz x nf x nn_jts x 3 #
  new_nearest_base_pts = [nearest_base_pts[:, 0:1, :]] # 
  new_nearest_base_normals = [nearest_base_normals[:, 0:1, :]]
  
  
  for i_f in range(1, nf):
      cur_nearest_base_pts = nearest_base_pts[:, i_f] # bsz x nn_jts x 3 # 
      cur_nearest_base_normals = nearest_base_normals[:, i_f] # bsz x nn_jts x 3 #
      prev_nearest_base_pts = nearest_base_pts[:, i_f - 1] # bsz x nn_jts x 3 #
      prev_nearest_base_normals = nearest_base_normals[:, i_f - 1] # bsz x nn_jts x 3 
      # 
      dot_cur_n_with_prev_n = torch.sum( # cur nearest base normals; nearest base normals #
          cur_nearest_base_normals * prev_nearest_base_normals, dim=-1 # bsz x nn_jts for the nearest base normals and prev normals # 
      )
      cur_new_nearest_pts = cur_nearest_base_pts.clone() # bsz x nn_jts x 3 # for the base normals #
      cur_new_nearest_normals = cur_nearest_base_normals.clone()  # bsz x nn_jts x 3  # for base normals # 
      # if less than dot_cur_n_with_prev_n # then use 
      thres = -0.3
      thres = 0.5
      cur_new_nearest_pts[dot_cur_n_with_prev_n < thres] = prev_nearest_base_pts[dot_cur_n_with_prev_n < thres] # the dot prev 
      cur_new_nearest_normals[dot_cur_n_with_prev_n < thres] = prev_nearest_base_normals[dot_cur_n_with_prev_n < thres]
      # new nearest base pts # #
      # # cur new nearest pts, normals #
      new_nearest_base_pts.append(cur_new_nearest_pts.unsqueeze(1)) 
      new_nearest_base_normals.append(cur_new_nearest_normals.unsqueeze(1)) # 
  new_nearest_base_pts = torch.cat(new_nearest_base_pts, dim=1) # bsz x nf x nn_jts x 3 --> for nearest base_pts and normals # 
  new_nearest_base_normals = torch.cat(new_nearest_base_normals, dim=1) # bsz x nf x nn_jts x 3 --> for nearest base pts and normals #
  return new_nearest_base_pts, new_nearest_base_normals
# how to project those penetrated 


def select_nearest_base_pts_via_normals_fr_mid( nearest_base_pts, nearest_base_normals):
    # nearest_base_pts: bsz x nf x nn_jts x 3 
    # nearest_base_normals: bsz x nf x nn_jts x 3 #
    bsz, nf, nn_jts = nearest_base_pts.size()[:3] # nearest_base_pts # bsz x nf x nn_jts x 3 #
    
    key_frame_idx = 50
    
    key_frame_idx = 30
    # key_frame_idx = 13
    key_frame_idx = 10
    
    ## select nearesst base pts via normals fr mid ##
    new_nearest_base_pts = [nearest_base_pts[:, key_frame_idx: key_frame_idx + 1, :]] # 
    new_nearest_base_normals = [nearest_base_normals[:, key_frame_idx: key_frame_idx + 1, :]]
    
    # new_nearest_base_pts = [nearest_base_pts[:, 0:1, :]] # 
    # new_nearest_base_normals = [nearest_base_normals[:, 0:1, :]]
    for i_f in range(key_frame_idx - 1, -1, -1):
        cur_nearest_base_pts = nearest_base_pts[:, i_f] # bsz x nn_jts x 3 # 
        cur_nearest_base_normals = nearest_base_normals[:, i_f] # bsz x nn_jts x 3 #
        prev_nearest_base_pts = new_nearest_base_pts[0].squeeze(1) # bsz x nn_jts x 3 #
        prev_nearest_base_normals = new_nearest_base_normals[0].squeeze(1) # bsz x nn_jts x 3 
        
        ## dot cur_n with prev_n ##
        dot_cur_n_with_prev_n = torch.sum( # cur nearest base normals; nearest base normals #
            cur_nearest_base_normals * prev_nearest_base_normals, dim=-1 # bsz x nn_jts for the nearest base normals and prev normals # 
        )
        cur_new_nearest_pts = cur_nearest_base_pts.clone() # bsz x nn_jts x 3 # for the base normals #
        cur_new_nearest_normals = cur_nearest_base_normals.clone()  # bsz x nn_jts x 3  # for base normals # 
        # if less than dot_cur_n_with_prev_n # then use 
        thres = -0.0
        # thres = 0.7
        # thres = 1.0
        cur_new_nearest_pts[dot_cur_n_with_prev_n < thres] = prev_nearest_base_pts[dot_cur_n_with_prev_n < thres] # the dot prev 
        cur_new_nearest_normals[dot_cur_n_with_prev_n < thres] = prev_nearest_base_normals[dot_cur_n_with_prev_n < thres]
        # new nearest base pts # #
        new_nearest_base_pts = [cur_new_nearest_pts.unsqueeze(1)] + new_nearest_base_pts
        new_nearest_base_normals = [cur_new_nearest_normals.unsqueeze(1)] + new_nearest_base_normals
    
    ### cur nearest base pts, base normals ###
    for i_f in range(key_frame_idx + 1, nf, 1):
        cur_nearest_base_pts = nearest_base_pts[:, i_f] # bsz x nn_jts x 3 #
        cur_nearest_base_normals = nearest_base_normals[:, i_f] # bsz x nn_jts x 3 #
        prev_nearest_base_pts = new_nearest_base_pts[-1].squeeze(1) # bsz x nn_jts x 3 # nn_jts x 3 #
        prev_nearest_base_normals = new_nearest_base_normals[-1].squeeze(1) # bsz x nn_jts x 3 ## bsz x nn_jts x 3 ##
        # 
        dot_cur_n_with_prev_n = torch.sum( # cur nearest base normals; nearest base normals #
            cur_nearest_base_normals * prev_nearest_base_normals, dim=-1 # bsz x nn_jts for the nearest base normals and prev normals # 
        ) 
        cur_new_nearest_pts = cur_nearest_base_pts.clone() # bsz x nn_jts x 3 # for the base normals # 
        cur_new_nearest_normals = cur_nearest_base_normals.clone()  # bsz x nn_jts x 3  # for base normals #
        # if less than dot_cur_n_with_prev_n # then use # then use the 
        thres = -0.0
        thres = 0.7
        thres = 0.8 # threshold # threshold 
        thres = 1.0
        # and with the physics projection for sampling
        # whether the guiding is enough #
        # animation and the sequence # 
        # decide the nearest points and nearest normals for the generation; prev nearest base pts #
        # motion planning and nearest pts #
        cur_new_nearest_pts[dot_cur_n_with_prev_n < thres] = prev_nearest_base_pts[dot_cur_n_with_prev_n < thres] # the dot prev 
        cur_new_nearest_normals[dot_cur_n_with_prev_n < thres] = prev_nearest_base_normals[dot_cur_n_with_prev_n < thres]
        # new nearest base pts # #
        # # cur new nearest pts, normals #
        new_nearest_base_pts.append(cur_new_nearest_pts.unsqueeze(1)) #
        new_nearest_base_normals.append(cur_new_nearest_normals.unsqueeze(1)) #
        # new nearest # nearest base pts; nearest_normals #
        
    # for i_f in range(1, nf): # 
    #     cur_nearest_base_pts = nearest_base_pts[:, i_f] # bsz x nn_jts x 3 # 
    #     cur_nearest_base_normals = nearest_base_normals[:, i_f] # bsz x nn_jts x 3 #
    #     prev_nearest_base_pts = nearest_base_pts[:, i_f - 1] # bsz x nn_jts x 3 #
    #     prev_nearest_base_normals = nearest_base_normals[:, i_f - 1] # bsz x nn_jts x 3 
    #     # 
    #     dot_cur_n_with_prev_n = torch.sum( # cur nearest base normals; nearest base normals #
    #         cur_nearest_base_normals * prev_nearest_base_normals, dim=-1 # bsz x nn_jts for the nearest base normals and prev normals # 
    #     )
    #     cur_new_nearest_pts = cur_nearest_base_pts.clone() # bsz x nn_jts x 3 # for the base normals #
    #     cur_new_nearest_normals = cur_nearest_base_normals.clone()  # bsz x nn_jts x 3  # for base normals # 
    #     # if less than dot_cur_n_with_prev_n # then use 
    #     thres = -0.3 # threshol
    #     cur_new_nearest_pts[dot_cur_n_with_prev_n < thres] = prev_nearest_base_pts[dot_cur_n_with_prev_n < thres] # the dot prev 
    #     cur_new_nearest_normals[dot_cur_n_with_prev_n < thres] = prev_nearest_base_normals[dot_cur_n_with_prev_n < thres]
    #     # new nearest base pts # #
    #     # # cur new nearest pts, normals #
    #     new_nearest_base_pts.append(cur_new_nearest_pts.unsqueeze(1)) 
    #     new_nearest_base_normals.append(cur_new_nearest_normals.unsqueeze(1)) # 
        
        
    
    # for i_f in range(1, nf):
    #     cur_nearest_base_pts = nearest_base_pts[:, i_f] # bsz x nn_jts x 3 # 
    #     cur_nearest_base_normals = nearest_base_normals[:, i_f] # bsz x nn_jts x 3 #
    #     prev_nearest_base_pts = nearest_base_pts[:, i_f - 1] # bsz x nn_jts x 3 #
    #     prev_nearest_base_normals = nearest_base_normals[:, i_f - 1] # bsz x nn_jts x 3 
    #     # 
    #     dot_cur_n_with_prev_n = torch.sum( # cur nearest base normals; nearest base normals #
    #         cur_nearest_base_normals * prev_nearest_base_normals, dim=-1 # bsz x nn_jts for the nearest base normals and prev normals # 
    #     )
    #     cur_new_nearest_pts = cur_nearest_base_pts.clone() # bsz x nn_jts x 3 # for the base normals #
    #     cur_new_nearest_normals = cur_nearest_base_normals.clone()  # bsz x nn_jts x 3  # for base normals # 
    #     # if less than dot_cur_n_with_prev_n # then use 
    #     thres = -0.3
    #     cur_new_nearest_pts[dot_cur_n_with_prev_n < thres] = prev_nearest_base_pts[dot_cur_n_with_prev_n < thres] # the dot prev 
    #     cur_new_nearest_normals[dot_cur_n_with_prev_n < thres] = prev_nearest_base_normals[dot_cur_n_with_prev_n < thres]
    #     # new nearest base pts # #
    #     # # cur new nearest pts, normals #
    #     new_nearest_base_pts.append(cur_new_nearest_pts.unsqueeze(1)) 
    #     new_nearest_base_normals.append(cur_new_nearest_normals.unsqueeze(1)) # 
        
    # new nearest base pts
    new_nearest_base_pts = torch.cat(new_nearest_base_pts, dim=1) # bsz x nf x nn_jts x 3 --> for nearest base_pts and normals # 
    new_nearest_base_normals = torch.cat(new_nearest_base_normals, dim=1) # bsz x nf x nn_jts x 3 --> for nearest base pts and normals #
    return new_nearest_base_pts, new_nearest_base_normals




def get_proj_losses(hand_verts, base_pts, base_normals, tot_penetration_masks, nearest_base_pts=None, nearest_base_normals=None):
  # nf x nn_verts x 3
  # nf x nn_base_pts x 3 
  dist_hand_verts_base_pts = torch.sum(
    (hand_verts.detach().unsqueeze(-2) - base_pts.unsqueeze(1)) ** 2, dim=-1 # nf x nn_verts x nn_base_pts 
  )
  
  if nearest_base_pts is None:
    nearest_base_dist, nearest_idxes = torch.min(dist_hand_verts_base_pts, dim=-1) # nf x nn_verts
    nearest_base_normals = model_util.batched_index_select_ours(base_normals, nearest_idxes, dim=1) # (nf - 1) x nf x 
    nearest_base_pts = model_util.batched_index_select_ours(base_pts, nearest_idxes, dim=1) # (nf - 1) x nf x 
    print(f"hand_verts: {hand_verts.size()}, nearest_base_pts: {nearest_base_pts.size()}, nearest_base_normals: {nearest_base_normals.size()}")
    # nearest_base_pts, nearest_base_normals = select_nearest_base_pts_via_normals(nearest_base_pts.unsqueeze(0), nearest_base_normals.unsqueeze(0))
    nearest_base_pts, nearest_base_normals = select_nearest_base_pts_via_normals_fr_mid(nearest_base_pts.unsqueeze(0), nearest_base_normals.unsqueeze(0))
    print(f"nearest_base_pts: {nearest_base_pts.size()}, nearest_base_normals: {nearest_base_normals.size()}")
    nearest_base_pts = nearest_base_pts.squeeze(0)
    nearest_base_normals = nearest_base_normals.squeeze(0)
  
  
  jts_to_base_pts = hand_verts - nearest_base_pts # from base pts to pred joints #
  dot_rel_with_normals = torch.sum(
      jts_to_base_pts * nearest_base_normals, dim=-1 # bsz x nf x nn_jts --> joints inside of the object #
  )
  
  # dot_rel_with_normals = torch.sum(
  #     jts_to_base_pts * nearest_base_normals, dim=-1 # bsz x nf x nn_jts --> joints inside of the object #
  # )
  # proj_loss f
  if tot_penetration_masks is not None:
    tot_proj_loss_masks = ((dot_rel_with_normals < 0.).float() + tot_penetration_masks.float()) > 1.5
  else:
    tot_proj_loss_masks = (dot_rel_with_normals < 0.)
  # proj_loss = torch.mean( # mean of loss #
  #     (jts_to_base_pts ** 2)[dot_rel_with_normals < 0.]
  # )
  proj_loss = torch.mean( # mean of loss #
      (jts_to_base_pts ** 2)[tot_proj_loss_masks]
  )
  # tot_penetration_masks
  # proj_loss = 

  return proj_loss, nearest_base_pts, nearest_base_normals
  
def get_optimized_hand_fr_joints_v4_fr_anchors(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False, with_proj=False):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  # tot_base_pts_trans, tot_base_normals_trans #
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  
  window_size = nn_frames
  
  # anchor_load_driver, masking_load_driver #
  inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
  palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
  hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
  # self.hand_palm_vertex_mask for hand palm mask #
  hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool().cuda() ## the mask for hand palm to get hand anchors #
      
  
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  ## tot base pts 
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  # s = 1.0
  # affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      
      hand_joints = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  # the direction of the normal vector and the moving direction of the object point -> whether the point should be selected
  # the contact maps of the object should be like? #
  # the direction of the normal vector and the moving direction 
  # define the attraction loss's weight; and attract points to the object surface #
  # 
  # 
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 1000
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      hand_joints = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # # attaction_loss = attaction_loss
      # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      # print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_ct_verts = hand_verts.detach().cpu().numpy()
  bf_ct_joints = hand_joints.detach().cpu().numpy()
  
  
  window_size = hand_verts.size(0)
  
  if with_contact_opt:
    num_iters = 2000
    num_iters = 1000 # seq 77 # if with contact opt #
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    # tot_base_pts_trans # nf x nn_base_pts x 3
    disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
    disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
      [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
    )
    
    rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

    dist_joints_to_base_pts = torch.sum(
      (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
    )
    
    nn_base_pts = dist_joints_to_base_pts.size(-1)
    nn_joints = dist_joints_to_base_pts.size(1)
    
    dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
    minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
    
    nk_contact_pts = 2
    minn_dist[:, :-5] = 1e9
    # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
    # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
    # minn_topk_mask = torch.zeros_like(minn_dist)
    # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
    # minn_topk_mask[:, -5: -3] = 1.
    basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
    minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
    # for seq 101
    # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
    minn_dist_mask = minn_dist_mask.float()
    
    attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
    
    
    
    # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
    dist_joints_to_base_pts_trans = torch.sum(
      (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
    )
    minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
    nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
    nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
    dot_nearest_base_normals_trans = torch.sum(
      nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
    )
    trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
    nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
    nearest_dist_mask = nearest_dist < 0.005 # hoi seq
    # nearest_dist_mask = nearest_dist < 0.1
    k_attr = 100.
    joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
    attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
    
    
    
    
    
    
    
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))


        hand_joints = rhand_anchors
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        
        
        
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        dist_joints_to_base_pts_sqr = torch.sum(
            (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        )
        # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # attaction_loss = attaction_loss
        # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 80
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 70
        # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # new version relying on new mask #
        # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        ### original version ###
        # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        
        attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        ### general ###
        # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        # tune for seq 140
        loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tAttraction Loss: {}'.format(attaction_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
  
  
    if with_proj:
      num_iters = 2000
      num_iters = 1000 # seq 77 # if with contact opt #
      # num_iters = 500 # seq 77
      ori_theta_var = theta_var.detach().clone()
      
      
      
      # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
      # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
      opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
      scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
      for i in range(num_iters):
          opt.zero_grad()
          # mano_layer
          hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
              beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
          hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
          # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
          
          # tot_base_pts_trans, tot_base_normals_trans #
          proj_loss = get_proj_losses(hand_verts, tot_base_pts_trans, tot_base_normals_trans)
          
          # rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))


          # hand_joints = rhand_anchors
          
          # joints_pred_loss = torch.sum(
          #   (hand_joints - joints) ** 2, dim=-1
          # ).mean()
          
          
          
          
          # # dist_joints_to_base_pts_sqr = torch.sum(
          # #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
          # # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
          # dist_joints_to_base_pts_sqr = torch.sum(
          #     (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
          # )
          # # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
          # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
          # # attaction_loss = attaction_loss
          # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
          
          # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
          # # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
          
          # # seq 80
          # # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
          
          # # seq 70
          # # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
          
          # # new version relying on new mask #
          # # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
          # ### original version ###
          # # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
          
          # # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
          
          
          
          # attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
          
          
          # seq mug
          # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
          
          
          # opt.zero_grad()
          pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
          # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
          shape_prior_loss = torch.mean(beta_var**2)
          pose_prior_loss = torch.mean(theta_var**2)
          joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
          # =0.05
          # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
          # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
          
          # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
          
          # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
          
          theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
          # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
          
          # attraction loss, joint prediction loss, joints smoothness loss #
          # loss = attaction_loss * 1000. + joints_pred_loss 
          ### general ###
          # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
          
          # tune for seq 140
          loss = proj_loss * 1. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
          # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
          # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
          
          # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
          
          # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
          
          # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
          # loss = joints_pred_loss * 30 + attaction_loss * 0.001
          
          opt.zero_grad()
          loss.backward()
          opt.step()
          scheduler.step()
          
          print('Iter {}: {}'.format(i, loss.item()), flush=True)
          print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
          print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
          print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
          # print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
          print('\tproj_loss Loss: {}'.format(proj_loss.item()))
          print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
          # theta_smoothness_loss
          print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
    
    
  # bf_ct_verts, bf_ct_joints #
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints



def get_rhand_joints_verts_fr_params(rhand_transl, rhand_rot, rhand_theta, rhand_beta):
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda() 
  nn_frames = rhand_rot.size(0) #### rhand_rot for glboal orientation ###
  # nframes for the joitns and th
  hand_verts, hand_joints = mano_layer(torch.cat([rhand_rot, rhand_theta], dim=-1),
      rhand_beta.view(-1, 10), rhand_transl)
  hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
  hand_joints = hand_joints.view( nn_frames, -1, 3) * 0.001
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()


  
def get_optimized_hand_fr_joints_v4_fr_anchors_wproj(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False, obj_verts=None):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = joints.size(0)
  
  window_size = nn_frames
  
  # anchor_load_driver, masking_load_driver #
  inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
  palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
  hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
  # self.hand_palm_vertex_mask for hand palm mask #
  hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool().cuda() ## the mask for hand palm to get hand anchors #
      
  
  

  # initialize variables
  beta_var = torch.randn([1, 10]).cuda()
  # first 3 global orientation
  rot_var = torch.randn([nn_frames, 3]).cuda()
  theta_var = torch.randn([nn_frames, 24]).cuda()
  transl_var = torch.randn([nn_frames, 3]).cuda()
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  dist_joints_to_base_pts = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  ## tot base pts 
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  print(f"minn_dist_mask: {minn_dist_mask.size()}")
  # s = 1.0
  # affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  num_iters = 200
  opt = optim.Adam([rot_var, transl_var], lr=learning_rate)
  for i in range(num_iters): #
      opt.zero_grad()
      # mano_layer #
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      
      hand_joints = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[:, 1:], theta_var.view(nn_frames, -1)[:, :-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      
      # pose_smoothness_loss = 
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      loss = joints_pred_loss * 30
      loss = joints_pred_loss * 1000
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
  
  # 
  print(tot_base_pts_trans.size())
  diff_base_pts_trans = torch.sum((tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1) # (nf - 1) x nn_base_pts
  print(f"diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_base_pts_trans = diff_base_pts_trans.mean(dim=-1)
  diff_base_pts_trans_threshold = 1e-20
  diff_base_pts_trans_mask = diff_base_pts_trans > diff_base_pts_trans_threshold # (nf - 1) ### the mask of the tranformed base pts
  diff_base_pts_trans_mask = diff_base_pts_trans_mask.float()
  print(f"diff_base_pts_trans_mask: {diff_base_pts_trans_mask.size()}, diff_base_pts_trans: {diff_base_pts_trans.size()}")
  diff_last_frame_mask = torch.tensor([0,], dtype=torch.float32).to(diff_base_pts_trans_mask.device) + diff_base_pts_trans_mask[-1]
  diff_base_pts_trans_mask = torch.cat(
    [diff_base_pts_trans_mask, diff_last_frame_mask], dim=0 # nf tensor
  )
  # attraction_mask = (diff_base_pts_trans_mask.unsqueeze(-1).unsqueeze(-1) + minn_topk_mask.float()) > 1.5
  attraction_mask = minn_topk_mask.float()
  attraction_mask = attraction_mask.float()
  
  # the direction of the normal vector and the moving direction of the object point -> whether the point should be selected
  # the contact maps of the object should be like? #
  # the direction of the normal vector and the moving direction 
  # define the attraction loss's weight; and attract points to the object surface #
  # 
  # 
  
  num_iters = 2000
  num_iters = 3000
  # num_iters = 100
  learning_rate = 0.01
  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      hand_joints = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      joints_pred_loss = torch.sum(
        (hand_joints - joints) ** 2, dim=-1
      ).mean()
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # # attaction_loss = attaction_loss
      # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      # print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      
      
  for i_meta_iter in range(0):
    dist_hand_verts_base_pts = torch.sum(
      (hand_verts.detach().unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
    )
    minn_dist_hand_verts_base_pts, minn_dist_idxes = torch.min(dist_hand_verts_base_pts, dim=-1) # nf x nn_verts; nf x nn_verts # 
      
    nn_proj_iters = 100
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=0.001)
    for i in range(nn_proj_iters):
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      # base_pts 
      # hand_verts: nf x nn_hand_verts x 3  #
      # base_pts: nf x nn_base_pts x 3 #
      
      ### 
      ## the same base pts for #
      # dist_hand_verts_base_pts = torch.sum(
      #   (hand_verts.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
      # )
      # minn_dist_hand_verts_base_pts, minn_dist_idxes = torch.min(dist_hand_verts_base_pts, dim=-1) # nf x nn_verts; nf x nn_verts # 
      
      nearest_base_pts = model_util.batched_index_select_ours(tot_base_pts_trans, minn_dist_idxes, dim=1)  # nf x nn_verts x 3--> nearest base pts # minn dist idxes; minn dist idxes; 
      nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, minn_dist_idxes, dim=1) # nf x nn_verts x 3 --> nearest base normals # 
      rel_base_pts_to_hand_verts = hand_verts - nearest_base_pts.detach()
      dot_rel_with_normals = torch.sum(
        rel_base_pts_to_hand_verts * nearest_base_normals.detach(), dim=-1
      )
      loss_proj = torch.sum(
        rel_base_pts_to_hand_verts ** 2, dim=-1
      )
      
      loss_proj = torch.mean(loss_proj[dot_rel_with_normals < 0.])
      print(f"i_meta_iter: {i_meta_iter}, loss_proj: {loss_proj}")
      opt.zero_grad()
      loss_proj.backward()
      opt.step()
      
      if i == nn_proj_iters - 1:
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
          
      # 
      print(f"i_meta_iter: {i_meta_iter}, loss_proj: {loss_proj.item()}")
    
  
    
      
      
  ### ### verts and joints before contact opt ### ###
  # bf_ct_verts, bf_ct_joints #
  bf_ct_verts = hand_verts.detach().cpu().numpy()
  bf_ct_joints = hand_joints.detach().cpu().numpy()
  
  
  window_size = hand_verts.size(0)
  if with_contact_opt:
    num_iters = 2000
    num_iters = 1000 # seq 77 # if with contact opt #
    # num_iters = 500 # seq 77
    ori_theta_var = theta_var.detach().clone()
    
    # tot_base_pts_trans # nf x nn_base_pts x 3
    disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
    disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
      [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
    )
    
    rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

    dist_joints_to_base_pts = torch.sum(
      (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
    )
    
    nn_base_pts = dist_joints_to_base_pts.size(-1)
    nn_joints = dist_joints_to_base_pts.size(1)
    
    dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
    minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
    
    nk_contact_pts = 2
    minn_dist[:, :-5] = 1e9
    # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
    # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
    # minn_topk_mask = torch.zeros_like(minn_dist)
    # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
    # minn_topk_mask[:, -5: -3] = 1.
    basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
    minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
    # for seq 101
    # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
    minn_dist_mask = minn_dist_mask.float()
    
    attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
    
    
    
    # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
    dist_joints_to_base_pts_trans = torch.sum(
      (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
    )
    minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
    nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
    nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
    dot_nearest_base_normals_trans = torch.sum(
      nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
    )
    trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
    nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
    nearest_dist_mask = nearest_dist < 0.005 # hoi seq
    # nearest_dist_mask = nearest_dist < 0.1
    k_attr = 100.
    joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
    attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
    
    
    
    
    
    
    
    # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
    # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
    opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        # mano_layer
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
        # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
        
        rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))


        hand_joints = rhand_anchors
        
        joints_pred_loss = torch.sum(
          (hand_joints - joints) ** 2, dim=-1
        ).mean()
        
        
        
        
        # dist_joints_to_base_pts_sqr = torch.sum(
        #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
        # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
        dist_joints_to_base_pts_sqr = torch.sum(
            (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
        )
        # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
        attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
        # attaction_loss = attaction_loss
        # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
        
        # attaction_loss = torch.mean(attaction_loss * attraction_mask)
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 80
        # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # seq 70
        # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        # new version relying on new mask #
        # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
        ### original version ###
        # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
        
        
        
        attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
        
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        
        # opt.zero_grad()
        pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
        # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
        # =0.05
        # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
        # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
        
        # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
        
        theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
        # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
        
        # attraction loss, joint prediction loss, joints smoothness loss #
        # loss = attaction_loss * 1000. + joints_pred_loss 
        ### general ###
        # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        
        # tune for seq 140
        loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
        # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
        
        # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
        
        # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
        # loss = joints_pred_loss * 30 + attaction_loss * 0.001
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
        print('\tAttraction Loss: {}'.format(attaction_loss.item()))
        print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
        # theta_smoothness_loss
        print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
  
  
  # bf_ct_verts, bf_ct_joints #
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(), bf_ct_verts, bf_ct_joints




## get optimized hand fr joints v4 anchors fr params ##
def get_optimized_hand_fr_joints_v4_anchors_fr_params(beta_var, rot_var, theta_var, transl_var, tot_base_pts_trans):
  # joints = torch.from_numpy(joints).float().cuda()
  # base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  # tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  # ### start optimization ###
  # # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root # # 
      ncomps=24,
      use_pca=True,
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  # nn_frames = joints.size(0)
  # under stand penetration and resolve penetrations # 
  
  # # anchor_load_driver, masking_load_driver #
  inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  # anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
  # palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
  # hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
  # # self.hand_palm_vertex_mask for hand palm mask #
  # hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool().cuda() ## the mask for hand palm to get hand anchors #
      
  
 

  # initialize variables
  # beta_var = torch.randn([1, 10]).cuda()
  # # first 3 global orientation
  # rot_var = torch.randn([nn_frames, 3]).cuda()
  # theta_var = torch.randn([nn_frames, 24]).cuda()
  # transl_var = torch.randn([nn_frames, 3]).cuda()
  
  beta_var = beta_var[0:1] # beta_var: 1 x 10 
  
  # transl_var = tot_rhand_transl.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  # ori_transl_var = transl_var.clone()
  # rot_var = tot_rhand_glb_orient.unsqueeze(0).repeat(args.num_init, 1, 1).contiguous().to(device).view(args.num_init * num_frames, 3).contiguous()
  
  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()
  
  learning_rate = 0.1
  
  nn_frames = theta_var.size(0)
  
  window_size = nn_frames
  
  
  hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1), # n
      beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
  hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
  hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
  
  
  
  
  # joints: nf x nnjoints x 3 #
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  # dist_joints_to_base_pts = torch.sum(
  #   (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  # )
  
  # nn_base_pts = dist_joints_to_base_pts.size(-1)
  # nn_joints = dist_joints_to_base_pts.size(1)
  
  # dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  # minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  # nk_contact_pts = 2
  # minn_dist[:, :-5] = 1e9
  # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  # minn_topk_mask = torch.zeros_like(minn_dist)
  # # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  # minn_topk_mask[:, -5: -3] = 1.
  # basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  # minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # # for seq 101
  # # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  # minn_dist_mask = minn_dist_mask.float()
  
  # ## tot base pts 
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  # attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  # minn_topk_mask = (minn_dist_mask + minn_topk_mask.float().unsqueeze(-1)) > 1.5
  # print(f"minn_dist_mask: {minn_dist_mask.size()}")
  # s = 1.0
  # # affinity_scores = get_affinity_fr_dist(dist_joints_to_base_pts, s=s)

  # opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)


  # ### ### verts and joints before contact opt ### ###
  # # bf_ct_verts, bf_ct_joints #
  # bf_ct_verts = hand_verts.detach().cpu().numpy()
  # bf_ct_joints = hand_joints.detach().cpu().numpy()
  
  
  window_size = hand_verts.size(0)
  # if with_contact_opt:
  num_iters = 2000
  num_iters = 1000 # seq 77 # if with contact opt #
  # num_iters = 500 # seq 77
  ori_theta_var = theta_var.detach().clone() # theta var # 
  
  # tot_base_pts_trans # nf x nn_base_pts x 3
  disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
  disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
    [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
  )
  
  rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

  dist_joints_to_base_pts = torch.sum(
    (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  # minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  # minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  minn_dist_mask[1:] = minn_dist_mask[0:1]
  minn_dist_mask = minn_dist_mask.float()
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
  dist_joints_to_base_pts_trans = torch.sum(
    (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
  )
  minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
  # nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
  # nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
  # dot_nearest_base_normals_trans = torch.sum(
  #   nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
  # )
  # trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
  nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
  nearest_dist_mask = nearest_dist < 0.002 # hoi seq
  # nearest_dist_mask = nearest_dist < 0.1
  k_attr = 100.
  joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
  # attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
  
  attraction_mask_new_new = (attraction_mask_new.float() + nearest_dist_mask.float().unsqueeze(-1)) > 1.5
  
  
  
  
  
  
  
  # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
  # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
  opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      # joints_pred_loss = torch.sum(
      #   (hand_joints - joints) ** 2, dim=-1
      # ).mean()
      
      
      rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
      dist_joints_to_base_pts_sqr = torch.sum(
          (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # seq 80
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # seq 70
      # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # new version relying on new mask #
      # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
      ### original version ###
      # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
      
      # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
      
      
      
      attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
      
      
      # seq mug
      # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)
      # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
      
      # attraction loss, joint prediction loss, joints smoothness loss #
      # loss = attaction_loss * 1000. + joints_pred_loss 
      ### general ###
      # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
      
      # tune for seq 140
      loss = attaction_loss * 10000. # + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      # print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      # theta_smoothness_loss
      print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))

  
  # bf_ct_verts, bf_ct_joints #
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy() # , bf_ct_verts, bf_ct_joints


def get_projection_one_frame(beta_var, rot_var, theta_var, transl_var, tot_base_pts_trans, tot_base_pts_normals, selected_frame=11):
  # beta_var.requires_grad_()
  # rot_var.requires_grad_()
  # theta_var.requires_grad_()
  # transl_var.requires_grad_()
  # beta_var, rot_var, theta_var, transl_var,

  nn_hand_params = theta_var.size(-1) # hand params
  use_pca = True if nn_hand_params < 45 else False
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  # nn_base_pts_trans: nn_frames x nn_pts x 3 # 
  # nn_frames = rot_var.size(0)

  

  num_iters = 1000
  learning_rate = 0.01
  
  rot_var = rot_var[selected_frame: selected_frame + 1]
  theta_var = theta_var[selected_frame: selected_frame + 1]
  transl_var = transl_var[selected_frame: selected_frame + 1]
  tot_base_pts_trans = tot_base_pts_trans[selected_frame: selected_frame + 1]
  tot_base_pts_normals = tot_base_pts_normals[selected_frame: selected_frame + 1]

  rot_var_ori = torch.randn_like(rot_var)
  theta_var_ori = torch.randn_like(theta_var)
  beta_var_ori = torch.randn_like(beta_var)
  transl_var_ori = torch.randn_like(transl_var)

  rot_var_ori.data[:] = rot_var.data[:].clone()
  theta_var_ori.data[:] = theta_var.data[:].clone()
  beta_var_ori.data[:] = beta_var.data[:].clone()
  transl_var_ori.data[:] = transl_var.data[:].clone()

  rot_var = rot_var_ori
  theta_var = theta_var_ori
  beta_var = beta_var_ori
  transl_var = transl_var_ori

  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()

  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)

  # rot_var_ori = rot_var.detach().clone()
  # transl_var_
  
  nn_frames = theta_var.size(0)

  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001

      if i == 0:
        hand_verts_ori = hand_verts.detach().clone()


      nearest_verts_to_base_pts = torch.sum(
        (hand_verts.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nn_frames x nn_verts x nn_base_pts # 
      )
      nearest_verts_to_base_pts_dist, nearest_verts_to_base_pts_idxes = torch.min(
        nearest_verts_to_base_pts, dim=-1 # nn_frames x nn_verts
      )
      nearest_verts_to_base_pts = model_util.batched_index_select_ours(tot_base_pts_trans, nearest_verts_to_base_pts_idxes, dim=1)
      nearest_verts_to_base_normals = model_util.batched_index_select_ours(
        tot_base_pts_normals, nearest_verts_to_base_pts_idxes, dim=1 # nnframes x nnverts x 3 
      )
      rel_pts_to_verts = hand_verts - nearest_verts_to_base_pts
      dot_rel_with_normals = torch.sum(
        rel_pts_to_verts * nearest_verts_to_base_normals, dim=-1 # nnframes x nnverts
      )
      tot_proj_loss_masks = dot_rel_with_normals < 0.
      proj_loss = torch.mean( # mean of loss #
        (rel_pts_to_verts ** 2).sum(dim=-1)[tot_proj_loss_masks]
      )

      
      hand_verts_smoothness_loss = torch.sum(
        (hand_verts_ori - hand_verts) ** 2, dim=-1 
      ).mean()

      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)

      loss = hand_verts_smoothness_loss * 0.1 + proj_loss + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 
      
      # joints_pred_loss = torch.sum(
      #   (hand_joints - joints) ** 2, dim=-1
      # ).mean() # theta var #
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # # attaction_loss = attaction_loss
      # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      # pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      # shape_prior_loss = torch.mean(beta_var**2)
      # pose_prior_loss = torch.mean(theta_var**2)
      # # joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # if not use_pca:
      #   loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.5 + shape_prior_loss * 0.001 + pose_prior_loss * 0.1 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tProjection Loss: {}'.format(proj_loss.item())) # hand_verts_smoothness_loss
      print('\thand_verts_smoothness_loss: {}'.format(hand_verts_smoothness_loss.item())) #
      # print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      # print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      # # print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      # print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))

      # beta_var, rot_var, theta_var, transl_var #
  
  return hand_verts.detach().cpu().numpy(), hand_verts_ori.detach().cpu().numpy(),  beta_var.detach(), rot_var.detach(), theta_var.detach(), transl_var.detach()
  

# and use attraction losses for the following vertices here #

def get_projection_multi_frames(beta_var, rot_var, theta_var, transl_var, tot_base_pts_trans, tot_base_pts_normals, selected_frame=11):
  # beta_var.requires_grad_()
  # rot_var.requires_grad_()
  # theta_var.requires_grad_()
  # transl_var.requires_grad_()
  # beta_var, rot_var, theta_var, transl_var,

  nn_hand_params = theta_var.size(-1) # hand params
  use_pca = True if nn_hand_params < 45 else False
  # setup MANO layer # tot_base_pts_trans #
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  # nn_base_pts_trans: nn_frames x nn_pts x 3 # 
  # nn_frames = rot_var.size(0)

  

  num_iters = 1000
  learning_rate = 0.01

  
  
  rot_var = rot_var[: selected_frame + 1]
  theta_var = theta_var[: selected_frame + 1]
  transl_var = transl_var[: selected_frame + 1]
  tot_base_pts_trans = tot_base_pts_trans[: selected_frame + 1]
  tot_base_pts_normals = tot_base_pts_normals[: selected_frame + 1]

  rot_var_ori = torch.randn_like(rot_var)
  theta_var_ori = torch.randn_like(theta_var)
  beta_var_ori = torch.randn_like(beta_var)
  transl_var_ori = torch.randn_like(transl_var)

  rot_var_ori.data[:] = rot_var.data[:].clone()
  theta_var_ori.data[:] = theta_var.data[:].clone()
  beta_var_ori.data[:] = beta_var.data[:].clone()
  transl_var_ori.data[:] = transl_var.data[:].clone()

  rot_var = rot_var_ori
  theta_var = theta_var_ori
  beta_var = beta_var_ori
  transl_var = transl_var_ori

  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()

  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)

  # rot_var_ori = rot_var.detach().clone()
  transl_var_ori = transl_var.detach().clone()
  
  nn_frames = theta_var.size(0)

  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001

      if i == 0:
        hand_verts_ori = hand_verts.detach().clone()


      nearest_verts_to_base_pts = torch.sum(
        (hand_verts.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nn_frames x nn_verts x nn_base_pts # 
      )
      nearest_verts_to_base_pts_dist, nearest_verts_to_base_pts_idxes = torch.min(
        nearest_verts_to_base_pts, dim=-1 # nn_frames x nn_verts
      )
      nearest_verts_to_base_pts = model_util.batched_index_select_ours(tot_base_pts_trans, nearest_verts_to_base_pts_idxes, dim=1)
      nearest_verts_to_base_normals = model_util.batched_index_select_ours(
        tot_base_pts_normals, nearest_verts_to_base_pts_idxes, dim=1 # nnframes x nnverts x 3 
      )
      rel_pts_to_verts = hand_verts - nearest_verts_to_base_pts
      dot_rel_with_normals = torch.sum(
        rel_pts_to_verts * nearest_verts_to_base_normals, dim=-1 # nnframes x nnverts
      )
      tot_proj_loss_masks = dot_rel_with_normals < 0.
      proj_loss = torch.mean( # mean of loss #
        (rel_pts_to_verts ** 2).sum(dim=-1)[tot_proj_loss_masks]
      )

      
      # hand_verts_smoothness_loss = torch.sum(
      #   (hand_verts_ori - hand_verts) ** 2, dim=-1 
      # ).mean()

      hand_verts_smoothness_loss = torch.sum(
        (hand_verts_ori[-1:] - hand_verts[-1:]) ** 2, dim=-1 
      ).mean()

      ## shape prior loss ##
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)

      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])

      transl_pred_loss = F.mse_loss(transl_var_ori, transl_var)

      loss = hand_verts_smoothness_loss * 0.1 + proj_loss + shape_prior_loss * 0.0001 + pose_prior_loss * 0.0001  + joints_smoothness_loss # + transl_pred_loss
      
      # joints_pred_loss = torch.sum(
      #   (hand_joints - joints) ** 2, dim=-1
      # ).mean() # theta var #
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # # attaction_loss = attaction_loss
      # # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :])
      
      
      # opt.zero_grad()
      # pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      # shape_prior_loss = torch.mean(beta_var**2)
      # pose_prior_loss = torch.mean(theta_var**2)
      # joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      # if not use_pca:
      #   loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.5 + shape_prior_loss * 0.001 + pose_prior_loss * 0.1 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tProjection Loss: {}'.format(proj_loss.item())) # hand_verts_smoothness_loss
      print('\thand_verts_smoothness_loss: {}'.format(hand_verts_smoothness_loss.item())) #
      # transl_pred_loss
      print('\ttransl_pred_loss: {}'.format(transl_pred_loss.item())) #
      # print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      # print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      # # print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      # print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))

      # beta_var, rot_var, theta_var, transl_var #
  
  return hand_verts.detach().cpu().numpy(), hand_verts_ori.detach().cpu().numpy(), hand_joints.detach().cpu().numpy(),  beta_var.detach(), rot_var.detach(), theta_var.detach(), transl_var.detach()


def get_attraction_multi_frames(beta_var, rot_var, theta_var, transl_var, tot_base_pts_trans, tot_base_pts_normals, selected_frame=11):
   
  nn_hand_params = theta_var.size(-1) # hand params
  use_pca = True if nn_hand_params < 45 else False
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()


  # anchor_load_driver, masking_load_driver #
  inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
  palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
  hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
  # self.hand_palm_vertex_mask for hand palm mask #
  hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool().cuda() ## the mask for hand palm to get hand anchors #
      
  
  ## tot base pts 
  tot_base_pts_trans_disp = torch.sum(
    (tot_base_pts_trans[1:, :, :] - tot_base_pts_trans[:-1, :, :]) ** 2, dim=-1 # (nf - 1) x nn_base_pts displacement 
  )
  ### tot base pts trans disp ###
  tot_base_pts_trans_disp = torch.sqrt(tot_base_pts_trans_disp).mean(dim=-1) # (nf - 1)
  # tot_base_pts_trans_disp_mov_thres = 1e-20
  tot_base_pts_trans_disp_mov_thres = 3e-4
  tot_base_pts_trans_disp_mask = tot_base_pts_trans_disp >= tot_base_pts_trans_disp_mov_thres
  tot_base_pts_trans_disp_mask = torch.cat(
    [tot_base_pts_trans_disp_mask, tot_base_pts_trans_disp_mask[-1:]], dim=0
  )
  
  # attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  

  # nn_base_pts_trans: nn_frames x nn_pts x 3 # 
  # nn_frames = rot_var.size(0)

  num_iters = 1000
  learning_rate = 0.01

  
  # rot_var = rot_var[: selected_frame + 1]
  # theta_var = theta_var[: selected_frame + 1]
  # transl_var = transl_var[: selected_frame + 1]
  # tot_base_pts_trans = tot_base_pts_trans[: selected_frame + 1]
  # tot_base_pts_normals = tot_base_pts_normals[: selected_frame + 1]

  rot_var_ori = torch.randn_like(rot_var)
  theta_var_ori = torch.randn_like(theta_var)
  beta_var_ori = torch.randn_like(beta_var)
  transl_var_ori = torch.randn_like(transl_var)

  rot_var_ori.data[:] = rot_var.data[:].clone()
  theta_var_ori.data[:] = theta_var.data[:].clone()
  beta_var_ori.data[:] = beta_var.data[:].clone()
  transl_var_ori.data[:] = transl_var.data[:].clone()

  rot_var = rot_var_ori
  theta_var = theta_var_ori
  beta_var = beta_var_ori
  transl_var = transl_var_ori

  nn_frames = rot_var.size(0)
  window_size = nn_frames

  # mano_layer
  hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
      beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
  hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
  hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001

  # # rhand_anchors: nnframes x nnanchors x 3 #
  # rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))
  
  # dist_rhand_anchors_base_pts = torch.sum(
  #   (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
  # )
  # minn_dist_rhand_anchors_base_pts, minn_dist_rhand_anchors_base_pts_idxes = torch.min(
  #   dist_rhand_anchors_base_pts, dim=-1 ## nn_frames x nn_anchors #
  # )

  beta_var.requires_grad_()
  rot_var.requires_grad_()
  theta_var.requires_grad_()
  transl_var.requires_grad_()

  

  opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=learning_rate)

  # rot_var_ori = rot_var.detach().clone()
  # transl_var_
  
  nn_frames = theta_var.size(0)

  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)



  # if with_ctx_mask:
  #   # obj_verts_trans, obj_faces
  #   # tot_penetration_masks_bf_contact_opt: nn_frames x nn_vert here for penetration masks ##
  #   tot_penetration_masks_bf_contact_opt = get_penetration_masks(obj_verts_trans, obj_faces, hand_verts)
  #   tot_penetration_masks_bf_contact_opt_frame = tot_penetration_masks_bf_contact_opt.float().sum(dim=-1) > 0.5 ### 
  #   tot_penetration_masks_bf_contact_opt_frame_nmask = (1. - tot_penetration_masks_bf_contact_opt_frame.float()).bool() ### nn_frames x nn_verts here
  
  # tot_penetration_masks_bf_contact_opt_nmask_frame
  
  
  window_size = hand_verts.size(0)
  # if with_contact_opt:
  # num_iters = 2000
  num_iters = 1000 # seq 77 # if with contact opt #
  # num_iters = 500 # seq 77
  ori_theta_var = theta_var.detach().clone()
  
  # tot_base_pts_trans # nf x nn_base_pts x 3
  disp_base_pts_trans = tot_base_pts_trans[1:] - tot_base_pts_trans[:-1] # (nf - 1) x nn_base_pts x 3
  disp_base_pts_trans = torch.cat( # nf x nn_base_pts x 3 
    [disp_base_pts_trans, disp_base_pts_trans[-1:]], dim=0
  )
  
  rhand_anchors = recover_anchor_batch(hand_verts.detach(), face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

  dist_joints_to_base_pts = torch.sum(
    (rhand_anchors.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nnjoints x nnbasepts #
  )
  
  nn_base_pts = dist_joints_to_base_pts.size(-1)
  nn_joints = dist_joints_to_base_pts.size(1)
  
  dist_joints_to_base_pts = torch.sqrt(dist_joints_to_base_pts) # nf x nnjoints x nnbasepts #
  minn_dist, minn_dist_idx = torch.min(dist_joints_to_base_pts, dim=-1) # nf x nnjoints #
  
  nk_contact_pts = 2
  minn_dist[:, :-5] = 1e9
  # minn_topk_dist, minn_topk_idx = torch.topk(minn_dist, k=nk_contact_pts, largest=False) # 
  # joints_idx_rng_exp = torch.arange(nn_joints).unsqueeze(0).cuda() == 
  # minn_topk_mask = torch.zeros_like(minn_dist)
  # minn_topk_mask[minn_topk_idx] = 1. # nf x nnjoints #
  # minn_topk_mask[:, -5: -3] = 1.
  basepts_idx_range = torch.arange(nn_base_pts).unsqueeze(0).unsqueeze(0).cuda()
  minn_dist_mask = basepts_idx_range == minn_dist_idx.unsqueeze(-1) # nf x nnjoints x nnbasepts
  # for seq 101
  # minn_dist_mask[31:, -5, :] = minn_dist_mask[30: 31, -5, :]
  minn_dist_mask = minn_dist_mask.float()
  
  # 
  # for seq 47
  # if cat_nm in ["Scissors"]:
  minn_dist_mask[selected_frame:] = minn_dist_mask[selected_frame: selected_frame + 1, :, :]
    # minn_dist_mask[:11] = False
  
  attraction_mask_new = (tot_base_pts_trans_disp_mask.float().unsqueeze(-1).unsqueeze(-1) + minn_dist_mask.float()) > 1.5
  
  
  
  # # joints: nf x nn_jts_pts x 3; nf x nn_base_pts x 3 
  dist_joints_to_base_pts_trans = torch.sum(
    (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1 # nf x nn_jts_pts x nn_base_pts
  )
  minn_dist_joints_to_base_pts, minn_dist_idxes = torch.min(dist_joints_to_base_pts_trans, dim=-1) # nf x nn_jts_pts # nf x nn_jts_pts # 
  # nearest_base_normals = model_util.batched_index_select_ours(tot_base_normals_trans, indices=minn_dist_idxes, dim=1) # nf x nn_base_pts x 3 --> nf x nn_jts_pts x 3 # # nf x nn_jts_pts x 3 #
  # nearest_base_pts_trans = model_util.batched_index_select_ours(disp_base_pts_trans, indices=minn_dist_idxes, dim=1) # nf x nn_jts_ts x 3 #
  # dot_nearest_base_normals_trans = torch.sum(
  #   nearest_base_normals * nearest_base_pts_trans, dim=-1 # nf x nn_jts 
  # )
  # trans_normals_mask = dot_nearest_base_normals_trans < 0. # nf x nn_jts # nf x nn_jts #
  nearest_dist = torch.sqrt(minn_dist_joints_to_base_pts)
  
  dist_thres = 0.005
  dist_thres = 0.01
  # dist_thres
  nearest_dist_mask = nearest_dist < dist_thres # hoi seq
  ## use selected frmae to set the nearest dist mask #
  nearest_dist_mask[selected_frame:] = nearest_dist_mask[selected_frame: selected_frame + 1]
  # nearest_dist_mask = nearest_dist < 0.005 # hoi seq
  

  
  # nearest_dist_mask = nearest_dist < 0.1
  k_attr = 100.
  joint_attraction_k = torch.exp(-1. * k_attr * nearest_dist)
  # attraction_mask_new_new = (attraction_mask_new.float() + trans_normals_mask.float().unsqueeze(-1) + nearest_dist_mask.float().unsqueeze(-1)) > 2.5
  
  joint_attraction_k = torch.ones_like(joint_attraction_k)
  attraction_mask_new_new = (attraction_mask_new.float() + nearest_dist_mask.float().unsqueeze(-1)) > 1.5
  

  # if cat_nm in ["ToyCar", "Pliers", "Bottle", "Mug"]:
  anchor_masks = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24]
  anchor_nmasks = [iid for iid in range(attraction_mask_new_new.size(1)) if iid not in anchor_masks]
  anchor_nmasks = torch.tensor(anchor_nmasks, dtype=torch.long).cuda()
  attraction_mask_new_new[:, anchor_nmasks, :] = False
  
  # # for seq 47
  # elif cat_nm in ["Scissors"]:
  #   anchor_masks = [2, 3, 4, 15, 16, 17, 22, 23, 24]
  #   anchor_nmasks = [iid for iid in range(attraction_mask_new_new.size(1)) if iid not in anchor_masks]
  #   anchor_nmasks = torch.tensor(anchor_nmasks, dtype=torch.long).cuda()
  #   attraction_mask_new_new[:, anchor_nmasks, :] = False
  
  # anchor_masks = torch.array([2, 3, 4, 15, 16, 17, 22, 23, 24], dtype=torch.long).cuda()
  # anchor_masks = torch.arange(attraction_mask_new_new.size(1)).unsqueeze(0).unsqueeze(-1).cuda() != 
  
  
  transl_var_ori = transl_var.clone().detach()
  # transl_var, theta_var, rot_var, beta_var # 
  # opt = optim.Adam([rot_var, transl_var, theta_var], lr=learning_rate)
  # opt = optim.Adam([transl_var, theta_var], lr=learning_rate)
  opt = optim.Adam([transl_var, theta_var, rot_var], lr=learning_rate)
  # opt = optim.Adam([theta_var, rot_var], lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=num_iters, gamma=0.5)
  for i in range(num_iters):
      opt.zero_grad()
      # mano_layer
      hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
          beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
      hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
      hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
      
      # joints_pred_loss = torch.sum(
      #   (hand_joints - joints) ** 2, dim=-1
      # ).mean()
      
      # 
      rhand_anchors = recover_anchor_batch(hand_verts, face_vertex_index, anchor_weight.unsqueeze(0).repeat(window_size, 1, 1))

      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # ) # nf x nnb x 3 ---- nf x nnj x 1 x 3 
      dist_joints_to_base_pts_sqr = torch.sum(
          (rhand_anchors.unsqueeze(2) - tot_base_pts_trans.unsqueeze(1)) ** 2, dim=-1
      )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # seq 80
      # attaction_loss = torch.mean(attaction_loss[46:, -5:-3, :] * minn_dist_mask[46:, -5:-3, :]) + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # seq 70
      # attaction_loss = torch.mean(attaction_loss[10:, -5:-3, :] * minn_dist_mask[10:, -5:-3, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      # new version relying on new mask #
      # attaction_loss = torch.mean(attaction_loss[:, -5:-3, :] * attraction_mask_new[:, -5:-3, :])
      ### original version ###
      # attaction_loss = torch.mean(attaction_loss[20:, -3:, :] * attraction_mask_new[20:, -3:, :])
      
      # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1))
      
      
      
      attaction_loss = torch.mean(attaction_loss[:, :, :] * attraction_mask_new_new[:, :, :] * joint_attraction_k[:, :].unsqueeze(-1))
      
      
      # seq mug
      # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
      
      
      # opt.zero_grad()
      pose_smoothness_loss = F.mse_loss(theta_var.view(nn_frames, -1)[1:], theta_var.view(nn_frames, -1)[:-1])
      # joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor.to(device))
      shape_prior_loss = torch.mean(beta_var**2)
      pose_prior_loss = torch.mean(theta_var**2)
      joints_smoothness_loss = F.mse_loss(hand_joints.view(nn_frames, -1, 3)[1:], hand_joints.view(nn_frames, -1, 3)[:-1])
      # =0.05
      # # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001 + joints_smoothness_loss * 100.
      # loss = joints_pred_loss * 30 + pose_smoothness_loss * 0.000001 + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001 + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.05 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + joints_smoothness_loss * 200.
      
      # loss = joints_pred_loss * 5000 + pose_smoothness_loss * 0.03 + shape_prior_loss * 0.0002 + pose_prior_loss * 0.0005 # + attaction_loss * 10000 # + joints_smoothness_loss * 200.
      
      theta_smoothness_loss = F.mse_loss(theta_var, ori_theta_var)

      transl_smoothness_loss = F.mse_loss(transl_var_ori, transl_var)
      # loss = attaction_loss * 1000. + theta_smoothness_loss * 0.00001
      
      # attraction loss, joint prediction loss, joints smoothness loss #
      # loss = attaction_loss * 1000. + joints_pred_loss 
      ### general ###
      # loss = attaction_loss * 1000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
      
      # tune for seq 140
      # loss = attaction_loss * 10000. + joints_pred_loss * 0.0001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
      # # ToyCar
      # loss = attaction_loss * 10000. + joints_pred_loss * 0.01 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
      
      # if cat_nm in ["Scissors"]:

      loss = transl_smoothness_loss * 0.05 + attaction_loss * 10000.  + joints_smoothness_loss * 0.005
          
      # loss = joints_pred_loss * 20 + joints_smoothness_loss * 200. + shape_prior_loss * 0.0001 + pose_prior_loss * 0.00001
      # loss = joints_pred_loss * 30  # + shape_prior_loss * 0.001 + pose_prior_loss * 0.0001
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5 + attaction_loss * 0.001 + joints_smoothness_loss * 1.0
      
      # loss = joints_pred_loss * 20 + pose_smoothness_loss * 0.5  + attaction_loss * 2000000.  # + joints_smoothness_loss * 10.0
      
      # loss = joints_pred_loss * 20 #  + pose_smoothness_loss * 0.5 + attaction_loss * 100. + joints_smoothness_loss * 10.0
      # loss = joints_pred_loss * 30 + attaction_loss * 0.001
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      scheduler.step()
      
      print('Iter {}: {}'.format(i, loss.item()), flush=True)
      print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
      print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
      print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
      # print('\tJoints Prediction Loss: {}'.format(joints_pred_loss.item()))
      print('\tAttraction Loss: {}'.format(attaction_loss.item()))
      print('\tJoint Smoothness Loss: {}'.format(joints_smoothness_loss.item()))
      # theta_smoothness_loss
      print('\tTheta Smoothness Loss: {}'.format(theta_smoothness_loss.item()))
      # transl_smoothness_loss
      print('\tTransl Smoothness Loss: {}'.format(transl_smoothness_loss.item()))
  


  return hand_verts.detach().cpu().numpy()

   

# do the blindera interpolation for the output vertices; 
# rot, trans --> the original values; 
# theta -> how to do the interpolation here #
# linear interpolations? 
def interpolate_hand_poses(rot_var, trans_var, beta_var, theta_var, selected_hand_rot, selected_hand_trans, selected_hand_theta, selected_frame_idx=21):
  interpolated_hand_theta = theta_var[:selected_frame_idx + 1] # selected_frame_idx x theta_dim here for the interpolated thetas # 
  interpolated_hand_rot = rot_var[:selected_frame_idx + 1] # rot_var
  # and another -> can we keep the rot and trans no change but only chagne theta here? 
  interpolated_hand_trans = trans_var[:selected_frame_idx + 1]
  
  interpolated_hand_theta[selected_frame_idx: selected_frame_idx + 1] = selected_hand_theta
  tot_delta_hand_theta = selected_hand_theta[0] - interpolated_hand_theta[0] # theta_dim for delta here
  delta_hand_theta = tot_delta_hand_theta / float(selected_frame_idx) # theta_dim for delta 
  for i_fr in range(1, selected_frame_idx): # selected frmae idx and interpolated hand thetas #
    interpolated_hand_theta[i_fr] = interpolated_hand_theta[0] + i_fr * delta_hand_theta # for the delta_hand_thetas
  
  # but still no planning here #
  nn_hand_params = theta_var.size(-1) # hand params
  use_pca = True if nn_hand_params < 45 else False
  # setup MANO layer
  mano_path = "/data1/xueyi/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()

  nn_frames = interpolated_hand_theta.size(0)
  hand_verts, hand_joints = mano_layer(torch.cat([interpolated_hand_rot, interpolated_hand_theta], dim=-1),
      beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), interpolated_hand_trans)
  hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
  hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001

  return hand_verts.detach().cpu().numpy(), interpolated_hand_rot.detach(), interpolated_hand_theta.detach(), interpolated_hand_trans.detach(), beta_var.detach()
# to the selected_frame_idx 


## optimization ##
if __name__=='__main__':
  
  tot_seq_nn = 246
  tot_seq_nn = 18
  
  
  st_idx = 0
  ed_idx = 44

  st_idx = 8
  ed_idx = 9
  
  ### Articulated ####
  ## Bucket ##
  cat_nm = "Bucket"
  # # /data2/xueyi/eval_save/HOI_Arti/Bucket/predicted_infos_seq_36_seed_88_tag_rep_res_jts_hoi4d_arti_bucket_t_400_.npy
  # st_idx = 41
  # ed_idx = 42
  # st_idx = 38
  # ed_idx = 39
  ## Bucket ##

  ## Pliers ##
  # /data2/xueyi/eval_save/HOI_Arti/Pliers/predicted_infos_seq_3_seed_44_tag_rep_res_jts_hoi4d_pliers_t_300_st_idx_30_.npy
  # cat_nm = "Pliers"
  # st_idx = 1
  # ed_idx = 4
  
  ## Scissors ##
  cat_nm = "Scissors" # scissors #
  st_idx = 0
  ed_idx = 94

  st_idx = 27
  ed_idx = 28

  st_idx = 11
  ed_idx = 12

  st_idx = 10
  ed_idx = 11

  st_idx = 24
  ed_idx = 25

  st_idx = 0
  ed_idx = 205

  # st_idx = 27
  # ed_idx = 28

  # st_idx = 30
  # ed_idx = 31

  # 23
  # st_idx = 23
  # ed_idx = 24
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_400_" 
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_50_"
  # /data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_11_seed_0_tag_rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_30_.npy
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_30_"
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_0_"
  # /data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_10_seed_11_tag_rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_30_.npy
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_30_"
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_400_"
  test_tag = "rep_only_real_mean_t_200_"
  # /data2/xueyi/eval_save/GRAB/predicted_infos_seq_1_seed_110_tag_rep_res_jts_grab_t_200_.npy
  test_tag = "rep_res_jts_grab_t_200_" # rep_res_jts_grab_t_200_ as the test tag #
  # test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_50_"
  ## Scissors ##
  # test_tag = "jts_hoi4d_arti_scissors_t_400_st_idx_0_"
  pred_infos_sv_folder = f"/data2/xueyi/eval_save/HOI_Arti/{cat_nm}"
  pred_infos_sv_folder = f"/data1/xueyi/mdm/eval_save/"
  pred_infos_sv_folder = f"/data2/xueyi/eval_save/GRAB"

  cat_nm = "GRAB"
  st_idx = 0
  ed_idx = 205

  st_idx = 2
  ed_idx = 205
  st_idx = 80
  ed_idx = 81
  st_idx = 5
  ed_idx = 6

  st_idx = 0
  ed_idx = 246
  test_tag = "jts_grab_t_400_scale_obj_"
  test_tag = "jts_grab_t_500_scale_obj_"
  test_tag = "jts_grab_t_700_scale_obj_"
  # test_tag = "jts_grab_t_400_scale_obj_1_"
  # test_tag = "rep_jts_grab_t_700_scale_obj_"
  # test_tag = "rep_jts_grab_t_400_scale_obj_1_"
  test_tag = "jts_grab_t_700_scale_obj_2_"
  test_tag = "jts_spatial_grab_t_200_test_"
  # st_idx = 1
  # ed_idx = 2
  # tot_rnd_seeds = range(0, 33, 11)
  tot_rnd_seeds = range(0, 121, 11)
  # tot_rnd_seeds = [55]
  # tot_rnd_seeds = range(77, 88, 11)
  # tot_rnd_seeds = range(33, 121, 11) # 
  ### Articulated ####


  #### Rigid #####
  # st_idx = 0
  # ed_idx = 4
  # st_idx = 2
  # ed_idx = 4
  # # st_idx = 0
  # # ed_idx = 2
  # # /data2/xueyi/eval_save/HOI_Rigid/ToyCar/predicted_infos_seq_3_seed_99_tag_rep_res_jts_hoi4d_toycar_t_300_st_idx_0_.npy
  # # cat_nm = "ToyCar"
  # # tot_rnd_seeds = range(0, 122, 11)
  # # test_tag = "rep_res_jts_hoi4d_toycar_t_300_st_idx_0_" # rep res jts
  # st_idx = 1
  # ed_idx = 54
  # # cat_nm = "Bottle"
  # # tot_rnd_seeds = range(0, 122, 11)
  # # pred_infos_sv_folder = f"/data2/xueyi/eval_save/HOI_Rigid/{cat_nm}"
  # # test_tag = "rep_res_jts_hoi4d_bottle_t_300_st_idx_0_" # rep res jts
  # st_idx = 0
  # ed_idx = 249
  # cat_nm = "Mug"
  # test_tag = "rep_res_jts_hoi4d_mug_t_300_st_idx_0_" # rep res jts
  # st_idx = 0
  # ed_idx = 249
  # cat_nm = "Bowl"
  # pred_infos_sv_folder = f"/data2/xueyi/eval_save/HOI_Rigid/{cat_nm}"
  # os.makedirs(pred_infos_sv_folder, exist_ok=True)
  # test_tag = "rep_res_jts_hoi4d_bowl_t_300_st_idx_0_"
  # st_idx = 0
  # ed_idx = 249
  # cat_nm = "Knife"
  # test_tag = "rep_res_jts_hoi4d_knife_t_300_st_idx_0_"
  # st_idx = 0
  # ed_idx = 60
  # cat_nm = "Chair" 
  # # rep_res_jts_hoi4d_chair_t_300_st_idx_0_
  # test_tag = "rep_res_jts_hoi4d_chair_t_300_st_idx_0_"
  # pred_infos_sv_folder = f"/data2/xueyi/eval_save/HOI_Rigid/{cat_nm}"
  # os.makedirs(pred_infos_sv_folder, exist_ok=True)
  # ##### Rigid #####


  # /data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_8_seed_66_tag_rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_0_.npy
  # /data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_6_seed_66_tag_jts_rep_hoi4d_arti_t_300_.npy
  ### the test seq ###
  # for i_test_seq in range(st_idx, ed_idx):
  for i_test_seq in range(st_idx, ed_idx): # 
    for seed in tot_rnd_seeds:

      pred_joints_info_nm = "predicted_infos.npy"

      pred_joints_info_nm = f"predicted_infos_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}.npy"

      pred_joints_info_fn = os.path.join(pred_infos_sv_folder, pred_joints_info_nm) #
      
      
      
      # not existing, continue for remaining sequences #
      if not os.path.exists(pred_joints_info_fn):
        continue
      
      data = np.load(pred_joints_info_fn, allow_pickle=True).item()
      targets = data['targets'] # ## targets -> targets and outputs ##
      outputs = data['outputs'] #  
      tot_base_pts = data["tot_base_pts"][0] # total base pts, total base normals #
      tot_base_normals = data['tot_base_normals'][0] # nn_base_normals #
      
      
      print(f"outputs: {outputs.shape}")
      # outputs: bsz x ws x (params_dim) #
      if len(outputs.shape) == 2:
        rhand_transl, rhand_rot, rhand_theta, rhand_betas = outputs[..., :3], outputs[..., 3:6], outputs[..., 6: 30], outputs[..., 30:]
        rhand_transl = torch.from_numpy(rhand_transl).float().cuda()
        rhand_rot = torch.from_numpy(rhand_rot).float().cuda()
        rhand_theta = torch.from_numpy(rhand_theta).float().cuda()
        rhand_betas = torch.from_numpy(rhand_betas).float().cuda()
        rhand_verts, rhand_joints = get_rhand_joints_verts_fr_params(rhand_transl, rhand_rot, rhand_theta, rhand_betas)
        
        
        
        tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
        tot_obj_transl = data['tot_obj_transl'][0]
        print(f"tot_obj_rot: {tot_obj_rot.shape}, tot_obj_transl: {tot_obj_transl.shape}")
        
        if len(tot_base_pts.shape) == 2:
          # numpy array #
          tot_base_pts_trans = np.matmul(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
          tot_base_pts = np.matmul(tot_base_pts, tot_obj_rot[0]) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])[0] 
          
          tot_base_normals_trans = np.matmul( # # 
            tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), tot_obj_rot
          ) 
        else:
          # numpy array #
          tot_base_pts_trans = np.matmul(tot_base_pts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
          tot_base_pts = np.matmul(tot_base_pts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
          
          tot_base_normals_trans = np.matmul( # # 
            tot_base_normals, tot_obj_rot
          ) 
          
        # rhand_verts, rhand_joints = get_optimized_hand_fr_joints_v4_anchors_fr_params(rhand_betas, rhand_rot, rhand_theta, rhand_transl, tot_base_pts_trans)
        # reconstruct data bundle
        optimized_sv_infos = {
          'optimized_out_hand_verts': rhand_verts,
          'optimized_out_hand_joints': rhand_joints,
          'tot_base_pts_trans': tot_base_pts_trans,
          'optimized_out_hand_verts_before_contact_opt': rhand_verts,
          'optimized_out_hand_joints_before_contact_opt': rhand_joints,
          'outputs': rhand_joints # .detach().cpu().numpy(),
        }
        pred_infos_sv_folder = "/data1/xueyi/mdm/eval_save/"
        # pred_joints_info_nm = f"predicted_infos_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}.npy"
        # /data1/xueyi/mdm/eval_save/optimized_infos_sv_dict_seq_7_seed_77_tag_rep_only_real_mean_t_200_.npy
        # optimized_infos_sv_dict_seq_12_seed_77_tag_rep_only_real_mean_same_noise_hoi4d_t_200_.npy
        optimized_sv_infos_sv_fn_nm = f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}.npy"
        optimized_sv_infos_sv_fn = os.path.join(pred_infos_sv_folder, optimized_sv_infos_sv_fn_nm)
        np.save(optimized_sv_infos_sv_fn, optimized_sv_infos)
        print(f"optimized infos saved to {optimized_sv_infos_sv_fn}")
        exit(0)
        
      
      ### obj verts ###
      obj_verts = data['obj_verts'] # perhaps base normals --> base normals #
      
      # outputs = targets #
      # outputs = targets #
      
      # pred_infos_sv_folder = "/data1/xueyi/mdm/eval_save/"
      # pred_infos_sv_folder = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512" # all noise #
      
      # pred_infos_sv_folder = f"/data2/xueyi/eval_save/HOI_Arti/{cat_nm}"
      
      # pred_joints_info_nm = "predicted_infos.npy"
      # /data1/xueyi/mdm/eval_save/predicted_infos_seq_300_seed_144_tag_rep_only_mean_shape_hoi4d_t_400_.npy
      
      pred_joints_info_nm = f"predicted_infos_seq_300_seed_{seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy"
      
      pred_joints_info_nm = f"predicted_infos_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}.npy"
      # pred_joints_info_nm = "predicted_infos.npy"
      
      
      # pred_joints_info_nm = "predicted_infos_hoi_seed_77_jts_only_t_300.npy"
      # pred_infos_sv_folder = "/data1/xueyi/mdm/eval_save/"
      # pred_joints_info_nm = "predicted_infos_seq_1_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
      # pred_joints_info_nm = "predicted_infos_seq_2_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
      # pred_joints_info_nm = "predicted_infos_seq_36_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
      ### conditions, conditions, conditions ##
      # pred_joints_info_nm = "predicted_infos_seq_80_wtrans.npy"
      # pred_joints_info_nm = "predicted_infos_80_wtrans_rep.npy"
      # pred_joints_info_nm = "predicted_infos_seq_80_seed_31_tag_jts_only.npy"
      # pred_joints_info_nm = "predicted_infos_seq_77.npy"
      pred_joints_info_fn = os.path.join(pred_infos_sv_folder, pred_joints_info_nm)
      data = np.load(pred_joints_info_fn, allow_pickle=True).item()
      # outputs = targets # outputs # targets # 
      
      tot_obj_rot = data['tot_obj_rot'][0] # ws x 3 x 3 ---> obj_rot; #
      tot_obj_transl = data['tot_obj_transl'][0]
      print(f"tot_obj_rot: {tot_obj_rot.shape}, tot_obj_transl: {tot_obj_transl.shape}")
      
      if len(tot_base_pts.shape) == 2:
        # numpy array # # tot base pts #
        tot_base_pts_trans = np.matmul(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
        tot_base_pts = np.matmul(tot_base_pts, tot_obj_rot[0]) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])[0] 
        
        tot_base_normals_trans = np.matmul( # # 
          tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), tot_obj_rot
        ) 
      else:
        # numpy array #
        tot_base_pts_trans = np.matmul(tot_base_pts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
        tot_base_pts = np.matmul(tot_base_pts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
        
        tot_base_normals_trans = np.matmul( # # 
          tot_base_normals, tot_obj_rot
        ) 
      
      # # numpy array #
      # tot_base_pts_trans = np.matmul(tot_base_pts.reshape(1, tot_base_pts.shape[0], 3), tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])
      # tot_base_pts = np.matmul(tot_base_pts, tot_obj_rot[0]) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1])[0] 
      
      # tot_base_normals_trans = np.matmul( # # 
      #   tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), tot_obj_rot
      # ) 
      
      # if pred_joints_info_nm != "predicted_infos.npy":
      outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # ws x nn_verts x 3 #
      
      targets = np.matmul(targets, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # ws x nn_verts x 3 #
      # denoise relative positions 
      print(f"tot_base_pts: {tot_base_pts.shape}")
      
      
      #### obj_verts_trans, obj_faces ####
      ## obj_verts, tot_obj_rot ##
      obj_verts_trans = np.matmul(obj_verts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) 
      obj_faces = data['obj_faces'] #
      print(f"obj_verts_trans: {obj_verts_trans.shape}, obj_faces: {obj_faces.shape}")
      
      # outputs = targets ## targets 
      with_contact_opt = False
      with_contact_opt = True
      
      with_proj = False
      # with_proj = True
      
      with_params_smoothing = False
      # with_params_smoothing = True
      
      with_ctx_mask = True
      
      # optimized_out_hand_verts, optimized_out_hand_joints, bf_ct_verts, bf_ct_joints = get_optimized_hand_fr_joints_v4(outputs, tot_base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=with_contact_opt)
      
      
      # nn_frames x 3; nn_frames x 45; 10
      # transl_var, theta_var, rot_var, beta_var # 
      # optimized out hand verts # 
      # rt_vars: transl_var, theta_var, rot_var, beta_var # 
      ## get hand fr joints v4 anchors ##
      #### optimized hand fr joints v4 anchors ####
      nn_hand_params = 24
      # nn_hand_params = 45
      
      dist_thres = 0.005
      
      tot_dist_thres = [0.005, 0.01, 0.02, 0.05, 0.1]
      # ToyCar
      tot_dist_thres = [0.001, 0.002, 0.005, 0.01, 0.02, 0.1]
      # tot_dist_thres = [0.001, 0.002, 0.005]

      if cat_nm in ["Scissors"]:
        tot_dist_thres = [0.005, 0.01]
        tot_dist_thres = [0.005, 0.01, 0.02]
        # tot dist thresholds #
        tot_dist_thres = [0.005, 0.01, 0.02, 0.05, 0.1]
        # for one frame tests 
        # tot_dist_thres = [0.005]

      # tot_dist_thres = [0.005]
      #### with_contact_opt = True #### with_ctx_
      # tot_dist_thres = [0.01, 0.02, 0.05, 0.1]
      # distances between 
      ###
      tot_dist_thres = [0.001]
      
      # with_ctx_mask
      tot_with_proj = [True, False]
      
      tot_with_proj = [ False]

      # ToyCar
      with_params_smoothing = True
      
      for with_proj in tot_with_proj:
        for dist_thres in tot_dist_thres:
          with_proj = False
          # with_proj = True

          # ToycAr
        #   with_proj = True
          if cat_nm in ["Scissors", "Chair"]: # no proj
            with_proj = False
            # with_proj = True
            # with_params_smoothing = False
          # with_contact_opt = False
          with_contact_opt = True
          with_ctx_mask = False
          # optimized_out_hand_verts, optimized_out_hand_joints, bf_ct_verts, bf_ct_joints, transl_var, theta_var, rot_var, beta_var = get_optimized_hand_fr_joints_v4_anchors(outputs, tot_base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=with_contact_opt, nn_hand_params=nn_hand_params, rt_vars=True, with_proj=with_proj, obj_verts_trans=obj_verts_trans, obj_faces=obj_faces, with_params_smoothing=with_params_smoothing, dist_thres=dist_thres)

          # get_optimized_hand_fr_joints_v4_anchors # for the anchors # # # #
          
          bf_ct_optimized_dict, bf_proj_optimized_dict, optimized_dict = get_optimized_hand_fr_joints_v4_anchors(outputs, tot_base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=with_contact_opt, nn_hand_params=nn_hand_params, rt_vars=True, with_proj=with_proj, obj_verts_trans=obj_verts_trans, obj_faces=obj_faces, with_params_smoothing=with_params_smoothing, dist_thres=dist_thres, with_ctx_mask=with_ctx_mask)

          


          # bf_ct_optimized_dict, bf_proj_optimized_dict, optimized_dict
          optimized_sv_infos = {}
          optimized_sv_infos.update(bf_ct_optimized_dict)
          optimized_sv_infos.update(bf_proj_optimized_dict)
          optimized_sv_infos.update(optimized_dict)
          optimized_sv_infos.update(
            {
              'tot_base_pts_trans': tot_base_pts_trans,
              'tot_base_normals_trans': tot_base_normals_trans
            }
          )
          
          
          optimized_sv_infos_sv_fn_nm = f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}.npy"
          optimized_sv_infos_sv_fn = os.path.join(pred_infos_sv_folder, optimized_sv_infos_sv_fn_nm)
          np.save(optimized_sv_infos_sv_fn, optimized_sv_infos)
          print(f"optimized infos saved to {optimized_sv_infos_sv_fn}")
        

