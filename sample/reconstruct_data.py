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
from psbody.mesh import Mesh
from manopth.manolayer import ManoLayer
# from dataloading import GRAB_Single_Frame, GRAB_Single_Frame_V6, GRAB_Single_Frame_V7, GRAB_Single_Frame_V8, GRAB_Single_Frame_V9, GRAB_Single_Frame_V9_Ours, GRAB_Single_Frame_V10
# use_trans_encoders
# from model import TemporalPointAE, TemporalPointAEV2, TemporalPointAEV5, TemporalPointAEV6, TemporalPointAEV7, TemporalPointAEV8, TemporalPointAEV9, TemporalPointAEV10, TemporalPointAEV4, TemporalPointAEV3_Real, TemporalPointAEV11, TemporalPointAEV12, TemporalPointAEV13, TemporalPointAEV14, TemporalPointAEV17, TemporalPointAEV19, TemporalPointAEV20, TemporalPointAEV21, TemporalPointAEV22, TemporalPointAEV23, TemporalPointAEV24, TemporalPointAEV25, TemporalPointAEV26
import trimesh
from utils import *
import utils
import utils.model_util as model_util
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
  mano_path = "/data1/sim/mano_models/mano/models"
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
  mano_path = "/data1/sim/mano_models/mano/models"
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
  mano_path = "/data1/sim/mano_models/mano/models"
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
  mano_path = "/data1/sim/mano_models/mano/models"
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
  
  dot_prod_base_pts_to_joints_with_normals = torch.sum(
    (joints.unsqueeze(-2) - tot_base_pts_trans.unsqueeze(1)) * tot_base_normals_trans.unsqueeze(1), dim=-1 # 
  )
  # dist_joints_to_base_pts[dot_prod_base_pts_to_joints_with_normals < 0.] = 1e9
  
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
  minn_dist_mask[:, -5:, :] = minn_dist_mask[30:31:, -5:, :] # set to the last frame mask #
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
      
      # dist_joints_to_base_pts_sqr = torch.sum(
      #     (hand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(1)) ** 2, dim=-1
      # )
      # attaction_loss = 0.5 * affinity_scores * dist_joints_to_base_pts_sqr
      # attaction_loss = 0.5 * dist_joints_to_base_pts_sqr
      # attaction_loss = attaction_loss
      # attaction_loss = torch.mean(attaction_loss[..., -5:, :] * minn_dist_mask[..., -5:, :])
      
      # attaction_loss = torch.mean(attaction_loss * attraction_mask)
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
      
  
  
  if with_contact_opt:
    num_iters = 2000
    # num_iters = 1000 # seq 77
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
        
        second_tips_idxes = [9, 12, 6, 3, 15]
        second_tips_idxes =torch.tensor(second_tips_idxes, dtype=torch.long).cuda()
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:, :] * attraction_mask_new_new[:, -5:, :] * joint_attraction_k[:, -5:].unsqueeze(-1)) # + torch.mean(attaction_loss[:, second_tips_idxes, :] * attraction_mask_new_new[:, second_tips_idxes, :] * joint_attraction_k[:, second_tips_idxes].unsqueeze(-1)) 
        
        
        # attaction_loss = torch.mean(attaction_loss[:, -5:-2, :] * attraction_mask_new_new[:, -5:-2, :]) # + torch.mean(attaction_loss[:, second_tips_idxes, :] * attraction_mask_new_new[:, second_tips_idxes, :] * joint_attraction_k[:, second_tips_idxes].unsqueeze(-1)) 
        
        # attraction_mask_new
        # attaction_loss = torch.mean(attaction_loss[:, -5:-2, :] * attraction_mask_new[:, -5:-2, :]) # + torch.mean(attaction_loss[:, second_tips_idxes, :] * attraction_mask_new_new[:, second_tips_idxes, :] * joint_attraction_k[:, second_tips_idxes].unsqueeze(-1)) 
        
        # seq mug
        # attaction_loss = torch.mean(attaction_loss[4:, -5:-4, :] * minn_dist_mask[4:, -5:-4, :]) # + torch.mean(attaction_loss[:40, -5:-3, :] * minn_dist_mask[:40, -5:-3, :])
        
        attaction_loss = torch.mean(attaction_loss[10:, -5:-2, :] * attraction_mask_new[10:, -5:-2, :]) # + torch.mean(attaction_loss[:, second_tips_idxes, :] * attraction_mask_new_new[:, second_tips_idxes, :] * joint_attraction_k[:, second_tips_idxes].unsqueeze(-1)) 
        
        
        
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
        
        # loss = attaction_loss * 1000. + joints_pred_loss 
        # loss = attaction_loss * 100000. + joints_pred_loss * 0.00000001 + joints_smoothness_loss * 0.5 # + pose_prior_loss * 0.00005  # + shape_prior_loss * 0.001 # + pose_smoothness_loss * 0.5
        loss = attaction_loss * 100000. + joints_pred_loss * 0.000 + joints_smoothness_loss * 0.000005 
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
  
  
  
  return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()




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
  mano_path = "/data1/sim/mano_models/mano/models"
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



## optimization ##
if __name__=='__main__':
  pred_infos_sv_folder = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
  # /data1/sim/mdm/eval_save/predicted_infos_seq_1_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy
  # pred_infos_sv_folder = "/data1/sim/mdm/eval_save/"
  pred_joints_info_nm = "predicted_infos.npy"
  # pred_joints_info_nm = "predicted_infos_hoi_seed_77_jts_only_t_300.npy"
  # pred_joints_info_nm = "predicted_infos_seq_1_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # pred_joints_info_nm = "predicted_infos_seq_2_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # pred_joints_info_nm = "predicted_infos_seq_36_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # # pred_joints_info_nm = "predicted_infos_seq_36_seed_77_tag_jts_only.npy"
  # pred_joints_info_nm = "predicted_infos_seq_80_wtrans.npy"
  # pred_joints_info_nm = "predicted_infos_80_wtrans_rep.npy"
  # pred_joints_info_nm = "predicted_infos_seq_70_seed_31_tag_jts_only.npy"
  # pred_joints_info_nm = "predicted_infos_seq_80_seed_31_tag_jts_only.npy"
  # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_17_seed_31_tag_jts_only.npy
  # pred_joints_info_nm = "predicted_infos_seq_87_seed_31_tag_jts_only.npy"
  # pred_joints_info_nm = "predicted_infos_seq_77.npy"
  
  # pred_joints_info_nm = "predicted_infos_seq_1_seed_31_tag_rep_only_real_sel_base_0.npy"
  # # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_1_seed_31_tag_rep_only_real_sel_base_mean.npy
  # pred_joints_info_nm = "predicted_infos_seq_1_seed_31_tag_rep_only_real_sel_base_mean.npy"
  # # pred_infos_sv_folder = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos.npy"
  # TODO: load the total data sequence and transform object shape using the loaded sample
  # TODO: please output hands and objects other than hands only in each frame #
  pred_joints_info_fn = os.path.join(pred_infos_sv_folder, pred_joints_info_nm)
  data = np.load(pred_joints_info_fn, allow_pickle=True).item()
  targets = data['targets'] # ## targets -> targets and outputs ##
  outputs = data['outputs'] # 
  tot_base_pts = data["tot_base_pts"][0]
  tot_base_normals = data['tot_base_normals'][0] # nn_base_normals #
  
  obj_verts = data['obj_verts']
  
  # outputs = targets
  
  pred_infos_sv_folder = "/data1/sim/mdm/eval_save/"
  pred_infos_sv_folder = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
  pred_joints_info_nm = "predicted_infos.npy"
  # pred_joints_info_nm = "predicted_infos_hoi_seed_77_jts_only_t_300.npy"
  # pred_infos_sv_folder = "/data1/sim/mdm/eval_save/"
  # pred_joints_info_nm = "predicted_infos_seq_1_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # pred_joints_info_nm = "predicted_infos_seq_2_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
  # pred_joints_info_nm = "predicted_infos_seq_36_seed_77_tag_rep_only_real_sel_base_mean_all_noise_.npy"
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
  
  # tot_base_normals_trans = np.matmul( # # 
  #   tot_base_normals.reshape(1, tot_base_normals.shape[0], 3), tot_obj_rot
  # ) 
  
  outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # ws x nn_verts x 3 #
  targets = np.matmul(targets, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # ws x nn_verts x 3 #
  # denoise relative positions 
  print(f"tot_base_pts: {tot_base_pts.shape}")
  
  # obj_verts_trans = np.matmul(obj_verts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) 
  
  
  # outputs = targets
  with_contact_opt = False
  with_contact_opt = True
  optimized_out_hand_verts, optimized_out_hand_joints = get_optimized_hand_fr_joints_v4(outputs, tot_base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=with_contact_opt)
  
  
  # 
  # predicted_joint_quants_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/pred_joint_quants.npy"
  # predicted_joint_quants = np.load(predicted_joint_quants_fn, allow_pickle=True).item()
  # predicted_joint_quants = predicted_joint_quants['dec_joint_quants']
  # print("predicted_joint_quants",  predicted_joint_quants.shape)
  # # print(predicted_joint_quants.keys()) 
  
  
  
  # # exit(0)
  
  # # gt
  # tot_gt_rhand_joints = data['tot_gt_rhand_joints'][0] # nf x nn_joints x 3 --> gt joints #
  # tot_gt_rhand_joints = np.matmul(tot_gt_rhand_joints, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) # ws x nn_verts x 3 #
  # optimized_out_hand_verts, optimized_out_hand_joints = get_optimized_hand_fr_joints_v5(outputs, tot_gt_rhand_joints, tot_base_pts, tot_base_pts_trans, predicted_joint_quants=predicted_joint_quants)
  
  
  
  # optimized_data_fn = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/optimized_joints.npy"
  # data = np.load(optimized_data_fn, allow_pickle=True).item()
  # outputs = data["optimized_joints"]
  
  # optimized_out_hand_verts, optimized_out_hand_joints = get_optimized_hand_fr_joints(outputs)
  # optimized_tar_hand_verts, optimized_tar_hand_joints = get_optimized_hand_fr_joints(targets)
  # optimized_out_hand_verts, optimized_out_hand_joints = get_optimized_hand_fr_joints_v2(outputs, tot_base_pts)
  
  optimized_sv_infos = {
    'optimized_out_hand_verts': optimized_out_hand_verts,
    'optimized_out_hand_joints': optimized_out_hand_joints,
    'tot_base_pts_trans': tot_base_pts_trans,
    # 'optimized_tar_hand_verts': optimized_tar_hand_verts,
    # 'optimized_tar_hand_joints': optimized_tar_hand_joints,
  }
  optimized_sv_infos_sv_fn = os.path.join(pred_infos_sv_folder, "optimized_infos_sv_dict.npy")
  np.save(optimized_sv_infos_sv_fn, optimized_sv_infos)
  print(f"optimized infos saved to {optimized_sv_infos_sv_fn}")
  

