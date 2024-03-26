import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from torch import optim, nn
import numpy as np
import os, argparse, copy, json
# import pickle as pkl
from scipy.spatial.transform import Rotation as R
# from psbody.mesh import Mesh
from manopth.manolayer import ManoLayer


import trimesh
from utils import *
import utils
import utils.model_util as model_util
from utils.anchor_utils import masking_load_driver, anchor_load_driver, recover_anchor_batch


from utils.parser_util import train_args



def get_optimized_hand_fr_joints(joints):
  joints = torch.from_numpy(joints).float().cuda()
  mano_path = "manopth/mano/models"
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


  
def get_optimized_hand_fr_joints_v4(joints, base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=False):
  joints = torch.from_numpy(joints).float().cuda()
  base_pts = torch.from_numpy(base_pts).float().cuda()
  
  tot_base_pts_trans = torch.from_numpy(tot_base_pts_trans).float().cuda()
  tot_base_normals_trans = torch.from_numpy(tot_base_normals_trans).float().cuda()
  ### start optimization ###
  # setup MANO layer
  mano_path = "manopth/mano/models"
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
  mano_path = "manopth/mano/models"
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
  mano_path = "manopth/mano/models"
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
  inpath = "assets" 
  fvi, aw, _, _ = anchor_load_driver(inpath)
  face_vertex_index = torch.from_numpy(fvi).long().cuda()
  anchor_weight = torch.from_numpy(aw).float().cuda()
  
  anchor_path = os.path.join("assets", "anchor")
  palm_path = os.path.join("assets", "hand_palm_full.txt")
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
    
    if cat_nm in ["ToyCar", "Pliers", "Bottle", "Mug", "Scissors"]:
      anchor_masks = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24]
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
        # ToyCar
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
  

def get_rhand_joints_verts_fr_params(rhand_transl, rhand_rot, rhand_theta, rhand_beta):
  # setup MANO layer
  mano_path = "manopth/mano/models"
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





## optimization ##
if __name__=='__main__':
    
    
  args = train_args()
  

  single_seq_path = args.single_seq_path
  
  case_idx = args.single_seq_path.split("/")[-2]
  case_idx = int(case_idx[4:]) ## get the case idx ## 
  tot_test_seq_idxes = range(case_idx, case_idx + 1, 1) ## get the case idx ## 
  cat_nm = args.single_seq_path.split("/")[-3] ## get the category name
  
  case_folder_nm =  f"data/hoi4d/{cat_nm}/case{case_idx}" 
  args.single_seq_path = single_seq_path
  args.cad_model_fn = f"obj_model.obj"
  args.corr_fn = f"data/hoi4d/{cat_nm}/case{case_idx}/merged_data.npy"

  ### save res dir ###
  if len(args.save_dir) > 0:
    args.save_dir = os.path.join(args.save_dir, f"{cat_nm}")
    os.makedirs(args.save_dir, exist_ok=True)
        
  test_tag = args.test_tag
            
  
  print(f"Reconstructing meshes for sequence: {single_seq_path}")
  
  
  
#   test_tag = "rep_res_jts_spatial_hoi4d_rigid_bowl_t_200_st_idx_0__sel_idx_0_thetadim_45_"


  tot_rnd_seeds = range(0, 121, 11)


  for seed in tot_rnd_seeds:

      
      pred_joints_info_nm = f"predicted_infos_seq_{case_idx}_seed_{seed}_tag_{test_tag}.npy"
      # pred_joints_info_nm = "predicted_infos.npy"


      pred_joints_info_fn = os.path.join(args.save_dir, pred_joints_info_nm)
      data = np.load(pred_joints_info_fn, allow_pickle=True).item()
      
      obj_verts = data['obj_verts'] # perhaps base normals --> base normals #
      
      
      targets = data['targets'] #
      outputs = data['outputs'] #  
      tot_obj_rot = data['tot_obj_rot'][0]
      tot_obj_transl = data['tot_obj_transl'][0]
      tot_base_pts = data["tot_base_pts"][0]
      tot_base_normals = data['tot_base_normals'][0]
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


      outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) 
      
      targets = np.matmul(targets, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) 
      
      
      print(f"tot_base_pts: {tot_base_pts.shape}")
      
      
      
      obj_verts_trans = np.matmul(obj_verts, tot_obj_rot) + tot_obj_transl.reshape(tot_obj_transl.shape[0], 1, tot_obj_transl.shape[1]) 
      obj_faces = data['obj_faces']
      print(f"obj_verts_trans: {obj_verts_trans.shape}, obj_faces: {obj_faces.shape}")
      
      
      with_contact_opt = False
      with_contact_opt = True
      
      # with_proj = False
      # with_proj = True
      
      with_params_smoothing = False
      # with_params_smoothing = True
      
      with_ctx_mask = True
      
      
      nn_hand_params = 24
      # nn_hand_params = 45
      
      dist_thres = 0.005
      
      tot_dist_thres = [0.005, 0.01, 0.02, 0.05, 0.1]
      # ToyCar
      tot_dist_thres = [0.001, 0.002, 0.005, 0.01]

      
      tot_with_proj = [ False]

      # ToyCar
      with_params_smoothing = True
      
      for with_proj in tot_with_proj:
        for dist_thres in tot_dist_thres:


          with_contact_opt = True
          with_ctx_mask = False
          

          bf_ct_optimized_dict, bf_proj_optimized_dict, optimized_dict = get_optimized_hand_fr_joints_v4_anchors(outputs, tot_base_pts, tot_base_pts_trans, tot_base_normals_trans, with_contact_opt=with_contact_opt, nn_hand_params=nn_hand_params, rt_vars=True, with_proj=with_proj, obj_verts_trans=obj_verts_trans, obj_faces=obj_faces, with_params_smoothing=with_params_smoothing, dist_thres=dist_thres, with_ctx_mask=with_ctx_mask)


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
          optimized_sv_infos.update(
              {
                  'obj_verts_trans': obj_verts_trans
              }
          )
          
          
          optimized_sv_infos_sv_fn_nm = f"optimized_infos_sv_dict_seq_{case_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}_with_proj_{with_proj}.npy"

          optimized_sv_infos_sv_fn = os.path.join(args.save_dir, optimized_sv_infos_sv_fn_nm)
          np.save(optimized_sv_infos_sv_fn, optimized_sv_infos)
          print(f"optimized infos saved to {optimized_sv_infos_sv_fn}")
        

