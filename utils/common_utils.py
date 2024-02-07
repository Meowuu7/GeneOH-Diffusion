import torch
import time
import numpy as np
import utils
# from common_utils import data_utils_torch as data_utils
# from common_utils.part_transform import revoluteTransform
# import random
import utils.model_util as model_util

# batched_index_select_ours


def get_random_rot_np():
  aa = np.random.randn(3)
  theta = np.sqrt(np.sum(aa**2))
  k = aa / np.maximum(theta, 1e-6)
  K = np.array([[0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]])
  R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K, K)
  R = R.astype(np.float32)
  return R

def get_faces_from_verts(verts, faces, sel_verts_idxes, sel_faces=None):
  
  ### n_sel_faces x 3 x 3 --> 
  # sel_faces = []
  if not isinstance(sel_verts_idxes, list):
    sel_verts_idxes = sel_verts_idxes.tolist() 
  sel_verts_idxes_dict = {sel_idx : 1 for sel_idx in sel_verts_idxes}
  if sel_faces is None:
    sel_faces = []
    for i_f in range(faces.size(0)):
      cur_f = faces[i_f]
      va, vb, vc = cur_f.tolist()
      # va = va - 1
      # vb = vb - 1
      # vc = vc - 1
      if va  in sel_verts_idxes_dict or vb in sel_verts_idxes_dict or vc in sel_verts_idxes_dict:
        sel_faces.append(faces[i_f].tolist()) ### sel_faces items...
    # print(f"number of sel_faces: {len(sel_faces)}")
    sel_faces = torch.tensor(sel_faces, dtype=torch.long).cuda() ### n_sel_faces x 3 ## sel_
  # print(f"verts: {verts.size()}, max_sel_faces: {torch.max(sel_faces)}, min_sel_faces: {torch.min(sel_faces)}")
  sel_faces_vals = model_util.batched_index_select_ours(values=verts, indices=sel_faces, dim=0) ### self_faces_vals: n_sel_faces x 3 x 3 ### sel_fces_vals...
  return sel_faces, sel_faces_vals


def sel_faces_values_from_sel_faces(verts, sel_faces):
  sel_faces_vals = model_util.batched_index_select_ours(values=verts, indices=sel_faces, dim=0) #
  return sel_faces_vals

def get_sub_verts_faces_from_pts(verts, faces, pts, rt_sel_faces=False, minn_dist_pts_verts_idx=None, sel_faces=None):
  ### return tyep: sel_verts: n_pts x 3; sel_faces: faces selected from sel_verts ###
  ## verts: n_verts x 3; pts: n_pts x 3
  dis_pts_verts = torch.sum((pts.unsqueeze(1) - verts.unsqueeze(0)) ** 2, dim=-1) ### n_pts x n_verts ###
  if minn_dist_pts_verts_idx is None:
    minn_dist_pts_verts, minn_dist_pts_verts_idx = torch.min(dis_pts_verts, dim=-1) ###
  sel_verts = verts[minn_dist_pts_verts_idx] ### should be close to pts in the euclidean distance
  sel_faces, sel_faces_vals = get_faces_from_verts(verts, faces, minn_dist_pts_verts_idx, sel_faces=sel_faces) ### sel_faces_vals: n_sel_faces x 3 x 3
  if rt_sel_faces:
    return sel_verts, sel_faces, sel_faces_vals, minn_dist_pts_verts_idx
  else:
    return sel_verts, sel_faces_vals, minn_dist_pts_verts_idx

##### distance of each sel_vert in mesh_2 to each face in sel_faces in mesh_1 ##### ---> 

def get_faces_normals(faces_vals):
  ### faces_vals: n_faces x 3 x 3
  vas = faces_vals[:, 0, :]
  vbs = faces_vals[:, 1, :]
  vcs = faces_vals[:, 2, :]
  vabs = vbs - vas
  vacs = vcs - vas ### n_faces x 3
  vns = torch.cross(vabs, vacs) ### n_faces x 3 ---> cross product between two vectors
  vns = vns / torch.clamp(torch.norm(vns, dim=-1, p=2, keepdim=True), min=1e-6) ### vns: n_faces x 3
  return vns

### 

def get_distance_pts_faces(pts, faces_vals, faces_vns):
  ### faces_vals: n_faces x 3 x 3  ## 
  ### faces_vns: n_faces x 3
  ### ax + by + cz = d  ### one pts and another pts --> faces_vals --> faces_ds
  faces_ds = torch.sum(faces_vals[:, 0, :] * faces_vns, dim=-1) ## n_faces x 3 xxx n_faces x 3 --> n_faces
  ### distance from one point to another point ###
  ### pts: n_pts x 3; faces_vns: n_faces x 3
  faces_pts_ds = torch.sum(pts.unsqueeze(1) * faces_vns.unsqueeze(0), dim=-1) ### n_pts x n_faces ### ### negative distances --> 
  delta_faces_pts_ds = faces_pts_ds - faces_ds.unsqueeze(0) ### n_pts x n_faces ### ### as an distance vector is pts can be projected to the faces ### pts_ds; 
  ### 1 x n_faces x 3 xxxxxx n_pts x n_faces x 1 --> n_pts x n_faces x 3
  projected_pts = pts.unsqueeze(1) - faces_vns.unsqueeze(0) * delta_faces_pts_ds.unsqueeze(-1)
  
  ### n_faces x 3 x 3
  ### vab vac ### 
  va, vb, vc = faces_vals[:, 0, :], faces_vals[:, 1, :], faces_vals[:, 2, :] ## n_faces x 3
  
  projected_pts = projected_pts - va.unsqueeze(0)
  
  vab, vac = vb - va, vc - va
  vab_norm, vac_norm = vab / torch.clamp(torch.norm(vab, dim=-1, p=2, keepdim=True), min=1e-7), vac / torch.clamp(torch.norm(vac, dim=-1, p=2, keepdim=True), min=1e-7)
  
  coeff_vab = torch.sum(vab_norm.unsqueeze(0) * projected_pts, dim=-1) / torch.clamp(torch.norm(vab, dim=-1, p=2, keepdim=False), min=1e-7)
  coeff_vac = torch.sum(vac_norm.unsqueeze(0) * projected_pts, dim=-1) / torch.clamp(torch.norm(vac, dim=-1, p=2, keepdim=False), min=1e-7)
  
  # coeff_vab = torch.sum(vab.unsqueeze(0) * projected_pts, dim=-1) ### n_pts x n_faces
  # coeff_vac = torch.sum(vac.unsqueeze(0) * projected_pts, dim=-1) ### n_pts x n_faces
  
  pts_in_faces = (((coeff_vab >= 0.).float() + (coeff_vac >= 0.).float() + (coeff_vab + coeff_vac <= 0.5).float()) > 2.5).float() ### n_pts x n_faces ### pts_in_faces --> 
  ### pts_in_faces and delta_faces_pts_ds 
  ### pts_in_faces: the projected pts in faces...
  return delta_faces_pts_ds, pts_in_faces ### delta_faces_pts_ds: n_pts x n_faces; pts_in_faces: n_pts x n_faces ###
   
  


### 
#### revolute joitns here ####
def collision_loss(mesh_1, mesh_2, keypts_1, keypts_2, joints, n_sim_steps=100, early_stop=False, penalize_largest=False, pts_loss=False, st_def_pcs=None):
  joint_dir, joint_pvp, joint_angle = joints
  joint_dir = joint_dir.cuda()
  joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  verts1, faces1 = mesh_1
  verts2, faces2 = mesh_2 # mesh 1 mesh 2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  keypts_2 = keypts_2.detach()
  ### verts2, keypts_2 ###
  
  ### not just a loss, but constraints ###
  ### iteratively projection ###
  
  # print(f"verts1: {verts1.size()}, verts2: {verts2.size()}, faces1: {faces1.size()}, faces2: {faces2.size()}, joint_dir: {joint_dir.size()}, joint_pvp: {joint_pvp.size()}")
  
  ### sel_verts, sel_faces_vals ###
  ### for sub_verts and sub_faces_vals ###
  sel_verts2, sel_faces_vals2 = get_sub_verts_faces_from_pts(verts2, faces2, keypts_2)
  sel_faces_vns2 =  get_faces_normals(sel_faces_vals2)
  
  sel_faces_vns2 = sel_faces_vns2.detach()
  
  
  ### pts_in_faces & delta_ds in two adjacent time stamps ###
  ### collision response for the loss term? ### 
  ### penetration depth * face_ns for all collided pts
  keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  ### pivot point prediction, joint axis direction prediction ### 
  # ### joint axis direction prediction ###
  
  delta_joint_angle = joint_angle / float(n_sim_steps) ### delta_joint_angles ###
  
  tot_collision_loss = 0.
  
  non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False).cuda()
  
  
  mesh_pts_sequence = []
  
  def_pcs_sequence = []
  
  for i in range(0, n_sim_steps): ### joint_angle ###
    cur_joint_angle = i * delta_joint_angle ### delta_joint_angle ###
    ### revoluteTransform ### joint_pvp
    pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_joint_angle)
    m = torch.from_numpy(m).float().cuda() ### 4 x 4
    
    kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
    pts = torch.matmul(kpts_expanded, m)
    # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
    pts = pts[:, :3] ### pts: n_keypts x 3
    
    # if st_def_pcs is not None:
    #   part1_pc = st_def_pcs[0][0]
    #   part1_pc_expanded = torch.cat([])
    
    
    #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds #### # distance pts faces #
    delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
    
    
    
    mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
    mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
    # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
    mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
    mesh_pts_sequence.append(mesh_pts_expanded.clone())
    
    ### 
    # if penalize_largest:
    #   ### delta_faces_pts_ds: n_pts x n_faces; pts_in_faces: n_pts x n_faces ### 
    #   abs_filtered_delta_faces_pts_ds = torch.abs(delta_faces_pts_ds) * pts_in_faces ### 
    #   maxx_abs_pts_faces_ds, maxx_abs_pts_faces_ds_idxes = torch.max(abs_filtered_delta_faces_pts_ds, dim=-1) ### 
    #   maxx_abs_pts_faces_ds_idxes = maxx_abs_pts_faces_ds_idxes.unsqueeze(-1).contiguous().repeat(1, pts_in_faces.size(1)) ###
    #   pts_faces_ds_idxes_range = torch.arange(pts_in_faces.size(1)).contiguous().unsqueeze(0).cuda()
    #   maxx_mask = (pts_faces_ds_idxes_range == maxx_abs_pts_faces_ds_idxes)
    #   cur_delta_faces_pts_ds = torch.zeros_like(delta_faces_pts_ds)
    #   cur_delta_faces_pts_ds[maxx_mask] = delta_faces_pts_ds[maxx_mask]
    #   delta_faces_pts_ds = cur_delta_faces_pts_ds
    
    
    # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
    if len(delta_faces_pts_ds_sequence) > 0:
      prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
      prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
      prev_pts = keypts_sequence[-1]
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      
      if penalize_largest:
        ### delta_faces_pts_ds: n_pts x n_faces; pts_in_faces: n_pts x n_faces ### 
        abs_filtered_delta_faces_pts_ds = torch.abs(delta_faces_pts_ds) * collision_pts_faces * pts_in_faces ### 
        maxx_abs_pts_faces_ds, maxx_abs_pts_faces_ds_idxes = torch.max(abs_filtered_delta_faces_pts_ds, dim=-1) ### 
        maxx_abs_pts_faces_ds_idxes = maxx_abs_pts_faces_ds_idxes.unsqueeze(-1).contiguous().repeat(1, pts_in_faces.size(1)) ###
        pts_faces_ds_idxes_range = torch.arange(pts_in_faces.size(1)).contiguous().unsqueeze(0).cuda()
        maxx_mask = (pts_faces_ds_idxes_range == maxx_abs_pts_faces_ds_idxes)
        cur_delta_faces_pts_ds = torch.zeros_like(delta_faces_pts_ds)
        cur_delta_faces_pts_ds[maxx_mask] = delta_faces_pts_ds[maxx_mask]
        # delta_faces_pts_ds = cur_delta_faces_pts_ds
      else:
        cur_delta_faces_pts_ds = delta_faces_pts_ds
      
        
      # collision_dists = collision_pts_faces * delta_faces_pts_ds ### n_pts x n_faces
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      
      
      if pts_loss:
        ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v2: for pts directly ###
      else:
        ### loss version v1: for pts directly ###
        delta_keypts = pts - prev_pts ### from the previous keypts to hte current keypts ### pts - prev_pts
        collision_loss = torch.sum(collision_pulse * delta_keypts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v1: for pts directly ###
        
      
      non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      non_collided_pts = non_collided_pts * non_collided_indicator
      # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
      if early_stop and collision_loss.item() > 0.0001:
        break


    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
    
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
    
  # print(f"tot_collision_loss: {tot_collision_loss}")
    
  ### can even test for one part at first
  return tot_collision_loss, keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###
    



### 
#### revolute joitns here ####
def collision_loss_prismatic(mesh_1, mesh_2, keypts_1, keypts_2, joints, n_sim_steps=100, early_stop=False, penalize_largest=False, pts_loss=False, st_def_pcs=None):
  joint_dir, joint_pvp, joint_angle = joints
  joint_dir = joint_dir.cuda()
  joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  verts1, faces1 = mesh_1
  verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  keypts_2 = keypts_2.detach()
  ### verts2, keypts_2 ###
  
  ### not just a loss, but constraints ###
  ### iteratively projection ###
  
  # print(f"verts1: {verts1.size()}, verts2: {verts2.size()}, faces1: {faces1.size()}, faces2: {faces2.size()}, joint_dir: {joint_dir.size()}, joint_pvp: {joint_pvp.size()}")
  
  ### sel_verts, sel_faces_vals ###
  ### for sub_verts and sub_faces_vals ###
  sel_verts2, sel_faces_vals2 = get_sub_verts_faces_from_pts(verts2, faces2, keypts_2)
  sel_faces_vns2 =  get_faces_normals(sel_faces_vals2)
  
  sel_faces_vns2 = sel_faces_vns2.detach()
  
  
  ### pts_in_faces & delta_ds in two adjacent time stamps ###
  ### collision response for the loss term? ### 
  ### penetration depth * face_ns for all collided pts
  keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  ### pivot point prediction, joint axis direction prediction ### 
  # ### joint axis direction prediction ###
  
  # delta_joint_angle = joint_angle / float(n_sim_steps) ### delta_joint_angles ###
  
  delta_joint_angle = 1.0 / float(n_sim_steps)
  
  tot_collision_loss = 0.
  
  non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False).cuda()
  
  
  mesh_pts_sequence = []
  
  def_pcs_sequence = []
  
  for i in range(0, n_sim_steps): ### joint_angle ###
    # cur_joint_angle = i * delta_joint_angle ### delta_joint_angle ###
    cur_delta_dis = i * delta_joint_angle
    
    moving_dis = joint_dir * cur_delta_dis ## (3,)
    moving_dis = moving_dis.unsqueeze(0)
    
    
    # ### revoluteTransform ### joint_pvp
    # pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_joint_angle)
    # m = torch.from_numpy(m).float().cuda() ### 4 x 4
    
    # kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
    # pts = torch.matmul(kpts_expanded, m)
    # # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
    # pts = pts[:, :3] ### pts: n_keypts x 3
    
    pts = moving_dis + keypts_1
    
    # if st_def_pcs is not None:
    #   part1_pc = st_def_pcs[0][0]
    #   part1_pc_expanded = torch.cat([])
    
    
    #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
    delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
    
    
    
    # mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
    # mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
    # # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
    # mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
    
    mesh_pts_expanded = verts1 + moving_dis
    
    mesh_pts_sequence.append(mesh_pts_expanded.clone())
    
    
    # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
    if len(delta_faces_pts_ds_sequence) > 0:
      prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
      prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
      prev_pts = keypts_sequence[-1]
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      
      if penalize_largest:
        ### delta_faces_pts_ds: n_pts x n_faces; pts_in_faces: n_pts x n_faces ### 
        abs_filtered_delta_faces_pts_ds = torch.abs(delta_faces_pts_ds) * collision_pts_faces * pts_in_faces ### 
        maxx_abs_pts_faces_ds, maxx_abs_pts_faces_ds_idxes = torch.max(abs_filtered_delta_faces_pts_ds, dim=-1) ### 
        maxx_abs_pts_faces_ds_idxes = maxx_abs_pts_faces_ds_idxes.unsqueeze(-1).contiguous().repeat(1, pts_in_faces.size(1)) ###
        pts_faces_ds_idxes_range = torch.arange(pts_in_faces.size(1)).contiguous().unsqueeze(0).cuda()
        maxx_mask = (pts_faces_ds_idxes_range == maxx_abs_pts_faces_ds_idxes)
        cur_delta_faces_pts_ds = torch.zeros_like(delta_faces_pts_ds)
        cur_delta_faces_pts_ds[maxx_mask] = delta_faces_pts_ds[maxx_mask]
        # delta_faces_pts_ds = cur_delta_faces_pts_ds
      else:
        cur_delta_faces_pts_ds = delta_faces_pts_ds
      
        
      # collision_dists = collision_pts_faces * delta_faces_pts_ds ### n_pts x n_faces
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      
      
      if pts_loss:
        ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v2: for pts directly ###
      else:
        ### loss version v1: for pts directly ###
        delta_keypts = pts - prev_pts ### from the previous keypts to hte current keypts ### pts - prev_pts
        collision_loss = torch.sum(collision_pulse * delta_keypts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v1: for pts directly ###
        
      
      non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      non_collided_pts = non_collided_pts * non_collided_indicator
      # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
      if early_stop and collision_loss.item() > 0.0001:
        break


    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
    
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
    
  # print(f"tot_collision_loss: {tot_collision_loss}")
    
  ### can even test for one part at first
  return tot_collision_loss, keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###
    

def collision_loss_sim_sequence_ours(verts1, verts2, faces1, faces2, base_pts, use_delta=False, sel_faces_values=None): # inputs are torch tensors #
  # joint_dir, joint_pvp, joint_angle = joints
  # joint_dir = joint_dir.cuda()
  # joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  # verts1, faces1 = mesh_1
  # verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  faces2_exp = faces2.unsqueeze(0).repeat(verts2.size(0), 1, 1).contiguous()
  faces_values2 = model_util.batched_index_select_ours(verts2, indices=faces2_exp, dim=1) # nf x nn_face x 3 x 3 # 
  faces_values2 = faces_values2.mean(dim=-2) # nf x nn_faces x 3 
  
  if sel_faces_values is None:
    dist_hand_to_obj_verts = torch.sum(
      (verts1.detach().cpu().unsqueeze(-2) - faces_values2.detach().cpu().unsqueeze(1)) ** 2, dim=-1 # ### nf x nn_hand_verts x nn_obj_verts 
    )
    minn_dist_obj_to_hand, _ = torch.min(dist_hand_to_obj_verts, dim=0)
    minn_dist_obj_to_hand, _ = torch.min(minn_dist_obj_to_hand, dim=0) # nn_obj_vert 
    minn_dist_obj_to_hand_argsort = torch.argsort(minn_dist_obj_to_hand, dim=0, descending=False)
    cur_p_sel_faces = minn_dist_obj_to_hand_argsort[:base_pts.size(1)]
    cur_p_sel_faces = model_util.batched_index_select_ours(faces2.detach().cpu(), indices=cur_p_sel_faces, dim=0)

    
    sel_verts = []
    sel_faces = []
    sel_faces_values = []
    minn_dist_pts_verts_idx = None
    # cur_p_sel_faces = None
    for i_fr in range(verts2.size(0)):
      cur_p_sel_verts, cur_p_sel_faces, cur_p_sel_faces_vals, minn_dist_pts_verts_idx = get_sub_verts_faces_from_pts(verts2[i_fr].detach().cpu(), faces2.detach().cpu(), base_pts[i_fr].detach().cpu(), rt_sel_faces=True, minn_dist_pts_verts_idx=minn_dist_pts_verts_idx, sel_faces=cur_p_sel_faces)
      sel_verts.append(cur_p_sel_verts)
      sel_faces.append(cur_p_sel_faces)
      sel_faces_values.append(cur_p_sel_faces_vals)
  
  # keypts_2 = keypts_2.detach()
  # faces: nf x 3 # verts: nn_verts x 3 #
  # verts1: nn_verts x 3; faces1_values: nn_faces x 3 x 3 -> faces vertices 
  # faces1_values = model_util.batched_index_select_ours(values=verts1, indices=faces1, dim=0) 
  # faces2_values = model_util.batched_index_select_ours(values=verts2, indices=faces2, dim=0) 
  # faces1_normals = get_faces_normals(faces1_values)
  # faces2_normals = get_faces_normals(faces2_values)
  # delta_verts1_pts_abs, verts1_in_feats = get_distance_pts_faces(verts1, faces2_values, faces2_normals)
  # verts1 = verts
  
  
  # get_distance_pts_faces(pts, faces_vals, faces_vns):

  # sel_faces_vns2 = sel_faces_vns2.detach()
  
  
  # ### pts_in_faces & delta_ds in two adjacent time stamps ###
  # ### collision response for the loss term? ### 
  # ### penetration depth * face_ns for all collided pts
  # # keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  # ### pivot point prediction, joint axis direction prediction ### 
  # # ### joint axis direction prediction ### #### get part joints...
  
  # # joint_dir = joints["axis"]["dir"]
  # # joint_pvp = joints["axis"]["center"]
  # # joint_a = joints["axis"]["a"]
  # # joint_b = joints["axis"]["b"]
  
  
  # delta_joint_angle = (float(joint_b) - float(joint_a)) / float(n_sim_steps - 1) ### delta_joint_angles ###
  
  tot_collision_loss = 0.
  
  # non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False) # .cuda()
  
  
  # mesh_pts_sequence = []
  
  # def_pcs_sequence = []
  
  sv_dict = {
    'hand_verts': verts1.detach().cpu().numpy(),
    'obj_verts': verts2.detach().cpu().numpy(),
    'obj_faces': faces2.detach().cpu().numpy(),
    'base_pts': base_pts.detach().cpu().numpy(),
  }
  sv_dict_fn = "tmp_dict.npy"
  np.save(sv_dict_fn, sv_dict)
  
  
  
  n_frames = verts1.shape[0]
  print(f"cur_nframes: {n_frames}")
  for i in range(0, n_frames): ### joint_angle ###
    # if not back_sim:
    #   cur_joint_angle = joint_a +  i * delta_joint_angle ### delta_joint_angle ###
    # else:
    #   cur_joint_angle = joint_b - i * delta_joint_angle  
      
    if use_delta:
      delta_joint_angle = (joint_b - joint_a) / 100.0
      
      ''' Prev. arti. state '''  
      cur_st_joint_angle = np.random.uniform(low=joint_a + delta_joint_angle, high=joint_b, size=(1,)).item() #### lower and upper limits of simulation angles
      
      ### revoluteTransform ### joint_pvp
      prev_pts, prev_m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      prev_m = torch.from_numpy(prev_m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      prev_pts = torch.matmul(kpts_expanded, prev_m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      prev_pts = prev_pts[:, :3] ### pts: n_keypts x 3
      
      # prev_pts = verts1[i]
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      prev_delta_faces_pts_ds, prev_pts_in_faces = get_distance_pts_faces(prev_pts, sel_faces_vals2, sel_faces_vns2)
      ''' Prev. arti. state ''' 
      
      ''' Current state ''' 
      cur_ed_joint_angle = cur_st_joint_angle - delta_joint_angle
      ### revoluteTransform ### joint_pvp
      pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      pts = pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
      ''' Current state ''' 
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      cur_delta_faces_pts_ds = delta_faces_pts_ds
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      # collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      collision_pulse = (collision_dists * pts_in_faces).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      # collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      ### loss version v2: for pts directly ###
      
      # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      # non_collided_pts = non_collided_pts * non_collided_indicator
      # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
    else:
      ### revoluteTransform ### joint_pvp
      # pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_joint_angle)
      # m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      # kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()
      #                            ], dim=-1) #### kpts_expanded
      # pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      # pts = pts[:, :3] ### pts: n_keypts x 3
      
      pts = verts1[i]
      
      cur_verts2 = verts2[i]
      # cur_face_values2 = model_util.batched_index_select_ours(values=cur_verts2, indices=faces2, dim=0) # nn_verts x 3 x 3 for the verts and the faces #
      # cur_face_normals = get_faces_normals(cur_face_values2)
      
      
      # #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      # delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), cur_face_values2.detach().cpu(), cur_face_normals.detach().cpu())
      
      
      cur_face_values2 = sel_faces_values[i]
      cur_face_normals = get_faces_normals(sel_faces_values[i])
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), sel_faces_values[i].detach().cpu(), cur_face_normals.detach().cpu())
      
      
      # mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      # mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
      # # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      # mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
      # mesh_pts_sequence.append(mesh_pts_expanded.clone())
    
    
      # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
      if len(delta_faces_pts_ds_sequence) > 0:
        prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
        # prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
        prev_pts = keypts_sequence[-1]
        
        sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
        sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
        ### different signs ###
        collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 

        cur_delta_faces_pts_ds = delta_faces_pts_ds

        collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
        ### collision_dists ### n_pts x n_faces
        ### i think the sim step 
        ### whether tow meshes collide with each other: in the target mesh, 
        ### collision_pulse, collision_dists
        ### collision pulse ####
        # collision_pulse = 1.0 * (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * cur_face_normals.unsqueeze(0).detach().cpu() ### n_pts x n_faces x 3 --> pulse
        
        # collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
        ### collision pulse ####
        
        
        ### ==== collision loss v1 ==== ###
        # # # pulse ! -> tegether wit hface normals # 
        # collision_pulse = 1.0 * (collision_dists * pts_in_faces).unsqueeze(-1) * cur_face_normals.unsqueeze(0).detach().cpu() ### n_pts x n_faces x 3 --> pulse
        
        # collided_indicator = ((pts_in_faces * collision_pts_faces).sum(-1) > 0.1).float()

        # ### loss version v2: for pts directly ### ### calculate collision_loss from collision_pulse and pts ###
        # collision_loss = torch.sum(collision_pulse.cuda() * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        # collision_loss = collision_loss.mean()
        ### loss version v2: for pts directly ###
        ### ==== collision loss v1 ==== ###
        
        ### ==== collision loss v2 ==== ###
        pts_in_faces = pts_in_faces.cuda()
        # collision_pulse = 1.0 * (collision_dists * pts_in_faces).unsqueeze(-1) * cur_face_normals.unsqueeze(0).detach()
        cur_face_avg_values = torch.mean(cur_face_values2.detach(), dim=-2) # nn_faces x 3 -> for the face_avg_values #
        face_avg_values_to_key_pts = pts.unsqueeze(1) - cur_face_avg_values.unsqueeze(0).cuda() # nn_ptss x nn_faces x 3 -> from face avg values to joints here #
        face_avg_values_to_key_pts = face_avg_values_to_key_pts * pts_in_faces.unsqueeze(-1) # * collision_pts_faces.unsqueeze(-1).cuda()
        collision_loss = torch.mean((face_avg_values_to_key_pts ** 2).sum(dim=-1)) # nn_pts x nn_faces -> a single value here #
        ### ==== collision loss v2 ==== ###
        
        
        
        # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
        # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
        # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
        # non_collided_pts = non_collided_pts * non_collided_indicator
        # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
        
        # ### loss version v2: for pts directly ###
        # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        # ### loss version v2: for pts directly ###
        # collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
        tot_collision_loss += collision_loss
        # if early_stop and collision_loss.item() > 0.0001:
        #   break

    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
  
  tot_collision_loss = tot_collision_loss / n_frames
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
    
  # print(f"tot_collision_loss: {tot_collision_loss}")
    
  ### can even test for one part at first
  return tot_collision_loss, sel_faces_values # , keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###

  
# collision loss and sim sequence   
# 
def collision_loss_sim_sequence_ours_ccd_rigid(verts1, verts2, faces1, faces2, base_pts, obj_rot, obj_trans,
                                         use_delta=False, sel_faces_values=None, canon_verts1=None, canon_sel_faces_values=None): # inputs are torch tensors #
  # joint_dir, joint_pvp, joint_angle = joints
  # joint_dir = joint_dir.cuda()
  # joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  # verts1, faces1 = mesh_1
  # verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  faces2_exp = faces2.unsqueeze(0).repeat(verts2.size(0), 1, 1).contiguous()
  faces_values2 = model_util.batched_index_select_ours(verts2, indices=faces2_exp, dim=1) # nf x nn_face x 3 x 3 # 
  faces_values2 = faces_values2.mean(dim=-2) # nf x nn_faces x 3 
  
  if sel_faces_values is None:
    dist_hand_to_obj_verts = torch.sum(
      (verts1.detach().cpu().unsqueeze(-2) - faces_values2.detach().cpu().unsqueeze(1)) ** 2, dim=-1 # ### nf x nn_hand_verts x nn_obj_verts 
    )
    minn_dist_obj_to_hand, _ = torch.min(dist_hand_to_obj_verts, dim=0)
    minn_dist_obj_to_hand, _ = torch.min(minn_dist_obj_to_hand, dim=0) # nn_obj_vert 
    minn_dist_obj_to_hand_argsort = torch.argsort(minn_dist_obj_to_hand, dim=0, descending=False)
    cur_p_sel_faces = minn_dist_obj_to_hand_argsort[:base_pts.size(1)]
    cur_p_sel_faces = model_util.batched_index_select_ours(faces2.detach().cpu(), indices=cur_p_sel_faces, dim=0)

    
    sel_verts = []
    sel_faces = []
    sel_faces_values = []
    canon_sel_faces_values = []
    canon_verts1 = []
    minn_dist_pts_verts_idx = None
    # cur_p_sel_faces = None
    for i_fr in range(verts2.size(0)):
      
      
      
      cur_p_sel_verts, cur_p_sel_faces, cur_p_sel_faces_vals, minn_dist_pts_verts_idx = get_sub_verts_faces_from_pts(verts2[i_fr].detach().cpu(), faces2.detach().cpu(), base_pts[i_fr].detach().cpu(), rt_sel_faces=True, minn_dist_pts_verts_idx=minn_dist_pts_verts_idx, sel_faces=cur_p_sel_faces)
      sel_verts.append(cur_p_sel_verts)
      sel_faces.append(cur_p_sel_faces)
      sel_faces_values.append(cur_p_sel_faces_vals)
      
      # cur_p_sel_faces_vals: nn_sel_faces x 3 x 3 
      cur_fr_obj_rot = obj_rot[i_fr].detach().cpu() # 3 x 3
      cur_fr_obj_trans = obj_trans[i_fr].detach().cpu() # 3
      cur_fr_verts1 = verts1[i_fr]
      # cur_fr_canon_faces_values: nn_sel_faces x 3 x 3; cur_
      cur_fr_canon_faces_values = torch.matmul(cur_p_sel_faces_vals -  cur_fr_obj_trans.unsqueeze(0).unsqueeze(0), cur_fr_obj_rot.transpose(1, 0).unsqueeze(0)) # 
      # nn_verts x 3 xxxx 3 x 3 -> nn_verts x 3 #
      cur_fr_canon_verts1 = torch.matmul(
        cur_fr_verts1 - cur_fr_obj_trans.unsqueeze(0).cuda(), cur_fr_obj_rot.transpose(1, 0).cuda()
      )
      canon_verts1.append(cur_fr_canon_verts1) # 
      canon_sel_faces_values.append(cur_fr_canon_faces_values)  # face values canonicalized #
    
    
    
       
  
  # ### pts_in_faces & delta_ds in two adjacent time stamps ###
  # ### collision response for the loss term? ### 
  # ### penetration depth * face_ns for all collided pts
  # # keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  # ### pivot point prediction, joint axis direction prediction ### 
  # # ### joint axis direction prediction ### #### get part joints...
  
  # # joint_dir = joints["axis"]["dir"]
  # # joint_pvp = joints["axis"]["center"]
  # # joint_a = joints["axis"]["a"]
  # # joint_b = joints["axis"]["b"]
  
  
  # delta_joint_angle = (float(joint_b) - float(joint_a)) / float(n_sim_steps - 1) ### delta_joint_angles ###
  
  # tot_collision_loss = 0.
  
  # non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False) # .cuda()
  
  
  # mesh_pts_sequence = []
  
  # def_pcs_sequence = []
  
  # sv_dict = {
  #   'hand_verts': verts1.detach().cpu().numpy(),
  #   'obj_verts': verts2.detach().cpu().numpy(),
  #   'obj_faces': faces2.detach().cpu().numpy(),
  #   'base_pts': base_pts.detach().cpu().numpy(),
  # }
  # sv_dict_fn = "tmp_dict.npy"
  # np.save(sv_dict_fn, sv_dict)
  
  
  # verts2: 
  
  n_frames = verts1.shape[0]
  print(f"cur_nframes: {n_frames}")
  for i in range(0, n_frames): ### joint_angle ###
    

    pts = canon_verts1[i]

    
    # cur_verts2 = verts2[i]
    # cur_face_values2 = model_util.batched_index_select_ours(values=cur_verts2, indices=faces2, dim=0) # nn_verts x 3 x 3 for the verts and the faces #
    # cur_face_normals = get_faces_normals(cur_face_values2)
    
    
    # #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
    # delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), cur_face_values2.detach().cpu(), cur_face_normals.detach().cpu())
    
    
    # cur_face_values2 = canon_sel_faces_values[i]
    cur_face_normals = get_faces_normals(canon_sel_faces_values[i])
    delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), canon_sel_faces_values[i].detach().cpu(), cur_face_normals.detach().cpu())
    
    
    # mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
    # mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
    # # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
    # mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
    # mesh_pts_sequence.append(mesh_pts_expanded.clone())
  
  
    # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
    if len(delta_faces_pts_ds_sequence) > 0:
      
      prev_pts = keypts_sequence[-1] # nn_tps x 3 #
      prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
      
      coef = 1.
      coef_step = 0.1
      while coef >= 0.:
        print(f"coef: {coef}")
        cur_step_pts = prev_pts + (pts - prev_pts) * coef
        cur_step_delta_faces_pts_ds, cur_step_pts_in_faces = get_distance_pts_faces(cur_step_pts.detach().cpu(), canon_sel_faces_values[i].detach().cpu(), cur_face_normals.detach().cpu())
        
        sgn_delta_faces_ds = torch.sign(cur_step_delta_faces_pts_ds) ### sign of faces_ds
        sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
        ### different signs ###
        collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
        collision_pts_faces = ((collision_pts_faces + cur_step_pts_in_faces.float()) > 1.5).float()
        collision_pts_faces_sum = collision_pts_faces.sum().item() #  pts in faces? #
        if collision_pts_faces_sum == 0:
          break
        coef = coef - coef_step 
      coef = max(0., coef)
      pts = prev_pts + (pts - prev_pts) * coef
      delta_faces_pts_ds = cur_step_delta_faces_pts_ds
      pts_in_faces = cur_step_pts_in_faces
      
      
      # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      # non_collided_pts = non_collided_pts * non_collided_indicator
      # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      # tot_collision_loss += collision_loss
      # if early_stop and collision_loss.item() > 0.0001:
      #   break

    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
  
  # tot_collision_loss = tot_collision_loss / n_frames
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
  
  keypts_sequence = torch.stack(keypts_sequence, dim=0) ### nn_frames x nn_keypts x 3 ###
  # print(f"tot_collision_loss: {tot_collision_loss}")
  # sel_faces_values=None, canon_verts1=None, canon_sel_faces_values=None
  ### can even test for one part at first
  return keypts_sequence, sel_faces_values, canon_verts1, canon_sel_faces_values # , keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###

  

def collision_loss_sim_sequence(verts1, 
                                keypts_1, verts2, sel_faces_vals2, sel_faces_vns2, joints, n_sim_steps=100, back_sim=False, use_delta=False):
  # joint_dir, joint_pvp, joint_angle = joints
  # joint_dir = joint_dir.cuda()
  # joint_pvp = joint_pvp.cuda()
  
  ### should save the sequence of transformed shapes ... ###
  
  # verts1, faces1 = mesh_1
  # verts2, faces2 = mesh_2
  
  ### verts2, keypts_2.detach() ###
  ### verts2, keypts_2 ###
  verts2 = verts2.detach()
  # keypts_2 = keypts_2.detach()

  sel_faces_vns2 = sel_faces_vns2.detach()
  
  
  ### pts_in_faces & delta_ds in two adjacent time stamps ###
  ### collision response for the loss term? ### 
  ### penetration depth * face_ns for all collided pts
  # keypts_sequence = [keypts_1.clone()]
  keypts_sequence = []
  delta_faces_pts_ds_sequence = []
  pts_in_faces_sequence = [] 
  ### pivot point prediction, joint axis direction prediction ### 
  # ### joint axis direction prediction ### #### get part joints...
  
  joint_dir = joints["axis"]["dir"]
  joint_pvp = joints["axis"]["center"]
  joint_a = joints["axis"]["a"]
  joint_b = joints["axis"]["b"]
  
  
  delta_joint_angle = (float(joint_b) - float(joint_a)) / float(n_sim_steps - 1) ### delta_joint_angles ###
  
  tot_collision_loss = 0.
  
  non_collided_pts = torch.ones((keypts_1.size(0), ), dtype=torch.float32, requires_grad=False) # .cuda()
  
  
  mesh_pts_sequence = []
  
  def_pcs_sequence = []
  
  
  
  
  for i in range(0, n_sim_steps): ### joint_angle ###
    if not back_sim:
      cur_joint_angle = joint_a +  i * delta_joint_angle ### delta_joint_angle ###
    else:
      cur_joint_angle = joint_b - i * delta_joint_angle  
      
    if use_delta:
      delta_joint_angle = (joint_b - joint_a) / 100.0
      
      ''' Prev. arti. state '''  
      cur_st_joint_angle = np.random.uniform(low=joint_a + delta_joint_angle, high=joint_b, size=(1,)).item() #### lower and upper limits of simulation angles
      
      ### revoluteTransform ### joint_pvp
      prev_pts, prev_m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      prev_m = torch.from_numpy(prev_m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      prev_pts = torch.matmul(kpts_expanded, prev_m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      prev_pts = prev_pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      prev_delta_faces_pts_ds, prev_pts_in_faces = get_distance_pts_faces(prev_pts, sel_faces_vals2, sel_faces_vns2)
      ''' Prev. arti. state ''' 
      
      ''' Current state ''' 
      cur_ed_joint_angle = cur_st_joint_angle - delta_joint_angle
      ### revoluteTransform ### joint_pvp
      pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_st_joint_angle) ### st_joint_angle
      m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      pts = pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts, sel_faces_vals2, sel_faces_vns2)
      ''' Current state ''' 
      
      sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
      sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
      ### different signs ###
      collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 
      
      cur_delta_faces_pts_ds = delta_faces_pts_ds
      
      collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
      ### collision_dists ### n_pts x n_faces
       ### i think the sim step 
       ### whether tow meshes collide with each other: in the target mesh, 
      ### collision_pulse, collision_dists
      # collision_pulse = (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      collision_pulse = (collision_dists * pts_in_faces).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0) ### n_pts x n_faces x 3 --> pulse
      
      # collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()
      
      ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      ### loss version v2: for pts directly ###
      
      # non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
      # # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
      # # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
      # non_collided_pts = non_collided_pts * non_collided_indicator
      # # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
      
      # ### loss version v2: for pts directly ###
      # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
      # ### loss version v2: for pts directly ###
      collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
      tot_collision_loss += collision_loss
    else:
      ### revoluteTransform ### joint_pvp
      pts, m = revoluteTransform(keypts_1.detach().cpu().numpy(), joint_pvp.detach().cpu().numpy(), joint_dir.detach().cpu().numpy(), cur_joint_angle)
      m = torch.from_numpy(m).float().cuda() ### 4 x 4
      
      kpts_expanded = torch.cat([keypts_1, torch.ones((keypts_1.size(0), 1), dtype=torch.float32).cuda()
                                 ], dim=-1) #### kpts_expanded
      pts = torch.matmul(kpts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      pts = pts[:, :3] ### pts: n_keypts x 3
      
      #### distance_pts_faces, pts_in_faces, delta_faces_pts_ds ####
      delta_faces_pts_ds, pts_in_faces = get_distance_pts_faces(pts.detach().cpu(), sel_faces_vals2.detach().cpu(), sel_faces_vns2.detach().cpu())
      
      
      mesh_pts_expanded = torch.cat([verts1, torch.ones((verts1.size(0), 1), dtype=torch.float32).cuda()], dim=-1) #### kpts_expanded
      mesh_pts_expanded = torch.matmul(mesh_pts_expanded, m)
      # pts = torch.matmul(keypts_1, m[:3]) ### n_keypts x 3 xxx 3 x 4 --> 
      mesh_pts_expanded = mesh_pts_expanded[:, :3] ### pts: n_keypts x 3
      mesh_pts_sequence.append(mesh_pts_expanded.clone())
    
    
      # if len(delta_faces_pts_ds_sequence) > 0 and i == 0 or i == n_sim_steps - 1:
      if len(delta_faces_pts_ds_sequence) > 0:
        prev_delta_faces_pts_ds = delta_faces_pts_ds_sequence[-1]
        # prev_pts_in_faces = pts_in_faces_sequence[-1] ### not important
        prev_pts = keypts_sequence[-1]
        
        sgn_delta_faces_ds = torch.sign(delta_faces_pts_ds) ### sign of faces_ds
        sgn_prev_delta_faces_ds = torch.sign(prev_delta_faces_pts_ds) ### sign of prev_faces_ds
        ### different signs ###
        collision_pts_faces = (sgn_delta_faces_ds != sgn_prev_delta_faces_ds).float() ### n_pts x n_faces 

        cur_delta_faces_pts_ds = delta_faces_pts_ds

        collision_dists = collision_pts_faces * cur_delta_faces_pts_ds ### n_pts x n_faces
        ### collision_dists ### n_pts x n_faces
        ### i think the sim step 
        ### whether tow meshes collide with each other: in the target mesh, 
        ### collision_pulse, collision_dists
        collision_pulse = 1.0 * (collision_dists * pts_in_faces * non_collided_pts.unsqueeze(-1)).unsqueeze(-1) * sel_faces_vns2.unsqueeze(0).detach().cpu() ### n_pts x n_faces x 3 --> pulse
        
        collided_indicator = ((pts_in_faces * non_collided_pts.unsqueeze(-1) * collision_pts_faces).sum(-1) > 0.1).float()

        ### loss version v2: for pts directly ### ### calculate collision_loss from collision_pulse and pts ###
        collision_loss = torch.sum(collision_pulse.cuda() * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        ### loss version v2: for pts directly ###
        
        
        non_collided_indicator = 1.0  - collided_indicator # - (collision_pulse.sum(-1).sum(-1) > 1e-6).float()
        # print(f"collision_pulse: {collision_pulse.size()}, collided_indicator: {collided_indicator.size()}, non_collided_pts: {non_collided_pts.size()}")
        # non_collided_pts[collided_indicator] = non_collided_pts[collided_indicator] * 0.
        non_collided_pts = non_collided_pts * non_collided_indicator
        # print(f"collision_loss: {collision_loss.sum().mean().item()}, collided_indicator: {collided_indicator.sum(-1).item()}, non_collided_pts: {non_collided_pts.sum(-1).item()}")
        
        # ### loss version v2: for pts directly ###
        # collision_loss = torch.sum(collision_pulse * pts.unsqueeze(1), dim=-1) ### n_pts x n_faces ### 
        # ### loss version v2: for pts directly ###
        collision_loss = torch.sum(collision_loss, dim=-1).sum() ###  ## for all faces ###
        tot_collision_loss += collision_loss
        # if early_stop and collision_loss.item() > 0.0001:
        #   break


    delta_faces_pts_ds_sequence.append(delta_faces_pts_ds.clone())
    pts_in_faces_sequence.append(pts_in_faces.clone())
    keypts_sequence.append(pts.clone())
    
  # tot_collision_loss /= n_sim_steps
    ### delta_faces_
    # if 
    
  # print(f"tot_collision_loss: {tot_collision_loss}")
    
  ### can even test for one part at first
  return tot_collision_loss, keypts_sequence, mesh_pts_sequence ### collision_loss for all sim steps ###
    


