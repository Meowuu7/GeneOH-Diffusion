# import tensorflow_probability as tfp

import numpy as np
import torch.nn as nn
# import layer_utils
import torch
# import data_utils_torch as data_utils
import math ## 

import pickle

import os
import utils.model_util as model_util
import time


class DMTet:
    def __init__(self): # triangle_table -> the triangle table #
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')
        # triangles table; # base tet edges #
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################
    # sorted edges #
    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4): # po
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        face_to_valid_tet = torch.cat((
            tet_gidx[num_triangles == 1],
            torch.stack((tet_gidx[num_triangles == 2], tet_gidx[num_triangles == 2]), dim=-1).view(-1)
        ), dim=0)

        valid_vert_idx = tet_fx4[tet_gidx[num_triangles > 0]].long().unique()

        return verts, faces, uvs, uv_idx, face_to_valid_tet.long(), valid_vert_idx



def test_pickle(pkl_fn):
  pkl_data = pickle.load(open(pkl_fn, "rb"), encoding='latin1')
  # encoding='latin1'
  print(pkl_data.keys())
  for k in pkl_data:
    print(f"key: {k}, value: {pkl_data[k].shape}")


def load_data_fr_th_sv(th_sv_fn, grid_res=64):
  # /home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00001.pt
  # th_sv_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00001.pt"
  th_data = torch.load(th_sv_fn, map_location="cpu") # map location #
  # repo # th_data # 
  sdf = th_data["sdf"]
  deform = th_data["deform"]
  deform_unmasked = th_data["deform_unmasked"]
  
  root = "/home/xueyi/sim/MeshDiffusion/nvdiffrec" # get the root path #
  # grid_res = 64
  # grid_res of tets to be loaded #
  tets = np.load(os.path.join(root, 'data/tets/{}_tets_cropped.npz'.format(grid_res)))
  tet_verts = tets['vertices'] # tet _verts -> pose 
  tet_indices = tets['indices'] # indices
  
  # sdf for each grids # 
  print(f"tet_verts: {tet_verts.shape}, tet_indices: {tet_indices.shape}")
  
  dmt_net = DMTet()
  # grid_res = 64

  # 1) deform but not deformed -> so tet_verts + deform as the deformed pos #
  # 2) deform but not 
  tet_verts = torch.from_numpy(tet_verts).float() # nn_verts x 3 
  # sdf = torch.from_numpy(sdf).float() # nn_verts --> the size of nn_verts # 
  tet_indices = torch.from_numpy(tet_indices).long() # nn_tets x 4 # 
  # __call__(self, pos_nx3, sdf_n, tet_fx4)
  
  deform = deform.float()
  deformed_verts = tet_verts + deform
  # deformed_verts =  deform
  
  print(deform_unmasked)
  
  deformed_verts[deform_unmasked.bool()] = tet_verts[deform_unmasked.bool()]
  # verts, faces, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), sdf.cuda(), tet_indices.cuda())
  verts, faces, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(deformed_verts.cuda(), sdf.cuda(), tet_indices.cuda())
  print(f"verts: {verts.size()}, faces: {faces.size()}")
  
  sv_mesh_dict = {
    'verts': verts.detach().cpu().numpy(),
    'faces': faces.detach().cpu().numpy(), # deformed_verts #
  }
  # sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000.npy"
  sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000_res_128.npy"
  sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results_seq/tets/mesh_extracted_00002_res_128.npy"
  np.save(sv_mesh_fn, sv_mesh_dict)
  print(f"extracted mesh saved to {sv_mesh_fn}")
  
  # self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') # * scale
  # self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda') # 
  # 
  # print()




def load_data_fr_th_sv_fr_pred(th_sv_fn, grid_res=64):
  # /home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00001.pt
  # th_sv_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00001.pt"
#   th_data = torch.load(th_sv_fn, map_location="cpu") # map location #
  # repo # th_data # 
  
#   cur_step_sv_dict = {
#         "obj_sdf_inputs": sdf_obj.detach().cpu().numpy(),
#         "hand_sdf_inputs": sdf_hand.detach().cpu().numpy(),
#         "obj_sdf_outputs": sdf.squeeze(-1).detach().cpu().numpy(),
#         "hand_sdf_outputs": sdf_obj2.squeeze(-1).detach().cpu().numpy(),
#     }
  cur_step_data = np.load(th_sv_fn, allow_pickle=True).item()
  obj_sdf_inputs = cur_step_data["obj_sdf_inputs"]
  obj_sdf_outputs = cur_step_data["obj_sdf_outputs"]
  
  hand_sdf_inputs = cur_step_data["hand_sdf_inputs"]
  hand_sdf_outputs = cur_step_data["hand_sdf_outputs"]
  
  
#   sdf = th_data["sdf"]
#   deform = th_data["deform"]
#   deform_unmasked = th_data["deform_unmasked"]
  
  root = "/home/xueyi/sim/MeshDiffusion/nvdiffrec" # get the root path #
  # grid_res = 64
  # grid_res of tets to be loaded #
  tets = np.load(os.path.join(root, 'data/tets/{}_tets_cropped.npz'.format(grid_res)))
  tet_verts = tets['vertices'] # tet _verts -> pose 
  tet_indices = tets['indices'] # indices
  
  # sdf for each grids # 
  print(f"tet_verts: {tet_verts.shape}, tet_indices: {tet_indices.shape}")
  
  dmt_net = DMTet()
  # grid_res = 64

  # 1) deform but not deformed -> so tet_verts + deform as the deformed pos #
  # 2) deform but not 
  tet_verts = torch.from_numpy(tet_verts).float() # nn_verts x 3 
  # sdf = torch.from_numpy(sdf).float() # nn_verts --> the size of nn_verts # 
  tet_indices = torch.from_numpy(tet_indices).long() # nn_tets x 4 # 
  # __call__(self, pos_nx3, sdf_n, tet_fx4)
  
  obj_sdf_inputs = torch.from_numpy(obj_sdf_inputs).float().squeeze(0)
  obj_sdf_outputs = torch.from_numpy(obj_sdf_outputs).float().squeeze(0)
  
  hand_sdf_inputs = torch.from_numpy(hand_sdf_inputs).float().squeeze(0)
  hand_sdf_outputs = torch.from_numpy(hand_sdf_outputs).float().squeeze(0)
  
#   print()

  print(f"hand_sdf_inputs: {hand_sdf_inputs.size()}, hand_sdf_outputs: {hand_sdf_outputs.size()}")
  hand_verts_inputs, hand_faces_inputs, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), hand_sdf_inputs.cuda(), tet_indices.cuda())
  print(f"verts: {hand_verts_inputs.size()}, faces: {hand_faces_inputs.size()}")
  
  hand_verts_outputs, hand_faces_outputs, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), hand_sdf_outputs.cuda(), tet_indices.cuda())
  print(f"hand_verts_outputs: {hand_verts_outputs.size()}, hand_faces_outputs: {hand_faces_outputs.size()}")
  
  obj_verts_inputs, obj_faces_inputs, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), obj_sdf_inputs.cuda(), tet_indices.cuda())
  print(f"verts: {obj_verts_inputs.size()}, faces: {obj_faces_inputs.size()}")
  
  obj_verts_outputs, obj_faces_outputs, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), obj_sdf_outputs.cuda(), tet_indices.cuda())
  print(f"obj_verts_outputs: {obj_verts_outputs.size()}, obj_faces_outputs: {obj_faces_outputs.size()}")
  
  
  
  sv_mesh_dict = {
    'obj_input_verts': obj_verts_inputs.detach().cpu().numpy(),
    'obj_input_faces': obj_faces_inputs.detach().cpu().numpy(), # deformed_verts #
    'obj_verts_outputs': obj_verts_outputs.detach().cpu().numpy(),
    'obj_faces_outputs': obj_faces_outputs.detach().cpu().numpy(), # deformed_verts #
    'hand_verts_inputs': hand_verts_inputs.detach().cpu().numpy(),
    'hand_faces_inputs': hand_faces_inputs.detach().cpu().numpy(), # deformed_verts #
    'hand_verts_outputs': hand_verts_outputs.detach().cpu().numpy(),
    'hand_faces_outputs': hand_faces_outputs.detach().cpu().numpy(), # deformed_verts #
  }
  # sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000.npy"
#   sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000_res_128.npy"
  logging_dir = "/data2/sim/implicit_ae/logging/00041-stylegan2-rendering-gpus1-batch4-gamma80"
#   sv_mesh_fn = os.path.join(logging_dir, "pred_out_iter_2_batch_5_extracted.npy")
#   sv_mesh_fn = os.path.join(logging_dir, "pred_out_iter_3_batch_0_extracted.npy")
  sv_mesh_fn = os.path.join(logging_dir, "pred_out_iter_56_batch_0_extracted.npy")
  # iter_39_batch_0_nreg
  sv_mesh_fn = os.path.join(logging_dir, "pred_out_iter_63_batch_0_nreg_extracted.npy") # 
  np.save(sv_mesh_fn, sv_mesh_dict)
  print(f"extracted mesh saved to {sv_mesh_fn}")
  
  
#   deform = deform.float()
#   deformed_verts = tet_verts + deform
#   # deformed_verts =  deform
  
#   print(deform_unmasked)
  
#   deformed_verts[deform_unmasked.bool()] = tet_verts[deform_unmasked.bool()]
  # verts, faces, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(tet_verts.cuda(), sdf.cuda(), tet_indices.cuda())
#   verts, faces, uvs, uv_idx, face_to_valid_tet, valid_vert_idx = dmt_net(deformed_verts.cuda(), sdf.cuda(), tet_indices.cuda())
#   print(f"verts: {verts.size()}, faces: {faces.size()}")
  
#   sv_mesh_dict = {
#     'verts': verts.detach().cpu().numpy(),
#     'faces': faces.detach().cpu().numpy(), # deformed_verts #
#   }
#   # sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000.npy"
#   sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/mesh_extracted_00000_res_128.npy"
#   sv_mesh_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results_seq/tets/mesh_extracted_00002_res_128.npy"
#   np.save(sv_mesh_fn, sv_mesh_dict)
#   print(f"extracted mesh saved to {sv_mesh_fn}")
  
  # self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') # * scale
  # self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda') # 
  # 
  # print()

import trimesh
from open3d import io as o3dio
def load_ply_data(ply_fn, rt_normals=False):
    # obj_mesh = o3dio.read_triangle_mesh(ply_fn)
    # obj_verts = np.array(obj_mesh.vertices, dtype=np.float32)
    # obj_faces = np.array(obj_mesh.triangles)
    # # obj_vertex_normals = np.array(obj_mesh.vertex_normals)
    # # obj_face_normals = np.array(obj_mesh.face_normals)

    obj_mesh = trimesh.load(ply_fn, process=False)
    # obj_mesh.remove_degenerate_faces(height=1e-06)

    verts_obj = np.array(obj_mesh.vertices)
    faces_obj = np.array(obj_mesh.faces)
    obj_face_normals = np.array(obj_mesh.face_normals)
    obj_vertex_normals = np.array(obj_mesh.vertex_normals)

    print(f"vertex: {verts_obj.shape}, obj_faces: {faces_obj.shape}, obj_face_normals: {obj_face_normals.shape}, obj_vertex_normals: {obj_vertex_normals.shape}")
    if not rt_normals:
      return verts_obj, faces_obj
    else:
      return verts_obj, faces_obj, obj_vertex_normals

def load_and_save_verts(rt_path):
  ws = 60
  tot_obj_verts =[]
  for i_fr in range(ws):
    cur_fr_obj_nm = f"object_{i_fr}.obj"
    cur_fr_obj_path = os.path.join(rt_path, cur_fr_obj_nm)
    cur_obj_verts, cur_obj_faces = load_ply_data(cur_fr_obj_path)
    tot_obj_verts.append(cur_obj_verts)
  tot_obj_verts = np.stack(tot_obj_verts, axis=0) # ws x nn_obj_verts x 3 -> for obj verts here #
  bundle_obj_verts_sv_fn = os.path.join(rt_path, f"obj_verts_ws_{ws}.npy")
  np.save(bundle_obj_verts_sv_fn, tot_obj_verts)
  print(f"Object vertices saved to {bundle_obj_verts_sv_fn}")
  

def get_penetration_depth_rnk_data(sv_dict_fn):
  sv_dict = np.load(sv_dict_fn, allow_pickle=True).item()
  pred_fn_to_APD = {}
  for cur_fn in sv_dict:
    pred_fn_to_APD[cur_fn] = sv_dict[cur_fn][0]
  sorted_pred_fn_with_APD = sorted(pred_fn_to_APD.items(), key=lambda i: i[1])
  # print(sorted_pred_fn_with_APD)
  # predicted_infos_seq_300_seed_169_tag_rep_res_jts_hoi4d_scissors_t_300_.npy
  sorted_seed_APD_pair = []
  selected_seeds = []
  for cur_item in sorted_pred_fn_with_APD:
    cur_fn = cur_item[0]
    cur_seed =cur_fn.split("/")[-1].split("_")[5]
    cur_seed = int(cur_seed)
    sorted_seed_APD_pair.append((cur_seed, cur_item[1]))
    if cur_item[1] == 0.:
      selected_seeds.append(cur_seed)
  print(sorted_seed_APD_pair)
  print(f"selected_seeds:")
  print(selected_seeds)

# case_flag
def get_meta_info(sv_dict_fn):
  meta_info = np.load(sv_dict_fn, allow_pickle=True).item()
  # ZY20210800001/H1/C9/N12/S33/s01/T2
  case_flag = meta_info["case_flag"]
  print(case_flag)
  return case_flag


# average on all hand vertices #
### not a very accurate metric for penetration here ###
def calculate_penetration_depth(subj_seq, obj_verts, obj_faces): # subj seq -> can be any vertices as well #
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


### not a very accurate metric for penetration here ###
def calculate_penetration_depth_obj_seq(subj_seq, tot_obj_verts, tot_frames_obj_normals, obj_faces): # subj seq -> can be any vertices as well #
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  
  
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  for i_f in range(subj_seq.shape[0]): ## total sequence length ##
    obj_verts = tot_obj_verts[i_f]
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
            )
    
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


### not a very accurate metric for penetration here ###
def calculate_penetration_depth_obj_seq_v2(subj_seq, tot_obj_verts, tot_frames_obj_normals, obj_faces): # subj seq -> can be any vertices as well #
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  
  
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  
  for i_f in range(subj_seq.shape[0]): ## total sequence length ##
    obj_verts = tot_obj_verts[i_f]
    obj_normals = tot_frames_obj_normals[i_f]
    # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
    #         )
    
    cur_subj_seq = subj_seq[i_f]
    # cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #
    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts #
      (np.reshape(cur_subj_seq, (cur_subj_seq.shape[0], 1, 3)) - np.reshape(obj_verts, (1, obj_verts.shape[0], 3))) ** 2, axis=-1
    )
    # dist_cur_subj_to_obj_verts 
    nearest_obj_pts_idx = np.argmin(dist_cur_subj_to_obj_verts, axis=-1) # nn_subj_pts #
    nearest_obj_dist = np.min(dist_cur_subj_to_obj_verts, axis=-1) # nn_subj_pts
    nearest_obj_dist = np.sqrt(nearest_obj_dist)
    cur_pene_depth = np.zeros_like(nearest_obj_dist)
    
    nearest_obj_pts_idx_th = torch.from_numpy(nearest_obj_pts_idx).long().cuda() ### 
    obj_verts_th = torch.from_numpy(obj_verts).float().cuda() ### nn_obj_verts x 3 
    obj_normals_th = torch.from_numpy(obj_normals).float().cuda() ### nn_obj_verts x 3 
    
    ## nn_hand_verts x 3 ##
    nearest_obj_pts = model_util.batched_index_select_ours(obj_verts_th, indices=nearest_obj_pts_idx_th, dim=0) ## nn_hand_verts x 3
    nearest_obj_normals = model_util.batched_index_select_ours(obj_normals_th, indices=nearest_obj_pts_idx_th, dim=0) ## nn_hand_verts x 3
    cur_subj_seq_th = torch.from_numpy(cur_subj_seq).float().cuda()
    
    rel_obj_verts_to_subj_pts = cur_subj_seq_th - nearest_obj_pts
    dot_rel_with_obj_normals = torch.sum(nearest_obj_normals * rel_obj_verts_to_subj_pts, dim=-1)
    cur_subj_seq_in_obj = dot_rel_with_obj_normals < 0.
    cur_subj_seq_in_obj = cur_subj_seq_in_obj.cpu().numpy().astype(np.bool8)
    
    
    cur_pene_depth[cur_subj_seq_in_obj] = nearest_obj_dist[cur_subj_seq_in_obj]
    
    
    tot_penetration_depth.append(cur_pene_depth)
  tot_penetration_depth = np.stack(tot_penetration_depth, axis=0) # nf x nn_subj_pts
  tot_penetration_depth = np.mean(tot_penetration_depth).item()
  return tot_penetration_depth



def calculate_subj_smoothness(joint_seq):
  # joint_seq: nf x nnjoints x 3
  disp_seq = joint_seq[1:] - joint_seq[:-1] # (nf - 1) x nnjoints x 3 #
  disp_seq = np.sum(disp_seq ** 2, axis=-1)
  disp_seq = np.mean(disp_seq)
  # disp_seq = np.
  disp_seq = disp_seq.item()
  return disp_seq


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

  return diff_disp_joints_base_pts.item()

# i_test_seq
def get_test_settings_to_statistics(i_test_seq, test_tag, start_idx=50, ws=60, use_toch=False):
  # optimized_sv_infos_sv_fn_nm = f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}_with_proj_{with_proj}.npy"
  # optimized_infos_sv_dict_seq_0_seed_0_tag_jts_hoi4d_arti_t_400__dist_thres_0.005_with_proj_False.npy
  tot_dist_thres = [0.005, 0.01, 0.02, 0.05, 0.1]
  tot_dist_thres = [0.001, 0.005]
  tot_with_proj = [True, False]
  tot_seeds = range(0, 122, 11)

  if use_toch:
    tot_seeds = [0]
    # tot_with_proj = [False] 
    tot_with_proj = [True, False]
    # tot_dist_thres = [0.005]
    tot_dist_thres = [0.001, 0.005]

  # test_tag = ""
  pred_infos_sv_folder = f"/data2/sim/eval_save/HOI_{cat_ty}/{cat_nm}"
  if cat_nm in ["Scissors"]:
    corr_infos_sv_folder = f"/data2/sim/HOI_Processed_Data_{cat_ty}/{cat_nm}/{cat_nm}"
  else:
    corr_infos_sv_folder = f"/data2/sim/HOI_Processed_Data_{cat_ty}/{cat_nm}"
  test_setting_to_pene_depth = {}
  
  corr_mesh_folder = os.path.join(corr_infos_sv_folder, f"case{i_test_seq}")
  corr_mesh_folder = os.path.join(corr_mesh_folder, "corr_mesh")
  
  tot_frames_obj_verts = []
  tot_frames_obj_normals = []
  # for i_idx in range(start_idx, start_idx + ws):
  #   cur_obj_fn = os.path.join(corr_mesh_folder, f"object_{i_idx}.obj")
  #   if not os.path.exists(cur_obj_fn):
  #     return []
  #   cur_obj_verts, cur_obj_faces, cur_obj_verts_normals = load_ply_data(cur_obj_fn, rt_normals=True) #### load verts and faces jfrom the ply data ##
  #   tot_frames_obj_verts.append(cur_obj_verts)
  #   tot_frames_obj_normals.append(cur_obj_verts_normals)
    
  # tot_frames_obj_verts_np = np.stack(tot_frames_obj_verts, axis=0)

  for seed in tot_seeds:
    for dist_thres in tot_dist_thres:
      for with_proj in tot_with_proj:
        cur_optimized_sv_infos_fn = f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}_with_proj_{with_proj}.npy"
        cur_optimized_sv_infos_fn = f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}_with_proj_{with_proj}_wjts_0.01.npy"
        # f"optimized_infos_sv_dict_seq_{i_test_seq}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}_with_proj_{with_proj}.npy"
        cur_optimized_sv_infos_fn = os.path.join(pred_infos_sv_folder, cur_optimized_sv_infos_fn)
        
        print(f"cur_optimized_sv_infos_fn: {cur_optimized_sv_infos_fn}")
        if not os.path.exists(cur_optimized_sv_infos_fn):
          continue
        
        optimized_infos = np.load(cur_optimized_sv_infos_fn, allow_pickle=True).item()
        if len(tot_frames_obj_verts) == 0:
          for i_idx in range(start_idx, start_idx + ws):
            cur_obj_fn = os.path.join(corr_mesh_folder, f"object_{i_idx}.obj")
            if not os.path.exists(cur_obj_fn):
              return []
            cur_obj_verts, cur_obj_faces, cur_obj_verts_normals = load_ply_data(cur_obj_fn, rt_normals=True) #### load verts and faces jfrom the ply data ##
            tot_frames_obj_verts.append(cur_obj_verts)
            tot_frames_obj_normals.append(cur_obj_verts_normals)
          tot_frames_obj_verts_np = np.stack(tot_frames_obj_verts, axis=0)
          tot_base_pts_trans = optimized_infos['tot_base_pts_trans'] # nn_frames x nn_base_pts x 3 
          tot_base_pts_trans_th = torch.from_numpy(tot_base_pts_trans).float()
          tot_frames_obj_verts_th = torch.from_numpy(tot_frames_obj_verts_np).float()
          diff_base_pts_trans_obj = torch.sum(
            (tot_base_pts_trans_th.unsqueeze(-2) - tot_frames_obj_verts_th.unsqueeze(1)) ** 2, dim=-1 # nn_frames x nn_base x nn_obj
          )
          minn_diff_base_pts_trans_obj, minn_diff_base_pts_trans_obj_idxes = torch.min(diff_base_pts_trans_obj, dim=-1) # nn_frames x nn_base
          tot_frames_obj_verts_th = model_util.batched_index_select_ours(tot_frames_obj_verts_th.cuda(), minn_diff_base_pts_trans_obj_idxes.cuda(), dim=1) # nn_frames x nn_base x 3 
          tot_frames_obj_normals_np = np.stack(tot_frames_obj_normals, axis=0)
          tot_frames_obj_normals_th = torch.from_numpy(tot_frames_obj_normals_np).float()
          tot_frames_obj_normals_th = model_util.batched_index_select_ours(tot_frames_obj_normals_th.cuda(), minn_diff_base_pts_trans_obj_idxes.cuda(), dim=1) # nn_frames x nn_base x 3 

          # tot_frames_obj_normals = tot_frames_obj_normals_th.cpu().numpy().tolist()

          tot_frames_obj_normals = [tot_frames_obj_normals_th[ii].cpu().numpy() for ii in range(tot_frames_obj_normals_th.size(0))]

          tot_frames_obj_verts_np = tot_frames_obj_verts_th.cpu().numpy()
          # tot_frames_obj_verts = tot_frames_obj_verts_np.tolist()
          tot_frames_obj_verts = [tot_frames_obj_verts_np[ii] for ii in range(tot_frames_obj_verts_np.shape[0])]

    

        # if tot_frames_obj_verts 
        
        
        optimized_verts = optimized_infos["hand_verts"]
        if use_toch:
          toch_eval_sv_fn = f"/data2/sim/eval_save/HOI_{cat_ty}_toch/{cat_nm}/{i_test_seq}.npy"
          toch_eval_sv_dict = np.load(toch_eval_sv_fn, allow_pickle=True).item()
          optimized_verts = toch_eval_sv_dict["hand_verts_tot"]
          
        ### calculate penetration depth obj seq ###
        # cur_penetration_depth = calculate_penetration_depth_obj_seq(optimized_verts, tot_frames_obj_verts, tot_frames_obj_normals, cur_obj_faces)
        # calculate_penetration_depth_obj_seq_v2
        st_time = time.time()
        cur_penetration_depth = calculate_penetration_depth_obj_seq_v2(optimized_verts, tot_frames_obj_verts, tot_frames_obj_normals, cur_obj_faces)
        cur_subj_smoothness = calculate_subj_smoothness(optimized_verts)
        cur_moving_consistency = calculate_moving_consistency(tot_frames_obj_verts_np, optimized_verts)
        ed_time = time.time()
        print(f"Time used for calculating penetration depth (v2): {ed_time - st_time}")
        test_setting_to_pene_depth[(seed, dist_thres, with_proj) ] = (cur_penetration_depth, cur_subj_smoothness, cur_moving_consistency)
        print(f"i_test_seq: {i_test_seq}, seed: {seed}, dist_thres: {dist_thres}, with_proj: {with_proj}, penetration_depth: {cur_penetration_depth}, smoothness: {cur_subj_smoothness}, cur_moving_consistency: {cur_moving_consistency}")
  sorted_setting_to_pene_depth = sorted(test_setting_to_pene_depth.items(), key=lambda ii: ii[1][0], reverse=False)
  print(sorted_setting_to_pene_depth[:5])
  return sorted_setting_to_pene_depth

def get_setting_to_stats(st_idx, ed_idx, use_toch=False):
  # f"/data2/sim/eval_save/HOI_Arti/Scissors/setting_to_stats_seq_{i_test_seq}_toch.npy"
  tot_penetrations = []
  tot_smoothness = []
  tot_moving_consistency = []

  for test_idx in range(st_idx, ed_idx):
    if test_idx == 12:
      continue

    if not use_toch:
      cur_sv_dict_fn = f"/data2/sim/eval_save/HOI_{cat_ty}/{cat_nm}/setting_to_stats_seq_{test_idx}.npy"
    else:
      cur_sv_dict_fn = f"/data2/sim/eval_save/HOI_{cat_ty}/{cat_nm}/setting_to_stats_seq_{test_idx}_toch.npy"
    sv_dict = np.load(cur_sv_dict_fn, allow_pickle=True)
    # print(f"Test idx: {test_idx}, statistics: {sv_dict}")

    # cur_sv_dict_fn = f"/data2/sim/eval_save/HOI_Arti/Scissors/setting_to_stats_seq_{test_idx}_toch.npy"
    # sv_dict = np.load(cur_sv_dict_fn, allow_pickle=True)
    # print(f"Test idx: {test_idx}, statistics: {sv_dict}")
    
    if len(sv_dict) > 0:
      print(sv_dict[0])
      cur_stats = sv_dict[0][1]

      tot_penetrations.append(cur_stats[0])
      tot_smoothness.append(cur_stats[1])
      tot_moving_consistency.append(cur_stats[2])

    # cur_sv_dict_fn = f"/data2/sim/eval_save/HOI_Arti/Scissors/setting_to_stats_seq_{test_idx}_toch.npy"
    # sv_dict = np.load(cur_sv_dict_fn, allow_pickle=True)
    # print(f"Test idx: {test_idx}, statistics: {sv_dict}")
    # the 
  avg_penetration = sum(tot_penetrations) / float(len(tot_penetrations))
  avg_smoothness = sum(tot_smoothness) / float(len(tot_smoothness))
  avg_consis = sum(tot_moving_consistency) / float(len(tot_moving_consistency))
  print(f"avg_penetration: {avg_penetration}, avg_smoothness: {avg_smoothness}, avg_consis: {avg_consis}")
  # [8, 11]
  # avg_penetration: 7.019575241429266e-05, avg_smoothness: 2.841970498934643e-06, avg_consis: 4.051069936394924e-06
  # avg_penetration: 4.745464866573457e-05, avg_smoothness: 8.250485916505568e-05, avg_consis: 4.235470646563044e-05

  # Toch
  # avg_penetration: 0.00011692142591831119, avg_smoothness: 6.375887634627968e-05, avg_consis: 3.023650837500706e-05
  # Ours
  # avg_penetration: 5.615660193143412e-05, avg_smoothness: 3.3161883834509354e-06, avg_consis: 3.632244261098094e-06

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



# load clean obj clip data #
import data_loaders.humanml.data.utils as utils
# 
def load_grab_clip_data_clean_obj(clip_seq_idx, more_pert=False, other_noise=False, split='train'):
  mano_model = get_mano_model()
  grab_path = "/data1/sim/GRAB_extracted" # extracted # extracted 
  # split = "test"
  window_size = 60
  singe_seq_path = f"/data1/sim/GRAB_processed/{split}/{clip_seq_idx}.npy"
  clip_clean = np.load(singe_seq_path)
  subj_root_path = '/data1/sim/GRAB_processed_wsubj'
  subj_seq_path = f"{clip_seq_idx}_subj.npy"  

  # load datas # grab path; grab sequences #
  grab_path = "/data1/sim/GRAB_extracted"
  obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
  id2objmesh = []
  obj_meshes = sorted(os.listdir(obj_mesh_path))
  for i, fn in enumerate(obj_meshes):
      id2objmesh.append(os.path.join(obj_mesh_path, fn))
  


  subj_params_fn = os.path.join(subj_root_path, split, subj_seq_path)

  subj_params = np.load(subj_params_fn, allow_pickle=True).item()
  rhand_transl = subj_params["rhand_transl"][:window_size].astype(np.float32)
  rhand_betas = subj_params["rhand_betas"].astype(np.float32)
  rhand_pose = clip_clean['f2'][:window_size].astype(np.float32) ## rhand pose ## # 
  rhand_global_orient = clip_clean['f1'][:window_size].astype(np.float32)
  # rhand_pose = clip_clean['f2'][:window_size].astype(np.float32)


  object_global_orient = clip_clean['f5'].astype(np.float32) ## clip_len x 3 --> orientation 
  object_trcansl = clip_clean['f6'].astype(np.float32)  ## cliplen x 3 --> translation
  
  object_idx = clip_clean['f7'][0].item() # clip len x 3 # clip len x 3 for translations #

  object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
  object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
  object_trcansl_th = torch.from_numpy(object_trcansl).float()
  
  obj_nm = id2objmesh[object_idx]
  obj_mesh = trimesh.load(obj_nm, process=False) # obj mesh obj verts 
  obj_verts = np.array(obj_mesh.vertices)
  obj_vertex_normals = np.array(obj_mesh.vertex_normals)
  obj_faces = np.array(obj_mesh.faces)

  
  obj_verts = torch.from_numpy(obj_verts).float() # 
  obj_verts = torch.matmul(obj_verts.unsqueeze(0), object_global_orient_mtx_th) + object_trcansl_th.unsqueeze(1) ### nn_frames x nn_obj x 3 ### as the object transformed meshes ##
  obj_verts = obj_verts.detach().cpu().numpy() ### nn_frames x nn_obj_verts x 3 ###

  return obj_verts, obj_faces
 


# 
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

#   aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
  aug_trans, aug_rot, aug_pose = 0.01, 0.1, 0.5
  # aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
  # aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3

  # 

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
        
  return rhand_verts.detach().cpu().numpy(), rhand_joints.detach().cpu().numpy()
  

def load_grab_clip_data(clip_seq_idx, more_pert=False, other_noise=False):
  mano_model = get_mano_model()
  grab_path = "/data1/sim/GRAB_extracted" # extracted 
  split = "test"
  window_size = 60
  singe_seq_path = f"/data1/sim/GRAB_processed/{split}/{clip_seq_idx}.npy"
  clip_clean = np.load(singe_seq_path)
  subj_root_path = '/data1/sim/GRAB_processed_wsubj'
  subj_seq_path = f"{clip_seq_idx}_subj.npy"  
  
  subj_params_fn = os.path.join(subj_root_path, split, subj_seq_path)

  subj_params = np.load(subj_params_fn, allow_pickle=True).item()
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
  rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 


  
  rhand_verts, rhand_joints = mano_model(
      torch.cat([rnd_aug_global_orient_var, rnd_aug_pose_var], dim=-1),
      rhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), rnd_aug_transl_var
  )
  ### rhand_joints: for joints ###
  rhand_verts = rhand_verts * 0.001
  rhand_joints = rhand_joints * 0.001
        
  return rhand_verts.detach().cpu().numpy(), rhand_joints.detach().cpu().numpy()
  
def get_idx_to_objname():
  # load datas # grab path; grab sequences #
  grab_path =  "/data1/sim/GRAB_extracted"
  ## grab contactmesh ## id2objmeshname
  obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
  id2objmeshname = []
  obj_meshes = sorted(os.listdir(obj_mesh_path))
  # objectmesh name #
  id2objmeshname = [obj_meshes[i].split(".")[0] for i in range(len(obj_meshes))]
  return id2objmeshname
  
def get_test_idx_to_obj_name():
  id2objmeshname = get_idx_to_objname()
  test_folder = "/data1/sim/GRAB_processed/test/"
  tot_test_seqs = os.listdir(test_folder)
  tot_test_seq_idxes = [ii for ii in range(246)]
  test_seq_idx_to_mesh_nm = {}
  for cur_test_seq_idx in tot_test_seq_idxes:
    cur_test_seq_path = os.path.join(test_folder, f"{cur_test_seq_idx}.npy")
    cur_test_seq = np.load(cur_test_seq_path)
    object_idx = cur_test_seq['f7'][0].item()
    cur_obj_mesh_nm = id2objmeshname[object_idx]
    test_seq_idx_to_mesh_nm[cur_test_seq_idx] = cur_obj_mesh_nm
  return test_seq_idx_to_mesh_nm

def get_category_nns():
  cat_root_folder = "/data2/sim/HOI_Processed_Data_Arti"
  tot_cat_nms = ["Bucket", "Laptop", "Pliers", "Scissors"]
  # Scissors/Scissors
  tot_cat_nms = ["Bucket", "Laptop", "Pliers", "Scissors/Scissors"]
  cat_nm_to_case_nns = {}
  for cur_cat_nm in tot_cat_nms:
    cur_cat_folder = os.path.join(cat_root_folder, cur_cat_nm)
    tot_cases = os.listdir(cur_cat_folder)
    tot_cases = [cur_case_nm for cur_case_nm in tot_cases if "case" in cur_case_nm]
    cat_nm_to_case_nns[cur_cat_nm] = len(tot_cases)
  print(cat_nm_to_case_nns)
  # {'Bucket': 42, 'Laptop': 155, 'Pliers': 187, 'Scissors/Scissors': 93}
  # {'Bottle': 214, 'Bowl': 217, 'Chair': 167, 'Knife': 58, 'Mug': 249, 'ToyCar': 257}
  cat_root_folder = "/data2/sim/HOI_Processed_Data_Rigid"
  tot_cat_nms = ["Bottle",  "Bowl",  "Chair",  "Knife",  "Mug",  "ToyCar"]
  cat_nm_to_case_nns = {}
  for cur_cat_nm in tot_cat_nms:
    cur_cat_folder = os.path.join(cat_root_folder, cur_cat_nm)
    tot_cases = os.listdir(cur_cat_folder)
    tot_cases = [cur_case_nm for cur_case_nm in tot_cases if "case" in cur_case_nm]
    cat_nm_to_case_nns[cur_cat_nm] = len(tot_cases)
  print(cat_nm_to_case_nns)

def get_cat_avg_values():
  # cat_nm_to_arti_objs_nn = {'Bucket': 42, 'Laptop': 155, 'Pliers': 187, 'Scissors': 93}
  cat_nm_to_arti_objs_nn = {'Laptop': 155, 'Pliers': 187, 'Scissors': 93}
  # cat_nm_to_rigid_objs_nn = {'Bottle': 214, 'Bowl': 217, 'Chair': 167, 'Knife': 58, 'Mug': 249, 'ToyCar': 257, 'Kettle': 58}
  cat_nm_to_rigid_objs_nn = {'Bottle': 214, 'Bowl': 217, 'Chair': 167, 'Mug': 249, 'ToyCar': 257, 'Kettle': 58}
  cat_nm_to_objs_nn = {}
  cat_nm_to_objs_nn.update(cat_nm_to_arti_objs_nn)
  cat_nm_to_objs_nn.update(cat_nm_to_rigid_objs_nn)
  ### TOCH 
  # cat_to_penetration_depth = {
  #   "Knife": 29.56, "Bottle": 187.69, "Pliers": 667.96, "Scissors": 11.69, "Bowl": 28.66, "Kettle": 34.3, "Mug": 21.47, "ToyCar": 60.42
  # }
  # cat_to_smoothness = {
  #   "Knife": 9.885, "Bottle": 3.5871, "Pliers": 5.594, "Scissors": 6.376, "Bowl": 10.54, "Kettle": 6.81, "Mug": 10.56, "ToyCar": 2.404
  # }
  # cat_to_ho_motion_consistency = {
  #   "Knife": 20.896, "Bottle": 55.083, "Pliers": 14.3, "Scissors": 3.024, "Bowl": 18.37, "Kettle": 19.7, "Mug": 26.89, "ToyCar": 15.45
  # }
  # tot_sum_value = 0.
  # tot_sum_nns = 0
  # for cat_nm in cat_to_penetration_depth:
  #   cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
  #   cur_cat_penetration_depth = cat_to_penetration_depth[cat_nm]
  #   cur_cat_tot_pene_depth = cur_cat_penetration_depth * float(cur_cat_nn)
  #   tot_sum_nns += cur_cat_nn
  #   tot_sum_value += cur_cat_tot_pene_depth
  # avg_pene_depth = tot_sum_value / float(tot_sum_nns)
  # print(f"Avg_pene_depth: {avg_pene_depth}")
  
  # tot_sum_value = 0.
  # tot_sum_nns = 0
  # for cat_nm in cat_to_smoothness:
  #   cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
  #   cur_cat_smoothness = cat_to_smoothness[cat_nm]
  #   cur_cat_tot_smoothness = cur_cat_smoothness * float(cur_cat_nn)
  #   tot_sum_nns += cur_cat_nn
  #   tot_sum_value += cur_cat_tot_smoothness
  # avg_smoothness = tot_sum_value / float(tot_sum_nns)
  # print(f"Avg_smoothness: {avg_smoothness}")


  # tot_sum_value = 0.
  # tot_sum_nns = 0
  # for cat_nm in cat_to_ho_motion_consistency:
  #   cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
  #   cur_cat_consistency = cat_to_ho_motion_consistency[cat_nm]
  #   cur_cat_consistency = cur_cat_consistency * float(cur_cat_nn)
  #   tot_sum_nns += cur_cat_nn
  #   tot_sum_value += cur_cat_consistency
  # avg_consistency = tot_sum_value / float(tot_sum_nns)
  # print(f"Avg_consistency: {avg_consistency}")
  # # Avg_pene_depth: 147.75575393848462
  # # Avg_smoothness: 6.683753488372093
  # # Avg_consistency: 23.818613653413355


  ### TOCH 
  cat_to_penetration_depth = {
    "Knife": 1.5044, "Bottle": 135.51, "Pliers": 389.75, "Scissors": 5.616, "Bowl": 23.53, "Kettle": 38.64, "Mug": 7.446, "ToyCar": 19.19
  }
  cat_to_smoothness = {
    "Knife":0.1232, "Bottle": 1.9689, "Pliers": 2.249, "Scissors": 0.3316, "Bowl": 1.186, "Kettle": 1.013, "Mug": 0.7445, "ToyCar": 1.066
  }
  cat_to_ho_motion_consistency = {
    "Knife": 5.0841, "Bottle": 35.11, "Pliers": 10.86, "Scissors": 3.632, "Bowl": 4.983, "Kettle": 4.687, "Mug": 3.07, "ToyCar": 2.722
  }
  tot_sum_value = 0.
  tot_sum_nns = 0
  for cat_nm in cat_to_penetration_depth:
    cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
    cur_cat_penetration_depth = cat_to_penetration_depth[cat_nm]
    cur_cat_tot_pene_depth = cur_cat_penetration_depth * float(cur_cat_nn)
    tot_sum_nns += cur_cat_nn
    tot_sum_value += cur_cat_tot_pene_depth
  avg_pene_depth = tot_sum_value / float(tot_sum_nns)
  print(f"Avg_pene_depth: {avg_pene_depth}")
  
  tot_sum_value = 0.
  tot_sum_nns = 0
  for cat_nm in cat_to_smoothness:
    cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
    cur_cat_smoothness = cat_to_smoothness[cat_nm]
    cur_cat_tot_smoothness = cur_cat_smoothness * float(cur_cat_nn)
    tot_sum_nns += cur_cat_nn
    tot_sum_value += cur_cat_tot_smoothness
  avg_smoothness = tot_sum_value / float(tot_sum_nns)
  print(f"Avg_smoothness: {avg_smoothness}")


  tot_sum_value = 0.
  tot_sum_nns = 0
  for cat_nm in cat_to_ho_motion_consistency:
    cur_cat_nn = cat_nm_to_objs_nn[cat_nm]
    cur_cat_consistency = cat_to_ho_motion_consistency[cat_nm]
    cur_cat_consistency = cur_cat_consistency * float(cur_cat_nn)
    tot_sum_nns += cur_cat_nn
    tot_sum_value += cur_cat_consistency
  avg_consistency = tot_sum_value / float(tot_sum_nns)
  print(f"Avg_consistency: {avg_consistency}")
  #   Avg_pene_depth: 87.49058304576144
  # Avg_smoothness: 1.241823330832708
  # Avg_consistency: 9.74805311327832


def get_obj_name_to_test_seqs():
  id2objmeshname = get_idx_to_objname()
  test_folder = "/data1/sim/GRAB_processed/test/"
  test_folder = "/data1/sim/GRAB_processed/train/"
  tot_test_seqs = os.listdir(test_folder)
  # tot_test_seq_idxes = [ii for ii in range(246)]
  tot_test_seq_idxes = [ii for ii in range(1392)]
  test_seq_idx_to_mesh_nm = {}
  mesh_nm_to_test_seqs = {}
  for cur_test_seq_idx in tot_test_seq_idxes:
    cur_test_seq_path = os.path.join(test_folder, f"{cur_test_seq_idx}.npy")
    cur_test_seq = np.load(cur_test_seq_path)
    object_idx = cur_test_seq['f7'][0].item()
    cur_obj_mesh_nm = id2objmeshname[object_idx]
    if cur_obj_mesh_nm in mesh_nm_to_test_seqs:
      mesh_nm_to_test_seqs[cur_obj_mesh_nm].append(cur_test_seq_idx)
    else:
      mesh_nm_to_test_seqs[cur_obj_mesh_nm] = [cur_test_seq_idx]
    # test_seq_idx_to_mesh_nm[cur_test_seq_idx] = cur_obj_mesh_nm
  return mesh_nm_to_test_seqs

# and the test seqs for


def calculate_max_penetration_depth(subj_seq, obj_verts, obj_faces):
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
  if len(obj_verts.shape) == 2:
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
                )
  # subj_seq: nf x nn_subj_pts x 3 #
  tot_penetration_depth = []
  for i_f in range(subj_seq.shape[0]): ## total sequence length ##
    # for i_f in range(10):
    if len(obj_verts.shape) == 3:
        obj_mesh = trimesh.Trimesh(vertices=obj_verts[i_f], faces=obj_faces,)
        cur_obj_verts = obj_verts[i_f]
    else:
        cur_obj_verts = obj_verts
    cur_subj_seq = subj_seq[i_f]
    cur_subj_seq_in_obj = obj_mesh.contains(cur_subj_seq) # nn_subj_pts #

    dist_cur_subj_to_obj_verts = np.sum( # nn_subj_pts x nn_obj_pts #
      (np.reshape(cur_subj_seq, (cur_subj_seq.shape[0], 1, 3)) - np.reshape(cur_obj_verts, (1, cur_obj_verts.shape[0], 3))) ** 2, axis=-1
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



### metrics ### # avg acc metrics ##
def get_acc_metrics(outputs, gt_joints):
  # 
  # outputs: ws x nn_jts x 3 #
  # gt_joints: ws x nn_jts x 3 #
  dist_outputs_gt_joints = np.sqrt(np.sum((outputs - gt_joints) ** 2, axis=-1)) # ws x nn_jts #
  avg_dist_outputs_gt_joints = np.mean(dist_outputs_gt_joints).item()
  return avg_dist_outputs_gt_joints #### m -> the average ###





def calculate_joint_smoothness(joint_seq):
  # joint_seq: nf x nnjoints x 3
  disp_seq = joint_seq[1:] - joint_seq[:-1] # (nf - 1) x nnjoints x 3 #
  disp_seq = np.sum(disp_seq ** 2, axis=-1)
  disp_seq = np.mean(disp_seq)
  # disp_seq = np.
  disp_seq = disp_seq.item()
  return disp_seq


def calculate_proximity_dist(subj_seq, subj_seq_gt, obj_verts, obj_faces):
  # obj_verts: nn_verts x 3 -> numpy array
  # obj_faces: nn_faces x 3 -> numpy array
  # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
  #           process=False, use_embree=True)
#   obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
#             )
  # subj_seq: nf x nn_subj_pts x 3 #
#   tot_penetration_depth = []

  if len(obj_verts.shape) == 2:
  
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
  else:
    # nf x nn_subj_pts x 3 # # nf x nn_subj_pts x nn_obj_pts
    dist_subj_seq_to_obj_verts_gt = np.sum(
        (np.reshape(subj_seq_gt, (subj_seq_gt.shape[0], subj_seq_gt.shape[1], 1, 3)) - np.reshape(obj_verts, (obj_verts.shape[0], 1, obj_verts.shape[1], 3))) ** 2, axis=-1
    )
    minn_dist_subj_seq_to_obj_verts_gt = np.min(dist_subj_seq_to_obj_verts_gt, axis=-1) # nf x nn_subj_pts 
    

    # nf x nn_subj_pts x 3 # # nf x nn_subj_pts x nn_obj_pts
    dist_subj_seq_to_obj_verts = np.sum(
        (np.reshape(subj_seq, (subj_seq.shape[0], subj_seq.shape[1], 1, 3)) - np.reshape(obj_verts, (obj_verts.shape[0], 1, obj_verts.shape[1], 3))) ** 2, axis=-1
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




def calculate_moving_consistency(base_pts_trans, joints_trans):
  # base_pts_trans: nf x nn_base_pts x 3 #
  # joints_trans: nf x nn_jts x 3 #
  base_pts_trans = torch.from_numpy(base_pts_trans).float() ##  
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




if __name__=='__main__':
  # get_cat_avg_values()
  # exit(0)
  # get_category_nns()
  # exit(0)
  # pkl_fn = "/data1/sim/oakink/OakInk-Shape/oakink_shape_v2/apple/C90001/0eec013c90/hand_param.pkl"
  # test_pickle(pkl_fn)

  # test_seq_idx_to_mesh_nm = get_test_idx_to_obj_name()
  # print(test_seq_idx_to_mesh_nm)
  # test_seq_idx_to_mesh_nm_sv_fn = "test_seq_idx_to_mesh_nm.npy"
  # np.save(test_seq_idx_to_mesh_nm_sv_fn, test_seq_idx_to_mesh_nm)
#   # exit(0)

#   mesh_nm_to_test_seqs = get_obj_name_to_test_seqs()
#   print(mesh_nm_to_test_seqs)

#   train_mesh_nm_to_test_seqs = {}
#   test_mesh_nm_to_test_seqs = {}
#   for cur_mesh_nm in mesh_nm_to_test_seqs:
#     cur_idxes = mesh_nm_to_test_seqs[cur_mesh_nm]
#     cur_idxes_nn = len(cur_idxes)
#     train_nns = int(float(cur_idxes_nn) * 0.8)
#     test_nns = cur_idxes_nn - train_nns
#     train_mesh_nm_to_test_seqs[cur_mesh_nm] = cur_idxes[:train_nns]
#     test_mesh_nm_to_test_seqs[cur_mesh_nm] = cur_idxes[train_nns:]
#   np.save("train_mesh_nm_to_test_seqs.npy", train_mesh_nm_to_test_seqs)
#   np.save("test_mesh_nm_to_test_seqs.npy", test_mesh_nm_to_test_seqs)
#   np.save("mesh_nm_to_test_seqs.npy", mesh_nm_to_test_seqs)
#   exit(0)

# /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_inputs.py

  split = 'train'
  clip_seq_idx = 3
  clip_seq_idx = 100

  use_other_noise = False
  use_other_noise = True
  ws = 60

  # split = 'test'
  # clip_seq_idx = 5

  ### get obj data ##
  # obj_verts, obj_faces = load_grab_clip_data_clean_obj(clip_seq_idx, more_pert=False, other_noise=False, split=split)
  # # load_grab_clip_data_clean_subj(clip_seq_idx, pert=False, more_pert=False, other_noise=False):
  # clean_rhand_verts = load_grab_clip_data_clean_subj(clip_seq_idx, pert=False, more_pert=False, other_noise=False, split=split)
  # pert_rhand_verts = load_grab_clip_data_clean_subj(clip_seq_idx, pert=True, more_pert=False, other_noise=False, split=split)

  # /home/xueyi/sim/motion-diffusion-model/utils/test_utils_bundle_inputs.py
  tot_APD = []
  tot_jts_acc = []
  tot_verts_acc = []

  tot_consistency_value = []
  tot_dist_dist_avg = []
  tot_smoothness = []
  for clip_seq_idx in range(246):
    ### get obj data ##
    obj_verts, obj_faces = load_grab_clip_data_clean_obj(clip_seq_idx, more_pert=False, other_noise=False, split=split)
    # load_grab_clip_data_clean_subj(clip_seq_idx, pert=False, more_pert=False, other_noise=False): # other noise = True #
    clean_rhand_verts, clean_hand_joints = load_grab_clip_data_clean_subj(clip_seq_idx, pert=False, more_pert=False, other_noise=use_other_noise, split=split)
    pert_rhand_verts, pert_hand_joints = load_grab_clip_data_clean_subj(clip_seq_idx, pert=True, more_pert=False, other_noise=use_other_noise, split=split)
    # APD = calculate_max_penetration_depth(pert_hand_joints, obj_verts, obj_faces)
    # tot_APD.append(APD)

    APD = 0.
    tot_APD.append(APD)

    jts_acc =  get_acc_metrics(pert_hand_joints, clean_hand_joints)
    verts_acc =  get_acc_metrics(pert_rhand_verts, clean_rhand_verts)
    tot_jts_acc.append(jts_acc)
    tot_verts_acc.append(verts_acc)

    smoothness = calculate_joint_smoothness(pert_hand_joints)
    dist_dist_avg = calculate_proximity_dist(pert_hand_joints[:ws], clean_hand_joints[:ws], obj_verts[:ws], obj_faces)
    consistency_value = calculate_moving_consistency(obj_verts[:ws], pert_hand_joints[:ws])

    tot_smoothness.append(smoothness)
    tot_dist_dist_avg.append(dist_dist_avg)
    tot_consistency_value.append(consistency_value)

    avg_tot_smoothness = sum(tot_smoothness) / float(len(tot_smoothness))
    avg_tot_dist_dist_avg = sum(tot_dist_dist_avg) / float(len(tot_dist_dist_avg))
    avg_tot_consistency_value = sum(tot_consistency_value) / float(len(tot_consistency_value))
    

    avg_APD = sum(tot_APD) / float(len(tot_APD))
    avg_jts_acc = sum(tot_jts_acc) / float(len(tot_jts_acc))
    avg_verts_acc = sum(tot_verts_acc) / float(len(tot_verts_acc))
    print(f"clip_seq_idx: {clip_seq_idx}, APD: {APD}, avg_APD: {avg_APD}, avg_jts_acc: {avg_jts_acc}, avg_verts_acc: {avg_verts_acc}, avg_tot_dist_dist_avg: {avg_tot_dist_dist_avg}, avg_tot_smoothness: {avg_tot_smoothness}, avg_tot_consistency_value: {avg_tot_consistency_value}")
  avg_APD = sum(tot_APD) / float(len(tot_APD))
  avg_jts_acc = sum(tot_jts_acc) / float(len(tot_jts_acc))
  avg_verts_acc = sum(tot_verts_acc) / float(len(tot_verts_acc))

  avg_tot_smoothness = sum(tot_smoothness) / float(len(tot_smoothness))
  avg_tot_dist_dist_avg = sum(tot_dist_dist_avg) / float(len(tot_dist_dist_avg))
  avg_tot_consistency_value = sum(tot_consistency_value) / float(len(tot_consistency_value))
    

  print(f"avg_APD: {avg_APD}, avg_jts_acc: {avg_jts_acc}, avg_verts_acc: {avg_verts_acc}, avg_tot_dist_dist_avg: {avg_tot_dist_dist_avg}, avg_tot_smoothness: {avg_tot_smoothness}, avg_tot_consistency_value: {avg_tot_consistency_value}")

  exit(0)


  cur_clip_obj_hand_data = {
    'pert_rhand_verts': pert_rhand_verts,
    'clean_rhand_verts': clean_rhand_verts,
    'obj_verts': obj_verts,
    'obj_faces': obj_faces,
  }
  
  sv_dict_fn = f"tmp_sv_dict_grab_split_{split}_seq_{clip_seq_idx}.npy"
  np.save(sv_dict_fn, cur_clip_obj_hand_data)
  print(f"data with clip obj and subj saved to {sv_dict_fn}")
  exit(0)

  test_seq = 2
  test_seq = 4
  test_seq = 40
  test_seq = 55
  test_seq = 8
  test_seq = 80
  test_seq = 23
  test_seq = 5
  test_seq = 38
  test_seq = 41
  test_seq = 119
  test_seq = 149
  test_seq = 98
  test_seq = 1
  test_seq = 2
  test_seq = 80
  test_seq = 5
  test_seq = 23
  other_noise = True
  pert_rhand_verts = load_grab_clip_data(test_seq, other_noise=other_noise)
  pert_rhand_verts_more = load_grab_clip_data(test_seq, more_pert=True, other_noise=other_noise)
  pert_rhand_verts = pert_rhand_verts.detach().cpu().numpy()
  pert_rhand_verts_more = pert_rhand_verts_more.detach().cpu().numpy()
  sv_dict = {
    'pert_rhand_verts': pert_rhand_verts,
    'pert_rhand_verts_more': pert_rhand_verts_more
  }
  sv_dict_fn = f"tmp_sv_dict_pert_grab_seq_{test_seq}_other_noise_{other_noise}.npy"
  np.save(sv_dict_fn, sv_dict)
  print(f"pert rhand verts saved to {sv_dict_fn}")
  exit(0)
  
  
  # RIGHT_HAND_POSE_ROOT = /data1/sim/handpose/refinehandpose_right
  # SERVER_DATA_ROOT = /share/datasets/HOI4D_overall/
  
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case7/meta_data.npy"
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case70/meta_data.npy"
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case175/meta_data.npy"
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case174/meta_data.npy"
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case173/meta_data.npy"
  # /data1/sim/HOI_Processed_Data_Arti/case194
  sv_dict_fn = "/data1/sim/HOI_Processed_Data_Arti/case194/meta_data.npy"
  sv_dict_fn = "/data2/sim/HOI_Processed_Data_Arti/Scissors/Scissors/case47/meta_data.npy"
  # /data2/sim/HOI_Processed_Data_Arti # 
  # get_meta_info(sv_dict_fn)
  # exit(0)
  tot_case_flag = []
  ##### Bucket #####
  # ['ZY20210800001/H1/C8/N11/S73/s01/T1', 'ZY20210800001/H1/C8/N12/S73/s01/T1', 'ZY20210800001/H1/C8/N13/S73/s02/T1', 'ZY20210800001/H1/C8/N13/S73/s02/T2', 'ZY20210800001/H1/C8/N14/S73/s02/T2', 'ZY20210800001/H1/C8/N15/S73/s03/T2', 'ZY20210800001/H1/C8/N19/S74/s02/T1', 'ZY20210800001/H1/C8/N21/S74/s03/T2', 'ZY20210800001/H1/C8/N23/S76/s01/T2', 'ZY20210800001/H1/C8/N24/S76/s02/T2', 'ZY20210800001/H1/C8/N25/S76/s02/T1', 'ZY20210800001/H1/C8/N25/S76/s02/T2', 'ZY20210800001/H1/C8/N26/S76/s03/T1', 'ZY20210800001/H1/C8/N28/S78/s01/T2', 'ZY20210800001/H1/C8/N29/S77/s05/T1', 'ZY20210800001/H1/C8/N31/S77/s04/T1', 'ZY20210800001/H1/C8/N31/S77/s04/T2', 'ZY20210800001/H1/C8/N32/S77/s03/T1', 'ZY20210800001/H1/C8/N32/S77/s03/T2', 'ZY20210800001/H1/C8/N33/S77/s03/T1', 'ZY20210800001/H1/C8/N33/S77/s03/T2', 'ZY20210800001/H1/C8/N34/S77/s04/T1', 'ZY20210800001/H1/C8/N34/S77/s04/T2', 'ZY20210800001/H1/C8/N40/S77/s01/T2', 'ZY20210800001/H1/C8/N41/S77/s02/T2', 'ZY20210800002/H2/C8/N11/S80/s01/T1', 'ZY20210800002/H2/C8/N11/S80/s01/T2', 'ZY20210800002/H2/C8/N12/S80/s01/T1', 'ZY20210800002/H2/C8/N12/S80/s01/T2', 'ZY20210800002/H2/C8/N13/S80/s02/T1', 'ZY20210800002/H2/C8/N13/S80/s02/T2', 'ZY20210800002/H2/C8/N14/S80/s02/T1', 'ZY20210800002/H2/C8/N15/S80/s03/T1', 'ZY20210800002/H2/C8/N15/S80/s03/T2', 'ZY20210800003/H3/C8/N38/S74/s02/T2', 'ZY20210800003/H3/C8/N39/S74/s02/T1', 'ZY20210800003/H3/C8/N42/S74/s04/T1', 'ZY20210800004/H4/C8/N12/S71/s02/T1', 'ZY20210800004/H4/C8/N12/S71/s02/T2', 'ZY20210800004/H4/C8/N13/S71/s02/T1', 'ZY20210800004/H4/C8/N14/S71/s03/T1', 'ZY20210800004/H4/C8/N14/S71/s03/T2']
  # T2: 10 - 70

  # st_idx = 8
  # ed_idx = 12
  # st_idx = 14
  # ed_idx = 15
  # cat_nm = "Scissors"
  # cat_nm = "Pliers"
  # cat_ty = "Arti"
  # cat_nm = "ToyCar"
  # cat_ty = "Rigid"
  # st_idx = 0
  # ed_idx = 4
  # use_toch = True
  # # use_toch = False
  # cat_nm = "Bottle"
  # cat_ty = "Rigid"
  # test_tag = "rep_res_jts_hoi4d_bottle_t_300_st_idx_0_"
  # st_idx = 1
  # ed_idx = 4
  # get_setting_to_stats(st_idx, ed_idx, use_toch=use_toch)
  # exit(0)
  
  # 154
  # for case_idx in range(92):
  # for case_idx in range(42):
  # for case_idx in range(187): # jpliers
  # for case_idx in range(154): # 
  #   try:
  #     print(f"Case idx: {case_idx}")
  #     # sv_dict_fn = f"/data2/sim/HOI_Processed_Data_Arti/Scissors/Scissors/case{case_idx}/meta_data.npy"
  #     # sv_dict_fn = f"/data2/sim/HOI_Processed_Data_Arti/Bucket/case{case_idx}/meta_data.npy"
  #     # sv_dict_fn = f"/data2/sim/HOI_Processed_Data_Arti/Pliers/case{case_idx}/meta_data.npy"
  #     sv_dict_fn = f"/data2/sim/HOI_Processed_Data_Arti/Laptop/case{case_idx}/meta_data.npy"
  #     # sv_dict_fn = f"/data2/sim/HOI_Processed_Data_Rigid/Mug/case{case_idx}/meta_data.npy"
  #     cur_case_flag = get_meta_info(sv_dict_fn)
  #     tot_case_flag.append(cur_case_flag)
  #   except:
  #     continue
  # print(f"tot_case_flase")
  # print(tot_case_flag)
  # exit(0)
  
  
  i_test_seq = 0
  test_tag = "jts_hoi4d_arti_t_400_"
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_400_"
  test_tag = "rep_res_jts_hoi4d_toycar_t_300_st_idx_0_"
  start_idx = 50
  ws = 60
  
  st_idx = 0
  ed_idx = 44

  st_idx = 8
  ed_idx = 12

  st_idx = 12
  ed_idx = 15
  cat_nm = "Scissors"
  cat_nm = "Pliers"
  cat_ty = "Arti"
  cat_nm = "ToyCar"
  cat_ty = "Rigid"
  st_idx = 0
  ed_idx = 4

  cat_nm = "Pliers"
  cat_ty = "Arti"
  test_tag = "rep_res_jts_hoi4d_pliers_t_300_st_idx_30_"
  st_idx = 1
  ed_idx = 2

  cat_nm = "Bottle"
  cat_ty = "Rigid"
  test_tag = "rep_res_jts_hoi4d_bottle_t_300_st_idx_0_"
  st_idx = 0
  ed_idx = 5

  cat_nm = "Scissors"
  cat_ty = "Arti"
  test_tag = "rep_res_jts_hoi4d_bottle_t_300_st_idx_0_"
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_30_"
  test_tag = "rep_res_jts_hoi4d_arti_scissors_t_300_st_idx_0_"
  st_idx = 11
  ed_idx = 12

  cat_nm = "Knife"
  cat_ty = "Rigid"
  test_tag = "rep_res_jts_hoi4d_knife_t_300_st_idx_0_"
  st_idx = 0
  ed_idx = 8

  cat_nm = "Chair"
  cat_ty = "Rigid"
  test_tag = "rep_res_jts_hoi4d_chair_t_300_st_idx_0_"
  st_idx = 0
  ed_idx = 8

  # st_idx = 2
  # ed_idx = 3

  # seq_idx_to_setting_to_stats = {} ## get test settings to statistics
  for i_test_seq in range(st_idx, ed_idx):
    ### ours ### # get_test_settings_to_statistics
    cur_seq_setting_to_stats = get_test_settings_to_statistics(i_test_seq, test_tag, start_idx=start_idx, ws=ws)
    # # seq_idx_to_setting_to_stats[i_test_seq] = cur_seq_setting_to_stats
    cur_stats_sv_fn = f"/data2/sim/eval_save/HOI_{cat_ty}/{cat_nm}/setting_to_stats_seq_{i_test_seq}.npy"
    np.save(cur_stats_sv_fn, cur_seq_setting_to_stats)
    print(f"Setting to stats file saved to {cur_stats_sv_fn}")

    # # (0.0016338544664904475, 1.5728545577076147e-06, 3.436529596001492e-06)

    #  (0.0017392054433003068, 4.550241555989487e-06, 7.405976703012129e-06))
    # (0, 0.005, False), (0.0017126222373917699, 3.0001126560819102e-06, 6.4585701693431474e-06))

    # knife

    # ours
    # i_test_seq: 2, seed: 0, dist_thres: 0.001, with_proj: True, penetration_depth: 1.5044701285660267e-05, smoothness: 1.232047111443535e-06, cur_moving_consistency: 5.0841890697483905e-06
    # toch
    # i_test_seq: 2, seed: 0, dist_thres: 0.005, with_proj: True, penetration_depth: 0.002956472337245941, smoothness: 9.885283361654729e-05, cur_moving_consistency: 2.0896763089695014e-05

    ### toch ###
    # use_toch = True
    # try:
    #   cur_seq_setting_to_stats = get_test_settings_to_statistics(i_test_seq, test_tag, start_idx=start_idx, ws=ws, use_toch=use_toch)
    #   # seq_idx_to_setting_to_stats[i_test_seq] = cur_seq_setting_to_stats
    #   cur_stats_sv_fn = f"/data2/sim/eval_save/HOI_{cat_ty}/{cat_nm}/setting_to_stats_seq_{i_test_seq}_toch.npy"
    #   np.save(cur_stats_sv_fn, cur_seq_setting_to_stats)
    #   print(f"Setting to stats file saved to {cur_stats_sv_fn}")
    # except:
    #   continue

  # /data2/sim/eval_save/HOI_Arti/Scissors
  ### stats to saved fn
  # seq_idx_to_setting_to_stats_sv_fn = f"/data2/sim/eval_save/HOI_Arti/Scissors/seq_idx_to_setting_to_stats_v2_basepts.npy"
  # np.save(seq_idx_to_setting_to_stats_sv_fn, seq_idx_to_setting_to_stats)
  # print(f"seq_idx_to_setting_to_stats saved to {seq_idx_to_setting_to_stats_sv_fn}")
  exit(0)
  
  # th_sv_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00000.pt"
  # dmt_dict_00000_res_128.pt
#   th_sv_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results/tets/dmt_dict_00000_res_128.pt"
#   th_sv_fn = "/home/xueyi/sim/MeshDiffusion/nvdiffrec/dmtet_results_seq/tets/dmt_dict_00002.pt"
#   load_data_fr_th_sv(th_sv_fn, grid_res=128)
  
  # rendering gpus, 
  # incorporate dynamics into the process # th_sv_fn # # th_sv_fn #
  # dneoise accleration and 
  # representations, voxels, 
  th_sv_fn = "/data2/sim/implicit_ae/logging/00041-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_2_batch_5.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00045-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_3_batch_0.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00050-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_23_batch_0.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00050-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_56_batch_0.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00052-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_39_batch_0_nreg.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00054-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_22_batch_0_nreg.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00065-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_63_batch_0_nreg.npy"
  th_sv_fn = "/data2/sim/implicit_ae/logging/00065-stylegan2-rendering-gpus1-batch4-gamma80/out_iter_9420_batch_0_nreg.npy"
  load_data_fr_th_sv_fr_pred(th_sv_fn, grid_res=128)
  exit(0)
  
  # meta_data.npy
  
  # rt_path = "/data1/sim/mdm/tmp_data/case5"
  # load_and_save_verts(rt_path)
  
  rt_path = "/home/xueyi/sim/motion-diffusion-model/predicted_infos_fn_to_statistics.npy"
  
  get_penetration_depth_rnk_data(rt_path)
  
