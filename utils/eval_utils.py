import os
import numpy as np
from manopth.manolayer import ManoLayer


import  trimesh

def get_data_cat_type_to_neval_idxes():
  ToyCar_neval_idxes = [
    0
  ]

def load_ply_data(ply_fn):
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
    return verts_obj, faces_obj


def save_obj_file(vertices, face_list, obj_fn, add_one=False):
  with open(obj_fn, "w") as wf:
    for i_v in range(vertices.shape[0]):
      cur_v_values = vertices[i_v]
      wf.write("v")
      for i_v_v in range(cur_v_values.shape[0]):
        wf.write(f" {float(cur_v_values[i_v_v].item())}")
      wf.write("\n")
    for i_f in range(len(face_list)):
      cur_face_idxes = face_list[i_f]
      wf.write("f")
      for cur_f_idx in range(len(cur_face_idxes)):
        wf.write(f" {cur_face_idxes[cur_f_idx] if not add_one else cur_face_idxes[cur_f_idx] + 1}")
      wf.write("\n")
    wf.close()

def get_binvox_data(root):
  obj_fns = os.listdir(root)
  obj_fns = [fn for fn in obj_fns if fn.endswith(".obj")]
  cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
  vox_size = 64
  for obj_fn in obj_fns:
    cur_obj_fn = os.path.join(root, obj_fn)
    os.system(f"{cuda_voxelizer_path} -f {cur_obj_fn} -s {vox_size}")
    
# def get_binvox_data(root="/share/xueyi/proj_data/Motion_Aligned_2", sv_root="", shape_type="oven", shape_idxes=None):
#     os.makedirs(sv_root, exist_ok=True)
#     root = os.path.join(root, shape_type)
#     shape_idxes = os.listdir(root) 
#     shape_idxes = sorted(shape_idxes)
#     # shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]
#     shape_idxes = [fn for fn in shape_idxes if os.path.isdir(os.path.join(root, fn))]

#     sv_root = os.path.join(sv_root, shape_type)
#     os.makedirs(sv_root, exist_ok=True) # make motion category folder
    
#     for shp_idx in shape_idxes:
#       cur_shape_folder = os.path.join(root, shp_idx)
#       cur_shape_sv_folder = os.path.join(sv_root, shp_idx)
#       os.makedirs(cur_shape_sv_folder, exist_ok=True) # make shape folder

#       cur_shape_part_folder = cur_shape_folder
#       cur_shape_part_sv_folder = cur_shape_sv_folder
      
#       # cur_parts = os.listdir(cur_shape_folder)
#       # cur_parts = [fn for fn in cur_parts if os.path.isdir(os.path.join(cur_shape_folder, fn))]
#       # for cur_part in cur_parts: # current parts...
#       # cur_shape_part_folder = os.path.join(cur_shape_folder, cur_part)
#       # cur_shape_part_sv_folder = os.path.join(cur_shape_sv_folder, cur_part)
#       # os.makedirs(cur_shape_part_sv_folder, exist_ok=True) # make part folder
      
#       obj_files = os.listdir(cur_shape_part_folder)
#       obj_files = [fn for fn in obj_files if fn.endswith(".obj")]
      
#       cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
      
#       for obj_fn in obj_files:

#         cur_obj_fn = os.path.join(cur_shape_part_folder, obj_fn)

#         if n_scales == 1: # no scale
#           cur_obj_sv_fn = os.path.join(cur_shape_part_sv_folder, obj_fn)

#           os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
#           # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_obj_sv_fn} -s {vox_size}")
#           os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")
#         else:
#           cur_verts, cur_faces = data_utils.read_obj_file_ours(cur_obj_fn)
#           if cur_verts.shape[0] == 0 or len(cur_faces) == 0:
#             continue
#           for i_s in range(n_scales + 1):
#             cur_scaled_sample_obj_fn = obj_fn.split(".")[0] + f"_s_{i_s}.obj"
#             cur_scaled_sample_fn = os.path.join(cur_shape_part_sv_folder, cur_scaled_sample_obj_fn)
#             cur_scaled_verts = apply_random_scaling(cur_verts, with_scale=True if i_s >= 1 else False)
#             data_utils.save_obj_file(cur_scaled_verts, cur_faces, obj_fn=cur_scaled_sample_fn)
#             # os.system(f"/home/xueyi/gen/cuda_voxelizer/build/cuda_voxelizer -f {cur_scaled_sample_fn} -s {vox_size}")
#             os.system(f"{cuda_voxelizer_path} -f {cur_scaled_sample_fn} -s {vox_size}")

#     # shape_idxes = 


## the scale of the binvox data ##
# how to get binvox data #
# how to get intersection binvox data # what 


def get_mano_model(nn_hand_params=24):
  ### start optimization ###
  # setup MANO layer
  use_pca = True if nn_hand_params < 45 else False
  mano_path = "/data1/sim/mano_models/mano/models"
  mano_layer = ManoLayer(
      flat_hand_mean=True,
      side='right',
      mano_root=mano_path, # mano_root #
      ncomps=nn_hand_params, # hand params # 
      use_pca=use_pca, # pca for pca #
      root_rot_mode='axisang',
      joint_rot_mode='axisang'
  ).cuda()
  return mano_layer

def extract_mesh_from_optimized_dict(root_path, seq_idx, seed, test_tag, dist_thres, mano_layer):
  optimized_sv_dict_fn = f"optimized_infos_sv_dict_seq_{seq_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}.npy"
  optimized_sv_dict_fn = os.path.join(root_path, optimized_sv_dict_fn)
  
  # hand_faces: nn_hand_faces x 3 #
  hand_faces = mano_layer.th_faces.squeeze(0).detach().cpu().numpy()
  
  # hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
  #     beta_var.unsqueeze(1).repeat(1, nn_frames, 1).view(-1, 10), transl_var)
  # hand_verts = hand_verts.view( nn_frames, 778, 3) * 0.001
  # hand_joints = hand_joints.view(nn_frames, -1, 3) * 0.001
  
  optimized_dict_infos = np.load(optimized_sv_dict_fn, allow_pickle=True).item()
  ### optimized infos after all optimizations 
  hand_verts = optimized_dict_infos["hand_verts"]
  # hand_joints = optimized_dict_infos["hand_joints"]
  cur_optimized_seq_sv_folder = f"seq_{seq_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}"
  cur_optimized_seq_sv_folder = os.path.join(root_path, cur_optimized_seq_sv_folder)
  os.makedirs(cur_optimized_seq_sv_folder, exist_ok=True)
  for i_fr in range(hand_verts.shape[0]):
    cur_hand_verts = hand_verts[i_fr]
    cur_hand_sv_fn = f"hand_{i_fr}_full_opt.obj"
    cur_hand_sv_fn = os.path.join(cur_optimized_seq_sv_folder, cur_hand_sv_fn)
    save_obj_file(cur_hand_verts, hand_faces.tolist(), cur_hand_sv_fn, add_one=True)
    
  ### optimized infos after all optimizations 
  hand_verts_bf_proj = optimized_dict_infos["bf_proj_verts"]
  # hand_joints = optimized_dict_infos["hand_joints"]
  # cur_optimized_seq_sv_folder = f"seq_{seq_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}"
  # cur_optimized_seq_sv_folder = os.path.join(root_path, cur_optimized_seq_sv_folder)
  # os.makedirs(cur_optimized_seq_sv_folder, exist_ok=True)
  for i_fr in range(hand_verts_bf_proj.shape[0]):
    cur_hand_verts = hand_verts_bf_proj[i_fr]
    cur_hand_sv_fn = f"hand_{i_fr}_bf_proj.obj"
    cur_hand_sv_fn = os.path.join(cur_optimized_seq_sv_folder, cur_hand_sv_fn)
    save_obj_file(cur_hand_verts, hand_faces.tolist(), cur_hand_sv_fn, add_one=True)
    
  ### optimized infos after all optimizations 
  hand_verts_ct_proj = optimized_dict_infos["bf_ct_verts"]
  # hand_joints = optimized_dict_infos["hand_joints"]
  # cur_optimized_seq_sv_folder = f"seq_{seq_idx}_seed_{seed}_tag_{test_tag}_dist_thres_{dist_thres}"
  # cur_optimized_seq_sv_folder = os.path.join(root_path, cur_optimized_seq_sv_folder)
  # os.makedirs(cur_optimized_seq_sv_folder, exist_ok=True)
  for i_fr in range(hand_verts_ct_proj.shape[0]):
    cur_hand_verts = hand_verts_ct_proj[i_fr]
    cur_hand_sv_fn = f"hand_{i_fr}_bf_ct.obj"
    cur_hand_sv_fn = os.path.join(cur_optimized_seq_sv_folder, cur_hand_sv_fn)
    save_obj_file(cur_hand_verts, hand_faces.tolist(), cur_hand_sv_fn, add_one=True)

def get_obj_verts_faces(obj_mesh_sv_folder, start_idx=50, ws=60):
  tot_obj_verts = []
  for i_fr in range(start_idx, start_idx + ws):
    cur_fr_obj_fn = os.path.join(obj_mesh_sv_folder, f"object_{i_fr}.obj")
    cur_fr_verts, cur_fr_faces = load_ply_data(cur_fr_obj_fn)
    tot_obj_verts.append(cur_fr_verts)
  return tot_obj_verts, cur_fr_faces

def merge_mesh_list(verts_list, faces_list):
  tot_verts = []
  tot_faces = []
  nn_tot_verts = 0
  for cur_verts, cur_faces in zip(verts_list, faces_list):
    tot_verts.append(cur_verts)
    tot_faces.append(cur_faces + nn_tot_verts)
    nn_tot_verts += cur_verts.shape[0]
  tot_verts = np.concatenate(tot_verts, axis=0)
  tot_faces = np.concatenate(tot_faces, axis=0)
  return tot_verts, tot_faces

def get_merged_fns(tot_obj_verts, obj_faces, root, start_idx=50, ws=60):
  for i_fr in range(ws):
    cur_hand_obj_fn = f"hand_{i_fr}_full_opt.obj"
    cur_hand_obj_fn = os.path.join(root, cur_hand_obj_fn)
    cur_hand_verts, cur_hand_faces = load_ply_data(cur_hand_obj_fn)
    tot_verts, tot_faces = merge_mesh_list([cur_hand_verts, tot_obj_verts[i_fr]], [cur_hand_faces, obj_faces])
    cur_merged_obj_fn = f"merged_{i_fr}_full_opt.obj"
    cur_merged_obj_fn = os.path.join(root, cur_merged_obj_fn)
    save_obj_file(tot_verts, tot_faces.tolist(), cur_merged_obj_fn, add_one=True)
    
    
    cur_hand_obj_fn = f"hand_{i_fr}_bf_proj.obj"
    cur_hand_obj_fn = os.path.join(root, cur_hand_obj_fn)
    cur_hand_verts, cur_hand_faces = load_ply_data(cur_hand_obj_fn)
    tot_verts, tot_faces = merge_mesh_list([cur_hand_verts, tot_obj_verts[i_fr]], [cur_hand_faces, obj_faces])
    cur_merged_obj_fn = f"merged_{i_fr}_bf_proj.obj"
    cur_merged_obj_fn = os.path.join(root, cur_merged_obj_fn)
    save_obj_file(tot_verts, tot_faces.tolist(), cur_merged_obj_fn, add_one=True)
    
def get_binvox_data_merged_files(root):
  obj_fns = os.listdir(root)
  obj_fns = [fn for fn in obj_fns if fn.endswith(".obj")]
  cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
  vox_size = 64
  obj_fns = [fn for fn in obj_fns if "merged_" in fn]
  ### use obj_fns for binvoxing ###
  for obj_fn in obj_fns:
    cur_obj_fn = os.path.join(root, obj_fn)
    os.system(f"{cuda_voxelizer_path} -f {cur_obj_fn} -s {vox_size}")

def get_normalized_volumes(verts, volumes):
  maxx_hand_verts = np.max(verts, axis=0)
  minn_hand_verts = np.min(verts, axis=0)
  extents_hand_verts = maxx_hand_verts - minn_hand_verts
  V_hand_verts = extents_hand_verts[0].item() * extents_hand_verts[1].item() * extents_hand_verts[2].item()
  
  N_hand_verts = np.sum(volumes.astype(np.float32)).item() # 
  V_voxes = volumes.shape[0] * volumes.shape[0] * volumes.shape[0]
  
  N_hand_verts_normed_volume = float(N_hand_verts) / float(V_voxes) * float(V_hand_verts)
  return N_hand_verts_normed_volume


import utils.binvox_rw as binvox_rw
### Get the obj_fn and xxx ###
def get_intersection_volumes_here(root_folder, start_idx=50):
  # voxel_model_file = open(name_list[idx], 'rb') ### voxel_model_file
  # 		voxel_model_64_crude = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)
  # /data2/sim/eval_save/HOI_Arti/Scissors/seq_6_seed_66_tag_jts_rep_hoi4d_arti_t_300__dist_thres_0.005/merged_59_bf_proj.obj_64.binvox
  cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
  ws = 60
  tot_overlapped_volumes = []
  #  for i_ws in range(ws):
  #  for i_ws in range(start_idx, start_idx + ws): ## start_idx + ws ##
  for i_ws in range(0, 0 + ws): ## start_idx + ws ##
    ### binvox fiels here ### ### root folder for full opt ##
    ### Load hand binvox ###
    cur_ws_hand_vox_full_opt = os.path.join(root_folder, f"hand_{i_ws}_full_opt.obj_64.binvox")
    cur_ws_hand_vox_full_opt = open(cur_ws_hand_vox_full_opt, "rb")
    cur_ws_hand_vox_full_opt = binvox_rw.read_as_3d_array(cur_ws_hand_vox_full_opt).data.astype(np.uint8)

    ### Load hand mesh ###
    cur_ws_hand_mesh = os.path.join(root_folder, f"hand_{i_ws}_full_opt.obj")
    cur_ws_hand_verts, _ = load_ply_data(cur_ws_hand_mesh) # 
    
    ## hand volumes ##
    N_hand_verts_normed_volume = get_normalized_volumes(cur_ws_hand_verts, cur_ws_hand_vox_full_opt)
    
    
    
    ### Load hand binvox ###
    cur_ws_object_vox = os.path.join(root_folder, f"object_{i_ws + start_idx}.obj_64.binvox")
    cur_ws_object_vox = open(cur_ws_object_vox, "rb")
    cur_ws_object_vox = binvox_rw.read_as_3d_array(cur_ws_object_vox).data.astype(np.uint8)
    # tot_V:  w x h x depth -> the volumne of jhe 
    
    ### Load hand mesh ###
    cur_ws_obj_mesh = os.path.join(root_folder, f"object_{i_ws + start_idx}.obj")
    cur_ws_obj_verts, _ = load_ply_data(cur_ws_obj_mesh) # 
    
    
    
    ## hand volumes ##
    N_obj_verts_normed_volume = get_normalized_volumes(cur_ws_obj_verts, cur_ws_object_vox)
    
    
    
    ### Load hand binvox ###
    cur_ws_merged_vox = os.path.join(root_folder, f"merged_{i_ws}_full_opt.obj_64.binvox")
    cur_ws_merged_vox = open(cur_ws_merged_vox, "rb")
    cur_ws_merged_vox = binvox_rw.read_as_3d_array(cur_ws_merged_vox).data.astype(np.uint8)
    # tot_V:  w x h x depth -> the volumne of jhe 
    
    ### Load hand mesh ###
    cur_ws_merged_mesh = os.path.join(root_folder, f"merged_{i_ws}_full_opt.obj")
    cur_ws_merged_verts, _ = load_ply_data(cur_ws_merged_mesh) # 
    
    
    
    ## hand volumes ##
    N_merged_verts_normed_volume = get_normalized_volumes(cur_ws_merged_verts, cur_ws_merged_vox)
    
    
    overlapped_volumes = max(0., N_hand_verts_normed_volume + N_obj_verts_normed_volume - N_merged_verts_normed_volume)
    
    print(f"i_ws: {i_ws}, overlapped_volumes: {overlapped_volumes}")
    tot_overlapped_volumes.append(overlapped_volumes)

  avg_overlapped_volumes = sum(tot_overlapped_volumes) / float(len(tot_overlapped_volumes))
  return avg_overlapped_volumes
  

## get tot obj voxes ##
## get tot obj voxes ##
def get_tot_obj_voxes(obj_fn_folder, root):
  # object_xxx int([7:])
  tot_obj_fns = os.listdir(obj_fn_folder)
  tot_obj_fns = [fn for fn in tot_obj_fns if fn.endswith(".obj") and "object" in fn]
  tot_obj_idxes = [int(fn.split(".")[0][7:]) for fn in tot_obj_fns]
  for cur_obj_idx in tot_obj_idxes:
    cur_obj_fn = os.path.join(obj_fn_folder, f"object_{cur_obj_idx}.obj")
    cur_obj_sv_fn = os.path.join(root, f"object_{cur_obj_idx}.obj")
    os.system(f"cp {cur_obj_fn} {cur_obj_sv_fn}")
    cuda_voxelizer_path = "/home/zhangji/equiapp/code/cuda_voxelizer-0.4.8/build/cuda_voxelizer"
    vox_size = 64
    os.system(f"{cuda_voxelizer_path} -f {cur_obj_sv_fn} -s {vox_size}")

## eval_utils 

if __name__=='__main__':
  mano_layer = get_mano_model()
  root_path = "/data2/sim/eval_save/HOI_Arti/Scissors"
  test_tag = "jts_hoi4d_arti_t_400_"
  test_tag = "jts_rep_hoi4d_arti_t_300_"
  seed = 66
  st_idx = 6
  ed_idx = 7
  tot_dist_thres = [0.005]
  
  # for seq_idx in range(st_idx, ed_idx):
  #   for dist_thres in tot_dist_thres:
  #     extract_mesh_from_optimized_dict(root_path, seq_idx, seed, test_tag, dist_thres, mano_layer)
  
  start_idx = 50
  root = "/data2/sim/eval_save/HOI_Arti/Scissors/seq_6_seed_66_tag_jts_rep_hoi4d_arti_t_300__dist_thres_0.005"
  # get_binvox_data(root)
  
  obj_mesh_sv_folder = "/data2/sim/HOI_Processed_Data_Arti/Scissors/Scissors/case6/corr_mesh"
  # tot_verts, obj_faces = get_obj_verts_faces(obj_mesh_sv_folder, start_idx=50, ws=60)
  # get_merged_fns(tot_verts, obj_faces, root, start_idx=50, ws=60)
  
  ### get binvox data ###
  # get_binvox_data_merged_files(root)
  
  ### obj fn folder and get obj voxes ##
  # obj_fn_folder = "/data2/sim/HOI_Processed_Data_Arti/Scissors/Scissors/case6/corr_mesh"
  # root = "/data2/sim/eval_save/HOI_Arti/Scissors/seq_6_seed_66_tag_jts_rep_hoi4d_arti_t_300__dist_thres_0.005"
  # get_tot_obj_voxes(obj_fn_folder, root)


  avg_intersection_volume = get_intersection_volumes_here(root, start_idx=50)
  print(f"avg_intersection_volume: {avg_intersection_volume}")