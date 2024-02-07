from manopth.manolayer import ManoLayer
import pickle
import torch
import os
import torch.nn as nn
import numpy as np

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


def read_obj_file_ours(obj_fn, minus_one=False):
  vertices = []
  faces = []
  with open(obj_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if items[0] == 'v':
        cur_verts = items[1:]
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif items[0] == 'f':
        cur_faces = items[1:]
        cur_face_idxes = []
        for cur_f in cur_faces:
          if len(cur_f) == 0:
              continue
          try:
            cur_f_idx = int(float(cur_f.split("/")[0]))
          except:
            cur_f_idx = int(float(cur_f.split("//")[0]))
          cur_face_idxes.append(cur_f_idx - 1 if minus_one else cur_f_idx)
        faces.append(cur_face_idxes)
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  return vertices, faces

def extract_hand_mesh_fr_pkl(pkl_fn):

  manolayer = ManoLayer(
          mano_root='mano/models', use_pca=False, ncomps=45, flat_hand_mean=True, side='right')
  f = open(pkl_fn, 'rb')
  hand_info = pickle.load(f, encoding='latin1')
  f.close()

  theta = nn.Parameter(torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0))
  beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0))
  trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0))
  hand_verts, hand_joints = manolayer(theta, beta)
  kps3d = hand_joints / 1000.0 + trans.unsqueeze(1) # in meters
  hand_transformed_verts = hand_verts / 1000.0 + trans.unsqueeze(1)
  
  faces = manolayer.th_faces
  return kps3d, hand_transformed_verts, faces

def extract_hand_mesh_fr_pkl_all(pkl_rt):
  folder_nm = pkl_rt.split("/")[-1]
  parent_path = "/".join(pkl_rt.split("/")[:-1]) # parent path 
  sv_mesh_folder_nm = "extracted"
  # sv_mesh_folder_nm = os.path.join(parent_path, sv_mesh_folder_nm)
  sv_mesh_folder_nm = "/".join(pkl_rt.split("/")) + "_" + sv_mesh_folder_nm
  os.makedirs(sv_mesh_folder_nm, exist_ok=True)
  
  tot_pkl_fns = os.listdir(pkl_rt)
  tot_pkl_fns = [fn for fn in tot_pkl_fns if fn.endswith(".pickle")]
  for cur_pkl_fn in tot_pkl_fns:
    print(f"processing {cur_pkl_fn}")
    full_pkl_fn = os.path.join(pkl_rt, cur_pkl_fn)
    raw_fn = cur_pkl_fn.split(".")[0]
    kps3d, hand_transformed_verts, faces = extract_hand_mesh_fr_pkl(full_pkl_fn)
    verts_sv_fn = f"{raw_fn}_mesh.obj"
    verts_sv_fn = os.path.join(sv_mesh_folder_nm, verts_sv_fn)
    
    hand_transformed_verts = hand_transformed_verts.detach().cpu().numpy().squeeze(0)
    faces = faces.detach().cpu().numpy()
    save_obj_file(hand_transformed_verts, faces.tolist(), verts_sv_fn, add_one=True)


def read_cad_model_rigid(obj_fn):
  obj_verts, obj_faces = read_obj_file_ours(obj_fn, minus_one=True)
  return obj_verts, obj_faces


from scipy.spatial.transform import Rotation as Rt
import numpy as np

def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
        cont = eval(cont)
    if "dataList" in cont:
        anno = cont["dataList"][num]
    else:
        anno = cont["objects"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_matrix()
    return np.array(rot, dtype=np.float32), trans, dim

# screen, keyboard
def save_transformed_objs(obj_verts, obj_faces, obj_pose_folder):
  folder_nm = obj_pose_folder.split("/")[-1]
  parent_path = "/".join(obj_pose_folder.split("/")[:-1]) # parent path 
  sv_mesh_folder_nm = "extracted"
  sv_mesh_folder_nm = os.path.join(parent_path, sv_mesh_folder_nm)
  os.makedirs(sv_mesh_folder_nm, exist_ok=True)
  
  tot_pkl_fns = os.listdir(obj_pose_folder)
  tot_pkl_fns = [fn for fn in tot_pkl_fns if fn.endswith(".json")]
  # tot_pkl_fns = ["66.json"]
  for cur_pkl_fn in tot_pkl_fns:
    print(f"processing {cur_pkl_fn}")
    full_pkl_fn = os.path.join(obj_pose_folder, cur_pkl_fn)
    raw_fn = cur_pkl_fn.split(".")[0]
    # kps3d, hand_transformed_verts, faces = extract_hand_mesh_fr_pkl(full_pkl_fn)
    # verts_sv_fn = f"{raw_fn}_mesh.obj"
    # verts_sv_fn = os.path.join(sv_mesh_folder_nm, verts_sv_fn)
    
    for i_part in range(len(obj_verts)):
      verts_sv_fn = f"{raw_fn}_part_{i_part}_mesh.obj"
      verts_sv_fn = os.path.join(sv_mesh_folder_nm, verts_sv_fn)
      
      cur_part_obj_verts = obj_verts[i_part]
      cur_part_obj_faces = obj_faces[i_part]
    
      rot, trans, scale = read_rtd(full_pkl_fn, num=i_part)
      transformed_verts = cur_part_obj_verts # * np.reshape(scale, (1, 3))
      print(f"rot: {rot.shape}, transformed_verts: {transformed_verts.shape}")
      transformed_verts = np.matmul(rot, transformed_verts.T).T
      transformed_verts = transformed_verts + np.reshape(trans, (1, 3))
      save_obj_file(transformed_verts, cur_part_obj_faces, verts_sv_fn, add_one=True)


def save_transformed_objs_all(cad_model_fn, obj_pose_folder):
  obj_verts = []
  obj_faces = []
  for i_part in range(len(cad_model_fn)):
    
    cur_part_obj_verts, cur_part_obj_faces = read_cad_model_rigid(cad_model_fn[i_part])
    obj_verts.append(cur_part_obj_verts)
    obj_faces.append(cur_part_obj_faces)
    
  save_transformed_objs(obj_verts, obj_faces, obj_pose_folder)
    

if __name__=='__main__':
  # pkl_rt = "/data1/sim/sample_ho/handpose/refinehandpose_right/ZY20210800001/H1/C1/N19/S100/s02/T1"
  pkl_rt = "/data1/sim/sample_ho/handpose/refinehandpose_right/ZY20210800001/H1/C3/N01/S54/s05/T2"
  # extract_hand_mesh_fr_pkl_all(pkl_rt)
  
  obj_pose_folder = "/data1/sim/sample_ho/HOI4D_annotations/ZY20210800001/H1/C1/N19/S100/s02/T1/objpose"
  cad_model_fn = ["/data1/sim/sample_ho/HOI4D_CAD_Model_for_release/rigid/ToyCar/019.obj"]
  
  cad_model_fn = ["/data1/sim/sample_ho/HOI4D_CAD_Model_for_release/articulated/Laptop/001/objs/new-1-align.obj", "/data1/sim/sample_ho/HOI4D_CAD_Model_for_release/articulated/Laptop/001/objs/new-0-align.obj" ]
  obj_pose_folder = "/data1/sim/sample_ho/HOI4D_annotations/ZY20210800001/H1/C3/N01/S54/s05/T2/objpose"
  
  save_transformed_objs_all(cad_model_fn, obj_pose_folder)
    
    
