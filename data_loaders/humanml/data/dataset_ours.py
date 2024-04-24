import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os, glob
from data_loaders.humanml.data.utils import random_rotate_np
from manopth.manolayer import ManoLayer
import utils
import pickle

import data_loaders.humanml.data.utils as utils
# cp -r 

import random
import trimesh
from scipy.spatial.transform import Rotation as R

import utils.common_utils as common_utils


from utils.anchor_utils import masking_load_driver, anchor_load_driver, recover_anchor_batch



def load_ply_data(ply_fn):
    obj_mesh = trimesh.load(ply_fn, process=False)
    # obj_mesh.remove_degenerate_faces(height=1e-06)

    verts_obj = np.array(obj_mesh.vertices)
    faces_obj = np.array(obj_mesh.faces)
    obj_face_normals = np.array(obj_mesh.face_normals)
    obj_vertex_normals = np.array(obj_mesh.vertex_normals)

    print(f"vertex: {verts_obj.shape}, obj_faces: {faces_obj.shape}, obj_face_normals: {obj_face_normals.shape}, obj_vertex_normals: {obj_vertex_normals.shape}")
    return verts_obj, faces_obj




class GRAB_Dataset_V19(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset ##
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts ## 0.01, 0.05, 0.3 ##
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.inst_normalization = args.inst_normalization
        self.args = args
        
        self.denoising_stra = args.denoising_stra
        
        # load datas
        # grab_path =  "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
        obj_mesh_path = "data/grab/object_meshes"
        id2objmeshname = [] 
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        # objectmesh name #
        id2objmeshname = [obj_meshes[i].split(".")[0] for i in range(len(obj_meshes))]
        self.id2objmeshname = id2objmeshname
        
        
        self.aug_trans_T = 0.05
        self.aug_rot_T = 0.3
        self.aug_pose_T = 0.5
        self.aug_zero = 1e-4 if self.model_type not in ['v1_wsubj_wjointsv24', 'v1_wsubj_wjointsv25'] else 0.01
        
        self.sigmas_trans = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_trans_T), self.num_sche_steps
        ))
        self.sigmas_rot = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_rot_T), self.num_sche_steps
        ))
        self.sigmas_pose = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_pose_T), self.num_sche_steps
        ))
        
        print(f"[Train dataset] data_folder: {data_folder}")
        
        self.data_folder = data_folder
        # self.subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
        self.subj_data_folder = data_folder + "_wsubj"
        # self.mano_path = "/data1/sim/mano_models/mano/models" 
        self.mano_path = "manopth/mano/models" ### mano_path
        # self.aug = True
        self.aug = args.augment
        # self.use_anchors = False
        self.use_anchors = args.use_anchors
        # self.args = args
        
        # grab path extracted #
        # self.grab_path = "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')
        id2objmesh = []
        ''' Get idx to mesh path ''' 
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        ''' Get idx to mesh path ''' 
        
        ## obj root folder; obj p
        # ### Load field data from root folders ###
        # self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        # self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        ## TODO: add thsoe params to args
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
    
        mano_pkl_path = os.path.join(self.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_pkl_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.template_verts = np.array(mano_model['v_template'])
        self.template_faces = np.array(mano_model['f'])
        self.template_joints = np.array(mano_model['J'])
        #### finger tips; ####
        self.template_tips = self.template_verts[[745, 317, 444, 556, 673]]
        self.template_joints = np.concatenate([self.template_joints, self.template_tips], axis=0)
        #### template verts ####
        self.template_verts = self.template_verts * 0.001
        #### template joints ####
        self.template_joints = self.template_joints * 0.001 # nn_joints x 3 #
        # condition on template joints for current joints #
        
        # normalization and so so # combnations of those quantities ######## 
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        self.fingers_stats = [
            [16, 15, 14, 13, 0],
            [17, 3, 2, 1, 0],
            [18, 6, 5, 4, 0],
            [19, 12, 11, 10, 0],
            [20, 9, 8, 7, 0]
        ]
        # 5 x 5 states, the first dimension is the finger index
        self.fingers_stats = np.array(self.fingers_stats, dtype=np.int32)
        self.canon_obj = True
        
        self.dir_stra = "vecs" # "rot_angles", "vecs"
        # self.dir_stra = "rot_angles"
        # distance # 21 distances? # --> texture map like data.. ## nanshou  ##
        
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=24,
            use_pca=True,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )
        
        # ### Load field data from root folders ### ## obj root folder ##
        # self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        # self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # anchor_load_driver, masking_load_driver #
        # use_anchors, self.hand_palm_vertex_mask #
        # if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
        #     # anchor_load_driver, masking_load_driver #
        #     inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
        #     fvi, aw, _, _ = anchor_load_driver(inpath)
        #     self.face_vertex_index = torch.from_numpy(fvi).long()
        #     self.anchor_weight = torch.from_numpy(aw).float()
            
        #     anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
        #     palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
        #     hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
        #     # self.hand_palm_vertex_mask for hand palm mask #
        #     self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
        
        ## actions taken 
        # self.clip_sv_folder = os.path.join(data_folder, f"{split}_clip")
        # os.makedirs(self.clip_sv_folder, exist_ok=True)
        
        # if args.train_all_clips:
        #     for split in ["train", "val", "test"]:
        #         files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        #         files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
        #         # if self.load_meta:
        #         for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
        #             # if split != 'train' and split != 'val' and i_f >= 100:
        #             #     break
        #             # if split == 'train':
        #             print(f"split: {split}, loading {i_f} / {len(files_clean)}")
        #             base_nm_f = os.path.basename(f)
        #             base_name_f = base_nm_f.split(".")[0]
        #             cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
        #             cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
        #             cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
        #             cur_clip_len = cur_clip_meta_data['clip_len']
        #             clip_len = (cur_clip_len - window_size) // step_size + 1
        #             clip_len = max(clip_len, 0)
        #             if self.args.only_first_clip:
        #                 clip_len = min(clip_len, 1)
        #             print(f"cur_clip_len: {cur_clip_len}, clip_len: {clip_len}, window_size: {window_size}")
        #             self.clips.append((self.len, self.len+clip_len,  f
        #                 ))
        #             self.len += clip_len # len clip len
        # else:
        files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        #### filter files_clean here ####
        files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
        
        # if self.load_meta:
        for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
            # if split != 'train' and split != 'val' and i_f >= 100:
            #     break
            if split == 'train':
                print(f"loading {i_f} / {len(files_clean)}")
            base_nm_f = os.path.basename(f)
            base_name_f = base_nm_f.split(".")[0]
            cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
            cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
            cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
            cur_clip_len = cur_clip_meta_data['clip_len']
            clip_len = (cur_clip_len - window_size) // step_size + 1
            clip_len = max(clip_len, 0)
            if self.args.only_first_clip:
                clip_len = min(clip_len, 1)
            print(f"cur_clip_len: {cur_clip_len}, clip_len: {clip_len}, window_size: {window_size}")
            self.clips.append((self.len, self.len+clip_len,  f
                ))
            self.len += clip_len # len clip len
        # else:
        #     for i_f, f in enumerate(files_clean):
        #         if split == 'train':
        #             print(f"loading {i_f} / {len(files_clean)}")
        #         if split != 'train' and i_f >= 100:
        #             break
        #         if args is not None and args.debug and i_f >= 10:
        #             break
        #         clip_clean = np.load(f)
        #         pert_folder_nm = split + '_pert'
        #         if args is not None and not args.use_pert:
        #             pert_folder_nm = split
        #         clip_pert = np.load(os.path.join(data_folder, pert_folder_nm, os.path.basename(f)))
        #         clip_len = (len(clip_clean) - window_size) // step_size + 1
        #         sv_clip_pert = {}
        #         for i_idx in range(6):
        #             sv_clip_pert[f'f{i_idx + 1}'] = clip_pert[f'f{i_idx + 1}']
                
        #         ### sv clip pert, 
        #         ##### load subj params #####
        #         pure_file_name = f.split("/")[-1].split(".")[0]
        #         pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                        
        #         subj_params_fn = os.path.join(self.subj_data_folder, split, pure_subj_params_fn)
        #         subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        #         rhand_transl = subj_params["rhand_transl"]
        #         rhand_betas = subj_params["rhand_betas"]
        #         rhand_pose = clip_clean['f2'] ## rhand pose ##
                
        #         pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        #         pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        #         ##### load subj params #####
        #         ## meta ##
        #         # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        #         # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        #         ### pert and clean pair for encoding and decoding ###
        #         self.clips.append((self.len, self.len+clip_len, clip_pert,
        #             [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #             # subj_corr_data, pert_subj_corr_data
        #             ))
        #         # self.clips.append((self.len, self.len+clip_len, sv_clip_pert,
        #         #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #         #     # subj_corr_data, pert_subj_corr_data
        #         #     )) ## object surface; grid positions; points sampled from the sapce; and you may need 3D conv nets; 
        #         # two objects and the change of the distance field; 
        #         # object surface points and the subject-related quantities grounded on it. 
        #         self.len += clip_len # len clip len
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    ## load clip data ##
    def load_clip_data(self, clip_idx):
        cur_clip = self.clips[clip_idx]
        if len(cur_clip) > 3:
            return
        f = cur_clip[2]
        clip_clean = np.load(f)
        # pert_folder_nm = self.split + '_pert'
        
        if self.args.train_all_clips:
            pert_folder_nm = f.split("/")[-2] # get the split folder name
        else:
            pert_folder_nm = self.split
        
        # if not self.use_pert:
        #     pert_folder_nm = self.split
        # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        ##### load subj params #####
        pure_file_name = f.split("/")[-1].split(".")[0]
        pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
        
        # subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        rhand_transl = subj_params["rhand_transl"]
        rhand_betas = subj_params["rhand_betas"]
        rhand_pose = clip_clean['f2']
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        loaded_clip = (
            cur_clip[0], cur_clip[1], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip # object idx? 
        
        return loaded_clip

        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]
            

    #### enforce correct contacts #### ## enforce correct contacts ##
    # normalize instances #
    def __getitem__(self, index):
        ## GRAB single frame ## # enumerate clips #
        for i_c, c in enumerate(self.clips):
            if index < c[1]:
                break
        if self.load_meta:
            # c = self.clips[i_c]
            c = self.load_clip_data(i_c)

        object_id = c[3][-1] ## object_idx here ##
        object_name = self.id2objmeshname[object_id]
        # TODO: add random noise settings for noisy input #
        start_idx = (index - c[0]) * self.step_size
        # data = c[2][start_idx:start_idx+self.window_size]
        data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        object_global_orient = data['f5']
        object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        ### global orientation; object trcansl ####
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        # object_global_orient_mtx_th, object_trcansl_th
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float() # object glbal # object_global_orient: ws x 3 x 3 --> object global transformation #
        # object transl th #
        object_trcansl_th = torch.from_numpy(object_trcansl).float() # object_transl_th # ws x 3 --> translations #
        
        # pert_subj_params = c[4]
        
        ### pts gt ###
        ## rhnad pose, rhand pose gt ##
        ## glboal orientation and hand pose #
        rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        rhand_transl, rhand_betas = c[3][5], c[3][6]
        rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        ### rhand transl for rhand transl # ####; rhand betas for rhand_betas ##
        # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        # # orientation rotation matrix #
        # rhand_global_orient_mtx_gt = utils.batched_get_orientation_matrices(rhand_global_orient_gt)
        # rhand_global_orient_mtx_gt_var = torch.from_numpy(rhand_global_orient_mtx_gt).float()
        # # orientation rotation matrix #
        
        rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
        rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
        rhand_beta_var = torch.from_numpy(rhand_betas).float()
        rhand_transl_var = torch.from_numpy(rhand_transl).float() # self.window_size x 3
        # R.from_rotvec(obj_rot).as_matrix()
        
        
        # rhand betas #
        # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_betas_var, rhand_beta_var
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var #
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        )
        ### rhand_joints: for joints ###
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        
        # rhand_anchors, canon_rhand_anchors #
        # use_anchors, self.hand_palm_vertex_mask #
        if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 #
            # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
            # if self.use_anchors:
            ### recover anchor batched ###
            rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
            # rhand anchors ##
            
        
        
        
        canon_rhand_verts, canon_rhand_joints = self.mano_layer(
            torch.cat([torch.zeros_like(rhand_global_orient_var), rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(rhand_transl_var)
        )
        ### rhand_joints: for joints ###
        canon_rhand_verts = canon_rhand_verts * 0.001
        canon_rhand_joints = canon_rhand_joints * 0.001
        
        
        if self.use_anchors:
            # canon_rhand_anchors = canon_rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 #
            canon_rhand_anchors = recover_anchor_batch(canon_rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
        
        
        # ### Relative positions from base points to rhand joints ###
        object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        # base_pts_feats_sv_dict = {}
        #### distance between rhand joints and obj pcs #### ## anchor points? # # load single clip data? # 
        # pert_rhand_joints_th = pert_rhand_joints
        # ws x nnjoints x nnobjpts #
        dist_rhand_joints_to_obj_pc = torch.sum(
            (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        )
        # dist_pert_rhand_joints_obj_pc = torch.sum(
        #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
        # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
        # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        dist_object_pc_nearest_pcs = torch.sum(
            (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
        )
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
        # # dist_threshold = 0.01
        dist_threshold = self.dist_threshold
        # # dist_threshold for pc_nearest_pcs #
        dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)

        # # base_pts_mask: nn_obj_pcs #
        base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)


        if not self.args.not_canon_rep: # not canonicalize representations #
            # # nn_base_pts x 3 -> torch tensor #
            base_pts = object_pc_th[0][base_pts_mask]
            # # base_pts_bf_sampling = base_pts.clone()
            base_normals = object_normal_th[0][base_pts_mask]
            base_pts_tot = object_pc_th[:, base_pts_mask] 
            base_normals_tot = object_normal_th[:, base_pts_mask]
        else:
            base_pts = object_pc_th[0].clone()
            # # base_pts_bf_sampling = base_pts.clone()
            base_normals = object_normal_th[0].clone()
            base_pts_tot = object_pc_th.clone()
            base_normals_tot = object_normal_th.clone()
            
        nn_base_pts = self.nn_base_pts
        base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
        base_pts_idxes = base_pts_idxes[:nn_base_pts]

        
        # ### get base points ### # base_pts and base_normals #
        base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
        base_normals = base_normals[base_pts_idxes]
        
        # base_pts_tot, base_normals_tot # 
        base_pts_tot = base_pts_tot[:, base_pts_idxes]
        base_normals_tot = base_normals_tot[:, base_pts_idxes]
        
        
        # # object_global_orient_mtx # nn_ws x 3 x 3 #
        base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
        base_pts_transl = object_trcansl_th[0] # 3
        
        
        if not self.args.not_canon_rep:
            base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
                )
            base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
                ) # .transpose(0, 1)
            
            base_pts_tot = torch.matmul((base_pts_tot - object_trcansl_th.unsqueeze(1)[0].unsqueeze(0)), object_global_orient_mtx_th.transpose(1, 2)[0].unsqueeze(0))
            base_normals_tot = torch.matmul(base_normals_tot, object_global_orient_mtx_th.transpose(1, 2)[0].unsqueeze(0))
            
            # base_pts, base_normals;  # base pts, base normals #
            # normalize via object poses # rhand joints; rhand joints #
            # normalized rhand joints #
            rhand_joints = torch.matmul(
                rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
            
            rhand_joints_ncanon = torch.matmul(
                rhand_joints - object_trcansl_th.unsqueeze(1)[0].unsqueeze(0), object_global_orient_mtx_th.transpose(1, 2)[0].unsqueeze(0)
            )
            
            # normalized anchros #
            if self.use_anchors: # rhand_anchors, canon_rhand_anchors #
                # rhand_anchors: nf x nn_anchors x 3 #
                rhand_anchors = torch.matmul(
                    rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
                )
        else:
            rhand_joints_ncanon = rhand_joints.clone()
        
        # base pts single values, nearest object pc and nearest object normal #
        
        # 
        sampled_base_pts = base_pts
        base_pts = sampled_base_pts
        # base_pts, base_normals #
        # rhand_joints - sampled_base_pts #
        
        
        ''' # ==== the normalization for rhand joints, anchors and base points ==== # '''
        # base pts, base normals, 
        if self.aug:
            rnd_R = common_utils.get_random_rot_np()
            R_th = torch.from_numpy(rnd_R).float()
            # base_pts: nn_base_pts x 3 #
            # augmentation for the base pts and normals, and rel from base pts to hand vertices # 
            base_pts = torch.matmul(base_pts, R_th)
            sampled_base_pts = base_pts
            base_normals = torch.matmul(base_normals, R_th)
            rhand_joints = torch.matmul(rhand_joints, R_th.unsqueeze(0))
            base_pts_tot = torch.matmul(base_pts_tot, R_th.unsqueeze(0))
            base_normals_tot = torch.matmul(base_normals_tot, R_th.unsqueeze(0))
            if self.use_anchors:
                rhand_anchors = torch.matmul(rhand_anchors, R_th.unsqueeze(0)) # for the rhand anchors and vertices #
                # rhand_joints =  # put it for anchors #
    

        
        # 
        # current states, current joints, moving attraction forces, energies and change of energies ##
        ''' Relative positions and distances normalization, strategy 1 '''
        # rhand_joints = rhand_joints*5.
        # base_pts = base_pts * 5.
        ''' Relative positions and distances normalization, strategy 1 '''
        # sampled_base_pts: nn_base_pts x 3 #
        # nf x nnj x nnb x 3 #
        # nf x nnj x nnb x 3 # # rel base pts to rhand joints # 
        rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(0).unsqueeze(0)
        # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
        dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        ''' Sample pts in the ambient space '''
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :] # needs to multiple with the object pose to get the relative velocity
        
        # obj_pts_disp, vel_obj_pts_to_hand_pts # 
        ### the object points displacements ###
        obj_pts_disp = base_pts_tot[1:, :, :] - base_pts_tot[:-1, :, :]
        
        ### the relative velocity from object points to the hand points ###
        rhand_joints_ncanon_disp = rhand_joints_ncanon[1:, :, :] - rhand_joints_ncanon[:-1, :, :]
        vel_obj_pts_to_hand_pts = rhand_joints_ncanon_disp.unsqueeze(2) - obj_pts_disp.unsqueeze(1)
        
        
        signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
            base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
        )
        
        
        rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        
        dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
            rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
        ))
        
        k_a = 1.
        k_b = 1.


        e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
        # (ws - 1) x nnj x nnb # -> dist vt normals #
        e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
        # base_pts; base_normals; 
        
        ''' normalization sstrategy 1 '''
        disp_ws, nnj, nnb = e_disp_rel_to_base_along_normals.shape[:3]
        # disp_ws x nnf x nnb x 3 #  -> disp_ws x nnj x nnb
        per_frame_avg_disp_along_normals = torch.mean(
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True 
        ) 
        per_frame_std_disp_along_normals = torch.std(
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) 
        per_frame_avg_disp_vt_normals = torch.mean(
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True 
        ) 
        per_frame_std_disp_vt_normals = torch.std(
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) 


        e_disp_rel_to_base_along_normals = (e_disp_rel_to_base_along_normals - per_frame_avg_disp_along_normals) / per_frame_std_disp_along_normals
        e_disp_rel_to_baes_vt_normals = (e_disp_rel_to_baes_vt_normals - per_frame_avg_disp_vt_normals) / per_frame_std_disp_vt_normals
        ''' normalization sstrategy 1 ''' # 
         
         
        ''' Relative positions and distances normalization, strategy 2 '''
        # # for each point normalize joints over all frames #
        
        # rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - self.avg_joints_rel.unsqueeze(-2)) / self.std_joints_rel.unsqueeze(-2)
        # dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - self.avg_joints_dists.unsqueeze(-1)) / self.std_joints_dists.unsqueeze(-1)
        ''' Relative positions and distances normalization, strategy 2 '''
        
        
        
        if self.denoising_stra == "rep":
            ''' Relative positions and distances normalization, strategy 3 '''
            # for each point normalize joints over all frames #
            # rel_base_pts_to_rhand_joints: nf x nnj x nnb x 3 #
            ''' Stra 1 -> per frame ''' # per-frame #
            per_frame_avg_joints_rel = torch.mean(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True # for each point #
            )
            per_frame_std_joints_rel = torch.std(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_avg_joints_dists_rel = torch.mean(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_std_joints_dists_rel = torch.std(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            # base pts #
            
            ''' Stra 2 -> per frame with joints '''
            # nf x nnj x nnb x 3 #
            # ws, nnf , nnb = rel_base_pts_to_rhand_joints.shape[:3]
            # per_frame_avg_joints_rel = torch.mean(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True # for each point #
            # ).unsqueeze(0)
            # per_frame_std_joints_rel = torch.std(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_avg_joints_dists_rel = torch.mean(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_std_joints_dists_rel = torch.std(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            
            
            # max xyz vlaues for the relative positions, maximum, minimum distances for them #
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        # 
        # nf x nnj x 3 -> 
        # 
        if self.denoising_stra == "motion_to_rep": # motion_to_rep # # rhand joints; 
            rhand_joints = (rhand_joints - self.avg_jts) / self.std_jts
        
        
        # self.maxx_rel, minn_rel, maxx_dists, minn_dists #
        # 
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        
        ''' Create captions and tokens for text-condtional settings '''
        # object_name
        # caption = f"{object_name}"
        # tokens = f"{object_name}/NOUN"
        
        # tokens = tokens.split(" ")
        # max_text_len = 20
        # if len(tokens) < max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        # pos_one_hots = [] ## pose one hots ##
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        caption = "apple"
        ''' Create captions and tokens for text-condtional settings '''
        
        
        ''' Obj data '''
        obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id)
        obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 # # obj verts; #
        obj_normals = torch.from_numpy(obj_normals).float() # 
        obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        # # object_global_orient_mtx_th, object_trcansl_th
        
        # base_pts, base_normals, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var #
        # rhand_transl, rhand_rot, rhand_theta # 
        # and only 
        # rhand_anchors, canon_rhand_anchors #
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            'rhand_joints': rhand_joints if not self.args.use_canon_joints else canon_rhand_joints,
            'rhand_verts': rhand_verts,
            'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
            'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
            'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
            'rhand_betas': rhand_beta_var,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'obj_verts': obj_verts,
            'obj_normals': obj_normals,
            'obj_faces': obj_faces, # nnfaces x 3 # nnfaces x 3 # -> obj faces #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, # not for the ambinet space valuess s#
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            ### === per frame avg disp along normals, vt normals === ###
            # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
            # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
            # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            # obj_pts_disp, vel_obj_pts_to_hand_pts # 
            'vel_obj_pts_to_hand_pts': vel_obj_pts_to_hand_pts,
            'obj_pts_disp': obj_pts_disp
            ## sampled; learn the 
        }
        # rhand_anchors, canon_rhand_anchors #
        if self.use_anchors:
            rt_dict.update(
                {   # rhand_anchors, canon_rhand_anchors ##
                    'rhand_anchors': rhand_anchors, 
                    'canon_rhand_anchors': canon_rhand_anchors, #### rt_dict for updating anchors ###
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] =  # realtive 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict
        

    def __len__(self):
        return self.len


## GRAB dataset V19 #
class GRAB_Dataset_V19_ARCTIC(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset ##
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.inst_normalization = args.inst_normalization
        self.args = args
        
        self.denoising_stra = args.denoising_stra
        
 
        self.aug_trans_T = 0.05
        self.aug_rot_T = 0.3
        self.aug_pose_T = 0.5
        self.aug_zero = 1e-4 if self.model_type not in ['v1_wsubj_wjointsv24', 'v1_wsubj_wjointsv25'] else 0.01
        
        self.sigmas_trans = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_trans_T), self.num_sche_steps
        ))
        self.sigmas_rot = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_rot_T), self.num_sche_steps
        ))
        self.sigmas_pose = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_pose_T), self.num_sche_steps
        ))
        
        
        self.data_folder = data_folder
        # self.aug = True
        self.aug = args.augment
        # self.use_anchors = False
        self.use_anchors = args.use_anchors
        # self.args = args
        
        # grab path extracted #
        self.grab_path = "/data1/sim/GRAB_extracted"
        self.mano_path = "/data1/sim/mano_models/mano/models"
        obj_mesh_path = os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')
        id2objmesh = []
        ''' Get idx to mesh path ''' 
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        ''' Get idx to mesh path ''' 
        
        ## obj root folder; obj p
        ### Load field data from root folders ###
        # self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        # self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        self.load_meta = True
        
        ## TODO: add thsoe params to args
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
    
        mano_pkl_path = os.path.join(self.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_pkl_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.template_verts = np.array(mano_model['v_template'])
        self.template_faces = np.array(mano_model['f'])
        self.template_joints = np.array(mano_model['J'])
        #### finger tips; ####
        self.template_tips = self.template_verts[[745, 317, 444, 556, 673]]
        self.template_joints = np.concatenate([self.template_joints, self.template_tips], axis=0)
        #### template verts ####
        self.template_verts = self.template_verts * 0.001
        #### template joints ####
        self.template_joints = self.template_joints * 0.001 # nn_joints x 3 #
        # condition on template joints for current joints #
        
        # normalization and so so # combnations of those quantities ######## 
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        self.fingers_stats = [
            [16, 15, 14, 13, 0],
            [17, 3, 2, 1, 0],
            [18, 6, 5, 4, 0],
            [19, 12, 11, 10, 0],
            [20, 9, 8, 7, 0]
        ]
        # 5 x 5 states, the first dimension is the finger index
        self.fingers_stats = np.array(self.fingers_stats, dtype=np.int32)
        self.canon_obj = True
        
        self.dir_stra = "vecs" # "rot_angles", "vecs"
        # self.dir_stra = "rot_angles"
        # distance # 21 distances? # --> texture map like data.. ## nanshou  ##
        
        
        # self.mano_layer = ManoLayer(
        #     flat_hand_mean=True,
        #     side='right',
        #     mano_root=self.mano_path, # mano_root #
        #     ncomps=24,
        #     use_pca=True,
        #     root_rot_mode='axisang',
        #     joint_rot_mode='axisang'
        # )
        
        self.rgt_mano_layer = ManoLayer(
            flat_hand_mean=False,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=45,
            use_pca=False,
            # root_rot_mode='axisang',
            # joint_rot_mode='axisang'
        )
        
        self.lft_mano_layer = ManoLayer(
            flat_hand_mean=False,
            side='left',
            mano_root=self.mano_path, # mano_root #
            ncomps=45,
            use_pca=False,
            # root_rot_mode='axisang',
            # joint_rot_mode='axisang'
        )
        
        ### Load field data from root folders ### ## obj root folder ##
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # anchor_load_driver, masking_load_driver #
        # use_anchors, self.hand_palm_vertex_mask #
        if self.use_anchors:
            # anchor_load_driver, masking_load_driver #
            inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
            fvi, aw, _, _ = anchor_load_driver(inpath)
            self.face_vertex_index = torch.from_numpy(fvi).long()
            self.anchor_weight = torch.from_numpy(aw).float()
            
            anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
            palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
            hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
            # self.hand_palm_vertex_mask for hand palm mask #
            self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
        
        ## actions taken 
        # self.clip_sv_folder = os.path.join(data_folder, f"{split}_clip")
        # os.makedirs(self.clip_sv_folder, exist_ok=True)
        
        train_subjs = ["s{:02d}".format(s_idx) for s_idx in range(2, 11) if s_idx != 3]
        self.clips = []
        self.clips_obj_name = []
        subj_processed_seqs_root = "/data/datasets/genn/sim/arctic_processed_data/processed_seqs/"
        obj_template_mesh_root = "/home/xueyi/sim/arctic/data/arctic_data/data/meta/object_vtemplates"
        self.obj_template_mesh_root = obj_template_mesh_root
        self.tot_idx_to_seq_idx_st_pos = {}
        tot_idx = 0
        self.subj_to_seq_to_st_valid_idxes = np.load("/data2/datasets/sim/arctic_processed_data/processed_split_seqs/valid_st_idxes.npy", allow_pickle=True).item()
        tot_seq_idx = 0
        for train_subj in train_subjs:
            print(f"Loading from {train_subj}")
            cur_subj_processed_seqs_root = os.path.join(subj_processed_seqs_root, train_subj)
            cur_subj_tot_seqs = os.listdir(cur_subj_processed_seqs_root)
            cur_subj_seq_to_st_idxes = self.subj_to_seq_to_st_valid_idxes[train_subj]
            # self.clips += cur_subj_tot_seqs
            for cur_seq_nm in cur_subj_tot_seqs:
                cur_seq_obj_template_nm = cur_seq_nm.split("_")[0]
                # cur_seq_obj_template_mesh_fn = os
                self.clips_obj_name.append(cur_seq_obj_template_nm)
                
                cur_subj_seq_full_path = os.path.join(cur_subj_processed_seqs_root, cur_seq_nm)
                self.clips.append(cur_subj_seq_full_path)
                
                cur_seq_valid_st_idxes = cur_subj_seq_to_st_idxes[cur_seq_nm]
                for cur_st_idx in cur_seq_valid_st_idxes:
                    self.tot_idx_to_seq_idx_st_pos[tot_idx] = (tot_seq_idx, cur_st_idx)
                    tot_idx += 1
                    
                tot_seq_idx += 1
                
        
        self.len = tot_idx
        self.obj_name_to_meshdata = {}
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    ## load clip data ##
    def load_clip_data(self, clip_idx):
        
        if isinstance(self.clips[clip_idx], dict):
            return self.clips[clip_idx]
        clip_path = self.clips[clip_idx]
        cur_clip_data = np.load(clip_path, allow_pickle=True).item()
        self.clips[clip_idx] = cur_clip_data
        return self.clips[clip_idx]
    
    # ketchup_grab_01.npy
        
        
    def get_idx_to_mesh_data(self, seq_index):
        cur_seq_obj_name = self.clips_obj_name[seq_index]
        
        if cur_seq_obj_name not in self.obj_name_to_meshdata:
            cur_seq_mesh_fn = os.path.join(self.obj_template_mesh_root, cur_seq_obj_name, "mesh.obj")
            obj_mesh = trimesh.load(cur_seq_mesh_fn, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.obj_name_to_meshdata[cur_seq_obj_name] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.obj_name_to_meshdata[cur_seq_obj_name]
            

    #### enforce correct contacts #### ## enforce correct contacts ##
    # normalize instances #
    def __getitem__(self, tot_index):
        ## GRAB single frame ## # enumerate clips #
        
        # i_c = index
        index, start_idx = self.tot_idx_to_seq_idx_st_pos[tot_index]
        # print(f"index: {index}, start_idx: {start_idx}")
        c = self.load_clip_data(index)

        # object_id = c[3][-1] ## object_idx here ##
        # object_name = self.id2objmeshname[object_id]
        # TODO: add random noise settings for noisy input # # fileter the data # # right hand 
        # start_idx = self.args.start_idx
        # start_idx = (index - c[0]) * self.step_size
        # data = c[2][start_idx:start_idx+self.window_size]
        # data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        # object_global_orient = data['f5'] ### get object global orientations ###
        # object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        object_global_orient = c["obj_rot"] # num_frames x 3 
        object_transl = c["obj_trans"] # num_frames x 3
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        # object_global_orient, object_transl #
        object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        object_transl = object_transl[start_idx: start_idx + self.window_size]
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # ### global orientation; object trcansl ####
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        # # object_global_orient_mtx_th, object_trcansl_th
        # object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float() # object glbal # object_global_orient: ws x 3 x 3 --> object global transformation #
        # # object transl th #
        # object_trcansl_th = torch.from_numpy(object_trcansl).float() # object_transl_th # ws x 3 --> translations #
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        object_pc_tmp = c["verts.object"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        object_transl = np.mean(object_pc_tmp, axis=1)
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        object_trcansl_th = torch.from_numpy(object_transl).float()
        # object_trcansl_th = torch.zeros_like(object_trcansl_th)
        # pert_subj_params = c[4]
        # 
        # pert_subj_params = c[4]
        
        ### pts gt ###
        ## rhnad pose, rhand pose gt ##
        ## glboal orientation and hand pose #
        rhand_global_orient_gt, rhand_pose_gt = c["rot_r"], c["pose_r"]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        rhand_transl, rhand_betas = c["trans_r"], c["shape_r"][0]
        rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        
        # if we use a sliding windo 
        # ### pts gt ###
        # ## rhnad pose, rhand pose gt ##
        # ## glboal orientation and hand pose #
        # rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        # rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        # rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        # rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        # rhand_transl, rhand_betas = c[3][5], c[3][6]
        # rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # ### rhand transl for rhand transl # ####; rhand betas for rhand_betas ##
        # # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        # rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        # rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        # # orientation rotation matrix #
        # rhand_global_orient_mtx_gt = utils.batched_get_orientation_matrices(rhand_global_orient_gt)
        # rhand_global_orient_mtx_gt_var = torch.from_numpy(rhand_global_orient_mtx_gt).float()
        # # orientation rotation matrix #
        
        rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
        rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
        rhand_beta_var = torch.from_numpy(rhand_betas).float()
        rhand_transl_var = torch.from_numpy(rhand_transl).float() # self.window_size x 3
        # R.from_rotvec(obj_rot).as_matrix()
        
        aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        if self.args.pert_type == "uniform":
            aug_pose_var = (torch.rand_like(rhand_pose_var) - 0.5) * aug_pose
            aug_global_orient_var = (torch.rand_like(rhand_global_orient_var) - 0.5) * aug_rot
        elif self.args.pert_type == "beta":
            dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
            # print(f"here!")
            aug_pose_var = dist_beta.sample(rhand_pose_var.size()).squeeze(-1) * aug_pose
            aug_global_orient_var = dist_beta.sample(rhand_global_orient_var.size()).squeeze(-1) * aug_rot
            # print(f"aug_pose_var: {aug_pose_var.size()}, aug_global_orient_var: {aug_global_orient_var.size()}")
        
        # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # ### creat augmneted orientations, pose, and transl ###
        rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        cur_mano_layer = self.rgt_mano_layer
        
        # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_betas_var, rhand_beta_var
        rhand_verts, rhand_joints = cur_mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        )
        ### rhand_joints: for joints ###
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        
        # rhand_anchors, pert_rhand_anchors #
        # rhand_anchors, canon_rhand_anchors #
        # use_anchors, self.hand_palm_vertex_mask #
        if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 #
            # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
            rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        
        
        if self.use_rnd_aug_hand: ## rnd aug pose var, transl var #
            # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
            pert_rhand_global_orient_var = rnd_aug_global_orient_var.clone()
            pert_rhand_pose_var = rnd_aug_pose_var.clone()
            pert_rhand_transl_var = rnd_aug_transl_var.clone()
            # pert_rhand_global_orient_mtx = utils.batched_get_orientation_matrices(pert_rhand_global_orient_var.numpy())
        
        # # pert_rhand_betas_var
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_joints --> for rhand joints in the camera frmae ###
        pert_rhand_verts, pert_rhand_joints = cur_mano_layer(
            torch.cat([pert_rhand_global_orient_var, pert_rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), pert_rhand_transl_var
        )
        pert_rhand_verts = pert_rhand_verts * 0.001 # verts 
        pert_rhand_joints = pert_rhand_joints * 0.001 # joints
        
        if self.use_anchors:
            # pert_rhand_anchors = pert_rhand_verts[:, self.hand_palm_vertex_mask]
            pert_rhand_anchors = recover_anchor_batch(pert_rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        # use_canon_joints
        
        canon_pert_rhand_verts, canon_pert_rhand_joints = cur_mano_layer(
            torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
        )
        canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
        canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
        # if self.use_anchors:
        #     # canon_pert_rhand_anchors = canon_pert_rhand_verts[:, self.hand_palm_vertex_mask]
        #     canon_pert_rhand_anchors = recover_anchor_batch(canon_pert_rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
        
        # canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
        #     torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
        #     pert_rhand_betas_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
        # )
        # canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
        # canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
        # ### Relative positions from base points to rhand joints ###
        object_pc = c["verts.object"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)

        # if self.args.scale_obj > 1:
        #     object_pc = object_pc * self.args.scale_obj
        # object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        
        object_normal = np.zeros_like(object_pc)
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        
        # ws x nnjoints x nnobjpts #
        dist_rhand_joints_to_obj_pc = torch.sum(
            (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        )
        # dist_pert_rhand_joints_obj_pc = torch.sum(
        #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
        # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # nearest_obj_pcs: ws x nn_jts x 3 --> for nearet obj pcs # 
        # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
            (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
        ) 
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # ws x nn_obj #
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
        # # dist_threshold = 0.01 # threshold 
        dist_threshold = self.dist_threshold
        # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
        dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
        
        # # base_pts_mask: nn_obj_pcs #
        base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold) # nn_obj_pcs -> nearest_pcs mask #
        base_pts = object_pc_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
        base_normals = object_normal_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
        nn_base_pts = self.nn_base_pts
        base_pts_idxes = utils.farthest_point_sampling(base_pts[0:1], n_sampling=nn_base_pts)
        base_pts_idxes = base_pts_idxes[:nn_base_pts]
        base_pts = base_pts[:, base_pts_idxes]
        base_normals = base_normals[:, base_pts_idxes]
        
        base_pts_global_orient_mtx = object_global_orient_mtx_th # ws x 3 x 3 #
        base_pts_transl = object_trcansl_th # ws x 3 # 
        base_pts = torch.matmul(
            (base_pts - base_pts_transl.unsqueeze(1)), base_pts_global_orient_mtx.transpose(1, 2) # ws x nn_base_pts x 3 --> ws x nn_base_pts x 3 #
        )
        base_normals = torch.matmul(
            base_normals, base_pts_global_orient_mtx.transpose(1, 2)  # ws x nn_base_pts x 3 
        )
        
        # # if self.debug:
        # #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
        
        # # # object_global_orient_mtx # nn_ws x 3 x 3 #
        # base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
        # base_pts_transl = object_trcansl_th[0] # 3
        
        # # if self.dir_stra == "rot_angles": ## rot angles ##
        # #     normals_rot_mtx = utils.batched_get_rot_mtx_fr_vecs_v2(base_normals)
        
        # # if self.canon_obj:
        #     ## reverse transform base points ###
        #     ## canonicalize base points and base normals ###
        # base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
        #     ) # .transpose(0, 1)
        # base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
        #     ) # .transpose(0, 1)
        
        
        rhand_joints = torch.matmul(
            rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        pert_rhand_joints = torch.matmul(
            pert_rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        if self.args.use_anchors:
            # rhand_anchors, pert_rhand_anchors #
            rhand_anchors = torch.matmul(
                rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
            pert_rhand_anchors = torch.matmul(
                pert_rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
        
        object_pc_th = torch.matmul(
            object_pc_th - object_trcansl_th.unsqueeze(1),  object_global_orient_mtx_th.transpose(1, 2)
        )
        
        ''' normalization strategy xxx --- data scaling '''
        # base_pts = base_pts * 5.
        # rhand_joints = rhand_joints * 5.
        ''' Normlization stratey xxx --- data scaling '''
        
        if self.args.use_left: ### rhand_joints: nn_frames x nn_joints x 3 # base_pts: nn_frames x nn_jbase_pts x 3 # 
            pert_rhand_joints[:, :, -1] = -1. * pert_rhand_joints[:, :, -1] 
            base_pts[:, :, -1] = -1. * base_pts[:, :, -1] 
            
        
        
        # base_pts = sampled_base_pts
        # sampled_base_pts = base_pts
        
        ''' Relative positions and distances normalization, strategy 1 '''
        # rhand_joints = rhand_joints * 5.
        # base_pts = base_pts * 5.
        ''' Relative positions and distances normalization, strategy 1 '''
        # sampled_base_pts: nn_base_pts x 3 #
        # nf x nnj x nnb x 3 #
        # nf x nnj x nnb x 3 #
        # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(0).unsqueeze(0)
        # # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
        # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
        # if not self.args.use_arti_obj:
        #     # nf x nnj x nnb x 3 # 
        #     rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        #     # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
        #     dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        # else:
        rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1) # ws x nn_joints x nn_base_pts x 3 #
        # dist_base_pts_to_rhand_joints: ws x nn_joints x nn_base_pts -> the distance from base points to joint points #
        dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
        # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        
        
        rhand_joints_disp = pert_rhand_joints[1:, :, :] - pert_rhand_joints[:-1, :, :]
        
        # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # 
        # if not self.args.use_arti_obj:
        #     # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
        #     signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
        #         base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
        #     )
            
        #     # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        #     rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        # else:
        signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
            base_normals.unsqueeze(1)[:-1] * rhand_joints_disp.unsqueeze(2), dim=-1
        )
        # unsqueeze the dimensiton 1 #
        rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(1)[:-1]
        # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
        # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
        
        # nf x nnj x nnb ---> dist_vt_normals -> nf x nnj x nnb # # torch.sqrt() ##
        dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
            rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
        ))
        
        k_a = 1.
        k_b = 1. 
        # k and # give me a noised sequence ... #
        # (ws - 1) x nnj x nnb # --> (ws - 1) x nnj x nnb # nnj x nnb # 
        # add noise -> chagne of the joints displacements 
        # -> change of along_normalss energies and vertical to normals energies #
        # -> change of energy taken to make the displacements #
        # jts_to_base_pts energy in the noisy sequence #
        # jts_to_base_pts energy in the clean sequence #
        # vt-normal, along_normal #
        # TODO: the normalization strategy: 1) per-instnace; 2) per-category; #3
        # att_forces: (ws - 1) x nnj x nnb # # 
        e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
        # (ws - 1) x nnj x nnb # -> dist vt normals #
        e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
        # base_pts; base_normals; 
        
        
        ''' normalization sstrategy 1 ''' # 
        # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
        # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
        # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
        # per_frame_avg_disp_along_normalss, per_frame_std_disp_along_normalss # 
        # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        disp_ws, nnj, nnb = e_disp_rel_to_base_along_normals.shape[:3]
        # disp_ws x nnf x nnb x 3 #  -> disp_ws x nnj x nnb
        per_frame_avg_disp_along_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_along_normals = torch.std( # std over all frames #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        per_frame_avg_disp_vt_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_vt_normals = torch.std( # std over all frames #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        # per_frame_avg_joints_dists_rel = torch.mean(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        # per_frame_std_joints_dists_rel = torch.std(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        ### normalizaed aong normals and vat normals  # ws x nnj x nnb 
        e_disp_rel_to_base_along_normals = (e_disp_rel_to_base_along_normals - per_frame_avg_disp_along_normals) / per_frame_std_disp_along_normals
        e_disp_rel_to_baes_vt_normals = (e_disp_rel_to_baes_vt_normals - per_frame_avg_disp_vt_normals) / per_frame_std_disp_vt_normals
        # enrgy temrs #
        ''' normalization sstrategy 1 ''' # 
        
        
        if self.denoising_stra == "rep":
            ''' Relative positions and distances normalization, strategy 3 '''
            # # for each point normalize joints over all frames #
            # # rel_base_pts_to_rhand_joints: nf x nnj x nnb x 3 #
            per_frame_avg_joints_rel = torch.mean(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_std_joints_rel = torch.std(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_avg_joints_dists_rel = torch.mean(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_std_joints_dists_rel = torch.std(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            # max xyz vlaues for the relative positions, maximum, minimum distances for them #
            
            
            if not self.args.use_arti_obj:
                # # nf x nnj x nnb x 3 # 
                rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
                # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
                dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
            else:
                # # nf x nnj x nnb x 3 # 
                rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1)
                # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
                dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
                
            
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        # if self.denoising_stra == "motion_to_rep": # motion_to_rep #
        #     pert_rhand_joints = (pert_rhand_joints - self.avg_jts) / self.std_jts
        
        
        
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        caption = "apple"
        ''' Create captions and tokens for text-condtional settings '''
        
        
        # ''' Obj data '''
        obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(index)
        obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 # # obj verts; #
        obj_normals = torch.from_numpy(obj_normals).float() # 
        obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        # ''' Obj data '''
        
        # object_global_orient_th, object_transl_th #
        object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        
        # # object_global_orient_mtx_th, object_trcansl_th
        
        # base_pts, base_normals, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var #
        # rhand_transl, rhand_rot, rhand_theta # 
        # and only 
        # rhand_anchors, canon_rhand_anchors #
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            'rhand_joints': rhand_joints, # if not self.args.use_canon_joints else canon_rhand_joints,
            'rhand_verts': rhand_verts,
            'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
            'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
            'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
            'rhand_betas': rhand_beta_var,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'obj_verts': obj_verts,
            'obj_normals': obj_normals,
            'obj_faces': obj_faces, # nnfaces x 3 # nnfaces x 3 # -> obj faces #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, # not for the ambinet space valuess s#
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            ### === per frame avg disp along normals, vt normals === ###
            # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
            # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
            # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            ## sampled; learn the 
        }
        # rhand_anchors, canon_rhand_anchors #
        if self.use_anchors:
            rt_dict.update(
                {   # rhand_anchors, canon_rhand_anchors ##
                    'rhand_anchors': rhand_anchors, 
                    'canon_rhand_anchors': canon_rhand_anchors, #### rt_dict for updating anchors ###
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] =  # realtive 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict
        

    def __len__(self):
        return self.len



# obj_fn, obj_rot=None, obj_trans=None #
def get_object_mesh_ours_arti(obj_fn, obj_rot, obj_trans):
    # object_id, object_rot, object_transl = d['f7'], d['f5'], d['f6']
    # is_left = d['f9']
    tot_obj_vertices = []
    tot_obj_normals = []
    tot_obj_faces = []
    nn_vertices = 0
    
    if obj_rot is not None:
      print(f"obj_rot: {len(obj_rot)}")
    
    for i_obj, cur_obj_fn in enumerate(obj_fn):
      if obj_rot is not None:
        cur_obj_rot, cur_obj_trans = obj_rot[i_obj], obj_trans[i_obj]
      else:
        # cur_obj_rot, cur_obj_trans = obj_rot[i_obj], obj_trans[i_obj]
        cur_obj_rot = np.eye(3, 3, dtype=np.float32) # 3 x 3 --> as obj_rot 
        cur_obj_trans = np.zeros((3,), dtype=np.float32)
      # cur_obj_mesh = trimesh.load_mesh(cur_obj_fn, process=False)
      ### cur_obj_rot, 
      # cur_obj_vertices, cur_obj_normals, cur_obj_faces = read_obj_with_normals(cur_obj_fn, minus_one=True)
      
      cur_obj_mesh = trimesh.load_mesh(cur_obj_fn, process=False)
      cur_obj_vertices = cur_obj_mesh.vertices
      cur_obj_normals = cur_obj_mesh.vertex_normals
      cur_obj_faces = cur_obj_mesh.faces
      
      # cur_obj_faces = np.array(cur_obj_faces, dtype=np.long) # nn_faces x 3 #
      cur_obj_vertices = np.matmul(
        cur_obj_rot, cur_obj_vertices.T
      ).T + np.reshape(cur_obj_trans, (1, 3))
      cur_obj_normals = np.matmul(
        cur_obj_rot, cur_obj_normals.T # nn_verts x 3 --> normals, verts
      ).T
      # nn_vertices = 
      cur_obj_faces = cur_obj_faces + nn_vertices
      nn_vertices += cur_obj_vertices.shape[0]
      tot_obj_vertices.append(cur_obj_vertices)
      tot_obj_normals.append(cur_obj_normals)
      tot_obj_faces.append(cur_obj_faces)
    # tot_obj_vertices: tot_nn_vertices x 3 #
    tot_obj_vertices = np.concatenate(tot_obj_vertices, axis=0)
    tot_obj_faces = np.concatenate(tot_obj_faces, axis=0)
    tot_obj_normals = np.concatenate(tot_obj_normals, axis=0)
    
    object_mesh = trimesh.Trimesh(vertices=tot_obj_vertices, faces=tot_obj_faces, vertex_normals=tot_obj_normals)
 
    return object_mesh    


## GRAB dataset V21 #
class GRAB_Dataset_V21(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset ##
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts ## 0.01, 0.05, 0.3 ##
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.inst_normalization = args.inst_normalization
        self.args = args
        
        self.denoising_stra = args.denoising_stra
        
        # load datas
        # grab_path =  "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
        # id2objmeshname = []
        # obj_meshes = sorted(os.listdir(obj_mesh_path))
        # # objectmesh name #
        # id2objmeshname = [obj_meshes[i].split(".")[0] for i in range(len(obj_meshes))]
        # self.id2objmeshname = id2objmeshname
        
        
        self.aug_trans_T = 0.05
        self.aug_rot_T = 0.3
        self.aug_pose_T = 0.5
        self.aug_zero = 1e-4 if self.model_type not in ['v1_wsubj_wjointsv24', 'v1_wsubj_wjointsv25'] else 0.01
        
        self.sigmas_trans = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_trans_T), self.num_sche_steps
        ))
        self.sigmas_rot = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_rot_T), self.num_sche_steps
        ))
        self.sigmas_pose = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_pose_T), self.num_sche_steps
        ))
        
        
        self.data_folder = data_folder
        self.subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "/data1/sim/mano_models/mano/models" ### mano_path
        # self.aug = True
        self.aug = args.augment
        # self.use_anchors = False
        self.use_anchors = args.use_anchors
        # self.args = args
        
        # grab path extracted #
        self.grab_path = "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')
        # id2objmesh = []
        # ''' Get idx to mesh path ''' 
        # obj_meshes = sorted(os.listdir(obj_mesh_path))
        # for i, fn in enumerate(obj_meshes):
        #     id2objmesh.append(os.path.join(obj_mesh_path, fn))
        # self.id2objmesh = id2objmesh
        # self.id2meshdata = {}
        ''' Get idx to mesh path ''' 
        
        ## obj root folder; obj p
        ### Load field data from root folders ###
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        ## TODO: add thsoe params to args
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
    
        mano_pkl_path = os.path.join(self.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_pkl_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.template_verts = np.array(mano_model['v_template'])
        self.template_faces = np.array(mano_model['f'])
        self.template_joints = np.array(mano_model['J'])
        #### finger tips; ####
        self.template_tips = self.template_verts[[745, 317, 444, 556, 673]]
        self.template_joints = np.concatenate([self.template_joints, self.template_tips], axis=0)
        #### template verts ####
        self.template_verts = self.template_verts * 0.001
        #### template joints ####
        self.template_joints = self.template_joints * 0.001 # nn_joints x 3 #
        # condition on template joints for current joints #
        
        # normalization and so so # combnations of those quantities ######## 
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        self.fingers_stats = [
            [16, 15, 14, 13, 0],
            [17, 3, 2, 1, 0],
            [18, 6, 5, 4, 0],
            [19, 12, 11, 10, 0],
            [20, 9, 8, 7, 0]
        ]
        # 5 x 5 states, the first dimension is the finger index
        self.fingers_stats = np.array(self.fingers_stats, dtype=np.int32)
        self.canon_obj = True
        
        self.dir_stra = "vecs" # "rot_angles", "vecs"
        # self.dir_stra = "rot_angles"
        # distance # 21 distances? # --> texture map like data.. ## nanshou  ##
        
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=24,
            use_pca=True,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )
        
        ### Load field data from root folders ### ## obj root folder ##
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # anchor_load_driver, masking_load_driver #
        # use_anchors, self.hand_palm_vertex_mask #
        if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
            # anchor_load_driver, masking_load_driver #
            inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
            fvi, aw, _, _ = anchor_load_driver(inpath)
            self.face_vertex_index = torch.from_numpy(fvi).long()
            self.anchor_weight = torch.from_numpy(aw).float()
            
            anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
            palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
            hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
            # self.hand_palm_vertex_mask for hand palm mask #
            self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
        
        
        #### Load data from single_seq_path ####
        self.single_seq_path = args.single_seq_path
        self.data = np.load(self.single_seq_path, allow_pickle=True) # .item()
        
        
        self.cad_model_fn = args.cad_model_fn
        self.corr_fn = args.corr_fn # corr_fn 
        if len(self.corr_fn) > 0:
            self.raw_corr_data = np.load(self.corr_fn, allow_pickle=True)
            self.obj_verts, self.obj_faces = None, None
        # self.dist_stra = args.dist_stra
        
        # evaluate from predicted infos #
        # args.predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" #
        self.tot_predicted_infos = []
        selected_seeds = [0, 2, 3, 4, 6, 8, 10, 11, 13, 14, 17, 18, 19, 22, 23, 27, 30, 31, 34, 36, 37, 39, 40, 41, 43, 44, 46, 47, 48, 50, 52, 53, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 73, 78, 80, 85, 87, 88, 89, 90, 91, 93, 95, 99, 100, 105, 106, 107, 110, 111, 112, 113, 115, 118, 119, 120, 121, 125, 127, 128, 130, 135, 136, 138, 139, 141, 143, 147, 149, 150, 151, 152, 155, 156, 158, 159, 163, 165, 167, 168, 169, 177, 178, 179, 182, 184, 186, 193, 198]
        # for cur_seed in range(0, 100):
        for cur_seed in selected_seeds:
            # ### TODO: modify this line to load predicted infos from rep_mean prediction ###
            # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" 
            # /data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_14_tag_rep_res_jts_hoi4d_scissors_t_300_.npy
            ####### predicted info fn #######
            # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy" 
            # f"predicted_infos_seq_300_seed_{seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy"
            # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" 
            # f"predicted_infos_seq_300_seed_{seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy"
            cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy" 
            cur_predicted_info = np.load(cur_predicted_info_fn, allow_pickle=True).item()
            self.tot_predicted_infos.append(cur_predicted_info)
        self.len = len(self.tot_predicted_infos)
        # evaluate from optimized infos #
        
        # if args.train_all_clips:
        #     for split in ["train", "val", "test"]:
        #         files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        #         files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
        #         # if self.load_meta:
        #         for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
        #             # if split != 'train' and split != 'val' and i_f >= 100:
        #             #     break
        #             # if split == 'train':
        #             print(f"split: {split}, loading {i_f} / {len(files_clean)}")
        #             base_nm_f = os.path.basename(f)
        #             base_name_f = base_nm_f.split(".")[0]
        #             cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
        #             cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
        #             cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
        #             cur_clip_len = cur_clip_meta_data['clip_len']
        #             clip_len = (cur_clip_len - window_size) // step_size + 1
        #             clip_len = max(clip_len, 0)
        #             if self.args.only_first_clip:
        #                 clip_len = min(clip_len, 1)
        #             print(f"cur_clip_len: {cur_clip_len}, clip_len: {clip_len}, window_size: {window_size}")
        #             self.clips.append((self.len, self.len+clip_len,  f
        #                 ))
        #             self.len += clip_len # len clip len
        # else:
        #     files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        #     #### filter files_clean here ####
        #     files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
            
        #     # if self.load_meta:
        #     for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
        #         # if split != 'train' and split != 'val' and i_f >= 100:
        #         #     break
        #         if split == 'train':
        #             print(f"loading {i_f} / {len(files_clean)}")
        #         base_nm_f = os.path.basename(f)
        #         base_name_f = base_nm_f.split(".")[0]
        #         cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
        #         cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
        #         cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
        #         cur_clip_len = cur_clip_meta_data['clip_len']
        #         clip_len = (cur_clip_len - window_size) // step_size + 1
        #         clip_len = max(clip_len, 0)
        #         if self.args.only_first_clip:
        #             clip_len = min(clip_len, 1)
        #         print(f"cur_clip_len: {cur_clip_len}, clip_len: {clip_len}, window_size: {window_size}")
        #         self.clips.append((self.len, self.len+clip_len,  f
        #             ))
        #         self.len += clip_len # len clip len
        # else:
        #     for i_f, f in enumerate(files_clean):
        #         if split == 'train':
        #             print(f"loading {i_f} / {len(files_clean)}")
        #         if split != 'train' and i_f >= 100:
        #             break
        #         if args is not None and args.debug and i_f >= 10:
        #             break
        #         clip_clean = np.load(f)
        #         pert_folder_nm = split + '_pert'
        #         if args is not None and not args.use_pert:
        #             pert_folder_nm = split
        #         clip_pert = np.load(os.path.join(data_folder, pert_folder_nm, os.path.basename(f)))
        #         clip_len = (len(clip_clean) - window_size) // step_size + 1
        #         sv_clip_pert = {}
        #         for i_idx in range(6):
        #             sv_clip_pert[f'f{i_idx + 1}'] = clip_pert[f'f{i_idx + 1}']
                
        #         ### sv clip pert, 
        #         ##### load subj params #####
        #         pure_file_name = f.split("/")[-1].split(".")[0]
        #         pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                        
        #         subj_params_fn = os.path.join(self.subj_data_folder, split, pure_subj_params_fn)
        #         subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        #         rhand_transl = subj_params["rhand_transl"]
        #         rhand_betas = subj_params["rhand_betas"]
        #         rhand_pose = clip_clean['f2'] ## rhand pose ##
                
        #         pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        #         pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        #         ##### load subj params #####
        #         ## meta ##
        #         # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        #         # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        #         ### pert and clean pair for encoding and decoding ###
        #         self.clips.append((self.len, self.len+clip_len, clip_pert,
        #             [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #             # subj_corr_data, pert_subj_corr_data
        #             ))
        #         # self.clips.append((self.len, self.len+clip_len, sv_clip_pert,
        #         #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #         #     # subj_corr_data, pert_subj_corr_data
        #         #     )) ## object surface; grid positions; points sampled from the sapce; and you may need 3D conv nets; 
        #         # two objects and the change of the distance field; 
        #         # object surface points and the subject-related quantities grounded on it. 
        #         self.len += clip_len # len clip len
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    ## load clip data ##
    def load_clip_data(self, clip_idx):
        cur_clip = self.clips[clip_idx]
        if len(cur_clip) > 3:
            return
        f = cur_clip[2]
        clip_clean = np.load(f)
        # pert_folder_nm = self.split + '_pert'
        
        if self.args.train_all_clips:
            pert_folder_nm = f.split("/")[-2] # get the split folder name
        else:
            pert_folder_nm = self.split
        
        # if not self.use_pert:
        #     pert_folder_nm = self.split
        # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        ##### load subj params #####
        pure_file_name = f.split("/")[-1].split(".")[0]
        pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
        
        # subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        rhand_transl = subj_params["rhand_transl"]
        rhand_betas = subj_params["rhand_betas"]
        rhand_pose = clip_clean['f2']
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        loaded_clip = (
            cur_clip[0], cur_clip[1], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip # object idx? 
        
        return loaded_clip

        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]
            

    def get_ari_obj_fr_x(self, i_frame_st, i_frame_ed):
        if self.obj_verts is not None: # tot verts, tot faces #
            # print(f"self.obj_verts nt None, returning...")
            return self.obj_verts, self.obj_faces # return obj_verts and obj_faces #
        
        # raw_corr_data
        tot_obj_verts = []
        tot_obj_faces = []
        for i_frame in range(i_frame_st, i_frame_ed):
            cur_obj_rot = self.raw_corr_data[i_frame]['obj_rot']
            cur_obj_trans = self.raw_corr_data[i_frame]['obj_trans']
            cad_model_fn = [
                "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-1-align.obj",  # 
                "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-0-align.obj" 
            ]
            cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
            # nn_verts x 3 
            # nn_faces x 3 
            
            cur_obj_verts, cur_obj_faces = cur_obj_mesh.vertices, cur_obj_mesh.faces
            obj_center = np.mean(cur_obj_verts, axis=0, keepdims=True)
            cur_obj_verts = cur_obj_verts - obj_center
            tot_obj_verts.append(cur_obj_verts)
            tot_obj_faces.append(cur_obj_faces)
            
        # self.obj_verts, self.obj_faces
        tot_obj_verts = np.stack(tot_obj_verts, axis=0)
        tot_obj_faces = np.stack(tot_obj_faces, axis=0)
        
        self.obj_verts = tot_obj_verts
        self.obj_faces = tot_obj_faces
        print(f"self.obj_verts: {self.obj_verts.shape}, self.obj_faces: {self.obj_faces.shape}")
        
        return tot_obj_verts, tot_obj_faces


    #### enforce correct contacts #### ## enforce correct contacts ##
    # normalize instances #
    def __getitem__(self, index):
        ## GRAB single frame ## # enumerate clips #
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        # if self.load_meta:
        #     # c = self.clips[i_c]
        #     c = self.load_clip_data(i_c)

        start_idx = 0
        
        if len(self.corr_fn) > 0:
            cur_obj_verts, cur_obj_faces = self.get_ari_obj_fr_x(start_idx, start_idx + self.window_size) # nn_obj_verts x 3; nn_obj_faces x 3 #
            # print(f"corr_fn: {self.corr_fn}, obj_verts: {cur_obj_verts.shape}, cur_obj_faces: {cur_obj_faces.shape}")
        
        
        object_pc = self.data['f1'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        object_vn = self.data['f2'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        rhand_joints = self.data['f11'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        
        # rhand_glb_rot, rhand_pose, rhand_joints_gt, minn_dists_rhand_joints_object_pc,
        # rhand_joints = self.data[index]['f11'].reshape(21, 3).astype(np.float32)
        # rhand_joints = rhand_joints * 0.001
        # rhand_joints = rhand_joints - obj_center
        
        # rhand_joints_fr_data = rhand_joints.copy() ## rhandjoints
        
        
        rhand_transl = self.data['f10'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
        # rhand_transl = rhand_transl - obj_center[0]
        rhand_beta = self.data['f9'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
        rhand_theta = self.data['f8'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
        
        
        # rhand_transl_clean = self.clean_data[index]['f10'].reshape(3).astype(np.float32)
        # rhand_theta_clean = self.clean_data[index]['f8'].reshape(-1).astype(np.float32)
        
        rhand_glb_rot = rhand_theta[:, :3]
        rhand_theta = rhand_theta[:, 3:]
        
        ##### rhand transl #####
        # rhand_glb_rot = rhand_theta_clean[:3]
        # rhand_transl = rhand_transl_clean
        ##### rhand transl #####
        
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var, rhand_beta_var #
        rhand_global_orient_var = torch.from_numpy(rhand_glb_rot).float() # .unsqueeze(0)
        rhand_pose_var = torch.from_numpy(rhand_theta).float() # . unsqueeze(0)
        rhand_transl_var = torch.from_numpy(rhand_transl).float() # .unsqueeze(0)
        rhand_beta_var = torch.from_numpy(rhand_beta).float() # .unsqueeze(0)
        
        
        
        # # rhand_global_orient = self.data[index]['f1'].reshape(-1).astype(np.float32)
        # rhand_pose = rhand_theta
        # # rhand_transl = self.subj_params['rhand_transl'][index].reshape(-1).astype(np.float32)
        # rhand_betas = rhand_beta
        
        
        ####### Get rhand_verts and rhand_joint #######
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.view(-1, 10), rhand_transl_var
        )
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        ####### Get rhand_verts and rhand_joint #######
        
        
        cur_predicted_info = self.tot_predicted_infos[index]
        # outputs = data['outputs']
        # self.predicted_hand_trans = data['rhand_trans'] # nframes x 3 
        # self.predicted_hand_rot = data['rhand_rot'] # nframes x 3 
        # self.predicted_hand_theta = data['rhand_theta']
        # self.predicted_hand_beta = data['rhand_beta']
        rhand_joints = cur_predicted_info["outputs"]
        rhand_joints = torch.from_numpy(rhand_joints).float()
        if "rhand_trans" in cur_predicted_info:
            rhand_transl_var = cur_predicted_info["rhand_trans"] # nframes x 3 for rhand trans 
            rhand_global_orient_var = cur_predicted_info["rhand_rot"]
            rhand_pose_var = cur_predicted_info["rhand_theta"] # thetas
            rhand_beta_var = cur_predicted_info["rhand_beta"] # thetas #
            
            
            rhand_verts, rhand_joints = self.mano_layer(
                torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
                rhand_beta_var.view(-1, 10), rhand_transl_var
            )
            rhand_verts = rhand_verts * 0.001
            rhand_joints = rhand_joints * 0.001
        
        
        if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 # # recover anchor 
            # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
            rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            pert_rhand_anchors = rhand_anchors
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        
        # rhand_transl = rhand_transl - obj_center[0]
        
        
        pert_rhand_joints = rhand_joints
        pert_rhand_verts = rhand_verts
        
        
        object_global_orient = self.data['f3'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3, 3).astype(np.float32) # nf x 
        object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
        object_transl = self.data['f4'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
        object_trcansl_th = torch.from_numpy(object_transl).float()
        
        
        
        
        object_normal = object_vn
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        
        # ws x nnjoints x nnobjpts #
        dist_rhand_joints_to_obj_pc = torch.sum( # dist rhand joints to obj pc
            (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        )
        # dist_pert_rhand_joints_obj_pc = torch.sum( # object
        #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
        # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        
        
        if not self.args.use_arti_obj:
            object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
            nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # object pc #
            # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
            dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
                (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
            ) 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs # nearest to all pts in all frames ## 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs # nn_obj_pcs # nn_obj_pcs #
            # # dist_threshold = 0.01 # threshold 
            dist_threshold = self.dist_threshold
            # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
            dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
            # # base_pts_mask: nn_obj_pcs #
            base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
            # # nn_base_pts x 3 -> torch tensor #
            base_pts = object_pc_th[0][base_pts_mask]
            # # base_pts_bf_sampling = base_pts.clone()
            base_normals = object_normal_th[0][base_pts_mask]
            
            nn_base_pts = self.nn_base_pts
            base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
            base_pts_idxes = base_pts_idxes[:nn_base_pts]
            
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
        else:
            # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
            nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # nearest_obj_pcs: ws x nn_jts x 3 --> for nearet obj pcs # 
            # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
            dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
                (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
            ) 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # ws x nn_obj #
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
            # # dist_threshold = 0.01 # threshold 
            dist_threshold = self.dist_threshold
            # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
            dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
            # # base_pts_mask: nn_obj_pcs #
            base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold) # nn_obj_pcs -> nearest_pcs mask #
            base_pts = object_pc_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
            base_normals = object_normal_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
            nn_base_pts = self.nn_base_pts
            base_pts_idxes = utils.farthest_point_sampling(base_pts[0:1], n_sampling=nn_base_pts)
            base_pts_idxes = base_pts_idxes[:nn_base_pts]
            base_pts = base_pts[:, base_pts_idxes]
            base_normals = base_normals[:, base_pts_idxes]
            
            base_pts_global_orient_mtx = object_global_orient_mtx_th # ws x 3 x 3 #
            base_pts_transl = object_trcansl_th # ws x 3 # 
            base_pts = torch.matmul(
                (base_pts - base_pts_transl.unsqueeze(1)), base_pts_global_orient_mtx.transpose(1, 2) # ws x nn_base_pts x 3 --> ws x nn_base_pts x 3 #
            )
            base_normals = torch.matmul(
                base_normals, base_pts_global_orient_mtx.transpose(1, 2)  # ws x nn_base_pts x 3 
            )
            
            
            
        
        
        rhand_joints = torch.matmul(
            rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        pert_rhand_joints = torch.matmul(
            pert_rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        if self.args.use_anchors:
            # rhand_anchors, pert_rhand_anchors #
            rhand_anchors = torch.matmul(
                rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
            pert_rhand_anchors = torch.matmul(
                pert_rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
        
        # if self.predicted_hand_joints is not None:
        #     # self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
        #     # self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
        #     # self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
        #     # self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
        #     pert_rhand_joints = self.predicted_hand_joints
        #     # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
        #     if self.predicted_hand_trans is not None:
        #         rhand_transl_var = self.predicted_hand_trans
        #         rhand_global_orient_var = self.predicted_hand_rot
        #         rhand_pose_var = self.predicted_hand_theta
        #         print(f"rhand_beta_var: {self.predicted_hand_beta.size()}")
        #         rhand_beta_var = self.predicted_hand_beta #.unsqueeze(0)

        
        
        if not self.args.use_arti_obj:
            # nf x nnj x nnb x 3 # 
            rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
            # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        else:
            rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1) # ws x nn_joints x nn_base_pts x 3 #
            # dist_base_pts_to_rhand_joints: ws x nn_joints x nn_base_pts -> the distance from base points to joint points #
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
        # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        
        # # nf x nnj x nnb x 3 # 
        # rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
        # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        
        
        rhand_joints_disp = pert_rhand_joints[1:, :, :] - pert_rhand_joints[:-1, :, :]
        
        
        # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # 
        if not self.args.use_arti_obj:
            # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        else:
            # base normals and normals #
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals[:-1].unsqueeze(1) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # unsqueeze the dimensiton 1 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals[:-1].unsqueeze(1)
        
        
        sampled_base_pts = base_pts
        ''' # ==== the normalization for rhand joints, anchors and base points ==== # '''
        # base pts, base normals, 
        if self.aug:
            rnd_R = common_utils.get_random_rot_np()
            R_th = torch.from_numpy(rnd_R).float()
            # base_pts: nn_base_pts x 3 #
            # augmentation for the base pts and normals, and rel from base pts to hand vertices # 
            base_pts = torch.matmul(base_pts, R_th)
            sampled_base_pts = base_pts
            base_normals = torch.matmul(base_normals, R_th)
            rhand_joints = torch.matmul(rhand_joints, R_th.unsqueeze(0))
            if self.use_anchors:
                rhand_anchors = torch.matmul(rhand_anchors, R_th.unsqueeze(0)) # for the rhand anchors and vertices #
                # rhand_joints =  # put it for anchors #
    

        
        # 
        # current states, current joints, moving attraction forces, energies and change of energies ##
        ''' Relative positions and distances normalization, strategy 1 '''
        # rhand_joints = rhand_joints*5.
        # base_pts = base_pts * 5.
        ''' Relative positions and distances normalization, strategy 1 '''
        # sampled_base_pts: nn_base_pts x 3 #
        # nf x nnj x nnb x 3 #
        # nf x nnj x nnb x 3 # # rel base pts to rhand joints # 
        if not self.args.use_arti_obj:
            rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(0).unsqueeze(0)
            # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        else:
            rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(1)
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
        ''' Sample pts in the ambient space '''
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
        if not self.args.use_arti_obj:
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
            # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        else:
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(1)[:-1] * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
            # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(1)[:-1]
        # nf x nnj x nnb ---> dist_vt_normals -> nf x nnj x nnb # # torch.sqrt() ##
        dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
            rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
        ))
        
        k_a = 1.
        k_b = 1.
        # k and # give me a noised sequence ... #
        # (ws - 1) x nnj x nnb # --> (ws - 1) x nnj x nnb # nnj x nnb # 
        # add noise -> chagne of the joints displacements 
        # -> change of along_normalss energies and vertical to normals energies #
        # -> change of energy taken to make the displacements #
        # jts_to_base_pts energy in the noisy sequence #
        # jts_to_base_pts energy in the clean sequence #
        # vt-normal, along_normal #
        # TODO: the normalization strategy: 1) per-instnace; 2) per-category; #3
        # att_forces: (ws - 1) x nnj x nnb # # 
        e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
        # (ws - 1) x nnj x nnb # -> dist vt normals #
        e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
        # base_pts; base_normals; 
        
        ''' normalization sstrategy 1 ''' # 
        # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
        # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
        # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
        # per_frame_avg_disp_along_normalss, per_frame_std_disp_along_normalss # 
        # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        disp_ws, nnj, nnb = e_disp_rel_to_base_along_normals.shape[:3]
        # disp_ws x nnf x nnb x 3 #  -> disp_ws x nnj x nnb
        per_frame_avg_disp_along_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_along_normals = torch.std( # std over all frames #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        per_frame_avg_disp_vt_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_vt_normals = torch.std( # std over all frames #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        # per_frame_avg_joints_dists_rel = torch.mean(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        # per_frame_std_joints_dists_rel = torch.std(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        ### normalizaed aong normals and vat normals  # ws x nnj x nnb 
        e_disp_rel_to_base_along_normals = (e_disp_rel_to_base_along_normals - per_frame_avg_disp_along_normals) / per_frame_std_disp_along_normals
        e_disp_rel_to_baes_vt_normals = (e_disp_rel_to_baes_vt_normals - per_frame_avg_disp_vt_normals) / per_frame_std_disp_vt_normals
        # enrgy temrs #
        ''' normalization sstrategy 1 ''' # 
         
        
        
        # ws x nnj x nnb x 3 # 
        # 1  x nnj x 3 
        ## self.avg_joints_rel, self.std_joints_rel
        ## self.avg_joints_dists, self.std_joints_dists
        # rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        ''' Relative positions and distances normalization, strategy 2 '''
        # # for each point normalize joints over all frames #
        
        # rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - self.avg_joints_rel.unsqueeze(-2)) / self.std_joints_rel.unsqueeze(-2)
        # dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - self.avg_joints_dists.unsqueeze(-1)) / self.std_joints_dists.unsqueeze(-1)
        ''' Relative positions and distances normalization, strategy 2 '''
        
        
        
        if self.denoising_stra == "rep":
            ''' Relative positions and distances normalization, strategy 3 '''
            # for each point normalize joints over all frames #
            # rel_base_pts_to_rhand_joints: nf x nnj x nnb x 3 #
            ''' Stra 1 -> per frame ''' # per-frame #
            per_frame_avg_joints_rel = torch.mean(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True # for each point #
            )
            per_frame_std_joints_rel = torch.std(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_avg_joints_dists_rel = torch.mean(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_std_joints_dists_rel = torch.std(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            # base pts #
            
            ''' Stra 2 -> per frame with joints '''
            # nf x nnj x nnb x 3 #
            # ws, nnf , nnb = rel_base_pts_to_rhand_joints.shape[:3]
            # per_frame_avg_joints_rel = torch.mean(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True # for each point #
            # ).unsqueeze(0)
            # per_frame_std_joints_rel = torch.std(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_avg_joints_dists_rel = torch.mean(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_std_joints_dists_rel = torch.std(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            
            
            # max xyz vlaues for the relative positions, maximum, minimum distances for them #
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        # 
        # nf x nnj x 3 -> 
        # 
        if self.denoising_stra == "motion_to_rep": # motion_to_rep # # rhand joints; 
            rhand_joints = (rhand_joints - self.avg_jts) / self.std_jts
        
        
        # self.maxx_rel, minn_rel, maxx_dists, minn_dists #
        # 
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        
        ''' Create captions and tokens for text-condtional settings '''
        # object_name
        # caption = f"{object_name}"
        # tokens = f"{object_name}/NOUN"
        
        # tokens = tokens.split(" ")
        # max_text_len = 20
        # if len(tokens) < max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        # pos_one_hots = [] ## pose one hots ##
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        caption = "apple"
        ''' Create captions and tokens for text-condtional settings '''
        
        
        ''' Obj data '''
        # obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id)
        # obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 # # obj verts; #
        # obj_normals = torch.from_numpy(obj_normals).float() # 
        # obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        # # object_global_orient_mtx_th, object_trcansl_th
        
        # base_pts, base_normals, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var #
        # rhand_transl, rhand_rot, rhand_theta # 
        # and only 
        # rhand_anchors, canon_rhand_anchors #
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            'rhand_joints': rhand_joints if not self.args.use_canon_joints else canon_rhand_joints,
            'rhand_verts': rhand_verts,
            'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
            'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
            'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
            'rhand_betas': rhand_beta_var,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            # cur_obj_verts, cur_obj_faces 
            'obj_verts': cur_obj_verts,
            # 'obj_normals': obj_normals,
            'obj_faces': cur_obj_faces, # nnfaces x 3 # nnfaces x 3 # -> obj faces #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, # not for the ambinet space valuess s#
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            ### === per frame avg disp along normals, vt normals === ###
            # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
            # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
            # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            ## sampled; learn the 
        }
        # rhand_anchors, canon_rhand_anchors #
        if self.use_anchors:
            rt_dict.update(
                {   # rhand_anchors, canon_rhand_anchors ##
                    'rhand_anchors': rhand_anchors, 
                    'canon_rhand_anchors': canon_rhand_anchors, #### rt_dict for updating anchors ###
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] =  # realtive 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict
        

    def __len__(self):
        return self.len


## GRAB dataset V22 #
class GRAB_Dataset_V22(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset ##
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts ## 0.01, 0.05, 0.3 ##
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.inst_normalization = args.inst_normalization
        self.args = args
        
        self.denoising_stra = args.denoising_stra
        
        # load datas
        # grab_path =  "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
        # id2objmeshname = []
        # obj_meshes = sorted(os.listdir(obj_mesh_path))
        # # objectmesh name #
        # id2objmeshname = [obj_meshes[i].split(".")[0] for i in range(len(obj_meshes))]
        # self.id2objmeshname = id2objmeshname
        
        
        self.aug_trans_T = 0.05
        self.aug_rot_T = 0.3
        self.aug_pose_T = 0.5
        self.aug_zero = 1e-4 if self.model_type not in ['v1_wsubj_wjointsv24', 'v1_wsubj_wjointsv25'] else 0.01
        
        self.sigmas_trans = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_trans_T), self.num_sche_steps
        ))
        self.sigmas_rot = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_rot_T), self.num_sche_steps
        ))
        self.sigmas_pose = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_pose_T), self.num_sche_steps
        ))
        
        
        self.data_folder = data_folder
        self.subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "/data1/sim/mano_models/mano/models" ### mano_path
        # self.aug = True
        self.aug = args.augment
        # self.use_anchors = False
        self.use_anchors = args.use_anchors
        # self.args = args
        
        # grab path extracted #
        self.grab_path = "/data1/sim/GRAB_extracted"
        # obj_mesh_path = os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')
        # id2objmesh = []
        # ''' Get idx to mesh path ''' 
        # obj_meshes = sorted(os.listdir(obj_mesh_path))
        # for i, fn in enumerate(obj_meshes):
        #     id2objmesh.append(os.path.join(obj_mesh_path, fn))
        # self.id2objmesh = id2objmesh
        # self.id2meshdata = {}
        ''' Get idx to mesh path ''' 
        
        ## obj root folder; obj p
        ### Load field data from root folders ###
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        ## TODO: add thsoe params to args
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
    
        mano_pkl_path = os.path.join(self.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_pkl_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.template_verts = np.array(mano_model['v_template'])
        self.template_faces = np.array(mano_model['f'])
        self.template_joints = np.array(mano_model['J'])
        #### finger tips; ####
        self.template_tips = self.template_verts[[745, 317, 444, 556, 673]]
        self.template_joints = np.concatenate([self.template_joints, self.template_tips], axis=0)
        #### template verts ####
        self.template_verts = self.template_verts * 0.001
        #### template joints ####
        self.template_joints = self.template_joints * 0.001 # nn_joints x 3 #
        # condition on template joints for current joints #
        
        # normalization and so so # combnations of those quantities ######## 
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        self.fingers_stats = [
            [16, 15, 14, 13, 0],
            [17, 3, 2, 1, 0],
            [18, 6, 5, 4, 0],
            [19, 12, 11, 10, 0],
            [20, 9, 8, 7, 0]
        ]
        # 5 x 5 states, the first dimension is the finger index
        self.fingers_stats = np.array(self.fingers_stats, dtype=np.int32)
        self.canon_obj = True
        
        self.dir_stra = "vecs" # "rot_angles", "vecs"
        # self.dir_stra = "rot_angles"
        # distance # 21 distances? # --> texture map like data.. ## nanshou  ##
        
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=24,
            use_pca=True,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )
        
        ### Load field data from root folders ### ## obj root folder ##
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        # anchor_load_driver, masking_load_driver #
        # use_anchors, self.hand_palm_vertex_mask #
        if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
            # anchor_load_driver, masking_load_driver #
            inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
            fvi, aw, _, _ = anchor_load_driver(inpath)
            self.face_vertex_index = torch.from_numpy(fvi).long()
            self.anchor_weight = torch.from_numpy(aw).float()
            
            anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
            palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
            hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
            # self.hand_palm_vertex_mask for hand palm mask #
            self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
        
        
        #### Load data from single_seq_path ####
        self.single_seq_path = args.single_seq_path
        self.data = np.load(self.single_seq_path, allow_pickle=True) # .item()
        
        
        self.cad_model_fn = args.cad_model_fn
        self.corr_fn = args.corr_fn # corr_fn 
        if len(self.corr_fn) > 0:
            self.raw_corr_data = np.load(self.corr_fn, allow_pickle=True)
            self.obj_verts, self.obj_faces = None, None
        # self.dist_stra = args.dist_stra
        
        # evaluate from predicted infos #
        # args.predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" #
        self.tot_predicted_infos = []
        # selected_seeds = [0, 2, 3, 4, 6, 8, 10, 11, 13, 14, 17, 18, 19, 22, 23, 27, 30, 31, 34, 36, 37, 39, 40, 41, 43, 44, 46, 47, 48, 50, 52, 53, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 73, 78, 80, 85, 87, 88, 89, 90, 91, 93, 95, 99, 100, 105, 106, 107, 110, 111, 112, 113, 115, 118, 119, 120, 121, 125, 127, 128, 130, 135, 136, 138, 139, 141, 143, 147, 149, 150, 151, 152, 155, 156, 158, 159, 163, 165, 167, 168, 169, 177, 178, 179, 182, 184, 186, 193, 198]
        # # for cur_seed in range(0, 100):
        # for cur_seed in selected_seeds:
        #     # ### TODO: modify this line to load predicted infos from rep_mean prediction ###
        #     # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" 
        #     # /data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_14_tag_rep_res_jts_hoi4d_scissors_t_300_.npy
        #     ####### predicted info fn #######
        #     # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy" 
        #     # f"predicted_infos_seq_300_seed_{seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy"
        #     # cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy" 
        #     # f"predicted_infos_seq_300_seed_{seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy"
        #     cur_predicted_info_fn = f"/data1/sim/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_res_jts_hoi4d_scissors_t_300_.npy" 
        #     cur_predicted_info = np.load(cur_predicted_info_fn, allow_pickle=True).item()
        #     self.tot_predicted_infos.append(cur_predicted_info)

        predicted_info_fn = "/data2/sim/eval_save/HOI_Arti/Scissors/one_frame_tot_proj_optimized_infos_sv_dict_seq_11_seed_22_tag_rep_res_jts_hoi4d_arti_scissors_t_400__dist_thres_0.01_with_proj_False_wjts_0.01.npy"
        predicted_info = np.load(predicted_info_fn, allow_pickle=True).item()
        ##### tot_predicted_infos #####
        self.tot_predicted_infos.append(predicted_info)

        self.len = len(self.tot_predicted_infos)

        
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    ## load clip data ##
    def load_clip_data(self, clip_idx):
        cur_clip = self.clips[clip_idx]
        if len(cur_clip) > 3:
            return
        f = cur_clip[2]
        clip_clean = np.load(f)
        # pert_folder_nm = self.split + '_pert'
        
        if self.args.train_all_clips:
            pert_folder_nm = f.split("/")[-2] # get the split folder name
        else:
            pert_folder_nm = self.split
        
        # if not self.use_pert:
        #     pert_folder_nm = self.split
        # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        ##### load subj params #####
        pure_file_name = f.split("/")[-1].split(".")[0]
        pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
        
        # subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        rhand_transl = subj_params["rhand_transl"]
        rhand_betas = subj_params["rhand_betas"]
        rhand_pose = clip_clean['f2']
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        loaded_clip = (
            cur_clip[0], cur_clip[1], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip # object idx? 
        
        return loaded_clip

        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]
            
    
    def get_ari_obj_fr_x_v2(self, i_frame_st, i_frame_ed, selected_frame_idx):
        if self.obj_verts is not None: # tot verts, tot faces #
            # print(f"self.obj_verts nt None, returning...")
            return self.obj_verts, self.obj_faces # return obj_verts and obj_faces #
        tot_obj_verts = []
        tot_obj_faces = []
        tot_selected_obj_idxes = [ii for ii in range(selected_frame_idx)] + [selected_frame_idx for _ in range(i_frame_ed - i_frame_st - selected_frame_idx)]
        for i_frame in tot_selected_obj_idxes:
            cur_obj_fn = f"/data2/sim/HOI_Processed_Data_Arti/Scissors/Scissors/case11/corr_mesh/object_{i_frame}.obj"
            cur_obj_verts, cur_obj_faces = load_ply_data(cur_obj_fn)
            obj_center = np.mean(cur_obj_verts, axis=0, keepdims=True)
            cur_obj_verts = cur_obj_verts - obj_center
            tot_obj_verts.append(cur_obj_verts)
            tot_obj_faces.append(cur_obj_faces)

        # self.obj_verts, self.obj_faces
        tot_obj_verts = np.stack(tot_obj_verts, axis=0)
        tot_obj_faces = np.stack(tot_obj_faces, axis=0)
        self.obj_verts = tot_obj_verts
        self.obj_faces = tot_obj_faces
        print(f"self.obj_verts: {self.obj_verts.shape}, self.obj_faces: {self.obj_faces.shape}")
        
        return tot_obj_verts, tot_obj_faces


    def get_ari_obj_fr_x(self, i_frame_st, i_frame_ed):
        if self.obj_verts is not None: # tot verts, tot faces #
            # print(f"self.obj_verts nt None, returning...")
            return self.obj_verts, self.obj_faces # return obj_verts and obj_faces #
        
        # raw_corr_data
        tot_obj_verts = []
        tot_obj_faces = []
        for i_frame in range(i_frame_st, i_frame_ed):
            cur_obj_rot = self.raw_corr_data[i_frame]['obj_rot']
            cur_obj_trans = self.raw_corr_data[i_frame]['obj_trans']
            cad_model_fn = [
                "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-1-align.obj",  # 
                "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-0-align.obj" 
            ]
            cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
            # nn_verts x 3 
            # nn_faces x 3 
            
            cur_obj_verts, cur_obj_faces = cur_obj_mesh.vertices, cur_obj_mesh.faces
            obj_center = np.mean(cur_obj_verts, axis=0, keepdims=True)
            cur_obj_verts = cur_obj_verts - obj_center
            tot_obj_verts.append(cur_obj_verts)
            tot_obj_faces.append(cur_obj_faces)
            
        # self.obj_verts, self.obj_faces
        tot_obj_verts = np.stack(tot_obj_verts, axis=0)
        tot_obj_faces = np.stack(tot_obj_faces, axis=0)
        
        self.obj_verts = tot_obj_verts
        self.obj_faces = tot_obj_faces
        print(f"self.obj_verts: {self.obj_verts.shape}, self.obj_faces: {self.obj_faces.shape}")
        
        return tot_obj_verts, tot_obj_faces


    #### enforce correct contacts #### ## enforce correct contacts ##
    # normalize instances #
    def __getitem__(self, index):
        ## GRAB single frame ## # enumerate clips #
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        # if self.load_meta:
        #     # c = self.clips[i_c]
        #     c = self.load_clip_data(i_c)

        start_idx = 0
        
        if len(self.corr_fn) > 0:
            # tot_obj_verts, tot_obj_faces
            selected_frame_idx = self.tot_predicted_infos[0]["selected_frame"]
            cur_obj_verts, cur_obj_faces = self.get_ari_obj_fr_x_v2(start_idx, start_idx + self.window_size, selected_frame_idx=selected_frame_idx) 

            # cur_obj_verts, cur_obj_faces = self.get_ari_obj_fr_x(start_idx, start_idx + self.window_size) # nn_obj_verts x 3; nn_obj_faces x 3 #
            # print(f"corr_fn: {self.corr_fn}, obj_verts: {cur_obj_verts.shape}, cur_obj_faces: {cur_obj_faces.shape}")
        
        
        object_pc = self.data['f1'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        object_vn = self.data['f2'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        rhand_joints = self.data['f11'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        
        # rhand_glb_rot, rhand_pose, rhand_joints_gt, minn_dists_rhand_joints_object_pc,
        # rhand_joints = self.data[index]['f11'].reshape(21, 3).astype(np.float32)
        # rhand_joints = rhand_joints * 0.001
        # rhand_joints = rhand_joints - obj_center
        
        # rhand_joints_fr_data = rhand_joints.copy() ## rhandjoints
        
        
        rhand_transl = self.data['f10'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
        # rhand_transl = rhand_transl - obj_center[0]
        rhand_beta = self.data['f9'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
        rhand_theta = self.data['f8'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
        
        
        # rhand_transl_clean = self.clean_data[index]['f10'].reshape(3).astype(np.float32)
        # rhand_theta_clean = self.clean_data[index]['f8'].reshape(-1).astype(np.float32)
        
        rhand_glb_rot = rhand_theta[:, :3]
        rhand_theta = rhand_theta[:, 3:]
        
        ##### rhand transl #####
        # rhand_glb_rot = rhand_theta_clean[:3]
        # rhand_transl = rhand_transl_clean
        ##### rhand transl #####
        
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var, rhand_beta_var #
        rhand_global_orient_var = torch.from_numpy(rhand_glb_rot).float() # .unsqueeze(0)
        rhand_pose_var = torch.from_numpy(rhand_theta).float() # . unsqueeze(0)
        rhand_transl_var = torch.from_numpy(rhand_transl).float() # .unsqueeze(0)
        rhand_beta_var = torch.from_numpy(rhand_beta).float() # .unsqueeze(0)
        
        
        
        # # rhand_global_orient = self.data[index]['f1'].reshape(-1).astype(np.float32)
        # rhand_pose = rhand_theta
        # # rhand_transl = self.subj_params['rhand_transl'][index].reshape(-1).astype(np.float32)
        # rhand_betas = rhand_beta
        
        
        print(f"rhand_transl_var: {rhand_transl_var.size()}, rhand_pose_var: {rhand_pose_var.size()}, rhand_global_orient_var: {rhand_global_orient_var.size()}, rhand_beta_var: {rhand_beta_var.size()}")
        ####### Get rhand_verts and rhand_joint #######
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.view(-1, 10), rhand_transl_var
        )
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        ####### Get rhand_verts and rhand_joint #######


        # tot_proj_sv_dict = {
        #     'tot_proj_rot_var': tot_proj_rot_var.detach().cpu().numpy(),
        #     'tot_proj_theta_var': tot_proj_theta_var.detach().cpu().numpy(),
        #     'tot_proj_transl_var': tot_proj_transl_var.detach().cpu().numpy(),
        #     'tot_proj_beta_var': tot_proj_beta_var.detach().cpu().numpy(),
        #     'tot_proj_hand_verts': tot_proj_hand_verts.detach().cpu().numpy(),
        #     'tot_proj_hand_verts_ori': tot_proj_hand_verts_ori.detach().cpu().numpy(),
        #     'tot_proj_hand_joints': tot_proj_hand_joints.detach().cpu().numpy(),
        #     'selected_frame': selected_frame, # selected frame #
        # }
        
        
        cur_predicted_info = self.tot_predicted_infos[index]
        # outputs = data['outputs']
        # self.predicted_hand_trans = data['rhand_trans'] # nframes x 3 
        # self.predicted_hand_rot = data['rhand_rot'] # nframes x 3 
        # self.predicted_hand_theta = data['rhand_theta']
        # self.predicted_hand_beta = data['rhand_beta']
        # rhand_joints = cur_predicted_info["outputs"]
        # rhand_joints = torch.from_numpy(rhand_joints).float()

        rhand_joints = cur_predicted_info["tot_proj_hand_joints"]
        rhand_joints = torch.from_numpy(rhand_joints).float()

        rhand_transl_var = cur_predicted_info["tot_proj_transl_var"]
        rhand_pose_var = cur_predicted_info["tot_proj_theta_var"]
        rhand_global_orient_var = cur_predicted_info["tot_proj_rot_var"]
        rhand_beta_var = cur_predicted_info["tot_proj_beta_var"]

        rhand_transl_var = torch.from_numpy(rhand_transl_var).float()
        rhand_pose_var = torch.from_numpy(rhand_pose_var).float()
        rhand_global_orient_var = torch.from_numpy(rhand_global_orient_var).float()
        rhand_beta_var = torch.from_numpy(rhand_beta_var).float()

        rhand_beta_var = rhand_beta_var.repeat(rhand_joints.size(0), 1).contiguous()
        print(f"rhand_transl_var: {rhand_transl_var.size()}, rhand_pose_var: {rhand_pose_var.size()}, rhand_global_orient_var: {rhand_global_orient_var.size()}, rhand_beta_var: {rhand_beta_var.size()}")

        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.view(-1, 10), rhand_transl_var
        )
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001


        # if "rhand_trans" in cur_predicted_info:
        #     rhand_transl_var = cur_predicted_info["rhand_trans"] # nframes x 3 for rhand trans 
        #     rhand_global_orient_var = cur_predicted_info["rhand_rot"]
        #     rhand_pose_var = cur_predicted_info["rhand_theta"] # thetas
        #     rhand_beta_var = cur_predicted_info["rhand_beta"] # thetas #
            
            
        #     rhand_verts, rhand_joints = self.mano_layer(
        #         torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
        #         rhand_beta_var.view(-1, 10), rhand_transl_var
        #     )
        #     rhand_verts = rhand_verts * 0.001
        #     rhand_joints = rhand_joints * 0.001
        
        
        if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 # # recover anchor 
            # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
            rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            pert_rhand_anchors = rhand_anchors
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        
        # rhand_transl = rhand_transl - obj_center[0]
        
        
        pert_rhand_joints = rhand_joints
        pert_rhand_verts = rhand_verts
        
        
        object_global_orient = self.data['f3'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3, 3).astype(np.float32) # nf x 
        object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
        object_transl = self.data['f4'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
        object_trcansl_th = torch.from_numpy(object_transl).float()
        
        
        
        
        object_normal = object_vn
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        
        # ws x nnjoints x nnobjpts #
        dist_rhand_joints_to_obj_pc = torch.sum( # dist rhand joints to obj pc
            (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        )
        # dist_pert_rhand_joints_obj_pc = torch.sum( # object
        #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
        # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        
        
        if not self.args.use_arti_obj:
            object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
            nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # object pc #
            # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
            dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
                (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
            ) 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs # nearest to all pts in all frames ## 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs # nn_obj_pcs # nn_obj_pcs #
            # # dist_threshold = 0.01 # threshold 
            dist_threshold = self.dist_threshold
            # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
            dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
            # # base_pts_mask: nn_obj_pcs #
            base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
            # # nn_base_pts x 3 -> torch tensor #
            base_pts = object_pc_th[0][base_pts_mask]
            # # base_pts_bf_sampling = base_pts.clone()
            base_normals = object_normal_th[0][base_pts_mask]
            
            nn_base_pts = self.nn_base_pts
            base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
            base_pts_idxes = base_pts_idxes[:nn_base_pts]
            
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
        else:
            # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
            nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # nearest_obj_pcs: ws x nn_jts x 3 --> for nearet obj pcs # 
            # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
            dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
                (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
            ) 
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # ws x nn_obj #
            dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
            # # dist_threshold = 0.01 # threshold 
            dist_threshold = self.dist_threshold
            # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
            dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
            # # base_pts_mask: nn_obj_pcs #
            base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold) # nn_obj_pcs -> nearest_pcs mask #
            base_pts = object_pc_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
            base_normals = object_normal_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
            nn_base_pts = self.nn_base_pts
            base_pts_idxes = utils.farthest_point_sampling(base_pts[0:1], n_sampling=nn_base_pts)
            base_pts_idxes = base_pts_idxes[:nn_base_pts]
            base_pts = base_pts[:, base_pts_idxes]
            base_normals = base_normals[:, base_pts_idxes]
            
            base_pts_global_orient_mtx = object_global_orient_mtx_th # ws x 3 x 3 #
            base_pts_transl = object_trcansl_th # ws x 3 # 
            base_pts = torch.matmul(
                (base_pts - base_pts_transl.unsqueeze(1)), base_pts_global_orient_mtx.transpose(1, 2) # ws x nn_base_pts x 3 --> ws x nn_base_pts x 3 #
            )
            base_normals = torch.matmul(
                base_normals, base_pts_global_orient_mtx.transpose(1, 2)  # ws x nn_base_pts x 3 
            )
            
            
            
        
        
        rhand_joints = torch.matmul(
            rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        pert_rhand_joints = torch.matmul(
            pert_rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        if self.args.use_anchors:
            # rhand_anchors, pert_rhand_anchors #
            rhand_anchors = torch.matmul(
                rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
            pert_rhand_anchors = torch.matmul(
                pert_rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
        
        # if self.predicted_hand_joints is not None:
        #     # self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
        #     # self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
        #     # self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
        #     # self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
        #     pert_rhand_joints = self.predicted_hand_joints
        #     # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
        #     if self.predicted_hand_trans is not None:
        #         rhand_transl_var = self.predicted_hand_trans
        #         rhand_global_orient_var = self.predicted_hand_rot
        #         rhand_pose_var = self.predicted_hand_theta
        #         print(f"rhand_beta_var: {self.predicted_hand_beta.size()}")
        #         rhand_beta_var = self.predicted_hand_beta #.unsqueeze(0)

        
        
        if not self.args.use_arti_obj:
            # nf x nnj x nnb x 3 # 
            rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
            # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        else:
            rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1) # ws x nn_joints x nn_base_pts x 3 #
            # dist_base_pts_to_rhand_joints: ws x nn_joints x nn_base_pts -> the distance from base points to joint points #
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
        # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        
        # # nf x nnj x nnb x 3 # 
        # rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
        # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        
        
        rhand_joints_disp = pert_rhand_joints[1:, :, :] - pert_rhand_joints[:-1, :, :]
        
        
        # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # 
        if not self.args.use_arti_obj:
            # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        else:
            # base normals and normals #
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals[:-1].unsqueeze(1) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # unsqueeze the dimensiton 1 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals[:-1].unsqueeze(1)
        
        
        sampled_base_pts = base_pts
        ''' # ==== the normalization for rhand joints, anchors and base points ==== # '''
        # base pts, base normals, 
        if self.aug:
            rnd_R = common_utils.get_random_rot_np()
            R_th = torch.from_numpy(rnd_R).float()
            # base_pts: nn_base_pts x 3 #
            # augmentation for the base pts and normals, and rel from base pts to hand vertices # 
            base_pts = torch.matmul(base_pts, R_th)
            sampled_base_pts = base_pts
            base_normals = torch.matmul(base_normals, R_th)
            rhand_joints = torch.matmul(rhand_joints, R_th.unsqueeze(0))
            if self.use_anchors:
                rhand_anchors = torch.matmul(rhand_anchors, R_th.unsqueeze(0)) # for the rhand anchors and vertices #
                # rhand_joints =  # put it for anchors #
    

        
        # 
        # current states, current joints, moving attraction forces, energies and change of energies ##
        ''' Relative positions and distances normalization, strategy 1 '''
        # rhand_joints = rhand_joints*5.
        # base_pts = base_pts * 5.
        ''' Relative positions and distances normalization, strategy 1 '''
        # sampled_base_pts: nn_base_pts x 3 #
        # nf x nnj x nnb x 3 #
        # nf x nnj x nnb x 3 # # rel base pts to rhand joints # 
        if not self.args.use_arti_obj:
            rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(0).unsqueeze(0)
            # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        else:
            rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(1)
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
        ''' Sample pts in the ambient space '''
        
        
        # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
        ## TODO: other choices of k_f? ##
        k_f = 1.
        # relative #
        l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
        att_forces = att_forces[:-1, :, :]
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
        if not self.args.use_arti_obj:
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
            # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
        else:
            signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
                base_normals.unsqueeze(1)[:-1] * rhand_joints_disp.unsqueeze(2), dim=-1
            )
            # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
            # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
            # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(1)[:-1]
        # nf x nnj x nnb ---> dist_vt_normals -> nf x nnj x nnb # # torch.sqrt() ##
        dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
            rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
        ))
        
        k_a = 1.
        k_b = 1.
        # k and # give me a noised sequence ... #
        # (ws - 1) x nnj x nnb # --> (ws - 1) x nnj x nnb # nnj x nnb # 
        # add noise -> chagne of the joints displacements 
        # -> change of along_normalss energies and vertical to normals energies #
        # -> change of energy taken to make the displacements #
        # jts_to_base_pts energy in the noisy sequence #
        # jts_to_base_pts energy in the clean sequence #
        # vt-normal, along_normal #
        # TODO: the normalization strategy: 1) per-instnace; 2) per-category; #3
        # att_forces: (ws - 1) x nnj x nnb # # 
        e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
        # (ws - 1) x nnj x nnb # -> dist vt normals #
        e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
        # base_pts; base_normals; 
        
        ''' normalization sstrategy 1 ''' # 
        # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
        # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
        # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
        # per_frame_avg_disp_along_normalss, per_frame_std_disp_along_normalss # 
        # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        disp_ws, nnj, nnb = e_disp_rel_to_base_along_normals.shape[:3]
        # disp_ws x nnf x nnb x 3 #  -> disp_ws x nnj x nnb
        per_frame_avg_disp_along_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_along_normals = torch.std( # std over all frames #
            e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        per_frame_avg_disp_vt_normals = torch.mean( # avg over all frmaes #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
        ) # .unsqueeze(0)
        per_frame_std_disp_vt_normals = torch.std( # std over all frames #
            e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
        ) # .unsqueeze(0)
        # per_frame_avg_joints_dists_rel = torch.mean(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        # per_frame_std_joints_dists_rel = torch.std(
        #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
        # ).unsqueeze(0)
        ### normalizaed aong normals and vat normals  # ws x nnj x nnb 
        e_disp_rel_to_base_along_normals = (e_disp_rel_to_base_along_normals - per_frame_avg_disp_along_normals) / per_frame_std_disp_along_normals
        e_disp_rel_to_baes_vt_normals = (e_disp_rel_to_baes_vt_normals - per_frame_avg_disp_vt_normals) / per_frame_std_disp_vt_normals
        # enrgy temrs #
        ''' normalization sstrategy 1 ''' # 
         
        
        
        # ws x nnj x nnb x 3 # 
        # 1  x nnj x 3 
        ## self.avg_joints_rel, self.std_joints_rel
        ## self.avg_joints_dists, self.std_joints_dists
        # rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        ''' Relative positions and distances normalization, strategy 2 '''
        # # for each point normalize joints over all frames #
        
        # rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - self.avg_joints_rel.unsqueeze(-2)) / self.std_joints_rel.unsqueeze(-2)
        # dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - self.avg_joints_dists.unsqueeze(-1)) / self.std_joints_dists.unsqueeze(-1)
        ''' Relative positions and distances normalization, strategy 2 '''
        
        
        
        if self.denoising_stra == "rep":
            ''' Relative positions and distances normalization, strategy 3 '''
            # for each point normalize joints over all frames #
            # rel_base_pts_to_rhand_joints: nf x nnj x nnb x 3 #
            ''' Stra 1 -> per frame ''' # per-frame #
            per_frame_avg_joints_rel = torch.mean(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True # for each point #
            )
            per_frame_std_joints_rel = torch.std(
                rel_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_avg_joints_dists_rel = torch.mean(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            per_frame_std_joints_dists_rel = torch.std(
                dist_base_pts_to_rhand_joints, dim=0, keepdim=True
            )
            # base pts #
            
            ''' Stra 2 -> per frame with joints '''
            # nf x nnj x nnb x 3 #
            # ws, nnf , nnb = rel_base_pts_to_rhand_joints.shape[:3]
            # per_frame_avg_joints_rel = torch.mean(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True # for each point #
            # ).unsqueeze(0)
            # per_frame_std_joints_rel = torch.std(
            #     rel_base_pts_to_rhand_joints.view(ws * nnf, nnb, 3), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_avg_joints_dists_rel = torch.mean(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            # per_frame_std_joints_dists_rel = torch.std(
            #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
            # ).unsqueeze(0)
            
            
            # max xyz vlaues for the relative positions, maximum, minimum distances for them #
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        # 
        # nf x nnj x 3 -> 
        # 
        if self.denoising_stra == "motion_to_rep": # motion_to_rep # # rhand joints; 
            rhand_joints = (rhand_joints - self.avg_jts) / self.std_jts
        
        
        # self.maxx_rel, minn_rel, maxx_dists, minn_dists #
        # 
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        
        ''' Create captions and tokens for text-condtional settings '''
        # object_name
        # caption = f"{object_name}"
        # tokens = f"{object_name}/NOUN"
        
        # tokens = tokens.split(" ")
        # max_text_len = 20
        # if len(tokens) < max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        # pos_one_hots = [] ## pose one hots ##
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        caption = "apple"
        ''' Create captions and tokens for text-condtional settings '''
        
        
        ''' Obj data '''
        # obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id)
        # obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 # # obj verts; #
        # obj_normals = torch.from_numpy(obj_normals).float() # 
        # obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        # # object_global_orient_mtx_th, object_trcansl_th
        
        # base_pts, base_normals, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints # 
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var #
        # rhand_transl, rhand_rot, rhand_theta # 
        # and only 
        # rhand_anchors, canon_rhand_anchors #
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            'rhand_joints': rhand_joints if not self.args.use_canon_joints else canon_rhand_joints,
            'rhand_verts': rhand_verts,
            'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
            'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
            'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
            'rhand_betas': rhand_beta_var,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            # cur_obj_verts, cur_obj_faces 
            'obj_verts': cur_obj_verts,
            # 'obj_normals': obj_normals,
            'obj_faces': cur_obj_faces, # nnfaces x 3 # nnfaces x 3 # -> obj faces #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, # not for the ambinet space valuess s#
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            ### === per frame avg disp along normals, vt normals === ###
            # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
            # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
            # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            ## sampled; learn the 
        }
        # rhand_anchors, canon_rhand_anchors #
        if self.use_anchors:
            rt_dict.update(
                {   # rhand_anchors, canon_rhand_anchors ##
                    'rhand_anchors': rhand_anchors, 
                    'canon_rhand_anchors': canon_rhand_anchors, #### rt_dict for updating anchors ###
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] =  # realtive 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict
        

    def __len__(self):
        return self.len


### GRAB dataset with pose ###
# GRAB dataset with pose # # GRAB_Dataset_V16 # # with subject and object distances 
class GRAB_Dataset_V20(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts ## 0.01, 0.05, 0.3 ##
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.inst_normalization = args.inst_normalization
        self.args = args
        
        self.denoising_stra = args.denoising_stra
        
        # load datas
        grab_path =  "/data1/sim/GRAB_extracted"
        obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
        id2objmeshname = []
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        # objectmesh name #
        # id2objmeshname = [obj_meshes[i].split(".")[0] for i in range(len(obj_meshes))]
        # idx to object mesh path here ##
        id2objmeshname = [os.path.join(obj_mesh_path, cur_obj_mesh_path) for cur_obj_mesh_path in obj_meshes]
        self.id2objmeshname = id2objmeshname
        
        self.grid_data_sv_root = args.grid_data_sv_root
        
        
        self.aug_trans_T = 0.05
        self.aug_rot_T = 0.3
        self.aug_pose_T = 0.5
        self.aug_zero = 1e-4 if self.model_type not in ['v1_wsubj_wjointsv24', 'v1_wsubj_wjointsv25'] else 0.01
        
        self.sigmas_trans = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_trans_T), self.num_sche_steps
        ))
        self.sigmas_rot = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_rot_T), self.num_sche_steps
        ))
        self.sigmas_pose = np.exp(np.linspace(
            np.log(self.aug_zero), np.log(self.aug_pose_T), self.num_sche_steps
        ))
        
        
        self.data_folder = data_folder
        self.subj_data_folder = '/data1/sim/GRAB_processed_wsubj'
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "/data1/sim/mano_models/mano/models" ### mano_path
        self.aug = True
        self.use_anchors = False
        # self.args = args
        
        
        self.grab_path = "/data1/sim/GRAB_extracted"
        obj_mesh_path = os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')
        id2objmesh = []
        ''' Get idx to mesh path ''' 
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        ''' Get idx to mesh path ''' 
        
        ## obj root folder; obj p
        ### Load field data from root folders ###
        self.obj_root_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/sim/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
        ''' Load avg, std statistics '''
        # avg_joints_motion_ours_fn = f"/home/xueyi/sim/motion-diffusion-model/avg_joints_motion_ours_nb_{700}_nth_{0.005}.npy"
        # std_joints_motion_ours_fn = f"/home/xueyi/sim/motion-diffusion-model/std_joints_motion_ours_nb_{700}_nth_{0.005}.npy"
        # avg_joints_motion_dists_ours_fn = f"/home/xueyi/sim/motion-diffusion-model/avg_joints_dist_motion_ours_nb_{700}_nth_{0.005}.npy"
        # std_joints_motion_dists_ours_fn = f"/home/xueyi/sim/motion-diffusion-model/std_joints_dist_motion_ours_nb_{700}_nth_{0.005}.npy"
        # avg_joints_rel = np.load(avg_joints_motion_ours_fn, allow_pickle=True)
        # std_joints_rel = np.load(std_joints_motion_ours_fn, allow_pickle=True)
        # avg_joints_dists = np.load(avg_joints_motion_dists_ours_fn, allow_pickle=True)
        # std_joints_dists = np.load(std_joints_motion_dists_ours_fn, allow_pickle=True)
        # ## self.avg_joints_rel, self.std_joints_rel
        # ## self.avg_joints_dists, self.std_joints_dists
        # self.avg_joints_rel = torch.from_numpy(avg_joints_rel).float()
        # self.std_joints_rel = torch.from_numpy(std_joints_rel).float()
        # self.avg_joints_dists = torch.from_numpy(avg_joints_dists).float()
        # self.std_joints_dists = torch.from_numpy(std_joints_dists).float()
        ''' Load avg, std statistics '''
        
        
        ''' Load avg, std statistics '''
        # # self.maxx_rel, minn_rel, maxx_dists, minn_dists #
        # rel_dists_stats_fn = "/home/xueyi/sim/motion-diffusion-model/base_pts_rel_dists_stats.npy"
        # rel_dists_stats = np.load(rel_dists_stats_fn, allow_pickle=True).item()
        # maxx_rel = rel_dists_stats['maxx_rel']
        # minn_rel = rel_dists_stats['minn_rel']
        # maxx_dists = rel_dists_stats['maxx_dists']
        # minn_dists = rel_dists_stats['minn_dists']
        # self.maxx_rel = torch.from_numpy(maxx_rel).float()
        # self.minn_rel = torch.from_numpy(minn_rel).float()
        # self.maxx_dists = torch.from_numpy(maxx_dists).float()
        # self.minn_dists = torch.from_numpy(minn_dists).float()
        ''' Load avg, std statistics '''
        
        
        ''' Load avg-jts, std-jts '''
        # avg_jts_fn = "/home/xueyi/sim/motion-diffusion-model/avg_joints_motion_ours.npy"
        # std_jts_fn = "/home/xueyi/sim/motion-diffusion-model/std_joints_motion_ours.npy"
        # avg_jts = np.load(avg_jts_fn, allow_pickle=True)
        # std_jts = np.load(std_jts_fn, allow_pickle=True)
        # # self.avg_jts, self.std_jts #
        # self.avg_jts = torch.from_numpy(avg_jts).float()
        # self.std_jts = torch.from_numpy(std_jts).float()
        ''' Load avg-jts, std-jts '''
        
        # hand and the normalize hands only to the cube in the space? #
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        ## TODO: add thsoe params to args
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
    
        mano_pkl_path = os.path.join(self.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_pkl_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        self.template_verts = np.array(mano_model['v_template'])
        self.template_faces = np.array(mano_model['f'])
        self.template_joints = np.array(mano_model['J'])
        #### finger tips; ####
        self.template_tips = self.template_verts[[745, 317, 444, 556, 673]]
        self.template_joints = np.concatenate([self.template_joints, self.template_tips], axis=0)
        #### template verts ####
        self.template_verts = self.template_verts * 0.001
        #### template joints ####
        self.template_joints = self.template_joints * 0.001 # nn_joints x 3 #
        # condition on template joints for current joints #
        
        # normalization and so so # combnations of those quantities ######## 
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        self.fingers_stats = [
            [16, 15, 14, 13, 0],
            [17, 3, 2, 1, 0],
            [18, 6, 5, 4, 0],
            [19, 12, 11, 10, 0],
            [20, 9, 8, 7, 0]
        ]
        # 5 x 5 states, the first dimension is the finger index
        self.fingers_stats = np.array(self.fingers_stats, dtype=np.int32)
        self.canon_obj = True
        
        self.dir_stra = "vecs" # "rot_angles", "vecs"
        # self.dir_stra = "rot_angles"
        # distance # 21 distances? # --> texture map like data.. ## nanshou  ##
        # 
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=24,
            use_pca=True,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )
        
        ## actions taken 
        # self.clip_sv_folder = os.path.join(data_folder, f"{split}_clip")
        # os.makedirs(self.clip_sv_folder, exist_ok=True)

        files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        #### filter files_clean here ####
        files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
        
        if self.load_meta:
            for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
                if split != 'train' and split != 'val' and i_f >= 100:
                    break
                if split == 'train':
                    print(f"loading {i_f} / {len(files_clean)}")
                base_nm_f = os.path.basename(f)
                base_name_f = base_nm_f.split(".")[0]
                cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
                cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
                cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
                cur_clip_len = cur_clip_meta_data['clip_len']
                clip_len = (cur_clip_len - window_size) // step_size + 1
                if self.args.only_first_clip:
                    clip_len = min(clip_len, 1)
                self.clips.append((self.len, self.len+clip_len,  f
                    ))
                self.len += clip_len # len clip len
        else:
            for i_f, f in enumerate(files_clean):
                if split == 'train':
                    print(f"loading {i_f} / {len(files_clean)}")
                if split != 'train' and i_f >= 100:
                    break
                if args is not None and args.debug and i_f >= 10:
                    break
                clip_clean = np.load(f)
                pert_folder_nm = split + '_pert'
                if args is not None and not args.use_pert:
                    pert_folder_nm = split
                clip_pert = np.load(os.path.join(data_folder, pert_folder_nm, os.path.basename(f)))
                clip_len = (len(clip_clean) - window_size) // step_size + 1
                sv_clip_pert = {}
                for i_idx in range(6):
                    sv_clip_pert[f'f{i_idx + 1}'] = clip_pert[f'f{i_idx + 1}']
                
                ### sv clip pert, 
                ##### load subj params #####
                pure_file_name = f.split("/")[-1].split(".")[0]
                pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                        
                subj_params_fn = os.path.join(self.subj_data_folder, split, pure_subj_params_fn)
                subj_params = np.load(subj_params_fn, allow_pickle=True).item()
                rhand_transl = subj_params["rhand_transl"]
                rhand_betas = subj_params["rhand_betas"]
                rhand_pose = clip_clean['f2'] ## rhand pose ##
                
                pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
                pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
                ##### load subj params #####
                
                # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
                # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
                ### pert and clean pair for encoding and decoding ###
                self.clips.append((self.len, self.len+clip_len, clip_pert,
                    [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
                    # subj_corr_data, pert_subj_corr_data
                    ))
                # self.clips.append((self.len, self.len+clip_len, sv_clip_pert,
                #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
                #     # subj_corr_data, pert_subj_corr_data
                #     )) ## object surface; grid positions; points sampled from the sapce; and you may need 3D conv nets; 
                # two objects and the change of the distance field; 
                # object surface points and the subject-related quantities grounded on it. 
                self.len += clip_len # len clip len
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx):
        cur_clip = self.clips[clip_idx]
        if len(cur_clip) > 3: # curclip? 
            return
        f = cur_clip[2]
        clip_clean = np.load(f) #  file name of the clip file #
        # pert_folder_nm = self.split + '_pert'
        pert_folder_nm = self.split
        # if not self.use_pert:
        #     pert_folder_nm = self.split
        # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        ##### load subj params #####
        pure_file_name = f.split("/")[-1].split(".")[0]
        pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                
        subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        rhand_transl = subj_params["rhand_transl"]
        rhand_betas = subj_params["rhand_betas"]
        rhand_pose = clip_clean['f2']
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        loaded_clip = (
            cur_clip[0], cur_clip[1], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip # object idx? 
        
        return loaded_clip

        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]
            

    def seal(self, mesh_to_seal, rh=True):
        circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
        center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

        sealed_mesh = mesh_to_seal.copy()
        sealed_mesh.vertices = np.vstack([mesh_to_seal.vertices, center])
        center_v_id = sealed_mesh.vertices.shape[0] - 1

        for i in range(circle_v_id.shape[0]):
            if rh:
                new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id]
            else:
                new_faces = [center_v_id, circle_v_id[i], circle_v_id[i-1]]
            sealed_mesh.faces = np.vstack([sealed_mesh.faces, new_faces])
        return sealed_mesh

    #### enforce correct contacts ####
    # normalize instances #
    def __getitem__(self, index):
        ## GRAB single frame ## # enumerate clips #
        for i_c, c in enumerate(self.clips):
            if index < c[1]:
                break
        if self.load_meta:
            # c = self.clips[i_c] 
            c = self.load_clip_data(i_c)
            file_name = self.clips[i_c][2] # get file name #
            file_idx = file_name.split("/")[-1].split(".")[0]
            file_idx = int(file_idx)
            grid_data_sv_fn = os.path.join(self.grid_data_sv_root, f"{file_idx}.npy") # file_idx for data_sv_root here #

        object_id = c[3][-1] ## object_idx here ##
        object_name = self.id2objmeshname[object_id]
        
        object_mesh = trimesh.load_mesh(object_name, process=False)
        
        # TODO: add random noise settings for noisy input #
        start_idx = (index - c[0]) * self.step_size
        # data = c[2][start_idx:start_idx+self.window_size]
        data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        object_global_orient = data['f5'] ### get object global orientations ###
        object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        # object glboal orientations # object transl # # object transl th #
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        object_trcansl_th = torch.from_numpy(object_trcansl).float()
        
        # pert_subj_params = c[4]
        
        
        
        
        ### pts gt ###
        ## rhnad pose, rhand pose gt ##
        ## glboal orientation and hand pose #
        rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        rhand_transl, rhand_betas = c[3][5], c[3][6]
        rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        # # orientation rotation matrix #
        # rhand_global_orient_mtx_gt = utils.batched_get_orientation_matrices(rhand_global_orient_gt)
        # rhand_global_orient_mtx_gt_var = torch.from_numpy(rhand_global_orient_mtx_gt).float()
        # # orientation rotation matrix #
        
        rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
        rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
        rhand_beta_var = torch.from_numpy(rhand_betas).float()
        rhand_transl_var = torch.from_numpy(rhand_transl).float() # self.window_size x 3
        # R.from_rotvec(obj_rot).as_matrix()
        
        
        # rhand joints; rhand joints #
        # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 # # ws x nnjoints x 3 ## --> ##
        # pert_rhand_betas_var, rhand_beta_var
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        )
        ### rhand_joints: for joints ###
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        
        
        canon_rhand_verts, canon_rhand_joints = self.mano_layer(
            torch.cat([torch.zeros_like(rhand_global_orient_var), rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(rhand_transl_var)
        )
        ### rhand_joints: for joints ###
        canon_rhand_verts = canon_rhand_verts * 0.001
        canon_rhand_joints = canon_rhand_joints * 0.001
        
        
        
        # ### Relative positions from base points to rhand joints ###
        object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        ## canonicalized hands to object pc ##
        # base_pts_feats_sv_dict = {}
        #### distance between rhand joints and obj pcs ####
        # pert_rhand_joints_th = pert_rhand_joints
        # ws x nnjoints x nnobjpts #
        dist_rhand_joints_to_obj_pc = torch.sum(
            (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        )
        # dist_pert_rhand_joints_obj_pc = torch.sum(
        #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
        # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
        # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        dist_object_pc_nearest_pcs = torch.sum(
            (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
        )
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
        dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
        # # dist_threshold = 0.01
        dist_threshold = self.dist_threshold
        # # dist_threshold for pc_nearest_pcs #
        dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
        
        # # base_pts_mask: nn_obj_pcs #
        base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
        # # nn_base_pts x 3 -> torch tensor #
        base_pts = object_pc_th[0][base_pts_mask]
        # # base_pts_bf_sampling = base_pts.clone()
        base_normals = object_normal_th[0][base_pts_mask]
        
        nn_base_pts = self.nn_base_pts
        base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
        base_pts_idxes = base_pts_idxes[:nn_base_pts]
        # if self.debug:
        #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
        
        # ### get base points ### # base_pts and base_normals #
        base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
        base_normals = base_normals[base_pts_idxes]
        
        
        # # object_global_orient_mtx # nn_ws x 3 x 3 #
        base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
        base_pts_transl = object_trcansl_th[0] # 3 # 
        
        
        
        base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
            )
        base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
            ) # .transpose(0, 1)
        
        # base_pts, base_normals; 
        
        ### rhand_joints ###
        rhand_joints = torch.matmul(
            rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        
        #### Normalize rhand joints here ####
        # rhand joints # nf x nn_verts x 3 #
        nf, nn_verts = rhand_verts.size()[:2]
        rhand_verts = torch.matmul( 
            rhand_verts - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2) # nf x nn_verts x 3 --> number of vertices and the transformed rhand_verts
        ) 
        rhand_verts_exp = rhand_verts.view(nf * nn_verts, 3)
        avg_rhand_verts = rhand_verts_exp.mean(dim=0, keepdim=True).unsqueeze(1)
        normed_rhand_verts = rhand_verts - avg_rhand_verts
        normed_rhand_verts_exp = normed_rhand_verts.view(normed_rhand_verts.size(0) * normed_rhand_verts.size(1), 3)
        maxx_normed_rhand_verts, _ = torch.max(normed_rhand_verts_exp, dim=0) ### 1 x 3
        minn_normed_rhand_verts, _ = torch.min(normed_rhand_verts_exp, dim=0) ### 1 x 3
        extent_normed_rhand_verts = maxx_normed_rhand_verts - minn_normed_rhand_verts # 3
        # extent_normed_rhand_verts: 1
        # 1 x 1 x 1
        extent_normed_rhand_verts = torch.sqrt(torch.sum(extent_normed_rhand_verts ** 2, dim=-1, keepdim=True)).unsqueeze(0).unsqueeze(0) # the std value for all points 
        normed_rhand_verts = normed_rhand_verts / extent_normed_rhand_verts ### normalize rhand joints 
        #### Normalize rhand joints here ####
        # avg_rhand_verts, extent_normed_rhand_verts #
        
        #### Sample points from -1 to 1 ####
        # sample points from a to b #
        # sample points from a to b #
        # -1, 1 for the joints # --> -1, 1 #
        print(f"Calculating grid pts and indices...")
        # res = 64
        res = 16
        lspace_pts_coords = [-0.5 + i * (1. / float(res - 1)) for i in range(res)] # from 0 to 63, calculate rss for lspace j
        tot_pts_coords = []
        vox_grid_indices = []
        for i_x, x in enumerate(lspace_pts_coords):
            for i_y, y in enumerate(lspace_pts_coords):
                for i_z, z in enumerate(lspace_pts_coords):
                    tot_pts_coords.append((x, y, z))
                    vox_grid_indices.append((i_x, i_y, i_z))
        tot_pts_coords = np.array(tot_pts_coords, dtype=np.float64) # n_tot_pts x 3 --> for total pts sampled in the space #
        # xyz values ## # implicit field values
        vox_grid_indices = np.array(vox_grid_indices, dtype=np.long) # nn_tot_pts x 3 # 
        print(f"Calculated grid pts and indices...")
        
        tot_space_pts_dist_to_hand_mesh_grids = []
        tot_space_pts_dist_to_hand_mesh = [] # dist to hand mesh #
        # tot_pts_coords #
        # # then we need to get the signed distances from each point to the hand mesh ## 
        nf = rhand_verts.size(0)
        for i_fr in range(nf):
            print(f"Current frame: {i_fr}")
            cur_dist_to_hand_mesh_grids = np.zeros((res, res, res), dtype=np.float32)
            cur_fr_rhand_verts = rhand_verts[i_fr].numpy() # nn_verts x 3 --> the numpy array of the hand mesh #
            faces = self.mano_layer.th_faces.squeeze(0).numpy() # nn_faces x 3 for the hand mesh
            rhand_mesh = trimesh.Trimesh(vertices=cur_fr_rhand_verts, faces=faces,
            process=False, use_embree=True)
            rhand_mesh = self.seal(rhand_mesh) # rhand_mesh: nn_mesh_verts x 3 #
            dist_rhand_verts_to_tot_pts = np.sum(
                (np.reshape(cur_fr_rhand_verts, (cur_fr_rhand_verts.shape[0], 1, 3)) - np.reshape(tot_pts_coords, (1, tot_pts_coords.shape[0], 3))) ** 2, axis=-1
            )
            dist_rhand_verts_to_tot_pts = np.min(dist_rhand_verts_to_tot_pts, axis=0)
            dist_rhand_verts_to_tot_pts = np.sqrt(dist_rhand_verts_to_tot_pts) # nn_tot_base_pts 
            tot_pts_in_rhand_mesh = rhand_mesh.contains(tot_pts_coords) # nn_tot_base_pts 
            dist_rhand_verts_to_tot_pts[tot_pts_in_rhand_mesh] = -1. * dist_rhand_verts_to_tot_pts[tot_pts_in_rhand_mesh] # nn_tot_base_pts for the base pts in the rhand mesh ##
            tot_space_pts_dist_to_hand_mesh.append(dist_rhand_verts_to_tot_pts)
            
            # assign them to dist grids #
            cur_dist_to_hand_mesh_grids[vox_grid_indices[:, 0], vox_grid_indices[:, 1], vox_grid_indices[:, 2]] = dist_rhand_verts_to_tot_pts
            # 
            tot_space_pts_dist_to_hand_mesh_grids.append(cur_dist_to_hand_mesh_grids)
            
        tot_space_pts_dist_to_hand_mesh = np.stack(tot_space_pts_dist_to_hand_mesh, axis=0) ## nf x nn_tot_pts ### --> distances # need to 
        tot_space_pts_dist_to_hand_mesh_grids = np.stack(tot_space_pts_dist_to_hand_mesh_grids, axis=0) # # nf x res x res x res # #

        # tot_space_pts_dist_to_hand_mesh: nf x nn_tot_pts ##
        # object_mesh: vertices, faces #  # tot_space_pts_dist_to_hand_mesh: nf x nn_tot_pts x 3 ##
        # avg_rhand_verts, extent_normed_rhand_verts #
        space_pts_dist_to_obj_mesh_grids = np.zeros((res, res, res), dtype=np.float32)
        # obj_vertices for object mesh # 
        obj_vertices = object_mesh.vertices # nn_verts x 3 #
        obj_vertices = torch.from_numpy(obj_vertices).float()
        # obj verts # 
        obj_vertices = (obj_vertices - avg_rhand_verts.squeeze(0)) / extent_normed_rhand_verts.squeeze(0) # ### nn_obj_verts x 3 --> obj verts #
        object_mesh.vertices = obj_vertices.numpy()
        obj_vertices = obj_vertices.numpy()
        # nn_obj_verts x nn_tot_pts #
        print(f"Calculating space pts to obj verts distances")
        dist_space_pts_to_obj_verts = np.sum(
            (np.reshape(obj_vertices, (obj_vertices.shape[0], 1, 3)) - np.reshape(tot_pts_coords, (1, tot_pts_coords.shape[0], 3))) ** 2, axis=-1
        )
        # nn_tot_pts #
        dist_space_pts_to_obj_verts = np.min(dist_space_pts_to_obj_verts, axis=0) # nn_obj_verts --> object vertices for the verts 
        print(f"Calculated space pts to obj verts distances")
        
        print(f"Calculating obj verts containing tot_pts")
        tot_space_pts_in_obj_verts = object_mesh.contains(tot_pts_coords)
        print(f"Calculated obj verts containing tot_pts")
        # dist space pts to obj verts ## 
        dist_space_pts_to_obj_verts[tot_space_pts_in_obj_verts] = -1. * dist_space_pts_to_obj_verts[tot_space_pts_in_obj_verts]
        
        # res x res x res ---> dist from space pts to object mesh in the voxel grids #
        space_pts_dist_to_obj_mesh_grids[vox_grid_indices[:, 0], vox_grid_indices[:, 1], vox_grid_indices[:, 2]] = dist_space_pts_to_obj_verts
        
        # tot_pts_coords, tot_space_pts_dist_to_hand_mesh, tot_space_pts_dist_to_hand_mesh_grids
        # dist_space_pts_to_obj_verts, space_pts_dist_to_obj_mesh_grids
        # obj_vertices, object_mesh.faces, 
        # rhand_verts.numpy(), faces
        
        ''' Construct data for returning '''
        rt_dict = {
            # 'base_pts': base_pts, # th
            # 'base_normals': base_normals, # th
            # 'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3
            # 'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            'rhand_joints': rhand_joints, #  if not self.args.use_canon_joints else canon_rhand_joints,
            'rhand_verts': rhand_verts,
            # 'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            # 'lengths': rel_base_pts_to_rhand_joints.size(0),
            # 'obj_verts': obj_verts,
            # 'obj_normals': obj_normals,
            # 'obj_faces': obj_faces,
            'tot_pts_coords': tot_pts_coords, # tot_pts_coords: totpts x 3 --> 
            'tot_space_pts_dist_to_hand_mesh': tot_space_pts_dist_to_hand_mesh,
            'tot_space_pts_dist_to_hand_mesh_grids': tot_space_pts_dist_to_hand_mesh_grids, # grids 
            'dist_space_pts_to_obj_verts': dist_space_pts_to_obj_verts, 
            'space_pts_dist_to_obj_mesh_grids': space_pts_dist_to_obj_mesh_grids,
            'obj_vertices': obj_vertices,
            'obj_faces': object_mesh.faces,
            'rhand_verts': rhand_verts.numpy(),
            'faces': faces,
            ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, # not for the ambinet space valuess s#
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            ### === per frame avg disp along normals, vt normals === ###
            # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
            # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
            # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
            # 'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            # 'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            # 'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            # 'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            # 'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            # 'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            ## sampled; learn the 
        }
        
        np.save(grid_data_sv_fn, rt_dict) # grid_data_sv_fn --> grid data sv fn # conv 
        print(f"grid data dict saved to {grid_data_sv_fn}") ### save grid data here ##
        
        
        # try:
        #     # rt_dict['per_frame_avg_joints_rel'] =  # realtive 
        #     rt_dict.update(stats_dict)
        # except:
        #     pass
        ''' Construct data for returning '''
        
        return rt_dict
        

    def __len__(self):
        return self.len


