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

import random
import trimesh
from scipy.spatial.transform import Rotation as R

import pickle as pkl

from utils.anchor_utils import masking_load_driver, anchor_load_driver, recover_anchor_batch



# GRAB #
class GRAB_Dataset_V19(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None): # 
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        
        split = args.single_seq_path.split("/")[-2].split("_")[0]
        self.split = split
        print(f"split: {self.split}")
        
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.args = args
        
        self.denoising_stra = args.denoising_stra ## denoising_stra!
        
        self.seq_path = args.single_seq_path ## single seq path ##
        
        self.inst_normalization = args.inst_normalization
        
        
        
        self.start_idx = args.start_idx # clip starting idxes #
        # self.start_idx = 0

        # obj_mesh_path = "data/grab/object_meshes"
        obj_mesh_path = "data/grab/object_meshes"
        id2objmesh = []
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        
        
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
        
        
        ## predicted infos fn ##
        self.data_folder = data_folder
        # self.subj_data_folder = data_folder
        self.subj_data_folder = data_folder + "_wsubj"
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        # manopth/mano/models
        self.mano_path = "manopth/mano/models" ### mano_path
        ## mano paths ##
        self.aug = True
        self.use_anchors = False
        # self.args = args
        
        self.use_anchors = args.use_anchors
        
        
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        self.dist_threshold = 0.005
        self.dist_threshold = 0.01
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
        print(f"nn_base_pts: {self.nn_base_pts}")
        
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
        
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=24,
            use_pca=True,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )
        
        
        files_clean = [self.seq_path]
        
        if self.load_meta:
          
            for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
            # for 
                # if split != 'train' and split != 'val' and i_f >= 100:
                #     break
                # if split == 'train':
                    # print(f"loading {i_f} / {len(files_clean)}")
                print(f"loading {i_f} / {len(files_clean)}")
                base_nm_f = os.path.basename(f)
                base_name_f = base_nm_f.split(".")[0]
                cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
                cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
                cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
                cur_clip_len = cur_clip_meta_data['clip_len']
                # clip_len = (cur_clip_len - window_size) // step_size + 1
                # clip_len = cur_clip_len
                
                self.clips.append(self.load_clip_data(i_f, f=f)) ## add current clip ##
                # self.clips.append((self.len, self.len+clip_len,  f
                #     ))
                clip_len = self.clips[i_f][3][3].shape[0]
                self.len += clip_len # len clip len
                self.len = 81
                
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
                self.len += clip_len # len clip len
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx, f=None):
        if f is None:
          cur_clip = self.clips[clip_idx]
          if len(cur_clip) > 3:
              return cur_clip
          f = cur_clip[2]
        clip_clean = np.load(f)
        # pert_folder_nm = self.split + '_pert'
        pert_folder_nm = self.split
        # if not self.use_pert:
        #     pert_folder_nm = self.split
        # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        ##### load subj params ##### # 
        pure_file_name = f.split("/")[-1].split(".")[0]
        pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                
        subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        rhand_transl = subj_params["rhand_transl"]
        rhand_betas = subj_params["rhand_betas"]
        rhand_pose = clip_clean['f2'] ## rhand pose ##
        
        object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
        object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()


        loaded_clip = (
            0, rhand_transl.shape[0], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip
        
        return loaded_clip
        
        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False) # obj mesh obj verts 
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]


    def __getitem__(self, index):

        i_c = 0
        c = self.clips[i_c]
        
        object_id = c[3][-1]
        # object_name = self.id2objmeshname[object_id]
        
        #  self.start_idx = args.start_idx
        # start_idx = 0  # 
        start_idx = self.start_idx

        # start_idx = (index - c[0]) * self.step_size
        print(f"start_idx: {start_idx}, window_size: {self.window_size}")
        data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        # object_global_orient = data['f5'] ### get object global orientations ###
        # object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        object_global_orient = c[3][-3] # num_frames x 3 
        object_transl = c[3][-2] # num_frames x 3
        
        print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        # object_global_orient, object_transl #
        object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        object_transl = object_transl[start_idx: start_idx + self.window_size]
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        object_trcansl_th = torch.from_numpy(object_transl).float()
        
        # pert_subj_params = c[4]
        
        st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx
        
        ### pts gt ###
        ## rhnad pose, rhand pose gt ##
        ## glboal orientation and hand pose #
        rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
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
        
        ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
        ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
        #### ==== get random augmented pose, rot, transl ==== ####
        # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        # aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
        # aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3
        # # noise scale #
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3 # scale 1 for the standard scale
        aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.4 ### scale 3 for the standard scale ###
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.5
        # cur_t = self.uinform_sample_t()
        # # aug_trans, aug_rot, aug_pose #
        # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
        # ### === get and save noise vectors === ###
        # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
        aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        if self.args.pert_type == "uniform":
            aug_pose_var = (torch.rand_like(rhand_pose_var) - 0.5) * aug_pose
            aug_global_orient_var = (torch.rand_like(rhand_global_orient_var) - 0.5) * aug_rot
        elif self.args.pert_type == "beta":
            dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
            print(f"here!")
            aug_pose_var = dist_beta.sample(rhand_pose_var.size()).squeeze(-1) * aug_pose
            aug_global_orient_var = dist_beta.sample(rhand_global_orient_var.size()).squeeze(-1) * aug_rot
            print(f"aug_pose_var: {aug_pose_var.size()}, aug_global_orient_var: {aug_global_orient_var.size()}")
            
        # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # ### creat augmneted orientations, pose, and transl ###
        rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        
        # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_betas_var, rhand_beta_var
        rhand_verts, rhand_joints = self.mano_layer(
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
        pert_rhand_verts, pert_rhand_joints = self.mano_layer(
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
        
        canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
            torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
        )
        canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
        canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
        if self.use_anchors:
            # canon_pert_rhand_anchors = canon_pert_rhand_verts[:, self.hand_palm_vertex_mask]
            canon_pert_rhand_anchors = recover_anchor_batch(canon_pert_rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
        
        # canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
        #     torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
        #     pert_rhand_betas_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
        # )
        # canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
        # canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
        # ### Relative positions from base points to rhand joints ###
        object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)

        if self.args.scale_obj > 1:
            object_pc = object_pc * self.args.scale_obj
        object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
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
        
        
        # if we just set a parameter `use_arti_obj`? #
        
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
        
        ''' normalization strategy xxx --- data scaling '''
        # base_pts = base_pts * 5.
        # rhand_joints = rhand_joints * 5.
        ''' Normlization stratey xxx --- data scaling '''
        
        
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
        
        if self.denoising_stra == "motion_to_rep": # motion_to_rep #
            pert_rhand_joints = (pert_rhand_joints - self.avg_jts) / self.std_jts
        
        

        ''' Obj data '''
        obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id) # obj_verts, normals #
        obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
        obj_normals = torch.from_numpy(obj_normals).float() # 
        obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''


        caption = "apple"
        # pose_one_hots, word_embeddings #
        
        # object_global_orient_th, object_transl_th #
        object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        
        # pert_rhand_anchors, rhand_anchors
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            # 'rhand_joints': rhand_joints,
            'gt_rhand_joints': rhand_joints, ## rhand joints ###
            'rhand_joints': pert_rhand_joints if not self.args.use_canon_joints else canon_pert_rhand_joints,
            'rhand_verts': rhand_verts,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'object_id': object_id, # int value
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'object_global_orient': object_global_orient_th,
            'object_transl': object_transl_th,
            'st_idx': st_idx,
            'ed_idx': ed_idx,
            'pert_verts': pert_rhand_verts,
            'verts': rhand_verts,
            'obj_verts': obj_verts,
            'obj_normals': obj_normals,
            'obj_faces': obj_faces, # nnfaces x 3 #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
        }
        
        if self.use_anchors: ## update rhand anchors here ##
            rt_dict.update(
                {
                    'rhand_anchors': rhand_anchors,
                    'pert_rhand_anchors': pert_rhand_anchors,
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] = 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict

    def __len__(self):
        cur_len = self.len // self.step_size
        if cur_len * self.step_size < self.len:
          cur_len += 1
        cur_len = 1
        return cur_len


# GRAB # 
class GRAB_Dataset_V19_From_Evaluated_Info(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        
        
        
        split = args.single_seq_path.split("/")[-2].split("_")[0]
        self.split = split
        print(f"split: {self.split}")
        
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.args = args
        
        self.denoising_stra = args.denoising_stra ## denoising_stra!
        
        self.seq_path = args.single_seq_path ## single seq path ##
        
        self.inst_normalization = args.inst_normalization
         
        
        ## the predicted_info_fn
        predicted_info_fn = args.predicted_info_fn
        # load data from predicted information #
        data = np.load(predicted_info_fn, allow_pickle=True).item()
        self.wpredverts = False
        if 'optimized_out_hand_joints_ne' in data: ### joints_ne for joints ##
            print("Loading from optimized_out_hand_joints_ne!!!")
            outputs = data['optimized_out_hand_joints_ne'] # outputs of the o
            self.wpredverts = False
        elif 'hand_joints' in data:
            print(f"Loading from hand_joints!!!!")
            outputs = data['hand_joints'] # ws x nn_joints x 3 #
            predicted_hand_verts = data['hand_verts'] 
            # 
            self.wpredverts = True
            if len(args.predicted_info_fn_jts_only) > 0:
                print(f"Loading from predicted_info_fn_jts_only!!!!!")
                cur_predicted_info_fn_jts_only = np.load(args.predicted_info_fn_jts_only, allow_pickle=True).item()
                tot_obj_rot = cur_predicted_info_fn_jts_only['tot_obj_rot'][0] # ws x 3 x 3 #
                tot_obj_transl = cur_predicted_info_fn_jts_only['tot_obj_transl'][0] # ws x 3 #
                ws = tot_obj_transl.shape[0]
                outputs = np.matmul( # the outputs: ws x nn_joints x 3 #
                    outputs - tot_obj_transl.reshape(ws, 1, 3), np.transpose(tot_obj_rot, (0, 2, 1)) #  
                )
                predicted_hand_verts = np.matmul( # the outputs: ws x nn_joints x 3 #
                    predicted_hand_verts - tot_obj_transl.reshape(ws, 1, 3), np.transpose(tot_obj_rot, (0, 2, 1)) #  
                )
            self.predicted_hand_verts = predicted_hand_verts
            self.predicted_hand_verts = torch.from_numpy(self.predicted_hand_verts).float()
        else:
            outputs = data['outputs']
        self.predicted_hand_joints = outputs # nf x nnjoints x 3 #
        self.predicted_hand_joints = torch.from_numpy(self.predicted_hand_joints).float()
        # obj_verts = data['obj_verts']
        # obj_faces = data['obj_faces']
        # tot_base_pts = data["tot_base_pts"][0]
        # single_obj_normals = data['single_obj_normals']
        # self.obj_ver
        
        print(f"predicted_hand_joints: {self.predicted_hand_joints.shape}")
        
        self.start_idx = args.start_idx
        
        
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
        self.subj_data_folder = data_folder + "_wsubj"
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "manopth/mano/models" ### mano_path
        self.aug = True
        self.use_anchors = False

        
        obj_mesh_path = "data/grab/object_meshes"
        id2objmesh = []
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        
        
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
        
        
        # self.dist_stra = args.dist_stra
        
        self.load_meta = True
        
        self.dist_threshold = 0.005
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
        print(f"nn_base_pts: {self.nn_base_pts}")
        
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

        # files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        # #### filter files_clean here ####
        # files_clean = [cur_f for cur_f in files_clean if ("meta_data" not in cur_f and "uvs_info" not in cur_f)]
        
        files_clean = [self.seq_path]
        
        if self.load_meta:
          
            for i_f, f in enumerate(files_clean): ### train, val, test clip, clip_len ###
            # for 
                # if split != 'train' and split != 'val' and i_f >= 100:
                #     break
                # if split == 'train':
                    # print(f"loading {i_f} / {len(files_clean)}")
                print(f"loading {i_f} / {len(files_clean)}")
                base_nm_f = os.path.basename(f)
                base_name_f = base_nm_f.split(".")[0]
                cur_clip_meta_data_sv_fn = f"{base_name_f}_meta_data.npy"
                cur_clip_meta_data_sv_fn = os.path.join(data_folder, split, cur_clip_meta_data_sv_fn)
                cur_clip_meta_data = np.load(cur_clip_meta_data_sv_fn, allow_pickle=True).item()
                cur_clip_len = cur_clip_meta_data['clip_len']
                # clip_len = (cur_clip_len - window_size) // step_size + 1
                # clip_len = cur_clip_len
                
                self.clips.append(self.load_clip_data(i_f, f=f)) ## add current clip ##
                # self.clips.append((self.len, self.len+clip_len,  f
                #     ))
                clip_len = self.clips[i_f][3][3].shape[0]
                self.len += clip_len # len clip len
                self.len = 81
                
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
                #     ))
                self.len += clip_len # len clip len
        self.clips.sort(key=lambda x: x[0])
    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx, f=None):
        if f is None:
          cur_clip = self.clips[clip_idx]
          if len(cur_clip) > 3:
              return cur_clip
          f = cur_clip[2]
        clip_clean = np.load(f)
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
        rhand_pose = clip_clean['f2'] ## rhand pose ##
        
        object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
        object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        
        # maxx_clip_len = 
        loaded_clip = (
            0, rhand_transl.shape[0], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip
        
        return loaded_clip
        
        # self.clips.append((self.len, self.len+clip_len, clip_pert,
        #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #     # subj_corr_data, pert_subj_corr_data
        #     ))
        
    def get_idx_to_mesh_data(self, obj_id):
        if obj_id not in self.id2meshdata:
            obj_nm = self.id2objmesh[obj_id]
            obj_mesh = trimesh.load(obj_nm, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)
            self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        return self.id2meshdata[obj_id]

    #### enforce correct contacts #### ### the sequence in the clip is what we want here #
    def __getitem__(self, index):
        ## GRAB single frame ##
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        i_c = 0
        # if self.load_meta:
        #     # self.load_clip_data(i_c)
        c = self.clips[i_c]
        # c = self.load_clip_data(i_c)

        object_id = c[3][-1]
        # object_name = self.id2objmeshname[object_id]
        
        # start_idx = index * self.step_size
        # if start_idx + self.window_size > self.len:
        #     start_idx = self.len - self.window_size
        
        start_idx = self.start_idx
            
        # TODO: add random noise settings for noisy input #
        
        # start_idx = (index - c[0]) * self.step_size
        
        data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        # object_global_orient = data['f5'] ### get object global orientations ###
        # object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        object_global_orient = c[3][-3] # num_frames x 3 
        object_transl = c[3][-2] # num_frames x 3
        
        
        # object_global_orient, object_transl #
        object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        object_transl = object_transl[start_idx: start_idx + self.window_size]
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        object_trcansl_th = torch.from_numpy(object_transl).float()
        
        # pert_subj_params = c[4]
        
        st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx
        
        ### pts gt ###
        ## rhnad pose, rhand pose gt ##
        ## glboal orientation and hand pose #
        rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
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
        
        ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
        ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
        #### ==== get random augmented pose, rot, transl ==== ####
        # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
        aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
        aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3
        # cur_t = self.uinform_sample_t()
        # # aug_trans, aug_rot, aug_pose #
        # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
        # ### === get and save noise vectors === ###
        # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
        aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # ### creat augmneted orientations, pose, and transl ###
        rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        
        # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_betas_var, rhand_beta_var
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        )
        ### rhand_joints: for joints ###
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        
        
        # ### pert_rhand_global_orient_var, pert_rhand_pose_var, pert_rhand_transl_var ###
        if self.use_rnd_aug_hand: ## rnd aug pose var, transl var #
            # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
            pert_rhand_global_orient_var = rnd_aug_global_orient_var.clone()
            pert_rhand_pose_var = rnd_aug_pose_var.clone()
            pert_rhand_transl_var = rnd_aug_transl_var.clone()
            # pert_rhand_global_orient_mtx = utils.batched_get_orientation_matrices(pert_rhand_global_orient_var.numpy())
        
        # # pert_rhand_betas_var
        # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
        # pert_rhand_joints --> for rhand joints in the camera frmae ###
        pert_rhand_verts, pert_rhand_joints = self.mano_layer(
            torch.cat([pert_rhand_global_orient_var, pert_rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), pert_rhand_transl_var
        )
        pert_rhand_verts = pert_rhand_verts * 0.001 # verts 
        pert_rhand_joints = pert_rhand_joints * 0.001 # joints


        if self.wpredverts:
            # print(f"ori_pert_rhand_verts: {pert_rhand_verts.s}")
            pert_rhand_joints = self.predicted_hand_joints
            rhand_joints = self.predicted_hand_joints

            rhand_verts = self.predicted_hand_verts
            pert_rhand_verts =  self.predicted_hand_verts
            
            pert_rhand_joints = torch.matmul(
                pert_rhand_joints, object_global_orient_mtx_th
            ) + object_trcansl_th.unsqueeze(1)

            rhand_joints = torch.matmul(
                rhand_joints, object_global_orient_mtx_th
            ) + object_trcansl_th.unsqueeze(1)

            rhand_verts = torch.matmul(
                rhand_verts, object_global_orient_mtx_th
            ) + object_trcansl_th.unsqueeze(1)

            pert_rhand_verts = torch.matmul(
                pert_rhand_verts, object_global_orient_mtx_th
            ) + object_trcansl_th.unsqueeze(1)
        
        
       
        
        # use_canon_joints
        
        canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
            torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
        )
        canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
        canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
        
        # ### Relative positions from base points to rhand joints ###
        object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
        # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        
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
        base_pts_transl = object_trcansl_th[0] # 3
        
        # if self.dir_stra == "rot_angles": ## rot angles ##
        #     normals_rot_mtx = utils.batched_get_rot_mtx_fr_vecs_v2(base_normals)
        
        # if self.canon_obj:
            ## reverse transform base points ###
            ## canonicalize base points and base normals ###
        base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
            ) # .transpose(0, 1)
        base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
            ) # .transpose(0, 1)
        
        
        rhand_joints = torch.matmul(
            rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        pert_rhand_joints = torch.matmul(
            pert_rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        
        if not self.wpredverts:
            pert_rhand_joints = self.predicted_hand_joints
        

        # nf x nnj x nnb x 3 # 
        rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
        dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
        
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
        # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
        signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
            base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
        )
        # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
        # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
        # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
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
            
            
            # # nf x nnj x nnb x 3 # 
            rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
            # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
            dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
            
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        if self.denoising_stra == "motion_to_rep": # motion_to_rep #
            pert_rhand_joints = (pert_rhand_joints - self.avg_jts) / self.std_jts
        
        
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        # 
        # rt_pert_rhand_verts =  pert_rhand_verts
        # rt_rhand_verts = rhand_verts
        # rt_pert_rhand_joints = pert_rhand_joints
        
        # rt_rhand_joints = rhand_joints ## rhand_joints ##
        # # rt_rhand_joints = pert_rhand_joints
        
        
        # # rt_rhand_joints: nf x nnjts x 3 # ### pertrhandjoints
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1s
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
        
        ''' Obj data '''
        obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id)
        obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
        obj_normals = torch.from_numpy(obj_normals).float() # 
        obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        
        # rt_rhand_joints: nf x nnjts x 3 #
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
            
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        caption = "apple"
        # pose_one_hots, word_embeddings #
        
        # object_global_orient_th, object_transl_th #
        object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            # 'rhand_joints': rhand_joints,
            'gt_rhand_joints': rhand_joints, ## rhand joints ###
            'rhand_joints': pert_rhand_joints if not self.args.use_canon_joints else canon_pert_rhand_joints,
            'rhand_verts': rhand_verts,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'object_id': object_id, # int value
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'object_global_orient': object_global_orient_th,
            'object_transl': object_transl_th,
            # 'st_idx': st_idx,
            # 'ed_idx': ed_idx,
            'st_idx': start_idx,
            'ed_idx': start_idx + self.window_size,
            'pert_verts': pert_rhand_verts,
            'verts': rhand_verts,
            'obj_verts': obj_verts,
            'obj_normals': obj_normals, # normals? 
            'obj_faces': obj_faces, # nnfaces x 3 #
             'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
        }
        try:
            # rt_dict['per_frame_avg_joints_rel'] = 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict


    def __len__(self):
        cur_len = self.len // self.step_size
        if cur_len * self.step_size < self.len:
          cur_len += 1
        cur_len = 1
        return cur_len




# Utils #
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



# for hoi4d dataset #
class GRAB_Dataset_V19_HOI4D(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset #### ## GRAB dataset
        self.clips = []
        self.len = 0
        
        # self.single_seq_path = args.single_seq_path
        # self.data = np.load(self.single_seq_path, allow_pickle=True) # .item()
        
        
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        
        self.cad_model_fn = args.cad_model_fn
        
        self.start_idx = args.start_idx
        
        self.hoi4d_cad_model_root = args.hoi4d_cad_model_root
        # split = args.single_seq_path.split("/")[-2].split("_")[0]
        # self.split = split
        # print(f"split: {self.split}")
        
        # self.model_type = 'v1_wsubj_wjointsv25'
        # self.debug = False
        # # self.use_ambient_base_pts = args.use_ambient_base_pts
        # # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        # self.num_sche_steps = 100
        # self.w_vectorizer = w_vectorizer
        # self.use_pert = True
        # self.use_rnd_aug_hand = True
        
        self.args = args
        self.use_anchors = self.args.use_anchors
        
        self.denoising_stra = args.denoising_stra ## denoising_stra!
        
        # self.data_folder = data_folder
        # self.subj_data_folder = '/data1/xueyi/GRAB_processed_wsubj'
        # # self.subj_corr_data_folder = args.subj_corr_data_folder
        # self.mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
        self.mano_path = "manopth/mano/models" 
        # self.aug = True
        # self.use_anchors = False
        # # self.args = args
        
        predicted_info_fn = args.predicted_info_fn
        # load data from predicted information #
        if len(predicted_info_fn) > 0:
            print(f"Loading preidcted info from {predicted_info_fn}")
            data = np.load(predicted_info_fn, allow_pickle=True).item()
            # /data1/xueyi/mdm/eval_save/optimized_infos_sv_dict_seq_scissors_optimized_aug.npy
            # data_opt_info_fn = "/data1/xueyi/mdm/eval_save/optimized_infos_sv_dict_seq_scissors_optimized_aug.npy" # scissors aug #
            # data_opt = np.load(data_opt_info_fn, allow_pickle=True).item()
            outputs = data['outputs']
            # nf x nnjoints x 3 #
            self.predicted_hand_joints = outputs # nf x nnjoints x 3 #
            self.predicted_hand_joints = torch.from_numpy(self.predicted_hand_joints).float()
            
            if 'rhand_trans' in data:
                # outputs = data['outputs']
                self.predicted_hand_trans = data['rhand_trans'] # nframes x 3 
                self.predicted_hand_rot = data['rhand_rot'] # nframes x 3 
                self.predicted_hand_theta = data['rhand_theta']
                self.predicted_hand_beta = data['rhand_beta']
                self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
                self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
                self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
                self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
                
                # self.predicted_hand_trans_opt = data_opt['rhand_trans'] # nframes x 3 
                # self.predicted_hand_rot_opt = data_opt['rhand_rot'] # nframes x 3 
                # self.predicted_hand_theta_opt = data_opt['rhand_theta']
                # self.predicted_hand_beta_opt = data_opt['rhand_beta']
                # self.predicted_hand_trans_opt = torch.from_numpy(self.predicted_hand_trans_opt).float() # nframes x 3 
                # self.predicted_hand_rot_opt = torch.from_numpy(self.predicted_hand_rot_opt).float() # nframes x 3 
                # self.predicted_hand_theta_opt = torch.from_numpy(self.predicted_hand_theta_opt).float() # nframes x 24 
                # self.predicted_hand_beta_opt = torch.from_numpy(self.predicted_hand_beta_opt).float() # 10,
                
                # self.predicted_hand_trans[9:] = self.predicted_hand_trans_opt[9:]
                # self.predicted_hand_rot[9:] = self.predicted_hand_rot_opt[9:]
                # self.predicted_hand_theta[ 9:] = self.predicted_hand_theta_opt[ 9:]
                # # self.predicted_hand_trans[:, 9:] = self.predicted_hand_trans_opt[:, 9:]
                
            else:
                self.predicted_hand_trans = None
                self.predicted_hand_rot = None
                self.predicted_hand_theta = None
                self.predicted_hand_beta = None
            
        else:
            self.predicted_hand_joints = None
        
        
        self.corr_fn = args.corr_fn # corr_fn 
        if len(self.corr_fn) > 0:
            self.raw_corr_data = np.load(self.corr_fn, allow_pickle=True)
        # self.dist_stra = args.dist_stra
        
        # self.load_meta = True
        
        self.dist_threshold = 0.005
        self.dist_threshold = 0.01
        self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
        print(f"nn_base_pts: {self.nn_base_pts}")
        
        
        self.theta_dim = args.theta_dim
        use_pca = True if self.theta_dim < 45 else False
        
        self.mano_layer = ManoLayer(
            flat_hand_mean=True,
            side='right',
            mano_root=self.mano_path, # mano_root #
            ncomps=self.theta_dim,
            use_pca=use_pca,
            root_rot_mode='axisang',
            joint_rot_mode='axisang'
        )


    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx, f=None):
        if f is None:
          cur_clip = self.clips[clip_idx]
          if len(cur_clip) > 3:
              return cur_clip
          f = cur_clip[2]
        clip_clean = np.load(f)
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
        rhand_pose = clip_clean['f2'] ## rhand pose ##
        
        object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
        object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        

        # maxx_clip_len = 
        loaded_clip = (
            0, rhand_transl.shape[0], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip
        
        return loaded_clip
        
        # self.clips.append((self.len, self.len+clip_len, clip_pert,
        #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #     # subj_corr_data, pert_subj_corr_data
        #     ))
        
    def get_idx_to_mesh_data(self):
        # if obj_id not in self.id2meshdata:
        #     obj_nm = self.id2objmesh[obj_id]
        #     obj_mesh = trimesh.load(obj_nm, process=False)
        #     obj_verts = np.array(obj_mesh.vertices)
        #     obj_vertex_normals = np.array(obj_mesh.vertex_normals)
        #     obj_faces = np.array(obj_mesh.faces)
        #     self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
        # cad_model_fn = self.cad_model_fn
        cad_model_fn = self.cad_model_fn
        obj_mesh = trimesh.load(cad_model_fn, process=False)
        obj_verts = np.array(obj_mesh.vertices)
        obj_vertex_normals = np.array(obj_mesh.vertex_normals)
        obj_faces = np.array(obj_mesh.faces)
        mesh_data = (obj_verts, obj_vertex_normals, obj_faces)
        return mesh_data
    
    def get_ari_obj_fr_x(self, i_frame_st, i_frame_ed):
        
        # raw_corr_data
        tot_obj_verts = []
        tot_obj_faces = []
        tot_obj_normals = []
        tot_obj_glb_rot = []
        tot_obj_glb_trans = []
        tot_hand_beta = []
        tot_hand_theta = []
        tot_hand_transl = []
        tot_hand_joints = []

        tot_full_obj_verts = []
        tot_full_obj_faces = []
        
        # single_seq_path #
        single_seq_folder = "/".join(self.args.single_seq_path.split("/")[:-1]) # single_seq_path
        single_seq_meta_data_fn = os.path.join(single_seq_folder, "meta_data.npy")
        single_seq_meta_data = np.load(single_seq_meta_data_fn, allow_pickle=True).item()
        series_tag = single_seq_meta_data["case_flag"]
        series_obj_category = series_tag.split("/")[2]
        series_obj_category = int(series_obj_category[1:])
        series_obj_inst_idx = series_tag.split("/")[3] # N17
        series_obj_inst_idx = int(series_obj_inst_idx[1:]) # idx of the instance
        cat_idx_to_obj_nm_mapping = [ # Bottle-3  Bowl-3  Chair  Kettle-3  Knife  Mug-2  ToyCar-1 # 
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
        ]
        cat_nm = cat_idx_to_obj_nm_mapping[series_obj_category]

        case_merged_data_fn = os.path.join(single_seq_folder, "merged_data.npy")
        case_merged_data = np.load(case_merged_data_fn, allow_pickle=True)

      
        
        for i_frame in range(i_frame_st, i_frame_ed):
            cur_obj_rot = self.raw_corr_data[i_frame]['obj_rot']
            cur_obj_trans = self.raw_corr_data[i_frame]['obj_trans']

            
            cur_arti_cat_nm = cat_nm
            cur_arti_inst_nm = int(series_obj_inst_idx) # ## series obj inst idxes ### 

            if not self.args.use_arti_obj:
                cad_model_fn = [ # get cad models 
                    # f"/share/datasets/HOI4D_CAD_Model_for_release/rigid/{cur_arti_cat_nm}/%03d.obj" % cur_arti_inst_nm, 
                    # f"data/hoi4d/CAD_Model/rigid/{cur_arti_cat_nm}/%03d.obj" % cur_arti_inst_nm, 
                    os.path.join(self.hoi4d_cad_model_root, f"rigid/{cur_arti_cat_nm}/%03d.obj" % cur_arti_inst_nm)
                ]
                if not isinstance(cur_obj_rot, list):
                    cur_obj_rot = [cur_obj_rot]
                    cur_obj_trans = [cur_obj_trans]
                self.cad_model_fn = cad_model_fn[0]
            else:
                if cat_nm in ["Scissors", "Laptop"]:
                    if self.args.use_reverse:
                        cad_model_fn = [ # get cad models 
                            f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
                            f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
                        ]
                    else:
                        cad_model_fn = [ # get cad models 
                            f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm, 
                            f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm 
                        ]
                        # cad_model_fn = [
                        #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
                        #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
                        # ]
                else:
                    cad_model_fn = [ # get cad models 
                        f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
                        f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
                    ]


            # object mesh ours arti ###
            full_cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
            # nn_verts x 3 
            # nn_faces x 3 
            full_cur_obj_verts, full_cur_obj_faces = full_cur_obj_mesh.vertices, full_cur_obj_mesh.faces

            # tot_full_obj_verts, tot_full_obj_faces
            tot_full_obj_verts.append(full_cur_obj_verts)
            tot_full_obj_faces.append(full_cur_obj_faces)

            
            ## se
            if self.args.select_part_idx != -1:
                cad_model_fn = [cad_model_fn[self.args.select_part_idx]]
                
                cur_obj_glb_rot = cur_obj_rot[self.args.select_part_idx].reshape(-1) # glb rot #
                cur_obj_glb_trans = cur_obj_trans[self.args.select_part_idx]

                cur_obj_rot = [cur_obj_rot[self.args.select_part_idx]]
                cur_obj_trans = [cur_obj_trans[self.args.select_part_idx]]

                cur_frame_data = case_merged_data[i_frame]
                cur_theta = cur_frame_data['theta'].squeeze(0).numpy() # 24 current theta 
                cur_beta = cur_frame_data['beta'].squeeze(0).numpy() ## beta 
                cur_rhand_transl = cur_frame_data['trans'].squeeze(0).numpy() # ## rhand trans for the cur_frame_data
                cur_rhand_joints = cur_frame_data['joints'].reshape(-1)


                tot_obj_glb_rot.append(cur_obj_glb_rot)
                # tot_obj_glb_trans.append(cur_obj_glb_trans)

            cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
            # nn_verts x 3 
            # nn_faces x 3 
            cur_obj_verts, cur_obj_faces = cur_obj_mesh.vertices, cur_obj_mesh.faces
            obj_center = np.mean(cur_obj_verts, axis=0, keepdims=True)

            obj_center = np.zeros_like(obj_center)

            cur_obj_verts = cur_obj_verts - obj_center

            if self.args.select_part_idx != -1:
                cur_rhand_transl = cur_rhand_transl - obj_center[0]
                cur_rhand_joints = cur_rhand_joints.reshape(21, 3) - obj_center
                cur_obj_glb_trans = cur_obj_glb_trans - obj_center[0]
                tot_hand_beta.append(cur_beta)
                tot_hand_theta.append(cur_theta)
                tot_hand_transl.append(cur_rhand_transl)
                tot_hand_joints.append(cur_rhand_joints)
                tot_obj_glb_trans.append(cur_obj_glb_trans)

            cur_obj_normals = cur_obj_mesh.vertex_normals
            tot_obj_normals.append(cur_obj_normals)

            tot_obj_verts.append(cur_obj_verts)
            tot_obj_faces.append(cur_obj_faces)
        
        tot_obj_verts = np.stack(tot_obj_verts, axis=0)
        tot_obj_faces = np.stack(tot_obj_faces, axis=0)
        tot_obj_normals  = np.stack(tot_obj_normals, axis=0)

        # # tot_full_obj_verts, tot_full_obj_faces
        
        if len(tot_obj_glb_rot) > 0:
            tot_obj_glb_rot = np.stack(tot_obj_glb_rot, axis=0)
            tot_obj_glb_trans = np.stack(tot_obj_glb_trans, axis=0)
            tot_hand_beta = np.stack(tot_hand_beta, axis=0)
            tot_hand_theta = np.stack(tot_hand_theta, axis=0)
            tot_hand_transl = np.stack(tot_hand_transl, axis=0)
            tot_hand_joints = np.stack(tot_hand_joints, axis=0)
            tot_full_obj_verts = np.stack(tot_full_obj_verts, axis=0)
            tot_full_obj_faces = np.stack(tot_full_obj_faces, axis=0)

            print(f"tot_hand_joints: {tot_hand_joints.shape}")
            
        return tot_obj_verts, tot_obj_faces, tot_obj_normals, tot_obj_glb_rot, tot_obj_glb_trans, tot_hand_beta, tot_hand_theta, tot_hand_transl, tot_hand_joints, tot_full_obj_verts, tot_full_obj_faces


    #### enforce correct contacts #### ### the sequence in the clip is what we want here #
    def __getitem__(self, index):
        ## GRAB single frame ##
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        # i_c = 0
        
        # start_idx = 0
        
        start_idx = self.start_idx
        if len(self.corr_fn) > 0:
            cur_obj_verts, cur_obj_faces, cur_obj_normals, cur_obj_glb_rot, cur_obj_glb_trans, tot_hand_beta, tot_hand_theta, tot_hand_transl, tot_hand_joints, tot_full_obj_verts, tot_full_obj_faces = self.get_ari_obj_fr_x(start_idx, start_idx + self.window_size) # nn_obj_verts x 3; nn_obj_faces x 3 #
            print(f"corr_fn: {self.corr_fn}, obj_verts: {cur_obj_verts.shape}, cur_obj_faces: {cur_obj_faces.shape}")
            
        # if self.load_meta:
        #     # self.load_clip_data(i_c)
        # c = self.clips[i_c]
        # c = self.load_clip_data(i_c)

        # object_id = c[3][-1]
        # object_name = self.id2objmeshname[object_id]
        
        # start_idx = index * self.step_size
        # if start_idx + self.window_size > self.len:
        #     start_idx = self.len - self.window_size
            
        # TODO: add random noise settings for noisy input #
        
        # start_idx = (index - c[0]) * self.step_size
        
        # num_points = self.data['f1'].shape[0] // 3
        # rhand_pc = self.data[index]['f0'].reshape(778, 3) # method to get such data cannot generalize well...
        # object_pc = self.data[index]['f1'].reshape(-1, 3)
        # object_vn = self.data[index]['f2'].reshape(-1, 3)
        # object_corr_mask = self.data[index]['f5'].reshape(-1)
        # object_corr_pts = self.data[index]['f7'].reshape(-1, 3)
        # object_corr_dist = self.data[index]['f6'].reshape(-1)
        
        
        if self.args.select_part_idx != -1:
            # tot_obj_verts_th = torch.from_numpy(cur_obj_verts).float()
            # tot_obj_faces_th = torch.from_numpy(cur_obj_faces).long()
            # tot_obj_normals_th = torch.from_numpy(tot_obj_normals).float()

            object_pc = cur_obj_verts.copy()
            object_vn = cur_obj_normals.copy()
            # object_pc = cur_obj_verts.copy()
        else:
            object_pc = self.data['f1'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
            object_vn = self.data['f2'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        
        if self.args.select_part_idx != -1:
            rhand_joints = tot_hand_joints
            rhand_transl = tot_hand_transl
            rhand_beta = tot_hand_beta
            rhand_theta = tot_hand_theta
            print(f"rhand_transl: {rhand_transl.shape}, rhand_beta: {rhand_beta.shape},rhand_beta: {rhand_beta.shape}, rhand_theta: {rhand_theta.shape} ")
        else:
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
        
        
        # dataset ours single seq #
        
        # # rhand_global_orient = self.data[index]['f1'].reshape(-1).astype(np.float32)
        # rhand_pose = rhand_theta
        # # rhand_transl = self.subj_params['rhand_transl'][index].reshape(-1).astype(np.float32)
        # rhand_betas = rhand_beta
        
        
        print(f"rhand_global_orient_var: {rhand_global_orient_var.size()}, rhand_pose_var: {rhand_pose_var.size()}, rhand_beta_var: {rhand_beta_var.size()}")
        ####### Get rhand_verts and rhand_joint #######
        rhand_verts, rhand_joints = self.mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.view(-1, 10), rhand_transl_var
        )
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        ####### Get rhand_verts and rhand_joint #######
        
        
        if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 #
            # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
            rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
            pert_rhand_anchors = rhand_anchors
            # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        
        # rhand_transl = rhand_transl - obj_center[0]
        
        
        pert_rhand_joints = rhand_joints
        pert_rhand_verts = rhand_verts
        
        
        # data = c[2][start_idx:start_idx+self.window_size]
        # # # object_global_orient = self.data[index]['f5']
        # # # object_transl = self.data[index]['f6'] #
        # # object_global_orient = data['f5'] ### get object global orientations ###
        # # object_trcansl = data['f6']
        # # # object_id = data['f7'][0].item() ### data_f7 item ###
        # # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        # object_global_orient = c[3][-3] # num_frames x 3 
        # object_transl = c[3][-2] # num_frames x 3
        
        
        # # object_global_orient, object_transl #
        # object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        # object_transl = object_transl[start_idx: start_idx + self.window_size]
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        # object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        # object_trcansl_th = torch.from_numpy(object_transl).float()
        
        
        if self.args.select_part_idx != -1:
            object_global_orient = cur_obj_glb_rot.reshape(self.window_size, 3, 3).astype(np.float32)
            object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
            object_transl = cur_obj_glb_trans.reshape(self.window_size, 3).astype(np.float32)
            object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
            object_trcansl_th = torch.from_numpy(object_transl).float()
        else:
            # transpose objects #
            object_global_orient = self.data['f3'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3, 3).astype(np.float32) # nf x 
            object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
            object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
            object_transl = self.data['f4'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
            object_trcansl_th = torch.from_numpy(object_transl).float()
        
        # # pert_subj_params = c[4]
        
        # st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx
        
        # ### pts gt ###
        # ## rhnad pose, rhand pose gt ##
        # ## glboal orientation and hand pose #
        # rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        # rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        # rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        # rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        # rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        
        # rhand_transl, rhand_betas = c[3][5], c[3][6]
        # rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        # rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        # rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        # # # orientation rotation matrix #
        # # rhand_global_orient_mtx_gt = utils.batched_get_orientation_matrices(rhand_global_orient_gt)
        # # rhand_global_orient_mtx_gt_var = torch.from_numpy(rhand_global_orient_mtx_gt).float()
        # # # orientation rotation matrix #
        
        # rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
        # rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
        # rhand_beta_var = torch.from_numpy(rhand_betas).float()
        # rhand_transl_var = torch.from_numpy(rhand_transl).float() # self.window_size x 3
        # # R.from_rotvec(obj_rot).as_matrix()
        
        # ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
        # ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
        # #### ==== get random augmented pose, rot, transl ==== ####
        # # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        # aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
        # aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3
        # # cur_t = self.uinform_sample_t()
        # # aug_trans, aug_rot, aug_pose #
        # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
        # # ### === get and save noise vectors === ###
        # # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
        # aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        # aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        # aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        # # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # # ### creat augmneted orientations, pose, and transl ###
        # rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        # rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        # rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        
        object_normal = object_vn
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
        
        
        # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        # nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
        # # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        # dist_object_pc_nearest_pcs = torch.sum(
        #     (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
        # )
        # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
        # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
        # # # dist_threshold = 0.01
        # dist_threshold = self.dist_threshold
        # # # dist_threshold for pc_nearest_pcs #
        # dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
        
        # # # base_pts_mask: nn_obj_pcs #
        # base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
        # # # nn_base_pts x 3 -> torch tensor #
        # base_pts = object_pc_th[0][base_pts_mask]
        # # # base_pts_bf_sampling = base_pts.clone()
        # base_normals = object_normal_th[0][base_pts_mask]
        
        # nn_base_pts = self.nn_base_pts
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
        
        # # if self.dir_stra == "rot_angles": ## rot angles ##
        # #     normals_rot_mtx = utils.batched_get_rot_mtx_fr_vecs_v2(base_normals)
        
        # # if self.canon_obj:
        #     ## reverse transform base points ###
        #     ## canonicalize base points and base normals ###
        # base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
        #     ) # .transpose(0, 1)
        # base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
        #     ) # .transpose(0, 1)
        
        
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
            # # nn_base_pts x 3 -> torch tensor # ## object_pc_th ###
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

        tot_full_obj_verts = torch.from_numpy(tot_full_obj_verts).float()
        # tot_full_obj_verts = torch.matmul(
        #     tot_full_obj_verts - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        # )
        
        if self.args.use_anchors:
            # rhand_anchors, pert_rhand_anchors #
            rhand_anchors = torch.matmul(
                rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
            pert_rhand_anchors = torch.matmul(
                pert_rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
            )
        
        if self.predicted_hand_joints is not None:
            # self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
            # self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
            # self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
            # self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
            pert_rhand_joints = self.predicted_hand_joints
            # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
            if self.predicted_hand_trans is not None:
                rhand_transl_var = self.predicted_hand_trans
                rhand_global_orient_var = self.predicted_hand_rot
                rhand_pose_var = self.predicted_hand_theta
                print(f"rhand_beta_var: {self.predicted_hand_beta.size()}")
                rhand_beta_var = self.predicted_hand_beta #.unsqueeze(0)
        
        ''' normalization strategy xxx --- data scaling '''
        # base_pts = base_pts * 5.
        # rhand_joints = rhand_joints * 5.
        ''' Normlization stratey xxx --- data scaling '''
        
        
        
        # ''' GET ambinet space base pts '''
        # # normalized #
        # # rhand_joints: nf x nnj x 3
        # # base_pts: nnb x 3 ## --> (nf x nnj + nnb) x 3 #
        # tot_pts = torch.cat(
        #     [rhand_joints.view(rhand_joints.size(0) * rhand_joints.size(1), 3), base_pts], dim=0
        # )
        # maxx_tot_pts, _ = torch.max(tot_pts, dim=0) ## 3
        # minn_tot_pts, _ = torch.min(tot_pts, dim=0) ## 3
        # # uniformly sample rand
        # # rand(0, 1)? # 0 and 1 
        # xyz_coords_coeffs = torch.rand((nn_base_pts, 3)) ## nn_base_pts x 3 #
        # # nn_base_pts x 3 #
        # sampled_base_pts = minn_tot_pts.unsqueeze(0) + xyz_coords_coeffs * (maxx_tot_pts - minn_tot_pts).unsqueeze(0)
        # # to object_pc
        # # nn_base_pts x nn_obj_pc # ---> nf x nnb x nn_obj
        # dist_sampled_base_pts_to_obj_pc = torch.sum(
        #     (sampled_base_pts.unsqueeze(0).unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
        # )
        # # nf x nnb; nf x nnb #
        # minn_dist_sampled_base_to_obj, minn_dist_idxes = torch.min(
        #     dist_sampled_base_pts_to_obj_pc, dim=-1
        # )
        
        # ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
        # ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
        # # TODO: should compare between raw pts and raltive positions to nearest obj pts #
        # # sampled_base...: nf x nnb x 3 #
        # sampled_base_pts_nearest_obj_pc = utils.batched_index_select_ours(
        #     values=object_pc_th, indices=minn_dist_idxes, dim=1
        # )
        # # sampled...: nf x nnb x 3 #
        # sampled_base_pts_nearest_obj_vns = utils.batched_index_select_ours(
        #     values=object_normal_th, indices=minn_dist_idxes, dim=1
        # )
        
        # # base pts single values, nearest object pc and nearest object normal #
        ''' GET ambinet space base pts '''
        
        
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
        # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
        # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
        
        # # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
        # # 
        # # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
        # signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
        #     base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
        # )
        # # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
        # # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
        # # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
        # rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
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
                
                
            # # nf x nnj x nnb x 3 # 
            # rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
            # # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
            # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
            
            rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
            dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
            stats_dict = {
                'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
                'per_frame_std_joints_rel': per_frame_std_joints_rel,
                'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
                'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
            }
            ''' Relative positions and distances normalization, strategy 3 '''
        
        if self.denoising_stra == "motion_to_rep": # motion_to_rep #
            pert_rhand_joints = (pert_rhand_joints - self.avg_jts) / self.std_jts
        
        
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        ''' Obj data '''
        obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data()
        obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
        obj_normals = torch.from_numpy(obj_normals).float() # 
        obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        
        # obj_verts = object_pc_th
        ''' Obj data '''
        
        object_id = 0
        caption = "apple"
        # pose_one_hots, word_embeddings #
        
        # object_global_orient_th, object_transl_th #
        object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        # should centralize via the moving part # # centralize via the moving part
        # rhand_global_orient_var, rhand_pose_var, rhand_transl_var, rhand_beta_var #
        ''' Construct data for returning '''
        rt_dict = {
            # object_pc_th
            # 'object_pcs': object_pc_th.detach().cpu().numpy(),
            'object_pcs': object_pc_th, # .detach().cpu().numpy(),
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            # 'rhand_joints': rhand_joints,
            'gt_rhand_joints': rhand_joints, ## rhand joints ###
            'rhand_joints': pert_rhand_joints, #  if not self.args.use_canon_joints else canon_pert_rhand_joints,
            'rhand_verts': rhand_verts,
            # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
            'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
            'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
            'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
            'rhand_betas': rhand_beta_var[0],
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            'object_id': object_id, # int value
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'object_global_orient': object_global_orient_th,
            'object_transl': object_transl_th,
            # 'st_idx': start_idx,
            # 'ed_idx': start_idx + self.window_size,
            'st_idx': 0,
            'ed_idx': 0 + self.window_size,
            'pert_verts': pert_rhand_verts,
            'verts': rhand_verts,
            'obj_verts': obj_verts,
            'obj_normals': obj_normals,
            # 'obj_faces': obj_faces, # nnfaces x 3 #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
            # # obj_verts, obj_faces
            # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
            # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
        }
        # obj_verts, obj_faces
        if len(self.corr_fn) > 0:
            rt_dict.update(
                {
                    # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
                    # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
                    'obj_verts': tot_full_obj_verts, # nn_verts x 3 #
                    'obj_faces': torch.from_numpy(tot_full_obj_faces).long(), # nn_faces x 3 
                }
                # tot_full_obj_verts, tot_full_obj_faces
            )
        if self.args.select_part_idx != -1:
            rt_dict.update(
                {
                    # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
                    # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
                    'obj_verts': tot_full_obj_verts, # nn_verts x 3 #
                    'obj_faces': torch.from_numpy(tot_full_obj_faces).long(), # nn_faces x 3 
                }
                # tot_full_obj_verts, tot_full_obj_faces
            )
        
        if self.use_anchors: ## update rhand anchors here ##
            rt_dict.update(
                {
                    'rhand_anchors': rhand_anchors,
                    'pert_rhand_anchors': pert_rhand_anchors,
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] = 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict


        # if self.dir_stra == 'rot_angles':
        #     # tangent_orient_vec # nn_base_pts x 3 #
        #     rt_dict['base_tangent_orient_vec'] = tangent_orient_vec.numpy() #
        
        rt_dict_th = {
            k: torch.from_numpy(rt_dict[k]).float() if not isinstance(rt_dict[k], torch.Tensor) else rt_dict[k] for k in rt_dict 
        }
        # rt_dict

        return rt_dict_th
        # return np.concatenate([window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, rel_pos_object_pc_joint_gt, dec_cond, rhand_feats_exp], axis=2)

    def __len__(self):
        cur_len = self.len // self.step_size
        if cur_len * self.step_size < self.len:
          cur_len += 1
        cur_len = 1
        return cur_len
        # return ceil(self.len / self.step_size)
        # return self.len



# # HOI4D #
# class GRAB_Dataset_V19_Ours(torch.utils.data.Dataset):
#     def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
#         #### GRAB dataset #### ## GRAB dataset
#         self.clips = []
#         self.len = 0
        
#         self.single_seq_path = args.single_seq_path
#         self.data = np.load(self.single_seq_path, allow_pickle=True)
        
        
#         self.window_size = window_size
#         self.step_size = step_size
#         self.num_points = num_points
#         self.split = split
        
#         self.cad_model_fn = args.cad_model_fn
        
#         self.start_idx = args.start_idx
        
#         # split = args.single_seq_path.split("/")[-2].split("_")[0]
#         # self.split = split
#         # print(f"split: {self.split}")
        
#         # self.model_type = 'v1_wsubj_wjointsv25'
#         # self.debug = False
#         # # self.use_ambient_base_pts = args.use_ambient_base_pts
#         # # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
#         # self.num_sche_steps = 100
#         # self.w_vectorizer = w_vectorizer
#         # self.use_pert = True
#         # self.use_rnd_aug_hand = True
        
#         self.args = args
#         self.use_anchors = self.args.use_anchors
        
#         self.denoising_stra = args.denoising_stra ## denoising_stra!
        
#         # self.data_folder = data_folder
#         # self.subj_data_folder = data_folder
#         # # self.subj_corr_data_folder = args.subj_corr_data_folder
#         self.mano_path = "manopth/mano/models" ### mano_path
#         # self.aug = True
#         # self.use_anchors = False
#         # # self.args = args
        
#         predicted_info_fn = args.predicted_info_fn
#         # load data from predicted information #
#         if len(predicted_info_fn) > 0:
#             print(f"Loading preidcted info from {predicted_info_fn}")
#             data = np.load(predicted_info_fn, allow_pickle=True).item()
#             data_opt_info_fn = "/data1/xueyi/mdm/eval_save/optimized_infos_sv_dict_seq_scissors_optimized_aug.npy"
#             data_opt = np.load(data_opt_info_fn, allow_pickle=True).item()
#             outputs = data['outputs']
#             self.predicted_hand_joints = outputs # nf x nnjoints x 3 #
#             self.predicted_hand_joints = torch.from_numpy(self.predicted_hand_joints).float()
            
#             if 'rhand_trans' in data:
#                 # outputs = data['outputs']
#                 self.predicted_hand_trans = data['rhand_trans'] # nframes x 3 
#                 self.predicted_hand_rot = data['rhand_rot'] # nframes x 3 
#                 self.predicted_hand_theta = data['rhand_theta']
#                 self.predicted_hand_beta = data['rhand_beta']
#                 self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
#                 self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
#                 self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
#                 self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
                
#                 self.predicted_hand_trans_opt = data_opt['rhand_trans'] # nframes x 3 
#                 self.predicted_hand_rot_opt = data_opt['rhand_rot'] # nframes x 3 
#                 self.predicted_hand_theta_opt = data_opt['rhand_theta']
#                 self.predicted_hand_beta_opt = data_opt['rhand_beta']
#                 self.predicted_hand_trans_opt = torch.from_numpy(self.predicted_hand_trans_opt).float() # nframes x 3 
#                 self.predicted_hand_rot_opt = torch.from_numpy(self.predicted_hand_rot_opt).float() # nframes x 3 
#                 self.predicted_hand_theta_opt = torch.from_numpy(self.predicted_hand_theta_opt).float() # nframes x 24 
#                 self.predicted_hand_beta_opt = torch.from_numpy(self.predicted_hand_beta_opt).float() # 10,
                
#                 self.predicted_hand_trans[9:] = self.predicted_hand_trans_opt[9:]
#                 self.predicted_hand_rot[9:] = self.predicted_hand_rot_opt[9:]
#                 self.predicted_hand_theta[ 9:] = self.predicted_hand_theta_opt[ 9:]
#                 # self.predicted_hand_trans[:, 9:] = self.predicted_hand_trans_opt[:, 9:]
                
#             else:
#                 self.predicted_hand_trans = None
#                 self.predicted_hand_rot = None
#                 self.predicted_hand_theta = None
#                 self.predicted_hand_beta = None
            
#         else:
#             self.predicted_hand_joints = None
        
        
#         self.corr_fn = args.corr_fn # corr_fn 
#         if len(self.corr_fn) > 0:
#             self.raw_corr_data = np.load(self.corr_fn, allow_pickle=True)
#         # self.dist_stra = args.dist_stra
        
#         # self.load_meta = True
        
#         self.dist_threshold = 0.005
#         self.dist_threshold = 0.01
#         self.nn_base_pts = 700
#         self.nn_base_pts = args.nn_base_pts
#         print(f"nn_base_pts: {self.nn_base_pts}")
        
        
#         self.theta_dim = args.theta_dim
#         use_pca = True if self.theta_dim < 45 else False
        
#         self.mano_layer = ManoLayer(
#             flat_hand_mean=True,
#             side='right',
#             mano_root=self.mano_path, # mano_root #
#             ncomps=self.theta_dim,
#             use_pca=use_pca,
#             root_rot_mode='axisang',
#             joint_rot_mode='axisang'
#         )
        
#         # # anchor_load_driver, masking_load_driver #
#         # # use_anchors, self.hand_palm_vertex_mask #
#         # if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
#         #     # anchor_load_driver, masking_load_driver #
#         #     inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
#         #     fvi, aw, _, _ = anchor_load_driver(inpath)
#         #     self.face_vertex_index = torch.from_numpy(fvi).long()
#         #     self.anchor_weight = torch.from_numpy(aw).float()
            
#         #     anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
#         #     palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
#         #     hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
#         #     # self.hand_palm_vertex_mask for hand palm mask #
#         #     self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
        
    
#     def uinform_sample_t(self):
#         t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
#         return t
    
#     def load_clip_data(self, clip_idx, f=None):
#         if f is None:
#           cur_clip = self.clips[clip_idx]
#           if len(cur_clip) > 3:
#               return cur_clip
#           f = cur_clip[2]
#         clip_clean = np.load(f)
#         # pert_folder_nm = self.split + '_pert'
#         pert_folder_nm = self.split
#         # if not self.use_pert:
#         #     pert_folder_nm = self.split
#         # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
#         ##### load subj params #####
#         pure_file_name = f.split("/")[-1].split(".")[0]
#         pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                
#         subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
#         subj_params = np.load(subj_params_fn, allow_pickle=True).item()
#         rhand_transl = subj_params["rhand_transl"]
#         rhand_betas = subj_params["rhand_betas"]
#         rhand_pose = clip_clean['f2'] ## rhand pose ##
        
#         object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
#         object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
#         object_idx = clip_clean['f7'][0].item()
        
#         pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
#         pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
#         ##### load subj params #####
        
#         # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
#         # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
#         ### pert and clean pair for encoding and decoding ###
        
#         # maxx_clip_len = 
#         loaded_clip = (
#             0, rhand_transl.shape[0], clip_clean,
#             [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
#         )
#         # self.clips[clip_idx] = loaded_clip
        
#         return loaded_clip
        
#         # self.clips.append((self.len, self.len+clip_len, clip_pert,
#         #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
#         #     # subj_corr_data, pert_subj_corr_data
#         #     ))
        
#     def get_idx_to_mesh_data(self):
#         cad_model_fn = self.cad_model_fn
#         obj_mesh = trimesh.load(cad_model_fn, process=False)
#         obj_verts = np.array(obj_mesh.vertices)
#         obj_vertex_normals = np.array(obj_mesh.vertex_normals)
#         obj_faces = np.array(obj_mesh.faces)
#         mesh_data = (obj_verts, obj_vertex_normals, obj_faces)
#         return mesh_data
    
#     def get_ari_obj_fr_x(self, i_frame_st, i_frame_ed):
        
#         # raw_corr_data
#         tot_obj_verts = []
#         tot_obj_faces = []
#         tot_obj_normals = []
#         tot_obj_glb_rot = []
#         tot_obj_glb_trans = []
#         tot_hand_beta = []
#         tot_hand_theta = []
#         tot_hand_transl = []
#         tot_hand_joints = []

#         tot_full_obj_verts = []
#         tot_full_obj_faces = []
        
#         # single_seq_path #
#         single_seq_folder = "/".join(self.args.single_seq_path.split("/")[:-1]) # single_seq_path
#         single_seq_meta_data_fn = os.path.join(single_seq_folder, "meta_data.npy")
#         single_seq_meta_data = np.load(single_seq_meta_data_fn, allow_pickle=True).item()
#         series_tag = single_seq_meta_data["case_flag"]
#         series_obj_category = series_tag.split("/")[2]
#         series_obj_category = int(series_obj_category[1:])
#         series_obj_inst_idx = series_tag.split("/")[3] # N17
#         series_obj_inst_idx = int(series_obj_inst_idx[1:]) # idx of the instance
#         cat_idx_to_obj_nm_mapping = [ # Bottle-3  Bowl-3  Chair  Kettle-3  Knife  Mug-2  ToyCar-1 # 
#             '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
#             'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
#             'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
#         ]
#         cat_nm = cat_idx_to_obj_nm_mapping[series_obj_category]

#         case_merged_data_fn = os.path.join(single_seq_folder, "merged_data.npy")
#         case_merged_data = np.load(case_merged_data_fn, allow_pickle=True)

      
        
#         for i_frame in range(i_frame_st, i_frame_ed):
#             cur_obj_rot = self.raw_corr_data[i_frame]['obj_rot']
#             cur_obj_trans = self.raw_corr_data[i_frame]['obj_trans']

            
#             cur_arti_cat_nm = cat_nm
#             cur_arti_inst_nm = int(series_obj_inst_idx) # ## series obj inst idxes ### 

#             if not self.args.use_arti_obj:
#                 cad_model_fn = [ # get cad models 
#                     f"/share/datasets/HOI4D_CAD_Model_for_release/rigid/{cur_arti_cat_nm}/%03d.obj" % cur_arti_inst_nm, 
#                 ]
#                 if not isinstance(cur_obj_rot, list):
#                     cur_obj_rot = [cur_obj_rot]
#                     cur_obj_trans = [cur_obj_trans]
#                 self.cad_model_fn = cad_model_fn[0]
#             else:
#                 if cat_nm in ["Scissors", "Laptop"]:
#                     if self.args.use_reverse:
#                         cad_model_fn = [
#                             f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
#                             f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
#                         ]
#                     else:
#                         cad_model_fn = [ # get cad models 
#                             f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm, 
#                             f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm 
#                         ]
#                         # cad_model_fn = [ # get cad models 
#                         #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
#                         #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
#                         # ]
#                 else:
#                     cad_model_fn = [ # get cad models 
#                         f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm, 
#                         f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm 
#                     ]

#                     # cad_model_fn = [ # get cad models 
#                     #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-1-align.obj" % cur_arti_inst_nm, 
#                     #     f"/share/datasets/HOI4D_CAD_Model_for_release/articulated/{cur_arti_cat_nm}/%03d/objs/new-0-align.obj" % cur_arti_inst_nm 
#                     # ]

#             full_cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
#             # nn_verts x 3 
#             # nn_faces x 3 
#             full_cur_obj_verts, full_cur_obj_faces = full_cur_obj_mesh.vertices, full_cur_obj_mesh.faces

#             # tot_full_obj_verts = []
#             # tot_full_obj_faces = []
#             # tot_full_obj_verts, tot_full_obj_faces
#             tot_full_obj_verts.append(full_cur_obj_verts)
#             tot_full_obj_faces.append(full_cur_obj_faces)

            
#             if self.args.select_part_idx != -1:
#                 cad_model_fn = [cad_model_fn[self.args.select_part_idx]]
                
#                 cur_obj_glb_rot = cur_obj_rot[self.args.select_part_idx].reshape(-1)
#                 cur_obj_glb_trans = cur_obj_trans[self.args.select_part_idx]

#                 cur_obj_rot = [cur_obj_rot[self.args.select_part_idx]]
#                 cur_obj_trans = [cur_obj_trans[self.args.select_part_idx]]

#                 cur_frame_data = case_merged_data[i_frame]
#                 cur_theta = cur_frame_data['theta'].squeeze(0).numpy() # 24 current theta 
#                 cur_beta = cur_frame_data['beta'].squeeze(0).numpy() ## beta 
#                 cur_rhand_transl = cur_frame_data['trans'].squeeze(0).numpy() # ## rhand trans for the cur_frame_data
#                 cur_rhand_joints = cur_frame_data['joints'].reshape(-1)

#                 ### tot obje rot ####
#                 tot_obj_glb_rot.append(cur_obj_glb_rot)
#                 # tot_obj_glb_trans.append(cur_obj_glb_trans)

#             cur_obj_mesh = get_object_mesh_ours_arti(cad_model_fn, cur_obj_rot, cur_obj_trans)
#             # nn_verts x 3 
#             # nn_faces x 3 
#             cur_obj_verts, cur_obj_faces = cur_obj_mesh.vertices, cur_obj_mesh.faces
#             obj_center = np.mean(cur_obj_verts, axis=0, keepdims=True)

#             obj_center = np.zeros_like(obj_center)

#             cur_obj_verts = cur_obj_verts - obj_center

#             if self.args.select_part_idx != -1:
#                 cur_rhand_transl = cur_rhand_transl - obj_center[0]
#                 cur_rhand_joints = cur_rhand_joints.reshape(21, 3) - obj_center
#                 cur_obj_glb_trans = cur_obj_glb_trans - obj_center[0]
#                 tot_hand_beta.append(cur_beta)
#                 tot_hand_theta.append(cur_theta)
#                 tot_hand_transl.append(cur_rhand_transl)
#                 tot_hand_joints.append(cur_rhand_joints)
#                 tot_obj_glb_trans.append(cur_obj_glb_trans)

#             cur_obj_normals = cur_obj_mesh.vertex_normals
#             tot_obj_normals.append(cur_obj_normals)

#             tot_obj_verts.append(cur_obj_verts)
#             tot_obj_faces.append(cur_obj_faces)
        
#         tot_obj_verts = np.stack(tot_obj_verts, axis=0)
#         tot_obj_faces = np.stack(tot_obj_faces, axis=0)
#         tot_obj_normals  = np.stack(tot_obj_normals, axis=0)

#         # # tot_full_obj_verts, tot_full_obj_faces
        
#         if len(tot_obj_glb_rot) > 0:
#             tot_obj_glb_rot = np.stack(tot_obj_glb_rot, axis=0)
#             tot_obj_glb_trans = np.stack(tot_obj_glb_trans, axis=0)
#             tot_hand_beta = np.stack(tot_hand_beta, axis=0)
#             tot_hand_theta = np.stack(tot_hand_theta, axis=0)
#             tot_hand_transl = np.stack(tot_hand_transl, axis=0)
#             tot_hand_joints = np.stack(tot_hand_joints, axis=0)
#             tot_full_obj_verts = np.stack(tot_full_obj_verts, axis=0)
#             tot_full_obj_faces = np.stack(tot_full_obj_faces, axis=0)

#             print(f"tot_hand_joints: {tot_hand_joints.shape}")
            
#         return tot_obj_verts, tot_obj_faces, tot_obj_normals, tot_obj_glb_rot, tot_obj_glb_trans, tot_hand_beta, tot_hand_theta, tot_hand_transl, tot_hand_joints, tot_full_obj_verts, tot_full_obj_faces


#     #### enforce correct contacts #### ### the sequence in the clip is what we want here #
#     def __getitem__(self, index):

        
#         start_idx = self.start_idx
#         if len(self.corr_fn) > 0:
#             cur_obj_verts, cur_obj_faces, cur_obj_normals, cur_obj_glb_rot, cur_obj_glb_trans, tot_hand_beta, tot_hand_theta, tot_hand_transl, tot_hand_joints, tot_full_obj_verts, tot_full_obj_faces = self.get_ari_obj_fr_x(start_idx, start_idx + self.window_size) # nn_obj_verts x 3; nn_obj_faces x 3 #
#             print(f"corr_fn: {self.corr_fn}, obj_verts: {cur_obj_verts.shape}, cur_obj_faces: {cur_obj_faces.shape}")
            
 
        
#         if self.args.select_part_idx != -1:
#             # tot_obj_verts_th = torch.from_numpy(cur_obj_verts).float()
#             # tot_obj_faces_th = torch.from_numpy(cur_obj_faces).long()
#             # tot_obj_normals_th = torch.from_numpy(tot_obj_normals).float()

#             object_pc = cur_obj_verts.copy()
#             object_vn = cur_obj_normals.copy()
#             # object_pc = cur_obj_verts.copy()
#         else:
#             object_pc = self.data['f1'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
#             object_vn = self.data['f2'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        
#         if self.args.select_part_idx != -1:
#             rhand_joints = tot_hand_joints
#             rhand_transl = tot_hand_transl
#             rhand_beta = tot_hand_beta
#             rhand_theta = tot_hand_theta
#             print(f"rhand_transl: {rhand_transl.shape}, rhand_beta: {rhand_beta.shape},rhand_beta: {rhand_beta.shape}, rhand_theta: {rhand_theta.shape} ")
#         else:
#             rhand_joints = self.data['f11'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
            
#             # rhand_glb_rot, rhand_pose, rhand_joints_gt, minn_dists_rhand_joints_object_pc,
#             # rhand_joints = self.data[index]['f11'].reshape(21, 3).astype(np.float32)
#             # rhand_joints = rhand_joints * 0.001
#             # rhand_joints = rhand_joints - obj_center
            
#             # rhand_joints_fr_data = rhand_joints.copy() ## rhandjoints
            
            
#             rhand_transl = self.data['f10'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
#             # rhand_transl = rhand_transl - obj_center[0]
#             rhand_beta = self.data['f9'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
#             rhand_theta = self.data['f8'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1).astype(np.float32)
            
        
#         # rhand_transl_clean = self.clean_data[index]['f10'].reshape(3).astype(np.float32)
#         # rhand_theta_clean = self.clean_data[index]['f8'].reshape(-1).astype(np.float32)
        
#         rhand_glb_rot = rhand_theta[:, :3]
#         rhand_theta = rhand_theta[:, 3:]
        
#         ##### rhand transl #####
#         # rhand_glb_rot = rhand_theta_clean[:3]
#         # rhand_transl = rhand_transl_clean
#         ##### rhand transl #####
        
#         # rhand_global_orient_var, rhand_pose_var, rhand_transl_var, rhand_beta_var #
#         rhand_global_orient_var = torch.from_numpy(rhand_glb_rot).float() # .unsqueeze(0)
#         rhand_pose_var = torch.from_numpy(rhand_theta).float() # . unsqueeze(0)
#         rhand_transl_var = torch.from_numpy(rhand_transl).float() # .unsqueeze(0)
#         rhand_beta_var = torch.from_numpy(rhand_beta).float() # .unsqueeze(0)
        
        
        
#         # # rhand_global_orient = self.data[index]['f1'].reshape(-1).astype(np.float32)
#         # rhand_pose = rhand_theta
#         # # rhand_transl = self.subj_params['rhand_transl'][index].reshape(-1).astype(np.float32)
#         # rhand_betas = rhand_beta
        
        
#         ####### Get rhand_verts and rhand_joint #######
#         rhand_verts, rhand_joints = self.mano_layer(
#             torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
#             rhand_beta_var.view(-1, 10), rhand_transl_var
#         )
#         rhand_verts = rhand_verts * 0.001
#         rhand_joints = rhand_joints * 0.001
#         ####### Get rhand_verts and rhand_joint #######
        
        
#         if self.use_anchors: # # rhand_anchors: bsz x nn_hand_anchors x 3 #
#             # rhand_anchors = rhand_verts[:, self.hand_palm_vertex_mask] # nf x nn_anchors x 3 --> for the anchor points ##
#             rhand_anchors = recover_anchor_batch(rhand_verts, self.face_vertex_index, self.anchor_weight.unsqueeze(0).repeat(self.window_size, 1, 1))
#             pert_rhand_anchors = rhand_anchors
#             # print(f"rhand_anchors: {rhand_anchors.size()}") ### recover rhand verts here ###
        
        
#         # rhand_transl = rhand_transl - obj_center[0]
        
        
#         pert_rhand_joints = rhand_joints
#         pert_rhand_verts = rhand_verts
        
        
        
#         if self.args.select_part_idx != -1:
#             object_global_orient = cur_obj_glb_rot.reshape(self.window_size, 3, 3).astype(np.float32)
#             object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
#             object_transl = cur_obj_glb_trans.reshape(self.window_size, 3).astype(np.float32)
#             object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
#             object_trcansl_th = torch.from_numpy(object_transl).float()
#         else:
#             # transpose objects #
#             object_global_orient = self.data['f3'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3, 3).astype(np.float32) # nf x 
#             object_global_orient = np.transpose(object_global_orient, (0, 2, 1))
#             object_global_orient_mtx_th = torch.from_numpy(object_global_orient).float()
#             object_transl = self.data['f4'][start_idx: start_idx + self.window_size].reshape(self.window_size, 3).astype(np.float32)
#             object_trcansl_th = torch.from_numpy(object_transl).float()
        
#         # # pert_subj_params = c[4]
        
#         # st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx
        
#         # ### pts gt ###
#         # ## rhnad pose, rhand pose gt ##
#         # ## glboal orientation and hand pose #
#         # rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
#         # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
#         # rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
#         # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
#         # rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
#         # rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
#         # rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        
#         # rhand_transl, rhand_betas = c[3][5], c[3][6]
#         # rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
#         # # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
#         # rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
#         # rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
#         # # # orientation rotation matrix #
#         # # rhand_global_orient_mtx_gt = utils.batched_get_orientation_matrices(rhand_global_orient_gt)
#         # # rhand_global_orient_mtx_gt_var = torch.from_numpy(rhand_global_orient_mtx_gt).float()
#         # # # orientation rotation matrix #
        
#         # rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
#         # rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
#         # rhand_beta_var = torch.from_numpy(rhand_betas).float()
#         # rhand_transl_var = torch.from_numpy(rhand_transl).float() # self.window_size x 3
#         # # R.from_rotvec(obj_rot).as_matrix()
        
#         # ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
#         # ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
#         # #### ==== get random augmented pose, rot, transl ==== ####
#         # # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
#         # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
#         # aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
#         # aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3
#         # # cur_t = self.uinform_sample_t()
#         # # aug_trans, aug_rot, aug_pose #
#         # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
#         # # ### === get and save noise vectors === ###
#         # # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
#         # aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
#         # aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
#         # aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
#         # # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
#         # # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
#         # # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
#         # # ### creat augmneted orientations, pose, and transl ###
#         # rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
#         # rnd_aug_pose_var = rhand_pose_var + aug_pose_var
#         # rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        
#         # rhand_joints --> ws x nnjoints x 3 --> rhandjoitns! #
#         # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
#         # pert_rhand_betas_var, rhand_beta_var
#         # rhand_verts, rhand_joints = self.mano_layer(
#         #     torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
#         #     rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
#         # )
#         # ### rhand_joints: for joints ###
#         # rhand_verts = rhand_verts * 0.001
#         # rhand_joints = rhand_joints * 0.001
        
        
#         # if self.use_rnd_aug_hand: ## rnd aug pose var, transl var #
#         #     # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
#         #     pert_rhand_global_orient_var = rnd_aug_global_orient_var.clone()
#         #     pert_rhand_pose_var = rnd_aug_pose_var.clone()
#         #     pert_rhand_transl_var = rnd_aug_transl_var.clone()
#         #     # pert_rhand_global_orient_mtx = utils.batched_get_orientation_matrices(pert_rhand_global_orient_var.numpy())
        
#         # # # pert_rhand_betas_var
#         # # pert_rhand_joints, rhand_joints -> ws x nn_joints x 3 #
#         # # pert_rhand_joints --> for rhand joints in the camera frmae ###
#         # pert_rhand_verts, pert_rhand_joints = self.mano_layer(
#         #     torch.cat([pert_rhand_global_orient_var, pert_rhand_pose_var], dim=-1),
#         #     rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), pert_rhand_transl_var
#         # )
#         # pert_rhand_verts = pert_rhand_verts * 0.001 # verts 
#         # pert_rhand_joints = pert_rhand_joints * 0.001 # joints
        
#         # use_canon_joints
        
#         # canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
#         #     torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
#         #     rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
#         # )
#         # canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
#         # canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
#         # canon_pert_rhand_verts, canon_pert_rhand_joints = self.mano_layer(
#         #     torch.cat([torch.zeros_like(pert_rhand_global_orient_var), pert_rhand_pose_var], dim=-1),
#         #     pert_rhand_betas_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), torch.zeros_like(pert_rhand_transl_var)
#         # )
#         # canon_pert_rhand_verts = canon_pert_rhand_verts * 0.001 # verts 
#         # canon_pert_rhand_joints = canon_pert_rhand_joints * 0.001 # joints
        
#         # ### Relative positions from base points to rhand joints ###
#         # object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)
#         # object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        
#         object_normal = object_vn
#         object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
#         # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
#         object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3
#         # # object_normal_th = object_normal_th[0].unsqueeze(0).repeat(rhand_verts.size(0),)
        
        
#         # ws x nnjoints x nnobjpts #
#         dist_rhand_joints_to_obj_pc = torch.sum(
#             (rhand_joints.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
#         )
#         # dist_pert_rhand_joints_obj_pc = torch.sum(
#         #     (pert_rhand_joints_th.unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
#         # )
#         _, minn_dists_joints_obj_idx = torch.min(dist_rhand_joints_to_obj_pc, dim=-1) # num_frames x nn_hand_verts 
#         # # nf x nn_obj_pc x 3 xxxx nf x nn_rhands -> nf x nn_rhands x 3
        
        
#         # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
#         # nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1)
#         # # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
#         # dist_object_pc_nearest_pcs = torch.sum(
#         #     (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1
#         # )
#         # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs
#         # dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
#         # # # dist_threshold = 0.01
#         # dist_threshold = self.dist_threshold
#         # # # dist_threshold for pc_nearest_pcs #
#         # dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
        
#         # # # base_pts_mask: nn_obj_pcs #
#         # base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
#         # # # nn_base_pts x 3 -> torch tensor #
#         # base_pts = object_pc_th[0][base_pts_mask]
#         # # # base_pts_bf_sampling = base_pts.clone()
#         # base_normals = object_normal_th[0][base_pts_mask]
        
#         # nn_base_pts = self.nn_base_pts
#         # base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
#         # base_pts_idxes = base_pts_idxes[:nn_base_pts]
#         # # if self.debug:
#         # #     print(f"base_pts_idxes: {base_pts.size()}, nn_base_sampling: {nn_base_pts}")
        
#         # # ### get base points ### # base_pts and base_normals #
#         # base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
#         # base_normals = base_normals[base_pts_idxes]
        
        
#         # # # object_global_orient_mtx # nn_ws x 3 x 3 #
#         # base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
#         # base_pts_transl = object_trcansl_th[0] # 3
        
#         # # if self.dir_stra == "rot_angles": ## rot angles ##
#         # #     normals_rot_mtx = utils.batched_get_rot_mtx_fr_vecs_v2(base_normals)
        
#         # # if self.canon_obj:
#         #     ## reverse transform base points ###
#         #     ## canonicalize base points and base normals ###
#         # base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
#         #     ) # .transpose(0, 1)
#         # base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
#         #     ) # .transpose(0, 1)
        
        
#         if not self.args.use_arti_obj:
#             object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
#             nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # object pc #
#             # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
#             dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
#                 (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
#             ) 
#             dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs # nearest to all pts in all frames ## 
#             dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs # nn_obj_pcs # nn_obj_pcs #
#             # # dist_threshold = 0.01 # threshold 
#             dist_threshold = self.dist_threshold
#             # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
#             dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
#             # # base_pts_mask: nn_obj_pcs #
#             base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
#             # # nn_base_pts x 3 -> torch tensor #
#             base_pts = object_pc_th[0][base_pts_mask]
#             # # base_pts_bf_sampling = base_pts.clone()
#             base_normals = object_normal_th[0][base_pts_mask]
            
#             nn_base_pts = self.nn_base_pts
#             base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
#             base_pts_idxes = base_pts_idxes[:nn_base_pts]
            
#             # ### get base points ### # base_pts and base_normals #
#             base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
#             base_normals = base_normals[base_pts_idxes]
            
            
#             # # object_global_orient_mtx # nn_ws x 3 x 3 #
#             base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
#             base_pts_transl = object_trcansl_th[0] # 3
            
#             base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
#                 ) # .transpose(0, 1)
#             base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
#                 ) # .transpose(0, 1)
#         else:
#             # object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
#             nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # nearest_obj_pcs: ws x nn_jts x 3 --> for nearet obj pcs # 
#             # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
#             dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
#                 (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
#             ) 
#             dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # ws x nn_obj #
#             dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs #
#             # # dist_threshold = 0.01 # threshold 
#             dist_threshold = self.dist_threshold
#             # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
#             dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
#             # # base_pts_mask: nn_obj_pcs #
#             base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold) # nn_obj_pcs -> nearest_pcs mask #
#             base_pts = object_pc_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
#             base_normals = object_normal_th[:, base_pts_mask] # ws x nn_valid_obj_pcs x 3 #
#             nn_base_pts = self.nn_base_pts
#             base_pts_idxes = utils.farthest_point_sampling(base_pts[0:1], n_sampling=nn_base_pts)
#             base_pts_idxes = base_pts_idxes[:nn_base_pts]
#             base_pts = base_pts[:, base_pts_idxes]
#             base_normals = base_normals[:, base_pts_idxes]
            
#             base_pts_global_orient_mtx = object_global_orient_mtx_th # ws x 3 x 3 #
#             base_pts_transl = object_trcansl_th # ws x 3 # 
#             base_pts = torch.matmul(
#                 (base_pts - base_pts_transl.unsqueeze(1)), base_pts_global_orient_mtx.transpose(1, 2) # ws x nn_base_pts x 3 --> ws x nn_base_pts x 3 #
#             )
#             base_normals = torch.matmul(
#                 base_normals, base_pts_global_orient_mtx.transpose(1, 2)  # ws x nn_base_pts x 3 
#             )
            
            
            
        
        
#         rhand_joints = torch.matmul(
#             rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
#         )
        
#         pert_rhand_joints = torch.matmul(
#             pert_rhand_joints - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
#         )

#         tot_full_obj_verts = torch.from_numpy(tot_full_obj_verts).float()
#         # tot_full_obj_verts = torch.matmul(
#         #     tot_full_obj_verts - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
#         # )
        
#         if self.args.use_anchors:
#             # rhand_anchors, pert_rhand_anchors #
#             rhand_anchors = torch.matmul(
#                 rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
#             )
#             pert_rhand_anchors = torch.matmul(
#                 pert_rhand_anchors - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
#             )
        
#         if self.predicted_hand_joints is not None:
#             # self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
#             # self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
#             # self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
#             # self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
#             pert_rhand_joints = self.predicted_hand_joints
#             # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
#             if self.predicted_hand_trans is not None:
#                 rhand_transl_var = self.predicted_hand_trans
#                 rhand_global_orient_var = self.predicted_hand_rot
#                 rhand_pose_var = self.predicted_hand_theta
#                 print(f"rhand_beta_var: {self.predicted_hand_beta.size()}")
#                 rhand_beta_var = self.predicted_hand_beta #.unsqueeze(0)
        
#         ''' normalization strategy xxx --- data scaling '''
#         # base_pts = base_pts * 5.
#         # rhand_joints = rhand_joints * 5.
#         ''' Normlization stratey xxx --- data scaling '''
        
        
        
#         # ''' GET ambinet space base pts '''
#         # # normalized #
#         # # rhand_joints: nf x nnj x 3
#         # # base_pts: nnb x 3 ## --> (nf x nnj + nnb) x 3 #
#         # tot_pts = torch.cat(
#         #     [rhand_joints.view(rhand_joints.size(0) * rhand_joints.size(1), 3), base_pts], dim=0
#         # )
#         # maxx_tot_pts, _ = torch.max(tot_pts, dim=0) ## 3
#         # minn_tot_pts, _ = torch.min(tot_pts, dim=0) ## 3
#         # # uniformly sample rand
#         # # rand(0, 1)? # 0 and 1 
#         # xyz_coords_coeffs = torch.rand((nn_base_pts, 3)) ## nn_base_pts x 3 #
#         # # nn_base_pts x 3 #
#         # sampled_base_pts = minn_tot_pts.unsqueeze(0) + xyz_coords_coeffs * (maxx_tot_pts - minn_tot_pts).unsqueeze(0)
#         # # to object_pc
#         # # nn_base_pts x nn_obj_pc # ---> nf x nnb x nn_obj
#         # dist_sampled_base_pts_to_obj_pc = torch.sum(
#         #     (sampled_base_pts.unsqueeze(0).unsqueeze(2) - object_pc_th.unsqueeze(1)) ** 2, dim=-1
#         # )
#         # # nf x nnb; nf x nnb #
#         # minn_dist_sampled_base_to_obj, minn_dist_idxes = torch.min(
#         #     dist_sampled_base_pts_to_obj_pc, dim=-1
#         # )
        
#         # ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
#         # ## sampled_base_pts, sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns #
#         # # TODO: should compare between raw pts and raltive positions to nearest obj pts #
#         # # sampled_base...: nf x nnb x 3 #
#         # sampled_base_pts_nearest_obj_pc = utils.batched_index_select_ours(
#         #     values=object_pc_th, indices=minn_dist_idxes, dim=1
#         # )
#         # # sampled...: nf x nnb x 3 #
#         # sampled_base_pts_nearest_obj_vns = utils.batched_index_select_ours(
#         #     values=object_normal_th, indices=minn_dist_idxes, dim=1
#         # )
        
#         # # base pts single values, nearest object pc and nearest object normal #
#         ''' GET ambinet space base pts '''
        
        
#         # base_pts = sampled_base_pts
#         # sampled_base_pts = base_pts
        
#         ''' Relative positions and distances normalization, strategy 1 '''
#         # rhand_joints = rhand_joints * 5.
#         # base_pts = base_pts * 5.
#         ''' Relative positions and distances normalization, strategy 1 '''
#         # sampled_base_pts: nn_base_pts x 3 #
#         # nf x nnj x nnb x 3 #
#         # nf x nnj x nnb x 3 #
#         # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - sampled_base_pts.unsqueeze(0).unsqueeze(0)
#         # # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
#         # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
#         if not self.args.use_arti_obj:
#             # nf x nnj x nnb x 3 # 
#             rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
#             # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
#             dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
#         else:
#             rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1) # ws x nn_joints x nn_base_pts x 3 #
#             # dist_base_pts_to_rhand_joints: ws x nn_joints x nn_base_pts -> the distance from base points to joint points #
#             dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
        
#         # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
        
#         # # nf x nnj x nnb x 3 # 
#         # rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
#         # # rel_base_pts_to_rhand_joints = rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
        
#         # # # dist_base_pts_to...: ws x nn_joints x nn_sampling # ### dit bae tps to rhand joints ###
#         # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
        
        
#         # k of the # # nf x nnj x nnb # # nnj x nnb # nnb -> 
#         ## TODO: other choices of k_f? ##
#         k_f = 1.
#         # relative #
#         l2_rel_base_pts_to_rhand_joints = torch.norm(rel_base_pts_to_rhand_joints, dim=-1)
#         ### att_forces ##
#         att_forces = torch.exp(-k_f * l2_rel_base_pts_to_rhand_joints) # nf x nnj x nnb #
        
#         att_forces = att_forces[:-1, :, :]
#         # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        
        
#         rhand_joints_disp = pert_rhand_joints[1:, :, :] - pert_rhand_joints[:-1, :, :]
        
        
#         # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
#         # 
#         if not self.args.use_arti_obj:
#             # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
#             signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
#                 base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
#             )
            
#             # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
#             rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
#         else:
#             # base normals and normals #
#             signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
#                 base_normals[:-1].unsqueeze(1) * rhand_joints_disp.unsqueeze(2), dim=-1
#             )
#             # unsqueeze the dimensiton 1 #
#             rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals[:-1].unsqueeze(1)
#         # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
#         # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
        
#         # # rhand_joints_disp = rhand_joints[1:, :, :] - rhand_joints[:-1, :, :]
#         # # 
#         # # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 -
#         # signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
#         #     base_normals.unsqueeze(0).unsqueeze(0) * rhand_joints_disp.unsqueeze(2), dim=-1
#         # )
#         # # nf x nnj x nnb x 3 --> rel_vt_normals ## nf x nnj x nnb
#         # # # (ws - 1) x nnj x nnb # # (ws - 1) x nnj x 3 --> 
#         # # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
#         # rel_base_pts_to_rhand_joints_vt_normal = rhand_joints_disp.unsqueeze(2) - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(0).unsqueeze(0)
#         # nf x nnj x nnb ---> dist_vt_normals -> nf x nnj x nnb # # torch.sqrt() ##
#         dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
#             rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
#         ))
        
#         k_a = 1.
#         k_b = 1.
#         # k and # give me a noised sequence ... #
#         # (ws - 1) x nnj x nnb # --> (ws - 1) x nnj x nnb # nnj x nnb # 
#         # add noise -> chagne of the joints displacements 
#         # -> change of along_normalss energies and vertical to normals energies #
#         # -> change of energy taken to make the displacements #
#         # jts_to_base_pts energy in the noisy sequence #
#         # jts_to_base_pts energy in the clean sequence #
#         # vt-normal, along_normal #
#         # TODO: the normalization strategy: 1) per-instnace; 2) per-category; #3
#         # att_forces: (ws - 1) x nnj x nnb # # 
#         e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
#         # (ws - 1) x nnj x nnb # -> dist vt normals #
#         e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
#         # base_pts; base_normals; 
        
        
#         ''' normalization sstrategy 1 ''' # 
#         # per_frame_avg_disp_along_normals, per_frame_std_disp_along_normals # 
#         # per_frame_avg_disp_vt_normals, per_frame_std_disp_vt_normals #
#         # e_disp_rel_to_base_along_normals, e_disp_rel_to_baes_vt_normals #
#         # per_frame_avg_disp_along_normalss, per_frame_std_disp_along_normalss # 
#         # rel_base_pts_to_rhand_joints_vt_normal -> disp_ws x nnj x nnb x 3 #
#         disp_ws, nnj, nnb = e_disp_rel_to_base_along_normals.shape[:3]
#         # disp_ws x nnf x nnb x 3 #  -> disp_ws x nnj x nnb
#         per_frame_avg_disp_along_normals = torch.mean( # avg over all frmaes #
#             e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
#         ) # .unsqueeze(0)
#         per_frame_std_disp_along_normals = torch.std( # std over all frames #
#             e_disp_rel_to_base_along_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
#         ) # .unsqueeze(0)
#         per_frame_avg_disp_vt_normals = torch.mean( # avg over all frmaes #
#             e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True # for each point #
#         ) # .unsqueeze(0)
#         per_frame_std_disp_vt_normals = torch.std( # std over all frames #
#             e_disp_rel_to_baes_vt_normals.view(disp_ws, nnj, nnb), dim=0, keepdim=True
#         ) # .unsqueeze(0)
#         # per_frame_avg_joints_dists_rel = torch.mean(
#         #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
#         # ).unsqueeze(0)
#         # per_frame_std_joints_dists_rel = torch.std(
#         #     dist_base_pts_to_rhand_joints.view(ws * nnf, nnb), dim=0, keepdim=True
#         # ).unsqueeze(0)
#         ### normalizaed aong normals and vat normals  # ws x nnj x nnb 
#         e_disp_rel_to_base_along_normals = (e_disp_rel_to_base_along_normals - per_frame_avg_disp_along_normals) / per_frame_std_disp_along_normals
#         e_disp_rel_to_baes_vt_normals = (e_disp_rel_to_baes_vt_normals - per_frame_avg_disp_vt_normals) / per_frame_std_disp_vt_normals
#         # enrgy temrs #
#         ''' normalization sstrategy 1 ''' # 
        
        
        
        
        
#         if self.denoising_stra == "rep":
#             ''' Relative positions and distances normalization, strategy 3 '''
#             # # for each point normalize joints over all frames #
#             # # rel_base_pts_to_rhand_joints: nf x nnj x nnb x 3 #
#             per_frame_avg_joints_rel = torch.mean(
#                 rel_base_pts_to_rhand_joints, dim=0, keepdim=True
#             )
#             per_frame_std_joints_rel = torch.std(
#                 rel_base_pts_to_rhand_joints, dim=0, keepdim=True
#             )
#             per_frame_avg_joints_dists_rel = torch.mean(
#                 dist_base_pts_to_rhand_joints, dim=0, keepdim=True
#             )
#             per_frame_std_joints_dists_rel = torch.std(
#                 dist_base_pts_to_rhand_joints, dim=0, keepdim=True
#             )
#             # max xyz vlaues for the relative positions, maximum, minimum distances for them #
            
#             if not self.args.use_arti_obj:
#                 # # nf x nnj x nnb x 3 # 
#                 rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
#                 # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
#                 dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
#             else:
#                 # # nf x nnj x nnb x 3 # 
#                 rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(1)
#                 # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
#                 dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(1) * rel_base_pts_to_rhand_joints, dim=-1)
                
                
#             # # nf x nnj x nnb x 3 # 
#             # rel_base_pts_to_rhand_joints = pert_rhand_joints.unsqueeze(2) - base_pts.unsqueeze(0).unsqueeze(0)
#             # # # dist_base_pts_to...: ws x nn_joints x nn_sampling #
#             # dist_base_pts_to_rhand_joints = torch.sum(base_normals.unsqueeze(0).unsqueeze(0) * rel_base_pts_to_rhand_joints, dim=-1)
            
#             rel_base_pts_to_rhand_joints = (rel_base_pts_to_rhand_joints - per_frame_avg_joints_rel) / per_frame_std_joints_rel
#             dist_base_pts_to_rhand_joints = (dist_base_pts_to_rhand_joints - per_frame_avg_joints_dists_rel) / per_frame_std_joints_dists_rel
#             stats_dict = {
#                 'per_frame_avg_joints_rel': per_frame_avg_joints_rel,
#                 'per_frame_std_joints_rel': per_frame_std_joints_rel,
#                 'per_frame_avg_joints_dists_rel': per_frame_avg_joints_dists_rel,
#                 'per_frame_std_joints_dists_rel': per_frame_std_joints_dists_rel,
#             }
#             ''' Relative positions and distances normalization, strategy 3 '''
        
#         if self.denoising_stra == "motion_to_rep": # motion_to_rep #
#             pert_rhand_joints = (pert_rhand_joints - self.avg_jts) / self.std_jts
        
        
#         ''' Relative positions and distances normalization, strategy 4 '''
#         # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
#         # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
#         ''' Relative positions and distances normalization, strategy 4 '''
        
#         ''' Obj data '''
#         obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data()
#         obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
#         obj_normals = torch.from_numpy(obj_normals).float() # 
#         obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
#         ''' Obj data '''
        
#         object_id = 0
#         caption = "apple"
#         # pose_one_hots, word_embeddings #
        
#         # object_global_orient_th, object_transl_th #
#         object_global_orient_th = torch.from_numpy(object_global_orient).float()
#         object_transl_th = torch.from_numpy(object_transl).float()
        
#         # should centralize via the moving part # # centralize via the moving part
#         # rhand_global_orient_var, rhand_pose_var, rhand_transl_var, rhand_beta_var #
#         ''' Construct data for returning '''
#         rt_dict = {
#             'base_pts': base_pts, # th
#             'base_normals': base_normals, # th
#             'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
#             'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
#             # 'rhand_joints': rhand_joints,
#             'gt_rhand_joints': rhand_joints, ## rhand joints ###
#             'rhand_joints': pert_rhand_joints, #  if not self.args.use_canon_joints else canon_pert_rhand_joints,
#             'rhand_verts': rhand_verts,
#             # rhand_transl_var, rhand_global_orient_var, rhand_pose_var, rhand_beta_var
#             'rhand_transl': rhand_transl_var, # nf x 3 for rhand transl #
#             'rhand_rot': rhand_global_orient_var, # nf x 3 for rhand global orientation # 
#             'rhand_theta': rhand_pose_var, # nf x 24 for rhand_pose; 
#             'rhand_betas': rhand_beta_var[0],
#             # 'word_embeddings': word_embeddings,
#             # 'pos_one_hots': pos_one_hots,
#             'caption': caption,
#             # 'sent_len': sent_len,
#             # 'm_length': m_length,
#             # 'text': '_'.join(tokens),
#             'object_id': object_id, # int value
#             'lengths': rel_base_pts_to_rhand_joints.size(0),
#             'object_global_orient': object_global_orient_th,
#             'object_transl': object_transl_th,
#             # 'st_idx': start_idx,
#             # 'ed_idx': start_idx + self.window_size,
#             'st_idx': 0,
#             'ed_idx': 0 + self.window_size,
#             'pert_verts': pert_rhand_verts,
#             'verts': rhand_verts,
#             'obj_verts': obj_verts,
#             'obj_normals': obj_normals,
#             'obj_faces': obj_faces, # nnfaces x 3 #
#             'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
#             'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
#             ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
#             # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
#             # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
#             'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
#             'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
#             'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
#             'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
#             'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
#             'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
#             # # obj_verts, obj_faces
#             # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
#             # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
#         }
#         # obj_verts, obj_faces
#         if len(self.corr_fn) > 0:
#             rt_dict.update(
#                 {
#                     # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
#                     # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
#                     'obj_verts': tot_full_obj_verts, # nn_verts x 3 #
#                     'obj_faces': torch.from_numpy(tot_full_obj_faces).long(), # nn_faces x 3 
#                 }
#                 # tot_full_obj_verts, tot_full_obj_faces
#             )
#         if self.args.select_part_idx != -1:
#             rt_dict.update(
#                 {
#                     # 'obj_verts': torch.from_numpy(cur_obj_verts).float(), # nn_verts x 3 #
#                     # 'obj_faces': torch.from_numpy(cur_obj_faces).long(), # nn_faces x 3 
#                     'obj_verts': tot_full_obj_verts, # nn_verts x 3 #
#                     'obj_faces': torch.from_numpy(tot_full_obj_faces).long(), # nn_faces x 3 
#                 }
#                 # tot_full_obj_verts, tot_full_obj_faces
#             )
        
#         if self.use_anchors: ## update rhand anchors here ##
#             rt_dict.update(
#                 {
#                     'rhand_anchors': rhand_anchors,
#                     'pert_rhand_anchors': pert_rhand_anchors,
#                 }
#             )
        
#         try:
#             # rt_dict['per_frame_avg_joints_rel'] = 
#             rt_dict.update(stats_dict)
#         except:
#             pass
#         ''' Construct data for returning '''
        
#         return rt_dict


#         # if self.dir_stra == 'rot_angles':
#         #     # tangent_orient_vec # nn_base_pts x 3 #
#         #     rt_dict['base_tangent_orient_vec'] = tangent_orient_vec.numpy() #
        
#         rt_dict_th = {
#             k: torch.from_numpy(rt_dict[k]).float() if not isinstance(rt_dict[k], torch.Tensor) else rt_dict[k] for k in rt_dict 
#         }
#         # rt_dict

#         return rt_dict_th
#         # return np.concatenate([window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, rel_pos_object_pc_joint_gt, dec_cond, rhand_feats_exp], axis=2)

#     def __len__(self):
#         cur_len = self.len // self.step_size
#         if cur_len * self.step_size < self.len:
#           cur_len += 1
#         cur_len = 1
#         return cur_len
#         # return ceil(self.len / self.step_size)
#         # return self.len


# ARCTIC #
class GRAB_Dataset_V19_Arctic(torch.utils.data.Dataset): # GRAB datasset #
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None): # 
        #### GRAB dataset #### GRAB dataset ##
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        
        split = args.single_seq_path.split("/")[-2].split("_")[0]
        self.split = split
        print(f"split: {self.split}")
        
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.args = args
        
        self.denoising_stra = args.denoising_stra ## denoising_stra!
        
        self.seq_path = args.single_seq_path ## single seq path ##
        
        self.inst_normalization = args.inst_normalization
        
        
        ### for starting idxes ###
        # self.start_idx = args.start_idx # clip starting idxes #
        self.start_idx = self.args.start_idx
        
        # # load datas # grab path; grab sequences #
        # grab_path =  "/data1/xueyi/GRAB_extracted"
        # ## grab contactmesh ## id2objmeshname
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
        
        
        ## predicted infos fn ##
        self.data_folder = data_folder
        self.subj_data_folder = data_folder + "_wsubj"
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "manopth/mano/models" ### mano_path
        ## mano paths ##
        self.aug = True
        self.use_anchors = False
        # self.args = args
        
        self.use_anchors = args.use_anchors
        
        self.grab_path = "/data1/xueyi/GRAB_extracted"
        obj_mesh_path = "data/grab/object_meshes"
        id2objmesh = []
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        
        ## obj root folder; ##
        ### Load field data from root folders ###
        self.obj_root_folder = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes_params" # # and base points 
        
        self.load_meta = True
        
        self.dist_threshold = 0.005
        self.dist_threshold = 0.01
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
        print(f"nn_base_pts: {self.nn_base_pts}")
        
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
        self.obj_root_folder = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes_objs"
        self.obj_params_folder = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes_params"
        
        
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
        
        files_clean = [self.seq_path]
        
        
        for i_f, f in enumerate(files_clean):
            cur_frame = np.load(f, allow_pickle=True).item()
            self.clips.append(cur_frame)

    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx, f=None):
        if f is None:
          cur_clip = self.clips[clip_idx]
          if len(cur_clip) > 3:
              return cur_clip
          f = cur_clip[2]
        clip_clean = np.load(f)
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
        # rhand_pose = clip_clean['f2'] ## rhand pose ##
        
        object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
        object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
        object_idx = clip_clean['f7'][0].item()
        
        pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        ##### load subj params #####
        
        # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        ### pert and clean pair for encoding and decoding ###
        
        # maxx_clip_len = 
        loaded_clip = (
            0, rhand_transl.shape[0], clip_clean,
            [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
        )
        # self.clips[clip_idx] = loaded_clip
        
        return loaded_clip
        
        # self.clips.append((self.len, self.len+clip_len, clip_pert,
        #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #     # subj_corr_data, pert_subj_corr_data
        #     ))
        
    # def get_idx_to_mesh_data(self, obj_id):
    #     if obj_id not in self.id2meshdata:
    #         obj_nm = self.id2objmesh[obj_id]
    #         obj_mesh = trimesh.load(obj_nm, process=False) # obj mesh obj verts 
    #         obj_verts = np.array(obj_mesh.vertices)
    #         obj_vertex_normals = np.array(obj_mesh.vertex_normals)
    #         obj_faces = np.array(obj_mesh.faces)
    #         self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
    #     return self.id2meshdata[obj_id]

    #### enforce correct contacts #### ### the sequence in the clip is what we want here #
    def __getitem__(self, index): # get item; articulated objects? #
        ## GRAB single frame ##
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        i_c = 0
        # if self.load_meta:
        #     # self.load_clip_data(i_c)
        c = self.clips[i_c]
        # c = self.load_clip_data(i_c)

        # object_id = c[3][-1]
        # object_name = self.id2objmeshname[object_id]
        
        #  self.start_idx = args.start_idx
        # start_idx = 0  # 
        start_idx = self.args.start_idx # start idx #
        # start_idx = index * self.step_size
        # if start_idx + self.window_size > self.len:
        #     start_idx = self.len - self.window_size
        
        # and crop data sequences here ### #
        # TODO: add random noise settings for noisy input #
        
        # start_idx = (index - c[0]) * self.step_size
        print(f"start_idx: {start_idx}, window_size: {self.window_size}")
        # data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        # object_global_orient = data['f5'] ### get object global orientations ###
        # object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        object_global_orient = c["obj_rot"] # num_frames x 3 
        object_transl = c["obj_trans"] # num_frames x 3
        
        print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        # object_global_orient, object_transl #
        object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        object_transl = object_transl[start_idx: start_idx + self.window_size]
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        object_pc_tmp = c["verts.object"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        object_transl = np.mean(object_pc_tmp, axis=1)
        
        object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        
        # object_global_orient_mtx_th = torch.eye(3).float().unsqueeze(0).repeat(object_global_orient_mtx_th.size(0), 1, 1).contiguous()
        
        object_trcansl_th = torch.from_numpy(object_transl).float()
        # object_trcansl_th = torch.zeros_like(object_trcansl_th)
        # pert_subj_params = c[4]
        
        # st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx ##
        
        if self.args.use_left:
            rhand_global_orient_gt, rhand_pose_gt = c["rot_l"], c["pose_l"]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
            rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
            rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
            
            rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
            rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
            
            rhand_transl, rhand_betas = c["trans_l"], c["shape_l"][0]
            rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
            
            # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
            rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
            rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        else:
            ### pts gt ###
            ## rhnad pose, rhand pose gt ##
            ## glboal orientation and hand pose #
            rhand_global_orient_gt, rhand_pose_gt = c["rot_r"], c["pose_r"]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
            rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
            rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
            
            rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
            rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
            
            rhand_transl, rhand_betas = c["trans_r"], c["shape_r"][0]
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
        
        ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
        ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
        #### ==== get random augmented pose, rot, transl ==== ####
        # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
        aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        aug_trans, aug_rot, aug_pose = 0.001, 0.05, 0.3
        aug_trans, aug_rot, aug_pose = 0.000, 0.05, 0.3
        aug_trans, aug_rot, aug_pose = 0.000, 0.00, 0.00
        # noise scale #
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3 # scale 1 for the standard scale
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.4 ### scale 3 for the standard scale ###
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.5
        # cur_t = self.uinform_sample_t()
        # # aug_trans, aug_rot, aug_pose #
        # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
        # ### === get and save noise vectors === ###
        # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
        aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        if self.args.pert_type == "uniform":
            aug_pose_var = (torch.rand_like(rhand_pose_var) - 0.5) * aug_pose
            aug_global_orient_var = (torch.rand_like(rhand_global_orient_var) - 0.5) * aug_rot
        elif self.args.pert_type == "beta":
            dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
            print(f"here!")
            aug_pose_var = dist_beta.sample(rhand_pose_var.size()).squeeze(-1) * aug_pose
            aug_global_orient_var = dist_beta.sample(rhand_global_orient_var.size()).squeeze(-1) * aug_rot
            print(f"aug_pose_var: {aug_pose_var.size()}, aug_global_orient_var: {aug_global_orient_var.size()}")
            
        # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # ### creat augmneted orientations, pose, and transl ###
        rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        if self.args.use_left:
            cur_mano_layer = self.lft_mano_layer
        else:
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
        
        
        # if we just set a parameter `use_arti_obj`? #
        
        # if not self.args.use_arti_obj:
        #     object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        #     nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # object pc #
        #     # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        #     dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
        #         (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
        #     ) 
        #     dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs # nearest to all pts in all frames ## 
        #     dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs # nn_obj_pcs # nn_obj_pcs #
        #     # # dist_threshold = 0.01 # threshold 
        #     dist_threshold = self.dist_threshold
        #     # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
        #     dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
        #     # # base_pts_mask: nn_obj_pcs #
        #     base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
        #     # # nn_base_pts x 3 -> torch tensor #
        #     base_pts = object_pc_th[0][base_pts_mask]
        #     # # base_pts_bf_sampling = base_pts.clone()
        #     base_normals = object_normal_th[0][base_pts_mask]
            
        #     nn_base_pts = self.nn_base_pts
        #     base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
        #     base_pts_idxes = base_pts_idxes[:nn_base_pts]
            
        #     # ### get base points ### # base_pts and base_normals #
        #     base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
        #     base_normals = base_normals[base_pts_idxes]
            
            
        #     # # object_global_orient_mtx # nn_ws x 3 x 3 #
        #     base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
        #     base_pts_transl = object_trcansl_th[0] # 3
            
        #     base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
        #         ) # .transpose(0, 1)
        #     base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
        #         ) # .transpose(0, 1)
        # else:
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
        
        # (object)
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
            
            
            # ''' Stra 2 -> per frame with joints '''
            # # nf x nnj x nnb x 3 #
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
        
        
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        # 
        # rt_pert_rhand_verts =  pert_rhand_verts
        # rt_rhand_verts = rhand_verts
        # rt_pert_rhand_joints = pert_rhand_joints
        
        # rt_rhand_joints = rhand_joints ## rhand_joints ##
        # # rt_rhand_joints = pert_rhand_joints
        
        
        # # rt_rhand_joints: nf x nnjts x 3 # ### pertrhandjoints
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1s
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
        
        ''' Obj data '''
        # obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id) # obj_verts, normals #
        # obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
        # obj_normals = torch.from_numpy(obj_normals).float() # 
        # obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        
        # rt_rhand_joints: nf x nnjts x 3 #
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
            
            
        
        caption = "apple"
        # pose_one_hots, word_embeddings #
        
        # object_global_orient_th, object_transl_th #
        object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        
        # pert_rhand_anchors, rhand_anchors
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            # 'rhand_joints': rhand_joints,
            'gt_rhand_joints': rhand_joints, ## rhand joints ###
            'rhand_joints': pert_rhand_joints if not self.args.use_canon_joints else canon_pert_rhand_joints, # rhand_joints #
            'rhand_verts': rhand_verts,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            # 'object_id': object_id, # int value
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'object_global_orient': object_global_orient_th,
            'object_transl': object_transl_th,
            'st_idx': 0,
            'ed_idx': self.window_size,
            'pert_verts': pert_rhand_verts,
            'verts': rhand_verts,
            # 'obj_verts': obj_verts,
            # 'obj_normals': obj_normals,
            # 'obj_faces': obj_faces, # nnfaces x 3 #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            'object_pc_th': object_pc_th, ### get the object_pc_th for object_pc_th 
            ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
        }
        
        if self.use_anchors: ## update rhand anchors here ##
            rt_dict.update(
                {
                    'rhand_anchors': rhand_anchors,
                    'pert_rhand_anchors': pert_rhand_anchors,
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] = 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict


        if self.dir_stra == 'rot_angles':
            # tangent_orient_vec # nn_base_pts x 3 #
            rt_dict['base_tangent_orient_vec'] = tangent_orient_vec.numpy() #
        
        rt_dict_th = {
            k: torch.from_numpy(rt_dict[k]).float() if not isinstance(rt_dict[k], torch.Tensor) else rt_dict[k] for k in rt_dict 
        }
        # rt_dict

        return rt_dict_th
        # return np.concatenate([window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, rel_pos_object_pc_joint_gt, dec_cond, rhand_feats_exp], axis=2)

    def __len__(self):
        cur_len = self.len // self.step_size
        if cur_len * self.step_size < self.len:
          cur_len += 1
        cur_len = 1
        return cur_len
        # return ceil(self.len / self.step_size)
        # return self.len




# TACO #
class GRAB_Dataset_V19_HHO(torch.utils.data.Dataset): # GRAB datasset #
    def __init__(self, data_folder, split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None):
        #### GRAB dataset ####
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split
        
        split = args.single_seq_path.split("/")[-2].split("_")[0]
        self.split = split
        print(f"split: {self.split}")
        
        self.model_type = 'v1_wsubj_wjointsv25'
        self.debug = False
        # self.use_ambient_base_pts = args.use_ambient_base_pts
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3
        self.num_sche_steps = 100
        self.w_vectorizer = w_vectorizer
        self.use_pert = True
        self.use_rnd_aug_hand = True
        
        self.args = args
        
        self.denoising_stra = args.denoising_stra ## denoising_stra!
        
        self.seq_path = args.single_seq_path ## single seq path ##
        
        self.inst_normalization = args.inst_normalization
        
        
        ### for starting idxes ###
        # self.start_idx = args.start_idx # clip starting idxes #
        self.start_idx = self.args.start_idx
        
        # load datas # grab path; grab sequences #
        obj_mesh_path = "data/grab/object_meshes"
        ## grab contactmesh ## id2objmeshname
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
        
        
        ## predicted infos fn ##
        self.data_folder = data_folder
        self.subj_data_folder = data_folder + "_wsubj"
        # self.subj_corr_data_folder = args.subj_corr_data_folder
        self.mano_path = "manopth/mano/models" ### mano_path
        ## mano paths ##
        self.aug = True
        self.use_anchors = False
        # self.args = args
        
        self.use_anchors = args.use_anchors
        
        
        obj_mesh_path = "data/grab/object_meshes"
        id2objmesh = []
        obj_meshes = sorted(os.listdir(obj_mesh_path))
        for i, fn in enumerate(obj_meshes):
            id2objmesh.append(os.path.join(obj_mesh_path, fn))
        self.id2objmesh = id2objmesh
        self.id2meshdata = {}
        
        
        self.load_meta = True
        
        self.dist_threshold = 0.005
        self.dist_threshold = 0.01
        # self.nn_base_pts = 700
        self.nn_base_pts = args.nn_base_pts
        print(f"nn_base_pts: {self.nn_base_pts}")
        
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
        
        # self.template_joints = self.template_verts[self.hand_palm_vertex_mask]
        
        ''' Get predicted infos '''
        predicted_info_fn = args.predicted_info_fn
        # load data from predicted information #
        if len(predicted_info_fn) > 0:
            print(f"Loading preidcted info from {predicted_info_fn}")
            data = np.load(predicted_info_fn, allow_pickle=True).item()
            outputs = data['outputs'] # 
            self.predicted_hand_joints = outputs # nf x nnjoints x 3 #
            self.predicted_hand_joints = torch.from_numpy(self.predicted_hand_joints).float()
            
            # if 'rhand_trans' in data:
            #     # outputs = data['outputs']
            #     self.predicted_hand_trans = data['rhand_trans'] # nframes x 3 
            #     self.predicted_hand_rot = data['rhand_rot'] # nframes x 3 
            #     self.predicted_hand_theta = data['rhand_theta']
            #     self.predicted_hand_beta = data['rhand_beta']
            #     self.predicted_hand_trans = torch.from_numpy(self.predicted_hand_trans).float() # nframes x 3 
            #     self.predicted_hand_rot = torch.from_numpy(self.predicted_hand_rot).float() # nframes x 3 
            #     self.predicted_hand_theta = torch.from_numpy(self.predicted_hand_theta).float() # nframes x 24 
            #     self.predicted_hand_beta = torch.from_numpy(self.predicted_hand_beta).float() # 10,
                
            #     self.predicted_hand_trans_opt = data_opt['rhand_trans'] # nframes x 3 
            #     self.predicted_hand_rot_opt = data_opt['rhand_rot'] # nframes x 3 
            #     self.predicted_hand_theta_opt = data_opt['rhand_theta']
            #     self.predicted_hand_beta_opt = data_opt['rhand_beta']
            #     self.predicted_hand_trans_opt = torch.from_numpy(self.predicted_hand_trans_opt).float() # nframes x 3 
            #     self.predicted_hand_rot_opt = torch.from_numpy(self.predicted_hand_rot_opt).float() # nframes x 3 
            #     self.predicted_hand_theta_opt = torch.from_numpy(self.predicted_hand_theta_opt).float() # nframes x 24 
            #     self.predicted_hand_beta_opt = torch.from_numpy(self.predicted_hand_beta_opt).float() # 10,
                
            #     self.predicted_hand_trans[9:] = self.predicted_hand_trans_opt[9:]
            #     self.predicted_hand_rot[9:] = self.predicted_hand_rot_opt[9:]
            #     self.predicted_hand_theta[ 9:] = self.predicted_hand_theta_opt[ 9:]
            #     # self.predicted_hand_trans[:, 9:] = self.predicted_hand_trans_opt[:, 9:]
                
            # else:
            #     self.predicted_hand_trans = None
            #     self.predicted_hand_rot = None
            #     self.predicted_hand_theta = None
            #     self.predicted_hand_beta = None
            
        else:
            self.predicted_hand_joints = None
        
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
            flat_hand_mean=True,
            side='left',
            mano_root=self.mano_path, # mano_root #
            ncomps=45,
            use_pca=False,
            # center_idx=0
            # root_rot_mode='axisang',
            # joint_rot_mode='axisang'
        )
        
        
        
        # # anchor_load_driver, masking_load_driver #
        # # use_anchors, self.hand_palm_vertex_mask #
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
        
        files_clean = [self.seq_path]
        
        
        for i_f, f in enumerate(files_clean):
            # cur_frame = np.load(f, allow_pickle=True).item()
            # self.clips.append(cur_frame)
            
            clip_data = pkl.load(open(f, 'rb'))
            self.clips.append(clip_data)

    
    def uinform_sample_t(self):
        t = np.random.choice(np.arange(0, self.sigmas_trans.shape[0]), 1).item()
        return t
    
    def load_clip_data(self, clip_idx, f=None):
        
        clip_data = pkl.load(open(f, 'rb'))
        # dict_keys(['hand_pose', 'hand_trans', 'hand_shape', 'hand_verts', 'hand_faces', 'obj_verts', 'obj_faces', 'obj_pose'])
        # (288, 48)
        # (288, 3)
        # (10,)
        # (288, 778, 3)
        # (1538, 3)
        # (288, 1000, 3)
        # (2000, 3)
        # (288, 4, 4)
        return clip_data
        
        # if f is None:
        #   cur_clip = self.clips[clip_idx]
        #   if len(cur_clip) > 3:
        #       return cur_clip
        #   f = cur_clip[2]
        # clip_clean = np.load(f)
        # # pert_folder_nm = self.split + '_pert'
        # pert_folder_nm = self.split
        # # if not self.use_pert:
        # #     pert_folder_nm = self.split
        # # clip_pert = np.load(os.path.join(self.data_folder, pert_folder_nm, os.path.basename(f)))
        
        
        # ##### load subj params #####
        # pure_file_name = f.split("/")[-1].split(".")[0]
        # pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
                
        # subj_params_fn = os.path.join(self.subj_data_folder, self.split, pure_subj_params_fn)
        # subj_params = np.load(subj_params_fn, allow_pickle=True).item()
        # rhand_transl = subj_params["rhand_transl"]
        # rhand_betas = subj_params["rhand_betas"]
        # # rhand_pose = clip_clean['f2'] ## rhand pose ##
        
        # object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
        # object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
        
        # object_idx = clip_clean['f7'][0].item()
        
        # pert_subj_params_fn = os.path.join(self.subj_data_folder, pert_folder_nm, pure_subj_params_fn)
        # pert_subj_params = np.load(pert_subj_params_fn, allow_pickle=True).item()
        # ##### load subj params #####
        
        # # meta data -> lenght of the current clip  -> construct meta data from those saved meta data -> load file on the fly # clip file name -> yes...
        # # print(f"rhand_transl: {rhand_transl.shape},rhand_betas: {rhand_betas.shape}, rhand_pose: {rhand_pose.shape} ")
        # ### pert and clean pair for encoding and decoding ###
        
        # # maxx_clip_len = 
        # loaded_clip = (
        #     0, rhand_transl.shape[0], clip_clean,
        #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx], pert_subj_params, 
        # )
        # # self.clips[clip_idx] = loaded_clip
        
        # return loaded_clip
        
        # self.clips.append((self.len, self.len+clip_len, clip_pert,
        #     [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas], pert_subj_params, 
        #     # subj_corr_data, pert_subj_corr_data
        #     ))
        
    
        
    # def get_idx_to_mesh_data(self, obj_id):
    #     if obj_id not in self.id2meshdata:
    #         obj_nm = self.id2objmesh[obj_id]
    #         obj_mesh = trimesh.load(obj_nm, process=False) # obj mesh obj verts 
    #         obj_verts = np.array(obj_mesh.vertices)
    #         obj_vertex_normals = np.array(obj_mesh.vertex_normals)
    #         obj_faces = np.array(obj_mesh.faces)
    #         self.id2meshdata[obj_id] = (obj_verts, obj_vertex_normals, obj_faces)
    #     return self.id2meshdata[obj_id]

    #### enforce correct contacts #### ### the sequence in the clip is what we want here #
    def __getitem__(self, index): # get item; articulated objects? #
        ## GRAB single frame ##
        # for i_c, c in enumerate(self.clips):
        #     if index < c[1]:
        #         break
        i_c = 0
        # if self.load_meta:
        #     # self.load_clip_data(i_c)
        c = self.clips[i_c]
        # c = self.load_clip_data(i_c)

        # object_id = c[3][-1]
        # object_name = self.id2objmeshname[object_id]
        
        #  self.start_idx = args.start_idx
        # start_idx = 0  # 
        start_idx = self.args.start_idx
        # start_idx = index * self.step_size
        # if start_idx + self.window_size > self.len:
        #     start_idx = self.len - self.window_size
        
        # and crop data sequences here ### 
        # TODO: add random noise settings for noisy input #
        
        # start_idx = (index - c[0]) * self.step_size
        print(f"start_idx: {start_idx}, window_size: {self.window_size}")
        
        self.window_size = min(self.window_size, c['obj_pose'].shape[0] - start_idx)
        
        # data = c[2][start_idx:start_idx+self.window_size]
        # # object_global_orient = self.data[index]['f5']
        # # object_transl = self.data[index]['f6'] #
        # object_global_orient = data['f5'] ### get object global orientations ###
        # object_trcansl = data['f6']
        # # object_id = data['f7'][0].item() ### data_f7 item ###
        # ## two variants: 1) canonicalized joints; 2) parameters directly; ##
        
        object_pose = c['obj_pose']
        
        object_pose = object_pose[start_idx: start_idx + self.window_size]
        
        object_transl = object_pose[:, :3, 3]
        
        # object_global_orient = c["obj_rot"] # num_frames x 3 
        # object_transl = c["obj_trans"] # num_frames x 3
        
        
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        # object_global_orient, object_transl #
        # object_global_orient = object_global_orient[start_idx: start_idx + self.window_size]
        # object_transl = object_transl[start_idx: start_idx + self.window_size]
        
        # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        object_transl = object_transl.reshape(self.window_size, -1).astype(np.float32)
        
        
        # object_global_orient = object_global_orient.reshape(self.window_size, -1).astype(np.float32)
        # object_trcansl = object_trcansl.reshape(self.window_size, -1).astype(np.float32)
        object_pc_tmp = c["obj_verts"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        # 
        # obje
        raw_hand_verts = c['hand_verts'][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        raw_hand_verts = torch.from_numpy(raw_hand_verts).float()
        object_transl = np.mean(object_pc_tmp, axis=1)
        
        # object_global_orient_mtx = utils.batched_get_orientation_matrices(object_global_orient)
        
        object_global_orient_mtx = object_pose[:, :3, :3]
        object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float()
        object_global_orient_mtx_th = object_global_orient_mtx_th.contiguous().transpose(2, 1)
        
        # object_global_orient_mtx_th = torch.eye(3).float().unsqueeze(0).repeat(object_global_orient_mtx_th.size(0), 1, 1).contiguous()
        
        object_trcansl_th = torch.from_numpy(object_transl).float()
        # object_trcansl_th = torch.zeros_like(object_trcansl_th)
        # pert_subj_params = c[4]
        
        # st_idx, ed_idx = start_idx, start_idx + self.window_size ## start idx and end idx ##
        
        # 'hand_pose', 'hand_trans', 'hand_shape'
        if self.args.use_left:
            rhand_global_orient_gt, rhand_pose_gt = c["hand_pose"][:, :3], c["hand_pose"][:, 3:]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
            rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
            rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
            
            rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
            rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
            
            rhand_transl, rhand_betas = c["hand_trans"], c["hand_shape"] # [0]
            rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
            
            # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
            rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
            rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        else:
            ### pts gt ###
            ## rhnad pose, rhand pose gt ##
            ## glboal orientation and hand pose #
            rhand_global_orient_gt, rhand_pose_gt = c["rot_r"], c["pose_r"]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
            rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
            print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
            rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
            
            rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
            rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
            
            rhand_transl, rhand_betas = c["trans_r"], c["shape_r"][0]
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
        
        ### rhand_global_orient_var, rhand_pose_var, rhand_transl_var ###
        ### aug_global_orient_var, aug_pose_var, aug_transl_var ###
        #### ==== get random augmented pose, rot, transl ==== ####
        # rnd_aug_global_orient_var, rnd_aug_pose_var, rnd_aug_transl_var #
        aug_trans, aug_rot, aug_pose = 0.000, 0.00, 0.00
        # noise scale #
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.3 # scale 1 for the standard scale
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.4 ### scale 3 for the standard scale ###
        # aug_trans, aug_rot, aug_pose = 0.01, 0.05, 0.5
        # cur_t = self.uinform_sample_t()
        # # aug_trans, aug_rot, aug_pose #
        # aug_trans, aug_rot, aug_pose = self.sigmas_trans[cur_t].item(), self.sigmas_rot[cur_t].item(), self.sigmas_pose[cur_t].item()
        # ### === get and save noise vectors === ###
        # ### aug_global_orient_var,  aug_pose_var, aug_transl_var ### # estimate noise # ###
        aug_global_orient_var = torch.randn_like(rhand_global_orient_var) * aug_rot ### sigma = aug_rot
        aug_pose_var =  torch.randn_like(rhand_pose_var) * aug_pose ### sigma = aug_pose
        aug_transl_var = torch.randn_like(rhand_transl_var) * aug_trans ### sigma = aug_trans
        if self.args.pert_type == "uniform":
            aug_pose_var = (torch.rand_like(rhand_pose_var) - 0.5) * aug_pose
            aug_global_orient_var = (torch.rand_like(rhand_global_orient_var) - 0.5) * aug_rot
        elif self.args.pert_type == "beta":
            dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
            print(f"here!")
            aug_pose_var = dist_beta.sample(rhand_pose_var.size()).squeeze(-1) * aug_pose
            aug_global_orient_var = dist_beta.sample(rhand_global_orient_var.size()).squeeze(-1) * aug_rot
            print(f"aug_pose_var: {aug_pose_var.size()}, aug_global_orient_var: {aug_global_orient_var.size()}")
            
        # # rnd_aug_global_orient_var = rhand_global_orient_var + torch.randn_like(rhand_global_orient_var) * aug_rot
        # # rnd_aug_pose_var = rhand_pose_var + torch.randn_like(rhand_pose_var) * aug_pose
        # # rnd_aug_transl_var = rhand_transl_var + torch.randn_like(rhand_transl_var) * aug_trans
        # ### creat augmneted orientations, pose, and transl ###
        rnd_aug_global_orient_var = rhand_global_orient_var + aug_global_orient_var
        rnd_aug_pose_var = rhand_pose_var + aug_pose_var
        rnd_aug_transl_var = rhand_transl_var + aug_transl_var ### aug transl 
        
        if self.args.use_left:
            cur_mano_layer = self.lft_mano_layer
        else:
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
        
        offset_cur_to_raw = raw_hand_verts[0, 0] - rhand_verts[0, 0]
        rhand_verts = rhand_verts + offset_cur_to_raw.unsqueeze(0)
        rhand_joints = rhand_joints + offset_cur_to_raw.unsqueeze(0)
        
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
        pert_rhand_verts = pert_rhand_verts + offset_cur_to_raw.unsqueeze(0)
        pert_rhand_joints = pert_rhand_joints + offset_cur_to_raw.unsqueeze(0)
        
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
        # object_pc = c["verts.object"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc = c["obj_verts"][start_idx: start_idx + self.window_size].reshape(self.window_size, -1, 3).astype(np.float32)

        # if self.args.scale_obj > 1:
        #     object_pc = object_pc * self.args.scale_obj
        # object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        
        pert_rhand_verts_ori = pert_rhand_verts.clone()
        
        
        object_normal = np.zeros_like(object_pc)
        object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
        
        object_pc_th_ori = object_pc_th.clone()
        # hand_ori_pc = 
        
        object_pc_th_ntrans = object_pc_th.clone()
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
        
        
        # if we just set a parameter `use_arti_obj`? #
        
        # if not self.args.use_arti_obj:
        #     object_pc_th = object_pc_th[0].unsqueeze(0).repeat(self.window_size, 1, 1).contiguous()
        #     nearest_obj_pcs = utils.batched_index_select_ours(values=object_pc_th, indices=minn_dists_joints_obj_idx, dim=1) # object pc #
        #     # # dist_object_pc_nearest_pcs: nf x nn_obj_pcs x nn_rhands
        #     dist_object_pc_nearest_pcs = torch.sum( # - nearesst obj pc # # ws x nn_obj x 1 x 3 --- ws x 1 x nnjts x 3 --> ws x nn_obj x nn_jts
        #         (object_pc_th.unsqueeze(2) - nearest_obj_pcs.unsqueeze(1)) ** 2, dim=-1 # ws x nn_obj x nn_jts #
        #     ) 
        #     dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=-1) # nf x nn_obj_pcs # nearest to all pts in all frames ## 
        #     dist_object_pc_nearest_pcs, _ = torch.min(dist_object_pc_nearest_pcs, dim=0) # nn_obj_pcs # nn_obj_pcs # nn_obj_pcs #
        #     # # dist_threshold = 0.01 # threshold 
        #     dist_threshold = self.dist_threshold
        #     # # dist_threshold for pc_nearest_pcs # dist object pc nearest pcs #
        #     dist_object_pc_nearest_pcs = torch.sqrt(dist_object_pc_nearest_pcs)
            
        #     # # base_pts_mask: nn_obj_pcs #
        #     base_pts_mask = (dist_object_pc_nearest_pcs <= dist_threshold)
        #     # # nn_base_pts x 3 -> torch tensor #
        #     base_pts = object_pc_th[0][base_pts_mask]
        #     # # base_pts_bf_sampling = base_pts.clone()
        #     base_normals = object_normal_th[0][base_pts_mask]
            
        #     nn_base_pts = self.nn_base_pts
        #     base_pts_idxes = utils.farthest_point_sampling(base_pts.unsqueeze(0), n_sampling=nn_base_pts)
        #     base_pts_idxes = base_pts_idxes[:nn_base_pts]
            
        #     # ### get base points ### # base_pts and base_normals #
        #     base_pts = base_pts[base_pts_idxes] # nn_base_sampling x 3 #
        #     base_normals = base_normals[base_pts_idxes]
            
            
        #     # # object_global_orient_mtx # nn_ws x 3 x 3 #
        #     base_pts_global_orient_mtx = object_global_orient_mtx_th[0] # 3 x 3
        #     base_pts_transl = object_trcansl_th[0] # 3
            
        #     base_pts =  torch.matmul((base_pts - base_pts_transl.unsqueeze(0)), base_pts_global_orient_mtx.transpose(1, 0)
        #         ) # .transpose(0, 1)
        #     base_normals = torch.matmul((base_normals), base_pts_global_orient_mtx.transpose(1, 0)
        #         ) # .transpose(0, 1)
        # else:
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
        
        rhand_verts = torch.matmul(
            rhand_verts - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
        )
        
        # raw_hand_verts
        raw_hand_verts = torch.matmul(
            raw_hand_verts - object_trcansl_th.unsqueeze(1), object_global_orient_mtx_th.transpose(1, 2)
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
        
        
        
        if self.predicted_hand_joints is not None:
            pert_rhand_joints = self.predicted_hand_joints.clone()
            rhand_joints = self.predicted_hand_joints.clone()
        
        
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
            
            
            # ''' Stra 2 -> per frame with joints '''
            # # nf x nnj x nnb x 3 #
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
        
        
        ''' Relative positions and distances normalization, strategy 4 '''
        # rel_base_pts_to_rhand_joints = rel_base_pts_to_rhand_joints / (self.maxx_rel - self.minn_rel).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dist_base_pts_to_rhand_joints = dist_base_pts_to_rhand_joints / (self.maxx_dists - self.minn_dists).unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(-1)
        ''' Relative positions and distances normalization, strategy 4 '''
        
        # 
        # rt_pert_rhand_verts =  pert_rhand_verts
        # rt_rhand_verts = rhand_verts
        # rt_pert_rhand_joints = pert_rhand_joints
        
        # rt_rhand_joints = rhand_joints ## rhand_joints ##
        # # rt_rhand_joints = pert_rhand_joints
        
        
        # # rt_rhand_joints: nf x nnjts x 3 # ### pertrhandjoints
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1s
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
        
        ''' Obj data '''
        # obj_verts, obj_normals, obj_faces = self.get_idx_to_mesh_data(object_id) # obj_verts, normals #
        # obj_verts = torch.from_numpy(obj_verts).float() # nn_verts x 3 #
        # obj_normals = torch.from_numpy(obj_normals).float() # 
        # obj_faces = torch.from_numpy(obj_faces).long() # nn_faces x 3 ## -> triangels indexes ##
        ''' Obj data '''
        
        # rt_rhand_joints: nf x nnjts x 3 #
        # exp_hand_joints = rt_rhand_joints.view(rt_rhand_joints.size(0) * rt_rhand_joints.size(1), 3).contiguous()
        # avg_joints = torch.mean(exp_hand_joints, dim=0, keepdim=True) # 1 x 3
        # # avg_joints = torch.mean(avg_joints, dim=)
        # std_joints = torch.std(exp_hand_joints.view(-1), dim=0, keepdim=True) # 1
        # if self.inst_normalization:
        #     if self.args.debug:
        #         print(f"normalizing joints using mean: {avg_joints}, std: {std_joints}")
        #     rt_rhand_joints = (rt_rhand_joints - avg_joints.unsqueeze(0)) / std_joints.unsqueeze(0).unsqueeze(0)
            
            
        
        caption = "apple"
        # pose_one_hots, word_embeddings #
        
        # object_global_orient_th, object_transl_th #
        # object_global_orient_th = torch.from_numpy(object_global_orient).float()
        object_global_orient_th = object_global_orient_mtx_th.clone()
        object_transl_th = torch.from_numpy(object_transl).float()
        
        
        # pert_rhand_anchors, rhand_anchors
        ''' Construct data for returning '''
        rt_dict = {
            'base_pts': base_pts, # th
            'base_normals': base_normals, # th
            'rel_base_pts_to_rhand_joints': rel_base_pts_to_rhand_joints, # th, ws x nnj x nnb x 3 
            'dist_base_pts_to_rhand_joints': dist_base_pts_to_rhand_joints, # th, ws x nnj x nnb
            # 'rhand_joints': rhand_joints,
            'gt_rhand_joints': rhand_joints, ## rhand joints ###
            'rhand_joints': pert_rhand_joints if not self.args.use_canon_joints else canon_pert_rhand_joints, # rhand_joints #
            'rhand_verts': raw_hand_verts,
            # 'word_embeddings': word_embeddings,
            # 'pos_one_hots': pos_one_hots,
            'caption': caption,
            # 'sent_len': sent_len,
            # 'm_length': m_length,
            # 'text': '_'.join(tokens),
            # 'object_id': object_id, # int value
            'lengths': rel_base_pts_to_rhand_joints.size(0),
            'object_global_orient': object_global_orient_th,
            'object_transl': object_transl_th,
            'st_idx': 0,
            'ed_idx': self.window_size,
            'pert_verts': raw_hand_verts,
            'verts': raw_hand_verts,
            # 'obj_verts': obj_verts,
            # 'obj_normals': obj_normals,
            # 'obj_faces': obj_faces, # nnfaces x 3 #
            'obj_rot': object_global_orient_mtx_th, # ws x 3 x 3 --> 
            'obj_transl': object_trcansl_th, # ws x 3 --> obj transl 
            'object_pc_th': object_pc_th, ### get the object_pc_th for object_pc_th 
            ## sampled_base_pts_nearest_obj_pc, sampled_base_pts_nearest_obj_vns ##
            # 'sampled_base_pts_nearest_obj_pc': sampled_base_pts_nearest_obj_pc, 
            # 'sampled_base_pts_nearest_obj_vns': sampled_base_pts_nearest_obj_vns,
            'per_frame_avg_disp_along_normals': per_frame_avg_disp_along_normals,
            'per_frame_std_disp_along_normals': per_frame_std_disp_along_normals,
            'per_frame_avg_disp_vt_normals': per_frame_avg_disp_vt_normals,
            'per_frame_std_disp_vt_normals': per_frame_std_disp_vt_normals,
            'e_disp_rel_to_base_along_normals': e_disp_rel_to_base_along_normals,
            'e_disp_rel_to_baes_vt_normals': e_disp_rel_to_baes_vt_normals, # 
        }
        
        if self.use_anchors: ## update rhand anchors here ##
            rt_dict.update(
                {
                    'rhand_anchors': rhand_anchors,
                    'pert_rhand_anchors': pert_rhand_anchors,
                }
            )
        
        try:
            # rt_dict['per_frame_avg_joints_rel'] = 
            rt_dict.update(stats_dict)
        except:
            pass
        ''' Construct data for returning '''
        
        return rt_dict


        if self.dir_stra == 'rot_angles':
            # tangent_orient_vec # nn_base_pts x 3 #
            rt_dict['base_tangent_orient_vec'] = tangent_orient_vec.numpy() #
        
        rt_dict_th = {
            k: torch.from_numpy(rt_dict[k]).float() if not isinstance(rt_dict[k], torch.Tensor) else rt_dict[k] for k in rt_dict 
        }
        # rt_dict

        return rt_dict_th
        # return np.concatenate([window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, rel_pos_object_pc_joint_gt, dec_cond, rhand_feats_exp], axis=2)

    def __len__(self):
        cur_len = self.len // self.step_size
        if cur_len * self.step_size < self.len:
          cur_len += 1
        cur_len = 1
        return cur_len

