# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_ours import TrainLoop
from data_loaders.get_data import get_dataset_loader
# from utils.model_util import create_model_and_diffusion
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_multiple_models_fr_path
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from tqdm import tqdm
from argparse import Namespace
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np

def get_args():
    args = Namespace()
    args.fps = 20
    args.model_path = './save/humanml_trans_enc_512/model000200000.pt'
    args.guidance_param = 2.5
    args.unconstrained = False
    args.dataset = 'humanml'

    args.cond_mask_prob = 1
    args.emb_trans_dec = False
    args.latent_dim = 512
    args.layers = 8
    args.arch = 'trans_enc'

    args.noise_schedule = 'cosine'
    args.sigma_small = True
    args.lambda_vel = 0.0
    args.lambda_rcxyz = 0.0
    args.lambda_fc   = 0.0
    return args


def sample_loop(model, diffusion, dataloader, device):
    for motion, cond in tqdm(dataloader): ## motion; cond; data ##
        # if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
        #     break
        # print(f"motion: {motion.size()}, ") ## motion.to(self.device)
        motion = motion.to(device)
        cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

def get_obj_sequences_from_data(data):
    tot_obj_idx = []
    tot_obj_global_orient = []
    tot_obj_global_transl = []
    for batch in data:
        obj_idx = batch['object_id']
        obj_global_orient = batch['object_global_orient']
        obj_global_transl = batch['object_transl']
        tot_obj_idx.append(obj_idx)
        tot_obj_global_orient.append(obj_global_orient)
        tot_obj_global_transl.append(obj_global_transl)
    tot_obj_idx = torch.cat(tot_obj_idx, dim=0)
    tot_obj_global_orient = torch.cat(tot_obj_global_orient, dim=0)
    tot_obj_global_transl = torch.cat(tot_obj_global_transl, dim=0)
    obj_idx = tot_obj_idx[0].item()
    return obj_idx, tot_obj_global_orient, tot_obj_global_transl

def get_base_pts_rhand_joints_from_data(data):
    
    tot_base_pts = []
    tot_rhand_joints = []
    tot_base_normals = []
    tot_gt_rhand_joints = []
    
    tot_obj_rot = []
    tot_obj_transl = []
    for batch in data:
        base_pts = batch['base_pts'] 
        rhand_joints = batch['rhand_joints']
        base_normals = batch['base_normals']
        gt_rhand_joints = batch['gt_rhand_joints']
        
        obj_rot = batch['obj_rot']
        obj_transl = batch['obj_transl']
        tot_base_pts.append(base_pts.detach().cpu())
        tot_rhand_joints.append(rhand_joints.detach().cpu())
        tot_base_normals.append(base_normals.detach().cpu())
        tot_gt_rhand_joints.append(gt_rhand_joints.detach().cpu())
        tot_obj_rot.append(obj_rot.detach().cpu())
        tot_obj_transl.append(obj_transl.detach().cpu())
        
        
    tot_base_pts = torch.cat(tot_base_pts, dim=0)
    tot_rhand_joints = torch.cat(tot_rhand_joints, dim=0)
    tot_base_normals = torch.cat(tot_base_normals, dim=0)
    tot_gt_rhand_joints = torch.cat(tot_gt_rhand_joints, dim=0)
    tot_obj_rot = torch.cat(tot_obj_rot, dim=0)
    tot_obj_transl = torch.cat(tot_obj_transl, dim=0)
    # tot_obj_idx = torch.cat(tot_obj_idx, dim=0)
    # tot_obj_global_orient = torch.cat(tot_obj_global_orient, dim=0)
    # tot_obj_global_transl = torch.cat(tot_obj_global_transl, dim=0)
    # obj_idx = tot_obj_idx[0].item()
    return tot_base_pts, tot_base_normals, tot_rhand_joints, tot_gt_rhand_joints, tot_obj_rot, tot_obj_transl


import time


def main():
  
    args = train_args()

    fixseed(args.seed)
    
    single_seq_path = args.single_seq_path
    test_seq_idx = single_seq_path.split("/")[-1].split(".")[0]
    test_seq_idx = int(test_seq_idx)
    
    ### and also add the arg -> prev test tag ###
    save_dir = args.save_dir
    

    dist_util.setup_dist(args.device)
    

    
    
    
    ##### Get st_idxes #####
    pure_file_name = single_seq_path.split("/")[-1].split(".")[0]
    split = single_seq_path.split("/")[-2]
    subj_data_folder = args.grab_processed_dir + "_wsubj"
    pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
            
    subj_params_fn = os.path.join(subj_data_folder, split, pure_subj_params_fn)
    subj_params = np.load(subj_params_fn, allow_pickle=True).item()
    rhand_transl = subj_params["rhand_transl"]
    nn_frames = rhand_transl.shape[0]
    
    nn_st_skip = 30
    num_cleaning_frames = 60
    num_ending_clearning_frames = nn_frames - num_cleaning_frames + 1

    print(range(0, num_ending_clearning_frames, nn_st_skip))
    
    st_idxes = list(range(0, num_ending_clearning_frames, nn_st_skip))
    if st_idxes[-1] + num_cleaning_frames < nn_frames:
        st_idxes.append(nn_frames - num_cleaning_frames)
    print(f"st_idxes: {st_idxes}")
    ##### Get st_idxes #####
    
    # use_reverse = args.use_reverse
    os.makedirs(args.save_dir, exist_ok=True)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    
    
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    print(f"save_dir: {save_dir}, single_seq_path: {single_seq_path}")
    
    test_seeds = range(0, 122, 11)
    # test_seeds = range(0, 1)
    
    for cur_seed in test_seeds:
    
        for st_fr in st_idxes:
            
            args.start_idx = st_fr
            
            cur_single_seq_path = single_seq_path
            args.single_seq_path = cur_single_seq_path
            print(f"cur_single_seq_path: {cur_single_seq_path}")
            
            # if not os.path.exists(cur_single_seq_path):
            #     continue
        
            args.seed = cur_seed # random seeds #


            args.predicted_info_fn = f"" 

            print(f"Current sequence path: {args.single_seq_path}, seed: {args.seed}")
            
            ## get dataest loader ##
            ### ==== DATA LOADER ==== ###  # DATA LOADER #
            print("creating data loader...") ## create model and diffusion ##
            data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, args=args)

            ### ==== CREATE MODEL AND DIFFUSION MODEL ==== ### # create model and diffusion model #
            # create model and diffusion #
            print("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(args, data)

            if ';' in args.model_path:
                print(f"Loading model with multiple activated settings from {args.model_path}")
                load_multiple_models_fr_path(args.model_path, model)
            else:
                print(f"Loading model with single activated setting from {args.model_path}")
                ### ==== STATE DICT ==== ###
                state_dict = torch.load(args.model_path, map_location='cpu')
                load_model_wo_clip(model, state_dict) ## load model wihtout clip #
            
            model.to(dist_util.dev()) ## model-to-the-target-device ##
            # model.rot2xyz.smpl_model.eval()
            model.eval() ## ddp_model = model
            try:
                model.set_bn_to_eval()
            except:
                pass
            ### === GET object global orientation, translations from data === ### # get obj sequences from data #
            obj_idx, tot_obj_global_orient, tot_obj_global_transl = get_obj_sequences_from_data(data) #
            tot_base_pts,  tot_base_normals, tot_rhand_joints, tot_gt_rhand_joints, tot_obj_rot, tot_obj_transl = get_base_pts_rhand_joints_from_data(data)
            
            
            # grab_path = args.grab_path
            # obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
            obj_mesh_path = "data/grab/object_meshes"
            id2objmesh = []
            obj_meshes = sorted(os.listdir(obj_mesh_path))
            for i, fn in enumerate(obj_meshes):
                id2objmesh.append(os.path.join(obj_mesh_path, fn))
            cur_obj_mesh_fn = id2objmesh[obj_idx]
            # cur_obj_mesh_fn = args.cad_model_fn
            obj_mesh = trimesh.load(cur_obj_mesh_fn, process=False)
            obj_verts = np.array(obj_mesh.vertices)
            obj_vertex_normals = np.array(obj_mesh.vertex_normals)
            obj_faces = np.array(obj_mesh.faces)

            print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
            print("Training...")


            if args.diff_basejtse:
                tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts, tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals = TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
            else:
                tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts = TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
                tot_dec_disp_e_along_normals = None
                tot_dec_disp_e_vt_normals = None
            
            # 
            print(f"tot_st_idxes: {tot_st_idxes.size()}, tot_ed_idxes: {tot_ed_idxes.size()}")
            
            ## predict ours objbase ##
            data_scale_factor = 1.0
            # shoudl reture
            n_batches = tot_targets.size(0)
            full_targets = []
            full_outputs = []
            full_pert_verts = []
            full_verts = []
            full_obj_verts = []
            
            
            full_dec_disp_e_along_normals = []
            full_dec_disp_e_vt_normals = []
            
            for i_b in range(n_batches):
                cur_targets = tot_targets[i_b]
                cur_outputs = tot_model_outputs[i_b] # ['sampled_rhand_joints']
                cur_pert_verts = tot_pert_verts[i_b]
                cur_verts = tot_verts[i_b]
                
                cur_obj_orient = tot_obj_global_orient[i_b]
                cur_obj_transl = tot_obj_global_transl[i_b]
                
                if args.diff_basejtse:
                    cur_dec_disp_e_along_normals = tot_dec_disp_e_along_normals[i_b]
                    cur_dec_disp_e_vt_normals = tot_dec_disp_e_vt_normals[i_b]
                    
                
                cur_st_idxes = tot_st_idxes[i_b].item()
                cur_ed_idxes = tot_ed_idxes[i_b].item()
                cur_len_full_targets = len(full_targets)
                # for i_ins in range(cur_len_full_targets, cur_ed_idxes):
                for i_ins in range(cur_targets.size(0)):
                    cur_ins_rel_idx = i_ins # - cur_ed_idxes # negative index here
                    cur_ins_targets = cur_targets[cur_ins_rel_idx] / data_scale_factor
                    print(f"cur_outputs: {cur_outputs.size()}")
                    cur_ins_outputs = cur_outputs[cur_ins_rel_idx] / data_scale_factor
                    cur_ins_pert_verts = cur_pert_verts[cur_ins_rel_idx, ...]
                    cur_ins_verts = cur_verts[cur_ins_rel_idx, ...]
                    
                    cur_ins_obj_orient = cur_obj_orient[cur_ins_rel_idx].numpy() # 3 --> torch.tensor
                    cur_ins_obj_transl = cur_obj_transl[cur_ins_rel_idx].numpy() # 3 --> torch.tensor
                    # R.from_rotvec(obj_rot).as_matrix() #
                    # ### cur_ins_obj_rot_mtx, cur_ins_obj_transl ### #
                    cur_ins_obj_rot_mtx = R.from_rotvec(cur_ins_obj_orient).as_matrix() # 3 x 3
                    cur_ins_obj_transl = cur_ins_obj_transl.reshape(1, 3)


                    transformed_obj_verts = obj_verts
                    full_obj_verts.append(transformed_obj_verts) ## obj_verts ##
                    
                    if args.diff_basejtse:
                        try:
                            full_dec_disp_e_along_normals.append(cur_dec_disp_e_along_normals[cur_ins_rel_idx].detach().cpu().numpy())
                            full_dec_disp_e_vt_normals.append(cur_dec_disp_e_vt_normals[cur_ins_rel_idx].detach().cpu().numpy())
                        except:
                            pass
                    full_targets.append(cur_ins_targets.detach().cpu().numpy())
                    full_outputs.append(cur_ins_outputs.detach().cpu().numpy())
                    full_pert_verts.append(cur_ins_pert_verts.detach().cpu().numpy())
                    full_verts.append(cur_ins_verts.detach().cpu().numpy())
                    
            full_targets = np.stack(full_targets, axis=0)
            full_outputs = np.stack(full_outputs, axis=0)
            full_pert_verts = np.stack(full_pert_verts, axis=0)
            full_verts = np.stack(full_verts, axis=0)
            full_obj_verts = np.stack(full_obj_verts, axis=0)
            
            if args.diff_basejtse:
                full_dec_disp_e_along_normals = np.stack(full_dec_disp_e_along_normals, axis=0)
                full_dec_disp_e_vt_normals = np.stack(full_dec_disp_e_vt_normals, axis=0)
            
            
            sv_dict = {
                'targets': full_targets,
                'outputs': full_outputs,
                'pert_verts': full_pert_verts,
                'verts': full_verts,
                'obj_verts': full_obj_verts,
                'obj_faces': obj_faces,
                'tot_base_pts': tot_base_pts.numpy() / data_scale_factor,
                'tot_rhand_joints': tot_rhand_joints.numpy() / data_scale_factor,
                'tot_base_normals': tot_base_normals.numpy(), 
                'tot_gt_rhand_joints': tot_gt_rhand_joints.numpy() / data_scale_factor,
                'tot_obj_rot': tot_obj_rot.numpy(), 
                'tot_obj_transl': tot_obj_transl.numpy(),
                'single_obj_normals': obj_vertex_normals,
            }
            
            
            seq_idx = test_seq_idx
            # if args.diff_basejtse:
            #     dec_e_dict = {
            #         'dec_disp_e_along_normals': full_dec_disp_e_along_normals,
            #         'dec_disp_e_vt_normals': full_dec_disp_e_vt_normals, 
            #     }
            #     sv_dict.update(dec_e_dict)
            #     e_dict_sv_fn = os.path.join(args.save_dir, "e_predicted_infos.npy")
            #     np.save(e_dict_sv_fn, dec_e_dict)
            #     print(f"e dict ssaved to {e_dict_sv_fn}")
            # else:
            
            sv_predicted_info_fn = f"predicted_infos_seq_{seq_idx}_seed_{args.seed}_st_{args.start_idx}_tag_{args.test_tag}.npy"
            sv_dict_sv_fn = os.path.join(args.save_dir, sv_predicted_info_fn) # save dict fn 
            np.save(sv_dict_sv_fn, sv_dict)
            print(f"Predicted infos saved to {sv_dict_sv_fn}")


    train_platform.close()

if __name__ == "__main__":
    main()
