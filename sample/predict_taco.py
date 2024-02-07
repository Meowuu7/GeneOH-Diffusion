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
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform
from tqdm import tqdm
from argparse import Namespace
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import data_loaders.humanml.data.utils as utils



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

        motion = motion.to(device)
        cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

def get_obj_sequences_from_data(data):
    # tot_obj_idx = []
    tot_obj_global_orient = []
    tot_obj_global_transl = []
    # object_pc_th
    tot_obj_pcs = []
    for batch in data:
        # obj_idx = batch['object_id']
        obj_global_orient = batch['object_global_orient']
        obj_global_transl = batch['object_transl']
        obj_pc = batch["object_pc_th"]
        # tot_obj_idx.append(obj_idx)
        tot_obj_global_orient.append(obj_global_orient)
        tot_obj_global_transl.append(obj_global_transl)
        tot_obj_pcs.append(obj_pc)
    # tot_obj_idx = torch.cat(tot_obj_idx, dim=0)
    tot_obj_global_orient = torch.cat(tot_obj_global_orient, dim=0)
    tot_obj_global_transl = torch.cat(tot_obj_global_transl, dim=0)
    tot_obj_pcs = torch.cat(tot_obj_pcs, dim=0)
    # obj_idx = tot_obj_idx[0].item()
    return tot_obj_global_orient, tot_obj_global_transl, tot_obj_pcs

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

def get_resplit_test_idxes():
    test_split_mesh_nm_to_seq_idxes = "/home/xueyi/sim/motion-diffusion-model/test_mesh_nm_to_test_seqs.npy"
    test_split_mesh_nm_to_seq_idxes = np.load(test_split_mesh_nm_to_seq_idxes, allow_pickle=True).item()
    tot_test_seq_idxes = []
    for tst_nm in test_split_mesh_nm_to_seq_idxes:
        tot_test_seq_idxes = tot_test_seq_idxes + test_split_mesh_nm_to_seq_idxes[tst_nm]
    return tot_test_seq_idxes


def get_arctic_seq_paths():
    processed_arctic_root = "/data/datasets/genn/sim/arctic_processed_data/processed_seqs"
    subj_folders = os.listdir(processed_arctic_root)
    tot_arctic_seq_paths = []
    tot_arctic_seq_tags = []
    for cur_subj_folder in subj_folders:
        full_cur_subj_folder = os.path.join(processed_arctic_root, cur_subj_folder)
        cur_subj_seq_nms = os.listdir(full_cur_subj_folder)
        cur_subj_seq_nms = [fn for fn in cur_subj_seq_nms if fn.endswith(".npy")]
        for cur_subj_seq_nm in cur_subj_seq_nms:
            full_seq_nm = os.path.join(full_cur_subj_folder, cur_subj_seq_nm)
            tot_arctic_seq_paths.append(full_seq_nm)
            cur_seq_tag = f"{cur_subj_folder}_{cur_subj_seq_nm.split('.')[0]}"
            tot_arctic_seq_tags.append(cur_seq_tag)
    return tot_arctic_seq_paths, tot_arctic_seq_tags

import pickle as pkl


def main():
  
    args = train_args()
    
    fixseed(args.seed)
    

    dist_util.setup_dist(args.device)


    # use_reverse = args.use_reverse
    os.makedirs(args.save_dir, exist_ok=True)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    
    ### get arg path ###
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
        
    tot_hho_seq_paths = [args.single_seq_path]
    tot_hho_seq_tags = ["test"]
    
    data_dict = pkl.load(open(args.single_seq_path, 'rb'))
    data_hand_verts = data_dict['hand_verts']
    nn_frames = data_hand_verts.shape[0]
    
    
    nn_st_skip = 30
    num_cleaning_frames = 60
    num_ending_clearning_frames = nn_frames - num_cleaning_frames + 1

    print(range(0, num_ending_clearning_frames, nn_st_skip))
    
    st_idxes = list(range(0, num_ending_clearning_frames, nn_st_skip))
    if st_idxes[-1] + num_cleaning_frames < nn_frames:
        st_idxes.append(nn_frames - num_cleaning_frames)
    print(f"st_idxes: {st_idxes}")
    
    for cur_seed in range(0, 122, 11):
        args.seed = cur_seed 
        
        
        for st_fr in st_idxes:
            args.start_idx = st_fr
            
            # random seeds #
            args.predicted_info_fn = ""
            

            obj_sv_path = "/".join(args.single_seq_path.split("/")[:-1])
            # obj_name = args.single_seq_path.split("/")[-1].split("_")[0]
            obj_name = args.single_seq_path.split("/")[-1].split(".")[0]
            
            obj_mesh_fn = os.path.join(obj_sv_path, obj_name + ".obj") # object mesh file #
            print(f"loading from {obj_mesh_fn}") ## loading 
            template_obj_vs, template_obj_fs = utils.read_obj_file_ours(obj_mesh_fn, sub_one=True)
            template_obj_fs = np.array(template_obj_fs, dtype=np.long)
            ## loaded the template obj
            print(f"Current sequence path: {args.single_seq_path}, seed: {args.seed}; Template obj loaded with verts: {template_obj_vs.shape}, template_obj_fs: {template_obj_fs.shape}")
            
            ## get dataest loader ##
            ### ==== DATA LOADER ==== ### # DATA loade r##
            print("creating data loader...")
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
            
            model.to(dist_util.dev())
            
            model.eval()
            try:
                model.set_bn_to_eval()
            except:
                pass
            ### === GET object global orientation, translations from data === ### # get obj sequences from data #
            tot_obj_global_orient, tot_obj_global_transl, tot_obj_pcs = get_obj_sequences_from_data(data) #
            tot_base_pts,  tot_base_normals, tot_rhand_joints, tot_gt_rhand_joints, tot_obj_rot, tot_obj_transl = get_base_pts_rhand_joints_from_data(data)
            
            
            
            ## predict_from_data
            print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))


            if args.diff_basejtse:
                # tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals #
                tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts, tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals = TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
            else:
                tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts = TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
                tot_dec_disp_e_along_normals = None
                tot_dec_disp_e_vt_normals = None
            
            print(f"tot_st_idxes: {tot_st_idxes.size()}, tot_ed_idxes: {tot_ed_idxes.size()}")
            
            ## predict ours objbase ##
            data_scale_factor = 1.0
            # shoudl reture
            n_batches = tot_targets.size(0)
            full_targets = []
            full_outputs = []
            full_pert_verts = []
            full_verts = []

            
            full_dec_disp_e_along_normals = []
            full_dec_disp_e_vt_normals = []
            
            for i_b in range(n_batches):
                
                cur_targets = tot_targets[i_b]
                cur_outputs = tot_model_outputs[i_b]
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
                for i_ins in range(cur_len_full_targets, cur_ed_idxes):
                    cur_ins_rel_idx = i_ins - cur_ed_idxes # negative index here
                    cur_ins_targets = cur_targets[cur_ins_rel_idx] / data_scale_factor

                    cur_ins_outputs = cur_outputs[cur_ins_rel_idx] / data_scale_factor
                    cur_ins_pert_verts = cur_pert_verts[cur_ins_rel_idx, ...]
                    cur_ins_verts = cur_verts[cur_ins_rel_idx, ...]
                    
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
                    
                    
                    
            ## full targets ##
            full_targets = np.stack(full_targets, axis=0)
            full_outputs = np.stack(full_outputs, axis=0)
            full_pert_verts = np.stack(full_pert_verts, axis=0)
            full_verts = np.stack(full_verts, axis=0)
            # full_obj_verts = np.stack(full_obj_verts, axis=0)
            
            if args.diff_basejtse:
                full_dec_disp_e_along_normals = np.stack(full_dec_disp_e_along_normals, axis=0)
                full_dec_disp_e_vt_normals = np.stack(full_dec_disp_e_vt_normals, axis=0)
            
            ### transform them ###
            if args.use_left:
                tot_base_pts[..., -1] = tot_base_pts[..., -1] * -1.
                tot_rhand_joints[..., -1] = tot_rhand_joints[..., -1] * -1.
                full_outputs[..., -1] = full_outputs[..., -1] * -1.
                full_targets[..., -1] = full_targets[..., -1] * -1.
            
            # penetration resolving
            sv_dict = {
                'targets': full_targets,
                'outputs': full_outputs, 
                'pert_verts': full_pert_verts,
                'verts': full_verts,
                'tot_base_pts': tot_base_pts.numpy() / data_scale_factor, ## total base pts ##
                'tot_rhand_joints': tot_rhand_joints.numpy() / data_scale_factor,
                'tot_base_normals': tot_base_normals.numpy(), 
                'tot_gt_rhand_joints': tot_gt_rhand_joints.numpy() / data_scale_factor,
                'tot_obj_rot': tot_obj_rot.numpy(),  # ws x 3 x 3 #
                'tot_obj_transl': tot_obj_transl.numpy(), # ws x 3 #
                'tot_obj_pcs': tot_obj_pcs.detach().cpu().numpy(), 
                'template_obj_fs': template_obj_fs,
            }
            
            
            if args.diff_basejtse:
                dec_e_dict = {
                    'dec_disp_e_along_normals': full_dec_disp_e_along_normals,
                    'dec_disp_e_vt_normals': full_dec_disp_e_vt_normals, 
                }
                sv_dict.update(dec_e_dict)
                e_dict_sv_fn = os.path.join(args.save_dir, "e_predicted_infos.npy")
                np.save(e_dict_sv_fn, dec_e_dict)
                print(f"e dict ssaved to {e_dict_sv_fn}")
            else:
                sv_predicted_info_fn = f"predicted_infos_seed_{args.seed}_tag_{args.test_tag}_st_{args.start_idx}.npy"
                sv_dict_sv_fn = os.path.join(args.save_dir, sv_predicted_info_fn)
                np.save(sv_dict_sv_fn, sv_dict)
                print(f"Predicted infos saved to {sv_dict_sv_fn}")


    train_platform.close()

if __name__ == "__main__":
    main()
