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
    tot_obj_verts = []
    tot_obj_faces = []
    for batch in data:
        obj_idx = batch['object_id']
        obj_global_orient = batch['object_global_orient']
        obj_global_transl = batch['object_transl']
        obj_verts = batch['obj_verts']
        obj_faces = batch['obj_faces']
        for i_obj, cur_obj_verts in enumerate(obj_verts):
            print(f"i_obj: {i_obj}, cur_obj_verts: {cur_obj_verts.shape}")
        tot_obj_verts += obj_verts
        tot_obj_faces += obj_faces
        tot_obj_idx.append(obj_idx)
        tot_obj_global_orient.append(obj_global_orient)
        tot_obj_global_transl.append(obj_global_transl)
    tot_obj_idx = torch.cat(tot_obj_idx, dim=0)
    tot_obj_global_orient = torch.cat(tot_obj_global_orient, dim=0)
    tot_obj_global_transl = torch.cat(tot_obj_global_transl, dim=0)
    obj_idx = tot_obj_idx[0].item()
    tot_obj_verts = torch.cat(tot_obj_verts, dim=0)
    tot_obj_faces = torch.cat(tot_obj_faces, dim=0)
    return obj_idx, tot_obj_global_orient, tot_obj_global_transl, tot_obj_verts, tot_obj_faces

def get_base_pts_rhand_joints_from_data(data):
    
    tot_base_pts = []
    tot_rhand_joints = []
    tot_base_normals = []
    tot_gt_rhand_joints = []
    
    tot_obj_rot = []
    tot_obj_transl = []
    
    tot_obj_pcs = []
    
    
    # rhand_verts # 
    tot_rhand_verts = []
    
    for batch in data:
        base_pts = batch['base_pts'] 
        rhand_joints = batch['rhand_joints']
        base_normals = batch['base_normals']
        gt_rhand_joints = batch['gt_rhand_joints']
        
        obj_rot = batch['obj_rot']
        obj_transl = batch['obj_transl']
        
        cur_batch_rhand_verts = batch['rhand_verts'].detach().cpu()
        tot_rhand_verts.append(cur_batch_rhand_verts)
        
        tot_base_pts.append(base_pts.detach().cpu())
        tot_rhand_joints.append(rhand_joints.detach().cpu())
        tot_base_normals.append(base_normals.detach().cpu())
        tot_gt_rhand_joints.append(gt_rhand_joints.detach().cpu())
        tot_obj_rot.append(obj_rot.detach().cpu())
        tot_obj_transl.append(obj_transl.detach().cpu())
        
        cur_batch_obj_pcs = batch['object_pcs'] # object_pcs #
        tot_obj_pcs.append(cur_batch_obj_pcs)
        
        
    tot_base_pts = torch.cat(tot_base_pts, dim=0)
    tot_rhand_joints = torch.cat(tot_rhand_joints, dim=0)
    tot_base_normals = torch.cat(tot_base_normals, dim=0)
    tot_gt_rhand_joints = torch.cat(tot_gt_rhand_joints, dim=0)
    tot_obj_rot = torch.cat(tot_obj_rot, dim=0)
    tot_obj_transl = torch.cat(tot_obj_transl, dim=0)
    tot_obj_pcs = torch.cat(tot_obj_pcs, dim=0)
    tot_rhand_verts = torch.cat(tot_rhand_verts, dim=0)
    # tot_obj_idx = torch.cat(tot_obj_idx, dim=0)
    # tot_obj_global_orient = torch.cat(tot_obj_global_orient, dim=0)
    # tot_obj_global_transl = torch.cat(tot_obj_global_transl, dim=0)
    # obj_idx = tot_obj_idx[0].item()
    return tot_base_pts, tot_base_normals, tot_rhand_joints, tot_gt_rhand_joints, tot_obj_rot, tot_obj_transl, tot_obj_pcs, tot_rhand_verts




def main():
  
    args = train_args()
    
    fixseed(args.seed)
    

    dist_util.setup_dist(args.device)
    
    

    
    # use_reverse = args.use_reverse
    os.makedirs(args.save_dir, exist_ok=True)
    train_platform_type = eval(args.train_platform_type) # platform_type #
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    
    args_path = os.path.join(args.save_dir, 'args_hoi4d.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    print(f"arguments saved to {args_path}")
    
    
    
    if len(args.single_seq_path) > 0:
        print(f"Single sequence evaluation setting")
        case_idx = args.single_seq_path.split("/")[-2]
        case_idx = int(case_idx[4:]) ## get the case idx ## 
        tot_test_seq_idxes = range(case_idx, case_idx + 1, 1) ## get the case idx ## 
        cat_nm = args.single_seq_path.split("/")[-3] ## get the category name
        
        data_root = args.hoi4d_data_root
        data_root = os.path.join(data_root, cat_nm) 
        # case_folder_nm = os.path.join(data_root, f"case{test_seq_idx}")
        
    else:
        print(f"Category evaluation setting")
        ## TODO: add an arugment to control the category name here ##3
        rigid_categories = ["Bottle", "Bowl", "Chair", "Kettle", "Knife", "Mug", "ToyCar"]
        cat_nm = args.hoi4d_category_name
        cat_inst_st_idx = args.hoi4d_eval_st_idx
        cat_inst_ed_idx = args.hoi4d_eval_ed_idx ## get the st and ed idxes
        tot_test_seq_idxes = range(cat_inst_st_idx, cat_inst_ed_idx + 1, 1) ## [st, ed_idx] 
        
        data_root = args.hoi4d_data_root
        # if cat_nm in rigid_categories:
        #     data_root = os.path.join(data_root, "HOI_Processed_Data_Rigid")
        # else:
        #     data_root = os.path.join(data_root, "HOI_Processed_Data_Arti")
        data_root = os.path.join(data_root, cat_nm) 
        tot_case_folders = os.listdir(data_root)
        tot_case_folders = [fn for fn in tot_case_folders if "case" in fn]
        tot_case_idxes = [int(fn[4:]) for fn in tot_case_folders]
        tot_case_idxes = [idx for idx in tot_case_idxes if idx >= cat_inst_st_idx and idx <= cat_inst_ed_idx]
        tot_test_seq_idxes = tot_case_idxes
        ##### 
    
    
    if len(args.save_dir) > 0:
        args.save_dir = os.path.join(args.save_dir, f"{cat_nm}")
        os.makedirs(args.save_dir, exist_ok=True)
        
        
    ### TODO: for hoi4d, the start idx should be carefully selected for each category ###
    args.start_idx = 0
    args.select_part_idx = 0
    
    for test_seq_idx in tot_test_seq_idxes:
        # try:
        print(f"test_seq_idx: {test_seq_idx}")
        for cur_seed in range(0, 122, 11): # cur_seed #
            
            
            # cur_single_seq_path = f"data/hoi4d/{cat_nm}/case{test_seq_idx}/merged_data.npy"
            
            # case_folder_nm =  f"data/hoi4d/{cat_nm}/case{test_seq_idx}" 
            
            case_folder_nm = os.path.join(data_root, f"case{test_seq_idx}")
            cur_single_seq_path = os.path.join(case_folder_nm, "merged_data.npy")
            args.single_seq_path = cur_single_seq_path
            args.cad_model_fn = f"obj_model.obj"
            # args.corr_fn = f"data/hoi4d/{cat_nm}/case{test_seq_idx}/merged_data.npy"
            args.corr_fn = cur_single_seq_path

            
            print(f"[Data loading]")
            print(f"case_folder_nm: {case_folder_nm}")
            print(f"cur_single_seq_path: {cur_single_seq_path}")
            
            
            args.seed = cur_seed
            
            
            args.predicted_info_fn = f"" 


            print(f"Current sequence path: {args.single_seq_path}, seed: {args.seed}")
            
            
            
            # print("creating data loader...") ## create model and diffusion ##
            data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, args=args)


            # print("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(args, data)


            state_dict = torch.load(args.model_path, map_location='cpu')
            load_model_wo_clip(model, state_dict)
            
            model.to(dist_util.dev())
            model.eval()
            try:
                model.set_bn_to_eval()
            except:
                pass
            
            obj_idx, tot_obj_global_orient, tot_obj_global_transl, tot_obj_verts, tot_obj_faces = get_obj_sequences_from_data(data)
            tot_base_pts,  tot_base_normals, tot_rhand_joints, tot_gt_rhand_joints, tot_obj_rot, tot_obj_transl, tot_obj_pcs, tot_rhand_verts = get_base_pts_rhand_joints_from_data(data)
            
            
            
            tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts = TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
            tot_dec_disp_e_along_normals = None
            tot_dec_disp_e_vt_normals = None
        

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
                    cur_ins_rel_idx = i_ins - cur_ed_idxes 
                    cur_ins_targets = cur_targets[cur_ins_rel_idx] / data_scale_factor
                    # print(f"cur_outputs: {cur_outputs.size()}")
                    cur_ins_outputs = cur_outputs[cur_ins_rel_idx] / data_scale_factor
                    cur_ins_pert_verts = cur_pert_verts[cur_ins_rel_idx, ...]
                    cur_ins_verts = cur_verts[cur_ins_rel_idx, ...]
                    
                    cur_ins_obj_orient = cur_obj_orient[cur_ins_rel_idx].numpy()
                    cur_ins_obj_transl = cur_obj_transl[cur_ins_rel_idx].numpy()
                    
                    cur_ins_obj_rot_mtx = R.from_rotvec(cur_ins_obj_orient).as_matrix()
                    cur_ins_obj_transl = cur_ins_obj_transl.reshape(1, 3)


                    # if args.diff_basejtse:
                    #     try:
                    #         full_dec_disp_e_along_normals.append(cur_dec_disp_e_along_normals[cur_ins_rel_idx].detach().cpu().numpy())
                    #         full_dec_disp_e_vt_normals.append(cur_dec_disp_e_vt_normals[cur_ins_rel_idx].detach().cpu().numpy())
                    #     except:
                    #         pass
                    full_targets.append(cur_ins_targets.detach().cpu().numpy())
                    full_outputs.append(cur_ins_outputs.detach().cpu().numpy())
                    full_pert_verts.append(cur_ins_pert_verts.detach().cpu().numpy())
                    full_verts.append(cur_ins_verts.detach().cpu().numpy())
                    

            full_targets = np.stack(full_targets, axis=0)
            full_outputs = np.stack(full_outputs, axis=0)
            full_pert_verts = np.stack(full_pert_verts, axis=0)
            full_verts = np.stack(full_verts, axis=0)
            
            # if args.diff_basejtse:
            #     full_dec_disp_e_along_normals = np.stack(full_dec_disp_e_along_normals, axis=0)
            #     full_dec_disp_e_vt_normals = np.stack(full_dec_disp_e_vt_normals, axis=0)
            
            
            tot_obj_verts = tot_obj_verts.detach().cpu().numpy()
            tot_obj_faces = tot_obj_faces.detach().cpu().numpy()
            
            
            sv_dict = {
                'targets': full_targets,
                'outputs': full_outputs,
                'pert_verts': full_pert_verts,
                'verts': full_verts,
                'tot_rhand_verts': tot_rhand_verts.detach().cpu().numpy(),
                'obj_verts': tot_obj_verts,
                'obj_faces': tot_obj_faces,
                'tot_base_pts': tot_base_pts.numpy() / data_scale_factor,
                'tot_obj_pcs': tot_obj_pcs.detach().cpu().numpy() / data_scale_factor,
                'tot_rhand_joints': tot_rhand_joints.numpy() / data_scale_factor,
                'tot_base_normals': tot_base_normals.numpy(), 
                'tot_gt_rhand_joints': tot_gt_rhand_joints.numpy() / data_scale_factor,
                'tot_obj_rot': tot_obj_rot.numpy(),
                'tot_obj_transl': tot_obj_transl.numpy(),
            }
            
            
            seq_idx = test_seq_idx
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
                sv_predicted_info_fn = f"predicted_infos_seq_{seq_idx}_seed_{args.seed}_tag_{args.test_tag}.npy"
                sv_dict_sv_fn = os.path.join(args.save_dir, sv_predicted_info_fn)
                np.save(sv_dict_sv_fn, sv_dict)
                print(f"Predicted infos saved to {sv_dict_sv_fn}")

    
    train_platform.close()

if __name__ == "__main__":
    main()
