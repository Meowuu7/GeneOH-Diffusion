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

## TODO: 1) how the current prediction process functions? 
##       2) sampling based prediction?

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

def get_resplit_test_idxes():
    test_split_mesh_nm_to_seq_idxes = "/home/xueyi/sim/motion-diffusion-model/test_mesh_nm_to_test_seqs.npy"
    test_split_mesh_nm_to_seq_idxes = np.load(test_split_mesh_nm_to_seq_idxes, allow_pickle=True).item()
    tot_test_seq_idxes = []
    for tst_nm in test_split_mesh_nm_to_seq_idxes:
        tot_test_seq_idxes = tot_test_seq_idxes + test_split_mesh_nm_to_seq_idxes[tst_nm]
    return tot_test_seq_idxes

    
# CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.predict_ours --model_path ${model_path}  --input_text ./assets/example_pro.txt --dataset motion_ours --save_dir ${save_dir}
# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset motion_ours
def main():
  
    args = train_args()
    # cur_args = get_args()
    # args.model_path = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/model000050000.pt" # model_path
    # args.model_path = "/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_unconstrainted_inst_norm_svi_500_/model000002500.pt"
    # args.input_text = "./assets/example_pro.txt"
    # args.input_text = ""
    
    # args.save_dir = "/data2/xueyi/eval_save/HOI_Arti/Scissors"
    # predicted_infos_seq_44_seed_99_tag_jts_hoi4d_arti_t_400_.npy # for the predicted info seq ##
    # args.test_tag
    # /home/xueyi/sim/motion-diffusion-model/sample/predict_ours_objbase_bundle_ours_rndseed_grab.py
    
    fixseed(args.seed)
    

    dist_util.setup_dist(args.device)
    

    # tot_test_seq_idxes = range(0, 205, 1)
    tot_test_seq_idxes = range(1, 205, 1)
    # tot_test_seq_idxes = range(5, 6, 1)

    seq_root = "/data1/xueyi/GRAB_processed/test"

    if args.resplit:
        tot_test_seq_idxes = get_resplit_test_idxes()
        seq_root = "/data1/xueyi/GRAB_processed/train"

    # use_reverse = True
    # use_reverse = args.use_reverse
    # args.save_dir = f"/data2/xueyi/eval_save/HOI_Arti/{cat_nm}"
    args.save_dir = "/data2/xueyi/eval_save/GRAB"
    args.start_idx = 0
    # args.start_idx = 200
    # tot_test_seq_idxes = range(44, 200, 1)

    use_reverse = args.use_reverse
    os.makedirs(args.save_dir, exist_ok=True)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args') # train platform
    
    # device = torch.device("cpu") # device for torch #
    # if torch.cuda.is_available() and dist_util.dev() != 'cpu':
    #     device = torch.device(dist_util.dev())

    # 
    # if args.save_dir is None: # save dir was not specified #
    #     raise FileNotFoundError('save_dir was not specified.')
    # else: 
    #     os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    for cur_seed in range(0, 122, 11):
    # for test_seq_idx in tot_test_seq_idxes:
        try:
            for test_seq_idx in tot_test_seq_idxes:

                cur_single_seq_path = os.path.join(seq_root, f"{test_seq_idx}.npy")
                args.single_seq_path = cur_single_seq_path
                print(f"cur_single_seq_path: {cur_single_seq_path}")
                
                if not os.path.exists(cur_single_seq_path):
                    continue
            
                args.seed = cur_seed # random seeds #
                # test_seq_idx # # args.seed, cur_seed, 
                # try:
                # /home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_70_seed_77_tag_jts_only.npy
                # args.predicted_info_fn = f"/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only.npy"
                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_300_seed_8_tag_rep_only_mean_shape_hoi4d_t_400_.npy
                
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_300_seed_{cur_seed}_tag_rep_only_mean_shape_hoi4d_t_400_.npy"
                # # args.predicted_info_fn = f"" # preidcted_info_fn
                
                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_arti_t_400_.npy"
                
                # # while not os.path.exists(args.predicted_info_fn):
                # #     secs = 60 * 1
                # #     print(f"Cannot load data from {args.predicted_info_fn}. Wait for 1 min...")
                # #     time.sleep(secs)
                
                # if not os.path.exists(args.predicted_info_fn):
                #     continue
                    
                    # /data1/xueyi/toch/ckpts/v1_wsubj_wjointsv11_joints_100_/toch_aug_679.pthn
                args.predicted_info_fn = f"" 
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_400_.npy"
                # # /data2/xueyi/eval_save/GRAB/predicted_infos_seq_1_seed_33_tag_jts_grab_t_400_scale_1_.npy
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_400_scale_1_.npy"
                # # /data2/xueyi/eval_save/GRAB/predicted_infos_seq_1_seed_22_tag_jts_grab_t_400_scale_2_.npy
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_400_scale_2_.npy"
                # # /data2/xueyi/eval_save/GRAB/predicted_infos_seq_1_seed_22_tag_jts_grab_t_400_scale_2_.npy
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_400_scale_2_.npy"
                # # jts_grab_t_700_scale_obj_
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_700_scale_obj_.npy"
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_grab_t_700_scale_obj_2_.npy"
                # # /data2/xueyi/eval_save/GRAB/predicted_infos_seq_1210_seed_22_tag_rep_jts_grab_t_400_resplit_.npy
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/GRAB/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_rep_jts_grab_t_400_resplit_.npy"
                # /data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_9_seed_22_tag_jts_hoi4d_arti_scissors_t_400_st_idx_0_.npy
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Pliers/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_pliers_t_400_st_idx_30_.npy" 
                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/ToyCar/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_rep_res_jts_hoi4d_arti_bucket_t_400_.npy" 

                # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_arti_t_400_.npy"
                # # 
                # # jts_hoi4d_arti_scissors_t_400_st_idx_0_
                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_arti_scissors_t_400_st_idx_0_.npy"
                # # # 

                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/Chair/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_chair_t_400_st_idx_0_.npy"

                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/Knife/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_knife_t_400_st_idx_0_.npy"

                # if not os.path.exists(args.predicted_info_fn):
                #     continue

                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/Bottle/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_bottle_t_400_st_idx_0_.npy"

                # # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/Mug/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_mug_t_400_st_idx_0_.npy"

                # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Rigid/Bowl/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_bowl_t_400_st_idx_0_.npy"

                # 
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Scissors/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_arti_scissors_t_400_st_idx_0_reverse_.npy"
                # args.predicted_info_fn = f"/data2/xueyi/eval_save/HOI_Arti/Pliers/predicted_infos_seq_{test_seq_idx}_seed_{cur_seed}_tag_jts_hoi4d_pliers_t_400_st_idx_130_.npy"

                # while not os.path.exists(args.predicted_info_fn):
                #     secs = 60 * 1
                #     print(f"Cannot load data from {args.predicted_info_fn}. Wait for 1 min...")
                #     time.sleep(secs)

                ## predicted info fn ##


                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_37_seed_77_tag_jts_only_gaussian_hoi4d_t_300_.npy
                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_110_seed_77_tag_jts_only_gaussian_hoi4d_t_300_.npy
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only_gaussian_hoi4d_t_300_.npy"
                # jts_only_gaussian_hoi4d_t_300_
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only_gaussian_hoi4d_t_300_.npy"
                
                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_102_seed_77_tag_jts_only.npy
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only.npy"
                # args.single_seq_path = f"/data1/xueyi/GRAB_processed/test/{test_seq_idx}.npy" ## test seq path #
                # /home/xueyi/sim/ContactOpt/./ours_data/case5/merged_data_with_corr.npy
                # f"/data1/xueyi/HOI_Processed_Data/case59/merged_data_with_corr.npy"
                # args.single_seq_path = f"/data1/xueyi/HOI_Processed_Data/case{test_seq_idx}/merged_data_with_corr.npy"
                # args.cad_model_fn = f"/data1/xueyi/HOI_Processed_Data/case{test_seq_idx}/obj_model.obj"
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only_beta_t_300_.npy"
                # args.predicted_info_fn = ""
                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_245_seed_77_tag_jts_only_uniform_t_300_.npy
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only_uniform_t_300_.npy"
                # /data1/xueyi/mdm/eval_save/predicted_infos_seq_102_seed_77_tag_jts_only.npy
                # args.predicted_info_fn = f"/data1/xueyi/mdm/eval_save/predicted_infos_seq_{test_seq_idx}_seed_77_tag_jts_only.npy"
                # args.single_seq_path = f"/data1/xueyi/GRAB_processed/test/{test_seq_idx}.npy" ## test seq path #
            

                # if "HOI_Rigid" in args.save_dir:
                #     args.single_seq_path = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/merged_data_with_corr.npy"
                #     args.cad_model_fn = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/obj_model.obj"
                #     args.corr_fn = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/merged_data.npy" # corr fn #

                # if args.use_arti_obj:
                #     args.single_seq_path = "/home/xueyi/sim/ContactOpt/./ours_data/case5/merged_data_with_corr.npy"
                #     args.cad_model_fn = "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-0-align.obj"
                #     args.corr_fn = "/home/xueyi/sim/ContactOpt/./ours_data/case5/merged_data.npy" # corr fn #
                
                # if args.use_arti_obj:
                #     # args.single_seq_path = f"/data1/xueyi/HOI_Processed_Data_Arti/case{test_seq_idx}/merged_data_with_corr.npy"
                #     # args.cad_model_fn = f"/data1/xueyi/HOI_Processed_Data_Arti/case{test_seq_idx}/obj_model.obj"
                #     # args.corr_fn = f"/data1/xueyi/HOI_Processed_Data_Arti/case{test_seq_idx}/merged_data.npy" # corr fn #
                    
                #     if "HOI_Rigid" in args.save_dir:
                #         args.single_seq_path = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/merged_data_with_corr.npy"
                #         args.cad_model_fn = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/obj_model.obj"
                #         args.corr_fn = f"/data2/xueyi/HOI_Processed_Data_Rigid/{cat_nm}/case{test_seq_idx}/merged_data.npy" # corr fn #
                #     else:
                #         if use_reverse:
                #             args.single_seq_path = f"/data2/xueyi/HOI_Processed_Data_Arti_Reverse/{cat_nm}/case{test_seq_idx}/merged_data_with_corr.npy"
                #             args.cad_model_fn = f"/data2/xueyi/HOI_Processed_Data_Arti_Reverse/{cat_nm}/case{test_seq_idx}/obj_model.obj"
                #             args.corr_fn = f"/data2/xueyi/HOI_Processed_Data_Arti_Reverse/{cat_nm}/case{test_seq_idx}/merged_data.npy" # corr fn #
                #         else:
                #             if cat_nm == "Scissors":
                #                 args.single_seq_path = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/{cat_nm}/case{test_seq_idx}/merged_data_with_corr.npy"
                #                 args.cad_model_fn = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/{cat_nm}/case{test_seq_idx}/obj_model.obj"
                #                 args.corr_fn = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/{cat_nm}/case{test_seq_idx}/merged_data.npy" # corr fn #
                #             else:
                #                 args.single_seq_path = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/case{test_seq_idx}/merged_data_with_corr.npy"
                #                 args.cad_model_fn = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/case{test_seq_idx}/obj_model.obj"
                #                 args.corr_fn = f"/data2/xueyi/HOI_Processed_Data_Arti/{cat_nm}/case{test_seq_idx}/merged_data.npy" # corr fn #


                # args.single_seq_path = "/home/xueyi/sim/ContactOpt/./ours_data/case5/merged_data_with_corr.npy"
                # args.cad_model_fn = "/share/datasets/HOI4D_CAD_Model_for_release/articulated/Scissors/011/objs/new-0-align.obj"
                
                
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
                
                
                grab_path = "/data1/xueyi/GRAB_extracted"
                obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
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
                # runloop
                ## load model and sample / predict denoised data via the model from noisy inputs ##
                
                
                ## predict_from_data
                print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
                print("Training...")
                # TrainLoop(args, train_platform, model, diffusion, data).run_loop()
                # predict_from_data
                
                ## TODO: should return used objects as well ##
                ## ==== predict from data ==== ##
                # TrainLoop(args, train_platform, model, diffusion, data).predict_from_data()
                
                if args.diff_basejtse:
                    # tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals #
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
                full_obj_verts = [] # nn_frames x nn_obj_verts x 3 here ! #
                
                
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
                    for i_ins in range(cur_len_full_targets, cur_ed_idxes):
                        cur_ins_rel_idx = i_ins - cur_ed_idxes # negative index here
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
                        # obj_verts: nn_obj_verts x 3 #

                        # transformed_obj_verts = np.matmul(
                        #     obj_verts, cur_ins_obj_rot_mtx
                        # ) + cur_ins_obj_transl
                        
                        transformed_obj_verts = obj_verts
                        full_obj_verts.append(transformed_obj_verts) ## obj_verts ##
                        
                        if args.diff_basejtse:
                            try:
                                full_dec_disp_e_along_normals.append(cur_dec_disp_e_along_normals[cur_ins_rel_idx].detach().cpu().numpy())
                                full_dec_disp_e_vt_normals.append(cur_dec_disp_e_vt_normals[cur_ins_rel_idx].detach().cpu().numpy())
                            except:
                                pass
                        # /data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_/model000001500.pt... 
                        full_targets.append(cur_ins_targets.detach().cpu().numpy())
                        full_outputs.append(cur_ins_outputs.detach().cpu().numpy())
                        full_pert_verts.append(cur_ins_pert_verts.detach().cpu().numpy())
                        full_verts.append(cur_ins_verts.detach().cpu().numpy())
                        
                    # for i_ins in range(cur_targets.shape[-1]):
                    #     cur_ins_rel_idx = i_ins
                    #     cur_ins_targets = cur_targets[..., cur_ins_rel_idx]
                    #     cur_ins_outputs = cur_outputs[..., cur_ins_rel_idx]
                    #     full_targets.append(cur_ins_targets.detach().cpu().numpy())
                    #     full_outputs.append(cur_ins_outputs.detach().cpu().numpy())
                ## full targets ##
                full_targets = np.stack(full_targets, axis=0)
                full_outputs = np.stack(full_outputs, axis=0)
                full_pert_verts = np.stack(full_pert_verts, axis=0)
                full_verts = np.stack(full_verts, axis=0)
                full_obj_verts = np.stack(full_obj_verts, axis=0)
                
                if args.diff_basejtse:
                    full_dec_disp_e_along_normals = np.stack(full_dec_disp_e_along_normals, axis=0)
                    full_dec_disp_e_vt_normals = np.stack(full_dec_disp_e_vt_normals, axis=0)
                
                # and just use 
                # penetration resolving # extract predictions? #
                sv_dict = {
                    'targets': full_targets, ## p
                    'outputs': full_outputs, ### joutput joints 
                    'pert_verts': full_pert_verts,
                    'verts': full_verts,
                    'obj_verts': full_obj_verts, ### full_obj_verts; obj_faces ###
                    'obj_faces': obj_faces,
                    'tot_base_pts': tot_base_pts.numpy() / data_scale_factor, ## total base pts ##
                    'tot_rhand_joints': tot_rhand_joints.numpy() / data_scale_factor,
                    'tot_base_normals': tot_base_normals.numpy(), 
                    'tot_gt_rhand_joints': tot_gt_rhand_joints.numpy() / data_scale_factor,
                    'tot_obj_rot': tot_obj_rot.numpy(),  # ws x 3 x 3 #
                    'tot_obj_transl': tot_obj_transl.numpy(), # ws x 3 
                    'single_obj_normals': obj_vertex_normals,
                }
                
                # seq_idx = int(args.single_seq_path.split("/")[-1].split(".")[0])
                seq_idx = test_seq_idx # test seq idx #
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
                    sv_dict_sv_fn = os.path.join(args.save_dir, sv_predicted_info_fn) # save dict fn 
                    np.save(sv_dict_sv_fn, sv_dict)
                    print(f"Predicted infos saved to {sv_dict_sv_fn}")

        except:
            continue

    train_platform.close()

if __name__ == "__main__":
    main()
