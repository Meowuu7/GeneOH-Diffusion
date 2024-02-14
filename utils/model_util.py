# from model.mdm import MDM
# from model.mdm_ours import MDM as MDM_Ours
# from model.mdm_ours import MDMV3 as MDM_Ours_V3
# from model.mdm_ours import MDMV4 as MDM_Ours_V4
# from model.mdm_ours import MDMV5 as MDM_Ours_V5
# from model.mdm_ours import MDMV6 as MDM_Ours_V6
# from model.mdm_ours import MDMV7 as MDM_Ours_V7
# from model.mdm_ours import MDMV8 as MDM_Ours_V8
# from model.mdm_ours import MDMV9 as MDM_Ours_V9
from model.mdm_ours import MDMV10 as MDM_Ours_V10
from model.mdm_ours import MDMV11 as MDM_Ours_V11
# MDM_Ours_V12
from model.mdm_ours import MDMV12 as MDM_Ours_V12
# MDM_Ours_V13
from model.mdm_ours import MDMV13 as MDM_Ours_V13
# MDM_Ours_V14
from model.mdm_ours import MDMV14 as MDM_Ours_V14
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
import torch
from torch import optim, nn
import torch.nn.functional as F
from manopth.manolayer import ManoLayer
import numpy as np
import trimesh
import os
from diffusion.respace_ours import SpacedDiffusion as SpacedDiffusion_Ours
# SpacedDiffusionV2
from diffusion.respace_ours import SpacedDiffusionV2 as SpacedDiffusion_OursV2
from diffusion.respace_ours import SpacedDiffusionV3 as SpacedDiffusion_OursV3
# SpacedDiffusionV4
from diffusion.respace_ours import SpacedDiffusionV4 as SpacedDiffusion_OursV4
# SpacedDiffusion_OursV5
from diffusion.respace_ours import SpacedDiffusionV5 as SpacedDiffusion_OursV5
# SpacedDiffusion_OursV6
from diffusion.respace_ours import SpacedDiffusionV6 as SpacedDiffusion_OursV6
# SpacedDiffusion_OursV7
from diffusion.respace_ours import SpacedDiffusionV7 as SpacedDiffusion_OursV7
from diffusion.respace_ours import SpacedDiffusionV9 as SpacedDiffusion_OursV9



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



def gaussian_entropy(logvar): # gaussian entropy ##
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z): # feature dim
    dim = z.size(-1) # dim size -1
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def load_multiple_models_fr_path(model_path, model):
    model_paths = model_path.split(";")
    print(f"Loading multiple models with split model_path: {model_paths}")
    setting_to_model_path = {}
    for cur_path in model_paths:
        cur_setting_nm, cur_model_path = cur_path.split(':')
        setting_to_model_path[cur_setting_nm] = cur_model_path
    loaded_dict = {}
    for cur_setting in setting_to_model_path:
        cur_model_path = setting_to_model_path[cur_setting]
        cur_model_state_dict = torch.load(cur_model_path, map_location='cpu')
        if cur_setting == 'diff_realbasejtsrel':
            interested_keys = [
                'real_basejtsrel_input_process', 'real_basejtsrel_sequence_pos_encoder', 'real_basejtsrel_seqTransEncoder', 'real_basejtsrel_embed_timestep', 'real_basejtsrel_sequence_pos_denoising_encoder', 'real_basejtsrel_denoising_seqTransEncoder', 'real_basejtsrel_output_process'
            ]
        elif cur_setting == 'diff_basejtsrel':
            interested_keys = [
                'avg_joints_sequence_input_process', 'joints_offset_input_process', 'sequence_pos_encoder', 'seqTransEncoder', 'logvar_seqTransEncoder', 'embed_timestep', 'basejtsrel_denoising_embed_timestep', 'sequence_pos_denoising_encoder', 'basejtsrel_denoising_seqTransEncoder', 'basejtsrel_glb_denoising_latents_trans_layer', 'avg_joint_sequence_output_process', 'joint_offset_output_process', 'output_process'
            ]
        elif cur_setting == 'diff_realbasejtsrel_to_joints':
            interested_keys = [
                'real_basejtsrel_to_joints_input_process', 'real_basejtsrel_to_joints_sequence_pos_encoder', 'real_basejtsrel_to_joints_seqTransEncoder', 'real_basejtsrel_to_joints_embed_timestep', 'real_basejtsrel_to_joints_sequence_pos_denoising_encoder', 'real_basejtsrel_to_joints_denoising_seqTransEncoder', 'real_basejtsrel_to_joints_output_process', 
            ]
        else:
            raise ValueError(f"cur_setting:{cur_setting} Not implemented yet")
        for k in cur_model_state_dict:
            for cur_inter_key in interested_keys:
                if cur_inter_key in k:
                    loaded_dict[k] = cur_model_state_dict[k]
    model_dict = model.state_dict()
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    
            
                


def load_model_wo_clip(model, state_dict): # missing_keys: in the current model but not found in the state_dict? # unexpected_keys: not in the current model but found inthe state_dict? 
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print(unexpected_keys)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

### create model and diffusion ## # 
def create_model_and_diffusion(args, data):
    if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist"]:
        model = MDM_Ours(**get_model_args(args, data))
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we"]:
        model = MDM_Ours_V3(**get_model_args(args, data))
    # MDM_Ours_V4
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj"]:
        model = MDM_Ours_V4(**get_model_args(args, data))
    # obj_base_rel_dist_we_wj_latents
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj_latents"]:
        if args.diff_spatial:
            if args.pred_joints_offset:
                if args.diff_joint_quants:
                    model =  MDM_Ours_V13(**get_model_args(args, data))
                elif args.diff_hand_params:
                    model =  MDM_Ours_V14(**get_model_args(args, data))
                else:
                    if args.finetune_with_cond:
                        print(f"Using MDM ours V12!!!!")
                        model =  MDM_Ours_V12(**get_model_args(args, data))
                    else:
                        print(f"Using MDM ours V10!!!!")
                        model =  MDM_Ours_V10(**get_model_args(args, data))
            else:
                print(f"Using MDM ours V9!!!!")
                model =  MDM_Ours_V9(**get_model_args(args, data))
        elif args.diff_latents:
            print(f"Using MDM ours V11!!!!")
            model =  MDM_Ours_V11(**get_model_args(args, data))
        elif args.use_sep_models:
            if args.use_vae:
                if args.pred_basejtsrel_avgjts:
                    print(f"Using MDM ours V8!!!!")
                    model = MDM_Ours_V8(**get_model_args(args, data))
                else:
                    model = MDM_Ours_V7(**get_model_args(args, data))
            else:
                model = MDM_Ours_V6(**get_model_args(args, data))
        else:
            model = MDM_Ours_V5(**get_model_args(args, data))
    else:
        model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

# give utils to models #
def get_model_args(args, data):
    # default_args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor' ## get model arguments ##
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml']: ## from 
        data_rep = 'hml_vec'
        njoints = 263 # joints
        nfeats = 1
    elif args.dataset in ['motion_ours']:
        data_rep = 'xyz'
        njoints = 21
        nfeats = 3
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    ## modeltype; 
    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'args': args}


## negative distances ##
## negative distances ##
## positive distances --> neareste points #
# approximate penetration detections #
## approximiate nearest poonts distances -> should not be negative ones here ##
# sampling -> should be added in the sampling process #
# physically realistic #
# optimize it frame-by-frame? #
# penetration penelty -> nearest points -> should not pentrate
# approximation pealty 
# human like realistic -> optimize throug the mano model #
# would such terms help with valid joints optimization? #
# TODO: other optimization strategies? e.g. sequential optimziation> #
def optimize_sampled_hand_joints(sampled_joints, rel_base_pts_to_joints, dists_base_pts_to_joints, base_pts, base_normals):
    # sampled_joints: bsz x ws x nnj x 3
    # signed distances 
    # smoothness 
    bsz, ws, nnj = sampled_joints.shape[:3]
    device = sampled_joints.device
    coarse_lr = 0.1
    num_iters = 100 # if i_iter > 0 else 1 ## nn-coarse-iters for global transformations #
    mano_path = "/data1/sim/mano_models/mano/models"
    
    base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    base_normals_exp = base_normals.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    
    signed_dist_e_coeff = 1.0
    signed_dist_e_coeff = 0.0
    
    
    ### start optimization ###
    # setup MANO layer
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_path for the mano model #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)
    
    ## random init variables ##
    beta_var = torch.randn([bsz, 10]).to(device)
    rot_var = torch.randn([bsz * ws, 3]).to(device)
    theta_var = torch.randn([bsz * ws, 24]).to(device)
    transl_var = torch.randn([bsz * ws, 3]).to(device)
    
    beta_var.requires_grad_()
    rot_var.requires_grad_()
    theta_var.requires_grad_()
    transl_var.requires_grad_()
    opt = optim.Adam([rot_var, transl_var], lr=coarse_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        if dists_base_pts_to_joints is not None:
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).to(device).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # rel_base_pts_to_hand_joints: bsz x ws x nnj x nnb x 3 #
        signed_dist_mask = signed_dist_base_pts_to_hand_joints < 0.
        l2_dist_rel_joints_to_base_pts_mask = torch.sqrt(
            torch.sum(rel_base_pts_to_hand_joints ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_base_pts_mask.float()) > 1.5
        dot_rel_with_normals = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #

        # dot_rel_with_normals: bsz x ws x nnj x nnb
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        signed_dist_e = dot_rel_with_normals * signed_dist_base_pts_to_hand_joints
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        
        
        
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # dist_rhand_joints_to_base_pts = torch.sum(
        #     (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        # )
        # # minn_dists_idxes: bsz x ws x nnj -->  
        # minn_dists_to_base_pts, minn_dists_idxes = torch.min(
        #     dist_rhand_joints_to_base_pts, dim=-1
        # )
        # # base_pts: bsz x nn_base_pts x 3 #
        # # base_pts: bsz x ws x nn_base_pts x 3 #
        # # bsz x ws x nnj 
        
        # # object verts and object faces #
        # ## other than the sampling process; not 
        # # bsz x ws x nnj x 3 ##
        # nearest_base_pts = batched_index_select_ours(
        #     base_pts_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 # # base normalse #
        # nearest_base_normals = batched_index_select_ours(
        #     base_normals_exp, indices=minn_dists_idxes, dim=2
        # )  
        # # bsz x ws x nnj x 3 #  # the nearest distance points may be of some ambiguous 
        # rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.05
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        # ### ==== mean of signed distances ==== ###
        # signed_dist_e = torch.sum( # penetration 
        #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # ) / torch.clamp(
        #     torch.sum(signed_dist_mask.float()), min=1e-5
        # ).item()
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        ### signed distance coeff -> the distance coeff #
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    fine_lr = 0.1
    num_iters = 1000
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        dist_rhand_joints_to_base_pts = torch.sum(
            (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        )
        # minn_dists_idxes: bsz x ws x nnj -->  
        minn_dists_to_base_pts, minn_dists_idxes = torch.min(
            dist_rhand_joints_to_base_pts, dim=-1
        )
        # base_pts: bsz x nn_base_pts x 3 #
        # base_pts: bsz x ws x nn_base_pts x 3 #
        # bsz x ws x nnj 
        # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # bsz x ws x nnj x 3 ##
        nearest_base_pts = batched_index_select_ours(
            base_pts_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        nearest_base_normals = batched_index_select_ours(
            base_normals_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # bsz x ws x nnj #
        signed_dist_joints_to_base_pts = torch.sum(
            rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        )
        # should not be negative
        signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
            torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        
        ### ==== mean of signed distances ==== ###
        signed_dist_e = torch.sum(
            -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        ) / torch.clamp(
            torch.sum(signed_dist_mask.float()), min=1e-5
        ).item()
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    
    ### refine the optimization with signed energy ##
    signed_dist_e_coeff = 1.0
    fine_lr = 0.1
    num_iters = 1000
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # rel_base_pts_to_hand_joints: bsz x ws x nnj x nnb x 3 #
        signed_dist_mask = signed_dist_base_pts_to_hand_joints < 0.
        l2_dist_rel_joints_to_base_pts_mask = torch.sqrt(
            torch.sum(rel_base_pts_to_hand_joints ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_base_pts_mask.float()) > 1.5
        ## === dot rel with normals === ##
        # dot_rel_with_normals = torch.sum(
        #     rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        # )
        ## === dot rel with normals === ##
        ## === dot rel with rel, strategy 3 === ##
        dot_rel_with_normals = torch.sum(
            -1.0 * rel_base_pts_to_hand_joints * rel_base_pts_to_hand_joints, dim=-1
        )
        ## === dot rel with rel, strategy 3 === ##
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #

        # dot_rel_with_normals: bsz x ws x nnj x nnb
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        signed_dist_e = dot_rel_with_normals * signed_dist_base_pts_to_hand_joints
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # hard projections for 
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        # ## base_pts: bsz x nn_base_pts x 3
        # ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        # ## bsz x ws x nnj x nnb ##
        # dist_rhand_joints_to_base_pts = torch.sum(
        #     (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        # )
        # # minn_dists_idxes: bsz x ws x nnj -->  
        # minn_dists_to_base_pts, minn_dists_idxes = torch.min(
        #     dist_rhand_joints_to_base_pts, dim=-1
        # )
        # 
        # # base_pts: bsz x nn_base_pts x 3 #
        # # base_pts: bsz x ws x nn_base_pts x 3 #
        # # bsz x ws x nnj 
        # # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # # bsz x ws x nnj x 3 ##
        # nearest_base_pts = batched_index_select_ours(
        #     base_pts_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 #
        # nearest_base_normals = batched_index_select_ours(
        #     base_normals_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 #
        # rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        # ## === luojisiwei and others === ##
        # # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        # #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # # ) < 0.05 
        # ## === luojisiwei and others === ##
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.1
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        
        # ### ==== mean of signed distances ==== ###
        # # signed_dist_e = torch.sum(
        # #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # # ) / torch.clamp(
        # #     torch.sum(signed_dist_mask.float()), min=1e-5
        # # ).item()
        
        # # signed_dist_joints_to_base_pts: bsz x ws x nnj # -> disstances 
        # signed_dist_joints_to_base_pts = signed_dist_joints_to_base_pts.detach()
        # # 
        
        ## penetraition resolving --- strategy 
        # dot_rel_with_normals = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # signed_dist_mask = signed_dist_mask.detach() # detach the mask #
        # # bsz x ws x nnj --> the loss term
        # ## signed distances 3 #### isgned distance 3 ###
        # ## dotrelwithnormals, ##
        # # # signed_dist_mask -> the distances 
        
        # # dot_rel_with_normals: bsz x ws x nnj x nnb
        # avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        
        # signed_dist_e = dot_rel_with_normals * signed_dist_joints_to_base_pts
        # signed_dist_e = torch.sum(
        #     signed_dist_e[signed_dist_mask]
        # ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        # ###### ====== get loss for signed distances ==== ###
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        
        ## judeg whether inside the object and only project those one inside of the object
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        # shoudl take a 
        # how to proejct the jvertex
        # hwo to project the veretex
        # weighted sum of the projectiondirection
        # weights of each base point
        # atraction field -> should be able to learn the penetration resolving strategy 
        # stochestic penetration resolving strategy #
        
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
        # avg_masks
        print('\tAvg masks: {}'.format(avg_masks.item()))
    
    
    
    ''' returning sampled_joints ''' 
    sampled_joints = hand_joints
    np.save("optimized_verts.npy", hand_verts.detach().cpu().numpy())
    print(f"Optimized verts saved to optimized_verts.npy")
    return sampled_joints.detach()



def get_obj_trimesh_list(obj_verts, obj_faces):
    tot_trimeshes = []
    tot_n = len(obj_verts)
    for i_obj in range(tot_n):
        cur_obj_verts, cur_obj_faces = obj_verts[i_obj], obj_faces[i_obj]
        if isinstance(cur_obj_verts, torch.Tensor):
            cur_obj_verts = cur_obj_verts.detach().cpu().numpy()
        if isinstance(cur_obj_faces, torch.Tensor):
            cur_obj_faces = cur_obj_faces.detach().cpu().numpy()
        cur_obj_mesh = trimesh.Trimesh(vertices=cur_obj_verts, faces=cur_obj_faces,
        process=False, use_embree=True)
        tot_trimeshes.append(cur_obj_mesh)
    return tot_trimeshes

def judge_penetrated_points(obj_mesh, subj_pts):
    # bsz 
    tot_pts_inside_objmesh_labels = []
    nn_bsz = len(obj_mesh)
    for i_bsz in range(nn_bsz):
        cur_obj_mesh = obj_mesh[i_bsz]
        cur_subj_pts = subj_pts[i_bsz].detach().cpu().numpy()
        ori_subj_pts_shape = cur_subj_pts.shape
        if len(cur_subj_pts.shape) > 2:
            cur_subj_pts = cur_subj_pts.reshape(cur_subj_pts.shape[0] * cur_subj_pts.shape[1], 3)
        # 
        pts_inside_objmesh = cur_obj_mesh.contains(cur_subj_pts)
        pts_inside_objmesh = pts_inside_objmesh.astype(np.float32)
        ### reshape inside_objmesh labels ###
        pts_inside_objmesh = pts_inside_objmesh.reshape(*ori_subj_pts_shape[:-1])
        
        tot_pts_inside_objmesh_labels.append(pts_inside_objmesh)
    tot_pts_inside_objmesh_labels = np.stack(tot_pts_inside_objmesh_labels, axis=0) # nn_bsz x nn_subj_pts
    tot_pts_inside_objmesh_labels = torch.from_numpy(tot_pts_inside_objmesh_labels).float()
    return tot_pts_inside_objmesh_labels.to(subj_pts.device) # gt inside objmesh labels and to the pts device #

# TODO: other optimization strategies? e.g. sequential optimziation> #
def optimize_sampled_hand_joints_wobj(sampled_joints, rel_base_pts_to_joints, dists_base_pts_to_joints, base_pts, base_normals, obj_verts, obj_normals, obj_faces):
    # sampled_joints: bsz x ws x nnj x 3
    # signed distances 
    
    # smoothness 
    # tot_n_objs #
    tot_obj_trimeshes = get_obj_trimesh_list(obj_verts, obj_faces)
    
    ## TODO: write the collect function for object verts, normals, faces ##
    
    
    ### A simple penetration resolving strategy is as follows:
    #### 1) get vertices in the object; 2) get nearest base points (for simplicity); 3) project the vertex to the base point ####
    ## 1) for joints only; 
    ## 2) for vertices;
    ## 3) for vertices ##
    ## TODO: optimzie the resolvign strategy stated above ##
    
    bsz, ws, nnj = sampled_joints.shape[:3]
    device = sampled_joints.device
    coarse_lr = 0.1
    num_iters = 100 # if i_iter > 0 else 1 ## nn-coarse-iters for global transformations #
    mano_path = "/data1/sim/mano_models/mano/models"
    
    # obj_verts: bsz x nnobjverts x 
    
    base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    base_normals_exp = base_normals.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    
    signed_dist_e_coeff = 1.0
    signed_dist_e_coeff = 0.0
    
    
    ### start optimization ###
    # setup MANO layer
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_path for the mano model #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)
    
    ## random init variables ##
    beta_var = torch.randn([bsz, 10]).to(device)
    rot_var = torch.randn([bsz * ws, 3]).to(device)
    theta_var = torch.randn([bsz * ws, 24]).to(device)
    transl_var = torch.randn([bsz * ws, 3]).to(device)
    
    beta_var.requires_grad_()
    rot_var.requires_grad_()
    theta_var.requires_grad_()
    transl_var.requires_grad_()
    opt = optim.Adam([rot_var, transl_var], lr=coarse_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        if dists_base_pts_to_joints is not None:
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).to(device).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # rel_base_pts_to_hand_joints: bsz x ws x nnj x nnb x 3 #
        signed_dist_mask = signed_dist_base_pts_to_hand_joints < 0.
        l2_dist_rel_joints_to_base_pts_mask = torch.sqrt(
            torch.sum(rel_base_pts_to_hand_joints ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_base_pts_mask.float()) > 1.5
        dot_rel_with_normals = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #

        # dot_rel_with_normals: bsz x ws x nnj x nnb
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        signed_dist_e = dot_rel_with_normals * signed_dist_base_pts_to_hand_joints
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        
        
        
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # dist_rhand_joints_to_base_pts = torch.sum(
        #     (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        # )
        # # minn_dists_idxes: bsz x ws x nnj -->  
        # minn_dists_to_base_pts, minn_dists_idxes = torch.min(
        #     dist_rhand_joints_to_base_pts, dim=-1
        # )
        # # base_pts: bsz x nn_base_pts x 3 #
        # # base_pts: bsz x ws x nn_base_pts x 3 #
        # # bsz x ws x nnj 
        
        # # object verts and object faces #
        # ## other than the sampling process; not 
        # # bsz x ws x nnj x 3 ##
        # nearest_base_pts = batched_index_select_ours(
        #     base_pts_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 # # base normalse #
        # nearest_base_normals = batched_index_select_ours(
        #     base_normals_exp, indices=minn_dists_idxes, dim=2
        # )  
        # # bsz x ws x nnj x 3 #  # the nearest distance points may be of some ambiguous 
        # rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.05
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        # ### ==== mean of signed distances ==== ###
        # signed_dist_e = torch.sum( # penetration 
        #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # ) / torch.clamp(
        #     torch.sum(signed_dist_mask.float()), min=1e-5
        # ).item()
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        ### signed distance coeff -> the distance coeff #
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    fine_lr = 0.1
    num_iters = 1000
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        dist_rhand_joints_to_base_pts = torch.sum(
            (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        )
        # minn_dists_idxes: bsz x ws x nnj -->  
        minn_dists_to_base_pts, minn_dists_idxes = torch.min(
            dist_rhand_joints_to_base_pts, dim=-1
        )
        # base_pts: bsz x nn_base_pts x 3 #
        # base_pts: bsz x ws x nn_base_pts x 3 #
        # bsz x ws x nnj 
        # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # bsz x ws x nnj x 3 ##
        nearest_base_pts = batched_index_select_ours(
            base_pts_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        nearest_base_normals = batched_index_select_ours(
            base_normals_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # bsz x ws x nnj #
        signed_dist_joints_to_base_pts = torch.sum(
            rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        )
        # should not be negative
        signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
            torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        
        ### ==== mean of signed distances ==== ###
        signed_dist_e = torch.sum(
            -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        ) / torch.clamp(
            torch.sum(signed_dist_mask.float()), min=1e-5
        ).item()
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    
    # tot_obj_trimeshes
    ### refine the optimization with signed energy ##
    signed_dist_e_coeff = 1.0 # 
    fine_lr = 0.1
    # num_iters = 1000 # 
    num_iters = 100 # reinement #
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters): # 
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # # rel_base_pts_to_hand_joints: bsz x ws x nnj x nnb x 3 #
        # signed_dist_mask = signed_dist_base_pts_to_hand_joints < 0.
        # l2_dist_rel_joints_to_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_base_pts_to_hand_joints ** 2, dim=-1)
        # ) < 0.05
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_base_pts_mask.float()) > 1.5
        # ## === dot rel with normals === ##
        # # dot_rel_with_normals = torch.sum(
        # #     rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        # # )
        # ## === dot rel with normals === ##
        # ## === dot rel with rel, strategy 3 === ##
        # dot_rel_with_normals = torch.sum(
        #     -1.0 * rel_base_pts_to_hand_joints * rel_base_pts_to_hand_joints, dim=-1
        # )
        # ## === dot rel with rel, strategy 3 === ##
        # signed_dist_mask = signed_dist_mask.detach() # detach the mask #

        # # dot_rel_with_normals: bsz x ws x nnj x nnb
        # avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        # signed_dist_e = dot_rel_with_normals * signed_dist_base_pts_to_hand_joints
        # signed_dist_e = torch.sum(
        #     signed_dist_e[signed_dist_mask]
        # ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        # ###### ====== get loss for signed distances ==== ###
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        
        ## use all base pts ##
        
        {
        # hard projections for 
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        # ## base_pts: bsz x nn_base_pts x 3
        # ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        # ## bsz x ws x nnj x nnb ##
        # dist_rhand_joints_to_base_pts = torch.sum(
        #     (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        # )
        # # minn_dists_idxes: bsz x ws x nnj -->  
        # minn_dists_to_base_pts, minn_dists_idxes = torch.min(
        #     dist_rhand_joints_to_base_pts, dim=-1
        # )
        # 
        # # base_pts: bsz x nn_base_pts x 3 #
        # # base_pts: bsz x ws x nn_base_pts x 3 #
        # # bsz x ws x nnj 
        # # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # # bsz x ws x nnj x 3 ##
        # nearest_base_pts = batched_index_select_ours(
        #     base_pts_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 #
        # nearest_base_normals = batched_index_select_ours(
        #     base_normals_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 #
        # rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        # ## === luojisiwei and others === ##
        # # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        # #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # # ) < 0.05 
        # ## === luojisiwei and others === ##
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.1
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        
        # ### ==== mean of signed distances ==== ###
        # # signed_dist_e = torch.sum(
        # #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # # ) / torch.clamp(
        # #     torch.sum(signed_dist_mask.float()), min=1e-5
        # # ).item()
        
        # # signed_dist_joints_to_base_pts: bsz x ws x nnj # -> disstances 
        # signed_dist_joints_to_base_pts = signed_dist_joints_to_base_pts.detach()
        # # 
        
        ## penetraition resolving --- strategy 
        # dot_rel_with_normals = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # signed_dist_mask = signed_dist_mask.detach() # detach the mask #
        # # bsz x ws x nnj --> the loss term
        # ## signed distances 3 #### isgned distance 3 ###
        # ## dotrelwithnormals, ##
        # # # signed_dist_mask -> the distances 
        
        # # dot_rel_with_normals: bsz x ws x nnj x nnb
        # avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        
        # signed_dist_e = dot_rel_with_normals * signed_dist_joints_to_base_pts
        # signed_dist_e = torch.sum(
        #     signed_dist_e[signed_dist_mask]
        # ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        # ###### ====== get loss for signed distances ==== ###
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        }
        
        # bsz x ws x nnj # --> objmesh insides pts labels 
        pts_inside_objmesh_labels = judge_penetrated_points(tot_obj_trimeshes, hand_joints)
        pts_inside_objmesh_labels_mask = pts_inside_objmesh_labels.bool()
        
        
        # {
        # hard projections for 
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        dist_rhand_joints_to_base_pts = torch.sum(
            (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        )
        # minn_dists_idxes: bsz x ws x nnj -->  
        # base_pts
        minn_dists_to_base_pts, minn_dists_idxes = torch.min(
            dist_rhand_joints_to_base_pts, dim=-1
        )
        
        # base_pts: bsz x nn_base_pts x 3 #
        # base_pts: bsz x ws x nn_base_pts x 3 #
        # bsz x ws x nnj 
        # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # bsz x ws x nnj x 3 ##
        # simple penetration ## 
        nearest_base_pts = batched_index_select_ours(
            base_pts_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        nearest_base_normals = batched_index_select_ours(
            base_normals_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        ## === luojisiwei and others === ##
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.05 
        ## === luojisiwei and others === ##
        ##### ===== GET l2_distance mask ===== #####
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.1
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        ##### ===== GET l2_distance mask ===== #####
        
        ### ==== mean of signed distances ==== ###
        # signed_dist_e = torch.sum(
        #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # ) / torch.clamp(
        #     torch.sum(signed_dist_mask.float()), min=1e-5
        # ).item()
        
        # signed_dist_joints_to_base_pts: bsz x ws x nnj # -> disstances 
        signed_dist_joints_to_base_pts = signed_dist_joints_to_base_pts.detach()
        # 
        
        # dot rel 
        # penetraition resolving --- strategy 
        # dot_rel_with_normals = torch.sum( # dot rhand joints with normals #
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # 
        dot_rel_with_normals = torch.sum( # dot rhand joints with normals #
            -rel_joints_to_nearest_base_pts * rel_joints_to_nearest_base_pts, dim=-1
        )
        #### Get masks for penetrated joint points ####
        # signed_dist_mask = (signed_dist_mask.float() + pts_inside_objmesh_labels_mask.float()) > 1.5
        signed_dist_mask = pts_inside_objmesh_labels_mask
        # bsz x ws x nnj 
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #
        # bsz x ws x nnj --> the loss term
        ## signed distances 3 #### isgned distance 3 ###
        ## dotrelwithnormals, ##
        # # signed_dist_mask -> the distances 
        
        # dot_rel_with_normals: bsz x ws x nnj x nnb # avg over windows and batches #
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        ## get singed distance energies ### ## projection ##
        # signed_dist_e = dot_rel_with_normals * signed_dist_joints_to_base_pts
        signed_dist_e = -1.0 * dot_rel_with_normals
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # cannot mask in some caes 
        # change of isgned distances #
        
        
        # intersection spline 
        ## judeg whether inside the object and only project those one inside of the object
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        #### ==== sv_dict ==== ####
        sv_dict = {
            'pts_inside_objmesh_labels_mask': pts_inside_objmesh_labels_mask.detach().cpu().numpy(),
            'hand_joints': hand_joints.detach().cpu().numpy(),
            'obj_verts': [cur_verts.detach().cpu().numpy() for cur_verts in obj_verts],
            'obj_faces': [cur_faces.detach().cpu().numpy() for cur_faces in obj_faces],
            'base_pts': base_pts.detach().cpu().numpy(),
            'base_normals': base_normals.detach().cpu().numpy(), # bsz x nnb x 3 -> bsz x nnb x 3 -> base normals #
            'nearest_base_pts': nearest_base_pts.detach().cpu().numpy(), # bsz x ws x nnj x 3 # 
            'nearest_base_normals': nearest_base_normals.detach().cpu().numpy(), # bsz x ws x nnj x 3 --> base normals and pts 
        }
        # 
        sv_dict_folder = "/data1/sim/mdm/tmp_saving"
        os.makedirs(sv_dict_folder, exist_ok=True)
        sv_dict_fn = os.path.join(sv_dict_folder, f"optim_iter_{i_iter}.npy")
        np.save(sv_dict_fn, sv_dict)
        print(f"Obj and subj saved to {sv_dict_fn}")
        #### ==== sv_dict ==== ####
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        # shoudl take a 
        # how to proejct the jvertex
        # hwo to project the veretex
        # weighted sum of the projectiondirection
        # weights of each base point
        # atraction field -> should be able to learn the penetration resolving strategy 
        # stochestic penetration resolving strategy #
        
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
        # avg_masks
        print('\tAvg masks: {}'.format(avg_masks.item()))
    
    
    
    ''' returning sampled_joints ''' 
    sampled_joints = hand_joints
    np.save("optimized_verts.npy", hand_verts.detach().cpu().numpy())
    print(f"Optimized verts saved to optimized_verts.npy")
    return sampled_joints.detach()


# TODO: other optimization strategies? e.g. sequential optimziation> #
def optimize_sampled_hand_joints_wobj_v2(sampled_joints, rel_base_pts_to_joints, dists_base_pts_to_joints, base_pts, base_normals, obj_verts, obj_normals, obj_faces):
    # sampled_joints: bsz x ws x nnj x 3 #
    # sampled_joints: bsz x ws x nnj x 3 # obj trimeshes #
    tot_obj_trimeshes = get_obj_trimesh_list(obj_verts, obj_faces)
    
    ## TODO: write the collect function for object verts, normals, faces ##
    
    ### A simple penetration resolving strategy is as follows:
    #### 1) get vertices in the object; 2) get nearest base points (for simplicity); 3) project the vertex to the base point ####
    ## 1) for joints only;
    ## 2) for vertices;
    ## 3) for vertices;
    ## TODO: optimzie the resolvign strategy stated above ##
    
    bsz, ws, nnj = sampled_joints.shape[:3]
    device = sampled_joints.device
    coarse_lr = 0.1
    num_iters = 100 # if i_iter > 0 else 1 ## nn-coarse-iters for global transformations #
    mano_path = "/data1/sim/mano_models/mano/models"
    
    # obj_verts: bsz x nnobjverts x 
    
    base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    base_normals_exp = base_normals.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
    
    signed_dist_e_coeff = 1.0
    signed_dist_e_coeff = 0.0
    
    
    ### start optimization ###
    # setup MANO layer
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_path for the mano model #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)
    
    ## random init variables ##
    beta_var = torch.randn([bsz, 10]).to(device)
    rot_var = torch.randn([bsz * ws, 3]).to(device)
    theta_var = torch.randn([bsz * ws, 24]).to(device)
    transl_var = torch.randn([bsz * ws, 3]).to(device)
    
    beta_var.requires_grad_()
    rot_var.requires_grad_()
    theta_var.requires_grad_()
    transl_var.requires_grad_()
    opt = optim.Adam([rot_var, transl_var], lr=coarse_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        if dists_base_pts_to_joints is not None:
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).to(device).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        # rel_base_pts_to_hand_joints: bsz x ws x nnj x nnb x 3 #
        signed_dist_mask = signed_dist_base_pts_to_hand_joints < 0.
        l2_dist_rel_joints_to_base_pts_mask = torch.sqrt(
            torch.sum(rel_base_pts_to_hand_joints ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_base_pts_mask.float()) > 1.5
        dot_rel_with_normals = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #

        # dot_rel_with_normals: bsz x ws x nnj x nnb
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        signed_dist_e = dot_rel_with_normals * signed_dist_base_pts_to_hand_joints
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 2: use all base pts, rel, dists for resolving '''
        
        
        
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # dist_rhand_joints_to_base_pts = torch.sum(
        #     (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        # )
        # # minn_dists_idxes: bsz x ws x nnj -->  
        # minn_dists_to_base_pts, minn_dists_idxes = torch.min(
        #     dist_rhand_joints_to_base_pts, dim=-1
        # )
        # # base_pts: bsz x nn_base_pts x 3 #
        # # base_pts: bsz x ws x nn_base_pts x 3 #
        # # bsz x ws x nnj 
        
        # # object verts and object faces #
        # ## other than the sampling process; not 
        # # bsz x ws x nnj x 3 ##
        # nearest_base_pts = batched_index_select_ours(
        #     base_pts_exp, indices=minn_dists_idxes, dim=2
        # )
        # # bsz x ws x nnj x 3 # # base normalse #
        # nearest_base_normals = batched_index_select_ours(
        #     base_normals_exp, indices=minn_dists_idxes, dim=2
        # )  
        # # bsz x ws x nnj x 3 #  # the nearest distance points may be of some ambiguous 
        # rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # # bsz x ws x nnj #
        # signed_dist_joints_to_base_pts = torch.sum(
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # # should not be negative
        # signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        # l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
        #     torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        # ) < 0.05
        # signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        # ### ==== mean of signed distances ==== ###
        # signed_dist_e = torch.sum( # penetration 
        #     -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        # ) / torch.clamp(
        #     torch.sum(signed_dist_mask.float()), min=1e-5
        # ).item()
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        ### signed distance coeff -> the distance coeff #
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    fine_lr = 0.1
    num_iters = 1000
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters):
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        dist_rhand_joints_to_base_pts = torch.sum(
            (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        )
        # minn_dists_idxes: bsz x ws x nnj -->  
        minn_dists_to_base_pts, minn_dists_idxes = torch.min(
            dist_rhand_joints_to_base_pts, dim=-1
        )
        # base_pts: bsz x nn_base_pts x 3 #
        # base_pts: bsz x ws x nn_base_pts x 3 #
        # bsz x ws x nnj 
        # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # bsz x ws x nnj x 3 ##
        nearest_base_pts = batched_index_select_ours(
            base_pts_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        nearest_base_normals = batched_index_select_ours(
            base_normals_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 #
        rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        # bsz x ws x nnj #
        signed_dist_joints_to_base_pts = torch.sum(
            rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        )
        # should not be negative
        signed_dist_mask = signed_dist_joints_to_base_pts < 0.
        l2_dist_rel_joints_to_nearest_base_pts_mask = torch.sqrt(
            torch.sum(rel_joints_to_nearest_base_pts ** 2, dim=-1)
        ) < 0.05
        signed_dist_mask = (signed_dist_mask.float() + l2_dist_rel_joints_to_nearest_base_pts_mask.float()) > 1.5
        
        ### ==== mean of signed distances ==== ###
        signed_dist_e = torch.sum(
            -1.0 * signed_dist_joints_to_base_pts[signed_dist_mask]
        ) / torch.clamp(
            torch.sum(signed_dist_mask.float()), min=1e-5
        ).item()
        
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
    
    
    # tot_obj_trimeshes
    ### refine the optimization with signed energy ##
    # 
    # signed_dist_jts_to_nearest_base_pts = []
    # tot_nearest_base_pts = []
    # tot_nearest_base_normals = []
    
    signed_dist_e_coeff = 1.0 # 
    fine_lr = 0.1
    # num_iters = 1000 # 
    num_iters = 100 # reinement #
    opt = optim.Adam([rot_var, transl_var, beta_var, theta_var], lr=fine_lr)
    for i_iter in range(num_iters): # 
        opt.zero_grad()
        # mano_layer #
        hand_verts, hand_joints = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, ws, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(bsz, ws, 778, 3) * 0.001 ## bsz x ws x nn
        hand_joints = hand_joints.view(bsz, ws, -1, 3) * 0.001
        
        ### === e1 should be close to predicted values === ###
        # bsz x ws x nnj x nnb x 3 #
        rel_base_pts_to_hand_joints = hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        # bs zx ws x nnj x nnb # 
        signed_dist_base_pts_to_hand_joints = torch.sum(
            rel_base_pts_to_hand_joints * base_normals.unsqueeze(1).unsqueeze(1), dim=-1
        )
        rel_e = torch.sum(
            (rel_base_pts_to_hand_joints - rel_base_pts_to_joints) ** 2, dim=-1
        ).mean()
        
        # dists_base_pts_to_joints ## dists_base_pts_to_joints ##
        if dists_base_pts_to_joints is not None: ## dists_base_pts_to_joints ##
            dist_e = torch.sum(
                (signed_dist_base_pts_to_hand_joints - dists_base_pts_to_joints) ** 2, dim=-1
            ).mean()
        else:
            dist_e = torch.zeros((1,), dtype=torch.float32).mean()
        
        ### ==== inside the objemesh labels ==== ###
        # bsz x ws x nnj # --> objmesh insides pts labels #
        pts_inside_objmesh_labels = judge_penetrated_points(tot_obj_trimeshes, hand_joints)
        pts_inside_objmesh_labels_mask = pts_inside_objmesh_labels.bool()
        
        
        # {
        # hard projections for 
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        ### === e2 the signed distances to nearest points should not be negative to the neareste === ###
        ## base_pts: bsz x nn_base_pts x 3
        ## bsz x ws x nnj x 1 x 3 -- bsz x 1 x 1 x nnb x 3 ##
        ## bsz x ws x nnj x nnb ##
        dist_rhand_joints_to_base_pts = torch.sum(
            (hand_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1
        )
        # minn_dists_idxes: bsz x ws x nnj #
        # base_pts
        minn_dists_to_base_pts, minn_dists_idxes = torch.min(
            dist_rhand_joints_to_base_pts, dim=-1
        )
        
        # base_pts: bsz x nn_base_pts x 3 #
        # base_pts: bsz x ws x nn_base_pts x 3 #
        # bsz x ws x nnj 
        # base_pts_exp = base_pts.unsqueeze(1).repeat(1, ws, 1, 1).contiguous()
        # bsz x ws x nnj x 3 ##
        # simple penetration ## 
        nearest_base_pts = batched_index_select_ours(
            base_pts_exp, indices=minn_dists_idxes, dim=2
        )
        # bsz x ws x nnj x 3 # # 
        nearest_base_normals = batched_index_select_ours(
            base_normals_exp, indices=minn_dists_idxes, dim=2
        )
        tot_masks = []
        tot_base_pts = []
        tot_base_normals = []
        tot_base_signed_dists = []
        ## === nearest base pts === ##
        for i_bsz in range(nearest_base_pts.size(0)):
            # masks, base_pts, base_normals for each frame here
            # cur_bsz_
            cur_bsz_masks = [pts_inside_objmesh_labels_mask[i_bsz][0]]
            cur_bsz_base_pts = [nearest_base_pts[i_bsz][0]]
            cur_bsz_base_normals = [nearest_base_normals[i_bsz][0]]
            # nnjts #
            ## st frame signed dist ##
            cur_bsz_st_frame_signed_dist = torch.sum(
                (hand_joints[i_bsz][0] - cur_bsz_base_pts[0]) * cur_bsz_base_normals[0], dim=-1
            )
            cur_bsz_signed_dist = [cur_bsz_st_frame_signed_dist]
            for i_fr in range(1, nearest_base_pts.size(1)):
                cur_bsz_cur_fr_jts = hand_joints[i_bsz][i_fr]
                # cur_bsz_cur_fr_base_pts = nearest_base_pts
                # cur_fr_jts - 
                cur_bsz_cur_fr_prev_fr_signed_dist = torch.sum(
                    (cur_bsz_cur_fr_jts - cur_bsz_base_pts[-1]) * cur_bsz_base_normals[-1], dim=-1
                )
                # nnjts # cur
                cur_bsz_cur_fr_mask = ((cur_bsz_signed_dist[-1] >= 0.).float() + (cur_bsz_cur_fr_prev_fr_signed_dist < 0.).float()) > 1.5
                cur_bsz_cur_fr_base_pts = nearest_base_pts[i_bsz][i_fr].clone()
                cur_bsz_cur_fr_base_pts[cur_bsz_cur_fr_mask] = cur_bsz_base_pts[-1][cur_bsz_cur_fr_mask]
                cur_bsz_cur_fr_base_normals = nearest_base_normals[i_bsz][i_fr].clone()
                # ### curbsz curfr base normals; ### #
                cur_bsz_cur_fr_base_normals[cur_bsz_cur_fr_mask] = cur_bsz_base_normals[-1][cur_bsz_cur_fr_mask]
                cur_bsz_cur_fr_signed_dist = torch.sum(
                    (cur_bsz_cur_fr_jts - cur_bsz_cur_fr_base_pts) * cur_bsz_cur_fr_base_normals, dim=-1
                )
                cur_bsz_cur_fr_signed_dist[cur_bsz_cur_fr_mask] = 0. # ot the bes points
                ### for masks ###
                cur_bsz_masks.append(cur_bsz_cur_fr_mask)
                cur_bsz_base_pts.append(cur_bsz_cur_fr_base_pts)
                cur_bsz_base_normals.append(cur_bsz_cur_fr_base_normals)
            # 
            cur_bsz_masks = torch.stack(cur_bsz_masks, dim=0)
            cur_bsz_base_pts = torch.stack(cur_bsz_base_pts, dim=0)
            cur_bsz_base_normals = torch.stack(cur_bsz_base_normals, dim=0)
            cur_bsz_signed_dist = torch.stack(cur_bsz_signed_dist, dim=0)
            tot_masks.append(cur_bsz_masks)
            tot_base_pts.append(cur_bsz_base_pts)
            tot_base_normals.append(cur_bsz_base_normals)
            tot_base_signed_dists.append(cur_bsz_signed_dist)
        # masks; 
        tot_masks = torch.stack(tot_masks, dim=0)
        tot_base_pts = torch.stack(tot_base_pts, dim=0)
        tot_base_normals = torch.stack(tot_base_normals, dim=0)
        tot_base_signed_dists = torch.stack(tot_base_signed_dists, dim=0)
        
        # 
        nearest_base_pts = tot_base_pts.clone() # tot base pts 
        nearest_base_normals = tot_base_normals.clone()
        pts_inside_objmesh_labels_mask = tot_masks.clone()
        
        # if len()
        # bsz x ws x nnj x 3 #
        rel_joints_to_nearest_base_pts = hand_joints - nearest_base_pts
        
        # signed_dist_joints_to_base_pts: bsz x ws x nnj # -> disstances 
        # signed_dist_joints_to_base_pts = signed_dist_joints_to_base_pts.detach()
        # 
        
        # dot rel 
        # penetraition resolving --- strategy 
        # dot_rel_with_normals = torch.sum( # dot rhand joints with normals #
        #     rel_joints_to_nearest_base_pts * nearest_base_normals, dim=-1
        # )
        # 
        dot_rel_with_normals = torch.sum( # dot rhand joints with normals #
            -rel_joints_to_nearest_base_pts * rel_joints_to_nearest_base_pts, dim=-1
        )
        #### Get masks for penetrated joint points ####
        # signed_dist_mask = (signed_dist_mask.float() + pts_inside_objmesh_labels_mask.float()) > 1.5
        signed_dist_mask = pts_inside_objmesh_labels_mask
        # bsz x ws x nnj 
        signed_dist_mask = signed_dist_mask.detach() # detach the mask #
        # bsz x ws x nnj --> the loss term
        ## signed distances 3 #### isgned distance 3 ###
        ## dotrelwithnormals, ##
        # # signed_dist_mask -> the distances 
        
        # dot_rel_with_normals: bsz x ws x nnj x nnb # avg over windows and batches #
        avg_masks = (signed_dist_mask.float()).sum(dim=-1).mean()
        
        ## get singed distance energies ### ## projection ##
        # signed_dist_e = dot_rel_with_normals * signed_dist_joints_to_base_pts
        ### dot_rel_with_normals --> 
        signed_dist_e = -1.0 * dot_rel_with_normals
        signed_dist_e = torch.sum(
            signed_dist_e[signed_dist_mask]
        ) / torch.clamp(torch.sum(signed_dist_mask.float()), min=1e-5).item()
        ###### ====== get loss for signed distances ==== ###
        ''' strategy 1: use nearest base pts, rel, dists for resolving '''
        # cannot mask in some caes 
        # change of isgned distances #
        
        
        # intersection spline 
        ## judeg whether inside the object and only project those one inside of the object
        ## === e3 smoothness and prior losses === ##
        pose_smoothness_loss = F.mse_loss(theta_var.view(bsz, ws, -1)[:, 1:], theta_var.view(bsz, ws, -1)[:, :-1])
        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)
        ## === e3 smoothness and prior losses === ##
        
        # points to object vertices 
        #### ==== sv_dict ==== ####
        sv_dict = {
            'pts_inside_objmesh_labels_mask': pts_inside_objmesh_labels_mask.detach().cpu().numpy(),
            'hand_joints': hand_joints.detach().cpu().numpy(),
            
            'obj_verts': [cur_verts.detach().cpu().numpy() for cur_verts in obj_verts],
            'obj_faces': [cur_faces.detach().cpu().numpy() for cur_faces in obj_faces],
            
            'base_pts': base_pts.detach().cpu().numpy(),
            'base_normals': base_normals.detach().cpu().numpy(), # bsz x nnb x 3 -> bsz x nnb x 3 -> base normals #
            'nearest_base_pts': nearest_base_pts.detach().cpu().numpy(), # bsz x ws x nnj x 3 # 
            'nearest_base_normals': nearest_base_normals.detach().cpu().numpy(), # bsz x ws x nnj x 3 --> base normals and pts 
        }
        # 
        sv_dict_folder = "/data1/sim/mdm/tmp_saving"
        os.makedirs(sv_dict_folder, exist_ok=True)
        sv_dict_fn = os.path.join(sv_dict_folder, f"optim_iter_{i_iter}.npy")
        np.save(sv_dict_fn, sv_dict)
        print(f"Obj and subj saved to {sv_dict_fn}")
        #### ==== sv_dict ==== ####
        
        ## === e4 hand joints should be close to sampled hand joints === ##
        dist_dec_jts_to_sampled_pts = torch.sum(
            (hand_joints - sampled_joints) ** 2, dim=-1
        ).mean()
        
        # shoudl take a 
        # how to proejct the jvertex
        # hwo to project the veretex
        # weighted sum of the projectiondirection
        # weights of each base point
        # atraction field -> should be able to learn the penetration resolving strategy 
        # stochestic penetration resolving strategy #
        
        
        loss = pose_smoothness_loss * 0.05 + shape_prior_loss*0.001 + pose_prior_loss * 0.0001 + signed_dist_e * signed_dist_e_coeff + rel_e + dist_e + dist_dec_jts_to_sampled_pts
        
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tsigned_dist_e Loss: {}'.format(signed_dist_e.item()))
        print('\trel_e Loss: {}'.format(rel_e.item()))
        print('\tdist_e Loss: {}'.format(dist_e.item()))
        print('\tdist_dec_jts_to_sampled_pts Loss: {}'.format(dist_dec_jts_to_sampled_pts.item()))
        # avg_masks
        print('\tAvg masks: {}'.format(avg_masks.item()))
    
    
    
    ''' returning sampled_joints ''' 
    sampled_joints = hand_joints
    np.save("optimized_verts.npy", hand_verts.detach().cpu().numpy())
    print(f"Optimized verts saved to optimized_verts.npy")
    return sampled_joints.detach()

    

## 
def create_gaussian_diffusion(args): ## create guassian diffusion ##
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000 # 
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False # learn sigma #
    rescale_timesteps = False

    ## noose schedule; steps; scale_beta ## ## MSE ##
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    print(f"dataset: {args.dataset}, rep_type: {args.rep_type}")
    if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist"]:
        print(f"here! dataset: {args.dataset}, rep_type: {args.rep_type}")
        cur_spaced_diffusion_model = SpacedDiffusion_Ours
    # SpacedDiffusion_OursV2
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we"]:
        cur_spaced_diffusion_model = SpacedDiffusion_OursV2
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj"]:
        cur_spaced_diffusion_model = SpacedDiffusion_OursV3
    # SpacedDiffusion_OursV4 
    elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj_latents"]:
        if args.diff_joint_quants:
            cur_spaced_diffusion_model = SpacedDiffusion_OursV7
        elif args.diff_hand_params:
            cur_spaced_diffusion_model = SpacedDiffusion_OursV9
        else:
            if args.diff_spatial:
                cur_spaced_diffusion_model = SpacedDiffusion_OursV5
            elif args.diff_latents:
                cur_spaced_diffusion_model = SpacedDiffusion_OursV6
            else:
                cur_spaced_diffusion_model = SpacedDiffusion_OursV4
    else:
        cur_spaced_diffusion_model = SpacedDiffusion
    ### ==== predict xstart other than the noise in the model === ###
    return cur_spaced_diffusion_model(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=( ## use fixed sigmas / variances ##
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL # fixed small #
            )
            if not learn_sigma ## use learned sigmas ##
            else gd.ModelVarType.LEARNED_RANGE
        ), ## modelvartype ##
        loss_type=loss_type, ## loss_type ##
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz, ## lambda
        lambda_fc=args.lambda_fc,
        # motion_to_rep
        denoising_stra=args.denoising_stra,
        inter_optim=args.inter_optim,
        args=args,
    )
    
### from decoded energies to optimized joints ###
## latent variables ##
## encoded energies ## from energies calculated from perturbed energies ##
## decoded energies should also match the clean energy term ##


## and those values should be all denormed ##
def optimize_joints_according_to_e(dec_joints, base_pts, base_normals, dec_e):
    # dec_e_along_normals: bsz x (ws - 1) x nnj x nnb
    dec_e_along_normals = dec_e['dec_e_along_normals']
    # dec_e_vt_normals: bsz x (ws - 1) x nnj x nnb
    dec_e_vt_normals  = dec_e['dec_e_vt_normals']
    
    nn_iters = 10
    coarse_lr = 0.001
    
    dec_joints.requires_grad_()
    opt = optim.Adam([dec_joints], lr=coarse_lr)
    
    for i_iter in range(nn_iters):
        # dec_joints: bsz x ws x nnj x 3 
        # base_pts: bsz x nnb x 3
        k_f = 1.
        # bsz x ws x nnj x nnb x 3 #
        denormed_rel_base_pts_to_rhand_joints = dec_joints.unsqueeze(-2) - base_pts.unsqueeze(1).unsqueeze(1)
        
        k_f = 1. ## l2 rel base pts to pert rhand joints ##
        # l2_rel_base_pts_to_pert_rhand_joints: bsz x nf x nnj x nnb #
        l2_rel_base_pts_to_pert_rhand_joints = torch.norm(denormed_rel_base_pts_to_rhand_joints, dim=-1)
        ### att_forces ##
        att_forces = torch.exp(-k_f * l2_rel_base_pts_to_pert_rhand_joints) # bsz x nf x nnj x nnb #
        # bsz x (ws - 1) x nnj x nnb #
        att_forces = att_forces[:, :-1, :, :] # attraction forces -1 #
        # rhand_joints: ws x nnj x 3 # -> (ws - 1) x nnj x 3 ## rhand_joints ##
        # bsz x (ws - 1) x nnj x 3 --> displacements s#
        denormed_rhand_joints_disp = dec_joints[:, 1:, :, :] - dec_joints[:, :-1, :, :]

        # distance -- base_normalss,; (ws - 1) x nnj x nnb x 3 --> bsz x (ws - 1) x nnj x nnb # 
        # signed_dist_base_pts_to_pert_rhand_joints_along_normal # bsz x (ws - 1) x nnj x nnb #
        signed_dist_base_pts_to_rhand_joints_along_normal = torch.sum(
            base_normals.unsqueeze(1).unsqueeze(1) * denormed_rhand_joints_disp.unsqueeze(-2), dim=-1
        )
        # rel_base_pts_to_pert_rhand_joints_vt_normal: bsz x (ws -1) x nnj x nnb x 3 -> the relative positions vertical to base normals #
        rel_base_pts_to_rhand_joints_vt_normal = denormed_rhand_joints_disp.unsqueeze(-2)  - signed_dist_base_pts_to_rhand_joints_along_normal.unsqueeze(-1) * base_normals.unsqueeze(1).unsqueeze(1)
        dist_base_pts_to_rhand_joints_vt_normal = torch.sqrt(torch.sum(
            rel_base_pts_to_rhand_joints_vt_normal ** 2, dim=-1
        ))
        k_a = 1.
        k_b = 1.
        
        ### bsz x (ws - 1) x nnj x nnb ###
        e_disp_rel_to_base_along_normals = k_a * att_forces * torch.abs(signed_dist_base_pts_to_rhand_joints_along_normal)
        # (ws - 1) x nnj x nnb # -> dist vt normals # ## 
        e_disp_rel_to_baes_vt_normals = k_b * att_forces * dist_base_pts_to_rhand_joints_vt_normal
        # nf x nnj x nnb ---> dist_vt_normals -> nf x nnj x nnb # # torch.sqrt() ##
        # 
        loss_cur_e_pred_e_along_normals = ((e_disp_rel_to_base_along_normals - dec_e_along_normals) ** 2).mean()
        loss_cur_e_pred_e_vt_normals = ((e_disp_rel_to_baes_vt_normals - dec_e_vt_normals) ** 2).mean()
        
        loss = loss_cur_e_pred_e_along_normals + loss_cur_e_pred_e_vt_normals
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print('Iter {}: {}'.format(i_iter, loss.item()), flush=True)
        print('\tloss_cur_e_pred_e_along_normals: {}'.format(loss_cur_e_pred_e_along_normals.item()))
        print('\tloss_cur_e_pred_e_vt_normals: {}'.format(loss_cur_e_pred_e_vt_normals.item()))
    return dec_joints.detach()