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
# import torch.nn.functional as F
# from manopth.manolayer import ManoLayer
import numpy as np
# import trimesh
# import os
# from diffusion.respace_ours import SpacedDiffusion as SpacedDiffusion_Ours
# SpacedDiffusionV2
# from diffusion.respace_ours import SpacedDiffusionV2 as SpacedDiffusion_OursV2
# from diffusion.respace_ours import SpacedDiffusionV3 as SpacedDiffusion_OursV3
# SpacedDiffusionV4
from diffusion.respace_ours import SpacedDiffusionV4 as SpacedDiffusion_OursV4
# SpacedDiffusion_OursV5
from diffusion.respace_ours import SpacedDiffusionV5 as SpacedDiffusion_OursV5
# SpacedDiffusion_OursV6
# from diffusion.respace_ours import SpacedDiffusionV6 as SpacedDiffusion_OursV6
# # SpacedDiffusion_OursV7
# from diffusion.respace_ours import SpacedDiffusionV7 as SpacedDiffusion_OursV7
# from diffusion.respace_ours import SpacedDiffusionV9 as SpacedDiffusion_OursV9



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


def standard_normal_logprob(z):
    dim = z.size(-1)
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
    # if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist"]:
    #     model = MDM_Ours(**get_model_args(args, data))
    # elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we"]:
    #     model = MDM_Ours_V3(**get_model_args(args, data))
    # # MDM_Ours_V4
    # elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj"]:
    #     model = MDM_Ours_V4(**get_model_args(args, data))
    # obj_base_rel_dist_we_wj_latents
    # if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj_latents"]:
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
        # else: ### create the model and the diffusion ##
        #     print(f"Using MDM ours V9!!!!")
        #     model =  MDM_Ours_V9(**get_model_args(args, data))
    elif args.diff_latents:
        print(f"Using MDM ours V11!!!!")
        model =  MDM_Ours_V11(**get_model_args(args, data))
        # elif args.use_sep_models:
        #     if args.use_vae:
        #         if args.pred_basejtsrel_avgjts:
        #             print(f"Using MDM ours V8!!!!")
        #             model = MDM_Ours_V8(**get_model_args(args, data))
        #         else:
        #             model = MDM_Ours_V7(**get_model_args(args, data))
        #     else:
        #         model = MDM_Ours_V6(**get_model_args(args, data))
        # else:
        #     model = MDM_Ours_V5(**get_model_args(args, data))
    # else:
    #     model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):
    clip_version = 'ViT-B/32'
    action_emb = 'tensor' ## get model arguments ##
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml']:
        data_rep = 'hml_vec'
        njoints = 263
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




def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    print(f"dataset: {args.dataset}, rep_type: {args.rep_type}")
    # if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist"]:
    #     print(f"here! dataset: {args.dataset}, rep_type: {args.rep_type}")
    #     cur_spaced_diffusion_model = SpacedDiffusion_Ours
    # elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we"]:
    #     cur_spaced_diffusion_model = SpacedDiffusion_OursV2
    # elif args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj"]:
    #     cur_spaced_diffusion_model = SpacedDiffusion_OursV3 
    # if args.dataset in ['motion_ours'] and args.rep_type in ["obj_base_rel_dist_we_wj_latents"]:
    #     if args.diff_joint_quants:
    #         cur_spaced_diffusion_model = SpacedDiffusion_OursV7
    #     elif args.diff_hand_params:
    #         cur_spaced_diffusion_model = SpacedDiffusion_OursV9
    #     else:
    if args.diff_spatial:
        cur_spaced_diffusion_model = SpacedDiffusion_OursV5
    # elif args.diff_latents:
    #     cur_spaced_diffusion_model = SpacedDiffusion_OursV6
    else:
        cur_spaced_diffusion_model = SpacedDiffusion_OursV4

    return cur_spaced_diffusion_model(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        denoising_stra=args.denoising_stra,
        inter_optim=args.inter_optim,
        args=args,
    )
    


def optimize_joints_according_to_e(dec_joints, base_pts, base_normals, dec_e):
    dec_e_along_normals = dec_e['dec_e_along_normals']
    dec_e_vt_normals  = dec_e['dec_e_vt_normals']
    
    nn_iters = 10
    coarse_lr = 0.001
    
    dec_joints.requires_grad_()
    opt = optim.Adam([dec_joints], lr=coarse_lr)
    
    for i_iter in range(nn_iters):
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