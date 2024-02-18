from argparse import ArgumentParser
import argparse
import os
import json

# 
def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    print(f"model: {model_path}, dir_name: {os.path.dirname(model_path)}")
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--debug", action='store_true',
                       help="If True, will run evaluation during training.")
    # rep_type
    group.add_argument("--rep_type", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--local_rank", default=0, type=int, help="Batch size during training.") ## for dist util ##
    group.add_argument("--nprocs", default=1, type=int, help="Batch size during training.") ## for dist util ##
    # denoising_stra
    ### 1) rep -> represetntions directly; 2) motion_to_rep
    group.add_argument("--denoising_stra", default="rep", type=str,
                       help="Denoising strategy")
    # inter_optim
    group.add_argument("--inter_optim", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_jts
    group.add_argument("--diff_jts", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_basejtsrel
    group.add_argument("--diff_basejtsrel", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_basejtse
    group.add_argument("--diff_basejtse", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_sep_models
    group.add_argument("--use_sep_models", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_vae
    group.add_argument("--use_vae", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # kl_weights
    group.add_argument("--kl_weights", default=0.0, type=float, help="Joint positions loss.")
    ### 1) rep -> represetntions directly; 2) motion_to_rep
    group.add_argument("--jts_sclae_stra", default="bbox", type=str,
                       help="Denoising strategy")
    # use_sigmoid
    group.add_argument("--use_sigmoid", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # train_enc
    group.add_argument("--train_enc", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # train_diff ## 
    group.add_argument("--train_diff", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    ## with_dec_pos_emb ---- whether to use pos emb ##
    group.add_argument("--without_dec_pos_emb", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # pred_diff_noise
    group.add_argument("--pred_diff_noise", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # pred_diff_noise
    group.add_argument("--deep_fuse_timeemb", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_ours_transformer_enc
    group.add_argument("--use_ours_transformer_enc", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--not_load_opt", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # resume_diff
    group.add_argument("--resume_diff", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # const_noise
    group.add_argument("--const_noise", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # set_attn_to_none
    group.add_argument("--set_attn_to_none", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # rnd_noise
    group.add_argument("--rnd_noise", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # jts_pred_loss_coeff
    # basejtsrel_pred_loss_coeff, 
    # basejtse_along_normal_loss_coeff, basejtse_vt_normal_loss_coeff
    # jts_pred_loss_coeff
    group.add_argument("--jts_pred_loss_coeff", default=20.0, type=float, help="Joint positions loss.")
    # basejtsrel_pred_loss_coeff
    group.add_argument("--basejtsrel_pred_loss_coeff", default=20.0, type=float, help="Joint positions loss.")
    # basejtse_along_normal_loss_coeff
    group.add_argument("--basejtse_along_normal_loss_coeff", default=20.0, type=float, help="Joint positions loss.")
    # basejtse_vt_normal_loss_coeff
    group.add_argument("--basejtse_vt_normal_loss_coeff", default=20.0, type=float, help="Joint positions loss.")
    # wo_e_normalization
    group.add_argument("--wo_e_normalization", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # wo_rel_normalization
    group.add_argument("--wo_rel_normalization", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_dec_rel_v2
    group.add_argument("--use_dec_rel_v2", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # pred_basejtsrel_avgjts ### pred_basejtsrel_avgjts ###
    group.add_argument("--pred_basejtsrel_avgjts", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
     #only_first_clip
    group.add_argument("--only_first_clip", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # single_frame_noise
    group.add_argument("--single_frame_noise", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_t
    group.add_argument("--use_t", default=1, type=int, help="Joint positions loss.")
    # not_add_noise
    group.add_argument("--not_add_noise", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # not_cond_base
    group.add_argument("--not_cond_base", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # not_pred_avg_jts
    group.add_argument("--not_pred_avg_jts", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_spatial
    group.add_argument("--diff_spatial", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # pred_joints_offset
    group.add_argument("--pred_joints_offset", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # not_diff_avgjts
    group.add_argument("--not_diff_avgjts", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    ## joint_std_v2
    group.add_argument("--joint_std_v2", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # joint_std_v3
    group.add_argument("--joint_std_v3", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_latents
    group.add_argument("--diff_latents", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_canon_joints
    group.add_argument("--use_canon_joints", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_var_sched
    group.add_argument("--use_var_sched", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # e_normalization_stra
    group.add_argument("--e_normalization_stra", default="cent", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # diff_realbasejtsrel
    group.add_argument("--diff_realbasejtsrel", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # real_basejtsrel_norm_stra
    group.add_argument("--real_basejtsrel_norm_stra", default="none", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # diff_realbasejtsrel_to_joints ## basejtsrel_to_joints
    group.add_argument("--diff_realbasejtsrel_to_joints", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_abs_jts_pos
    group.add_argument("--use_abs_jts_pos", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_abs_jts_for_encoding
    group.add_argument("--use_abs_jts_for_encoding", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_abs_jts_for_encoding_obj_base
    group.add_argument("--use_abs_jts_for_encoding_obj_base", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_objbase_v2
    group.add_argument("--use_objbase_v2", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_objbase_v3
    group.add_argument("--use_objbase_v3", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_jts_pert_realbasejtsrel
    group.add_argument("--use_jts_pert_realbasejtsrel", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_objbase_out_v3
    group.add_argument("--use_objbase_out_v3", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # nn_base_pts
    group.add_argument("--nn_base_pts", default=700, type=int, help="Joint positions loss.")
    # use_objbase_v4, use_objbase_out_v4
    group.add_argument("--use_objbase_v4", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--use_objbase_out_v4", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # ### objbase_v5, use_objbase_out_v5 ###
    group.add_argument("--use_objbase_v5", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--use_objbase_out_v5", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # out_objbase_v5_bundle_out
    group.add_argument("--out_objbase_v5_bundle_out", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # add_noise_onjts
    group.add_argument("--add_noise_onjts", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # add_noise_onjts_single
    group.add_argument("--add_noise_onjts_single", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # v5_out_not_cond_base
    group.add_argument("--v5_out_not_cond_base", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # v5_out_not_cond_base
    group.add_argument("--use_objbase_v6", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_objbase_v7
    group.add_argument("--use_objbase_v7", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # predicted_info_fn
    group.add_argument("--predicted_info_fn", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # only_cmb_finger
    group.add_argument("--only_cmb_finger", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_vox_data
    group.add_argument("--use_vox_data", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # v5_in_not_base_pos
    group.add_argument("--v5_in_not_base_pos", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # v5_in_not_base
    group.add_argument("--v5_in_not_base", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # v5_in_without_glb
    group.add_argument("--v5_in_without_glb", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # finetune_with_cond
    group.add_argument("--finetune_with_cond", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # in_eval
    group.add_argument("--in_eval", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # finetune_with_cond_rel; finetune_with_cond_jtsobj
    group.add_argument("--finetune_with_cond_rel", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # finetune_with_cond_jtsobj
    group.add_argument("--finetune_with_cond_jtsobj", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # sel_basepts_idx
    group.add_argument("--sel_basepts_idx", default=0, type=int, help="Joint positions loss.")
    # test_tag
    group.add_argument("--test_tag", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # finetune_cond_obj_feats_dim
    group.add_argument("--finetune_cond_obj_feats_dim", default=3, type=int, help="Joint positions loss.")
    # cad_model_fn
    group.add_argument("--cad_model_fn", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # cad_model_fn
    # group.add_argument("--cad_model_fn", default="", type=str,
    #                    help="If empty, will use defaults according to the specified dataset.")
    # diff_joint_quants
    group.add_argument("--diff_joint_quants", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # joint_quants_nn
    group.add_argument("--joint_quants_nn", default=2, type=int, help="Joint positions loss.")
    # use_same_noise_for_rep ### whether to use the same noise for representations denoising ## # 
    group.add_argument("--use_same_noise_for_rep", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_temporal_rep_v2
    group.add_argument("--use_temporal_rep_v2", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_arti_obj
    group.add_argument("--use_arti_obj", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # pert_type
    group.add_argument("--pert_type", default="gaussian", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # use_anchors #
    group.add_argument("--use_anchors", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # with_glb_info
    group.add_argument("--with_glb_info", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # phy_guided_sampling
    group.add_argument("--phy_guided_sampling", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # diff_hand_params
    group.add_argument("--diff_hand_params", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--corr_fn", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    
    group.add_argument("--prev_test_tag", default="", type=str,)
    # 
    group.add_argument("--augment", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # train_all_clips
    group.add_argument("--train_all_clips", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_predicted_infos
    group.add_argument("--use_predicted_infos", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # start_idx  ### start_idx here for the starting idxes 3 
    group.add_argument("--start_idx", default=50, type=int, help="Joint positions loss.")
    # theta_dim
    group.add_argument("--theta_dim", default=24, type=int, help="Joint positions loss.")
    # use_interpolated_infos
    group.add_argument("--use_interpolated_infos", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_reverse
    group.add_argument("--use_reverse", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # predicted_info_fn_jts_only
    group.add_argument("--predicted_info_fn_jts_only", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # select_part_idx
    group.add_argument("--select_part_idx", default=-1, type=int, help="Joint positions loss.")
    # not_canon_rep
    group.add_argument("--not_canon_rep", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # scale_obj
    group.add_argument("--scale_obj", default=1, type=int, help="Joint positions loss.")
    group.add_argument("--resplit", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_arctic
    group.add_argument("--use_arctic", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--use_left", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # use_pose_pred
    group.add_argument("--use_pose_pred", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--use_hho", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    # seq_root
    group.add_argument("--seq_root", default="", type=str,)
    group.add_argument("--grab_path", default="", type=str,)
    


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'motion_ours'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--single_seq_path", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # window_size
    group.add_argument("--window_size", default=30, type=int, help="Number of learning rate anneal steps.")
    # inst_normalization; inst_normalization
    group.add_argument("--inst_normalization", action='store_true', default=False,
                       help="If True, will enable to use an already existing save_dir.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--model_path", default="", type=str, ## model_path --> model_path 
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--grab_processed_dir", type=str, default="data/grab/source_data",)
    group.add_argument("--data_root", type=str, default="data", help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    # group.add_argument("--debug", action='store_true',
    #                    help="If True, will run evaluation during training.")
    
    ## with_dec_pos_emb


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    # group.add_argument("--debug", action='store_true',
    #                    help="If True, will run evaluation during training.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    # unconstrained, constrained, text conditional #
    if args.unconstrained:
        cond_mode = 'no_cond'
    # elif args.dataset in ['kit', 'humanml', 'motion_ours']:
    #     cond_mode = 'text'
    elif args.dataset in ['kit', 'humanml', 'motion_ours']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    print(f"dataset: {args.dataset}, cond_mode: {cond_mode}")
    return cond_mode


def train_args():
    #### === useufl arguments for training === ####
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)
    print(f"cond_mode: {cond_mode}")
    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif args.action_file or args.action_name and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)