export overwrite=""
export train_platform_type="NoPlatform"
export lr=0.0001
export weight_decay=0.0
export lr_anneal_steps=0
export eval_batch_size=32
export eval_split="test"
export eval_during_training=""
export eval_rep_times=3
export eval_num_samples=1000
export log_interval=1000
export save_interval=50000
export num_steps=600000000
export num_frames=60
export resume_checkpoint=""

#### Model settings ####
export cuda="--cuda"
export device=0
export seed=77
export batch_size=10
export debug=""
export rep_type="obj_base_rel_dist_we_wj_latents"
export local_rank=0
export nprocs=1
export denoising_stra="rep"
export inter_optim=""
export diff_jts=""
export diff_basejtsrel="--diff_basejtsrel"
export diff_basejtse=""
export use_sep_models="--use_sep_models"
export use_vae=""
export kl_weights=0.0
export jts_sclae_stra="std"
export use_sigmoid=""
export train_enc=""
export train_diff=""
export without_dec_pos_emb="--without_dec_pos_emb"
export pred_diff_noise="--pred_diff_noise"
export deep_fuse_timeemb="--deep_fuse_timeemb"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export not_load_opt="--not_load_opt"
export resume_diff=""
export const_noise=""
export set_attn_to_none="--set_attn_to_none"
export rnd_noise=""
export jts_pred_loss_coeff=20.0
export basejtsrel_pred_loss_coeff=20.0
export basejtse_along_normal_loss_coeff=20.0
export basejtse_vt_normal_loss_coeff=20.0
export wo_e_normalization=""
export wo_rel_normalization="--wo_rel_normalization"
export use_dec_rel_v2="--use_dec_rel_v2"
export pred_basejtsrel_avgjts="--pred_basejtsrel_avgjts"
export only_first_clip=""
export single_frame_noise=""
export use_t=400
export not_add_noise=""
export not_cond_base="--not_cond_base"
export not_pred_avg_jts="--not_pred_avg_jts"
export diff_spatial="--diff_spatial"
export pred_joints_offset="--pred_joints_offset"
export not_diff_avgjts="--not_diff_avgjts"
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
export diff_latents=""
export use_canon_joints=""
export use_var_sched="--use_var_sched"
export e_normalization_stra="cent"
export diff_realbasejtsrel=""
export real_basejtsrel_norm_stra="none"
export diff_realbasejtsrel_to_joints=""
export use_abs_jts_pos=""
export use_abs_jts_for_encoding=""
export use_abs_jts_for_encoding_obj_base=""
export use_objbase_v2=""
export use_objbase_v3=""
export use_jts_pert_realbasejtsrel=""
export use_objbase_out_v3=""
export nn_base_pts=700
export use_objbase_v4=""
export use_objbase_out_v4=""
export use_objbase_v5="--use_objbase_v5"
export use_objbase_out_v5="--use_objbase_out_v5"
export out_objbase_v5_bundle_out="--out_objbase_v5_bundle_out"
export add_noise_onjts="--add_noise_onjts"
export add_noise_onjts_single=""
export v5_out_not_cond_base="--v5_out_not_cond_base"
export use_objbase_v6=""
export use_objbase_v7=""
export predicted_info_fn=""
export only_cmb_finger=""
export use_vox_data=""
export v5_in_not_base_pos=""
export v5_in_not_base="--v5_in_not_base"
export v5_in_without_glb="--v5_in_without_glb"
export finetune_with_cond=""
export in_eval=""
export finetune_with_cond_rel=""
export finetune_with_cond_jtsobj=""
export sel_basepts_idx=-1
export finetune_cond_obj_feats_dim=3
export cad_model_fn=""
export diff_joint_quants=""
export joint_quants_nn=2
export inst_normalization="--inst_normalization"
export arch="trans_enc"
export emb_trans_dec=""
export layers=8
export latent_dim=512
export cond_mask_prob=0.1
export lambda_rcxyz=0.0
export lambda_vel=0.0
export lambda_fc=0.0
export unconstrained="--unconstrained"
export noise_schedule="linear"
export diffusion_steps=1000
export sigma_small="--sigma_small"
#### Model settings ####


#### Eval settings ####
export use_same_noise_for_rep=""
export use_temporal_rep_v2=""
export use_arti_obj=""
export use_anchors=""
export with_glb_info=""
export phy_guided_sampling=""
export diff_hand_params=""
export corr_fn=""
export test_tag="jts_grab_t_400_test_"
export prev_test_tag=""
export augment=""
export train_all_clips=""
export use_predicted_infos=""
export start_idx=0
export theta_dim=24
export use_interpolated_infos=""
export use_reverse=""
export predicted_info_fn_jts_only=""
export select_part_idx=-1
export not_canon_rep=""
export scale_obj=1
export resplit=""
export use_arctic=""
export use_left=""
export use_pose_pred=""
export use_hho=""
#### Eval settings ####





#### Data and exp folders ####
export data_root="data"
export dataset="motion_ours"
export data_dir=""
export single_seq_path="data/grab/source_data/test/14.npy"
export window_size=60
#### Data and exp folders ####



################# Set to your paths #################
#### Data and exp folders ####
export seq_root=""
export grab_path=""
export save_dir="data/grab/result"
export grab_processed_dir="data/grab/source_data"

export single_seq_path=data/grab/source_data/test/14.npy
export save_dir=data/grab/result
export grab_processed_dir=data/grab/source_data



################# Evaluation setting #################
# set `pert_type` to `gaussian` to add Gaussian noise
# set `pert_type` to `beta` to add noise from a beta distribution (GRAB (Beta) test set)`
export pert_type="gaussian"

################# Set to your model path #################
#### Model path ####
# export model_path="ckpts/model_spatial.pt"
export model_path=ckpts/model.pt





export cuda_ids=7

# 

CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.predict_grab --dataset motion_ours --save_dir ${save_dir} --single_seq_path ${single_seq_path} --window_size ${window_size} ${unconstrained} ${inst_normalization}  --model_path ${model_path} --rep_type ${rep_type} --batch_size=${batch_size}  --denoising_stra ${denoising_stra} ${inter_optim} --seed ${seed} ${diff_jts} ${diff_basejtsrel} ${diff_basejtse}  ${use_sep_models} --jts_sclae_stra ${jts_sclae_stra} ${use_vae} ${use_sigmoid} ${train_enc} ${without_dec_pos_emb} ${pred_diff_noise} ${resume_diff} ${not_load_opt} ${deep_fuse_timeemb}  ${use_ours_transformer_enc} ${const_noise} ${set_attn_to_none} ${rnd_noise} ${wo_e_normalization} ${wo_rel_normalization} ${use_dec_rel_v2} ${pred_basejtsrel_avgjts} ${single_frame_noise} --use_t ${use_t} ${not_add_noise} ${not_cond_base}  ${not_pred_avg_jts} --latent_dim ${latent_dim} ${diff_spatial} --noise_schedule ${noise_schedule} ${pred_joints_offset}  ${not_diff_avgjts}  ${joint_std_v3}  ${joint_std_v2}  ${diff_latents}  ${use_canon_joints} ${use_var_sched} --e_normalization_stra ${e_normalization_stra} --real_basejtsrel_norm_stra ${real_basejtsrel_norm_stra} ${diff_realbasejtsrel}  ${diff_realbasejtsrel_to_joints} ${use_abs_jts_for_encoding} ${use_abs_jts_for_encoding_obj_base} ${use_abs_jts_pos} ${use_objbase_v2} ${use_objbase_out_v3} ${use_objbase_v4} ${use_objbase_out_v4} ${use_objbase_v5} ${use_objbase_out_v5} ${out_objbase_v5_bundle_out} --nn_base_pts ${nn_base_pts} ${add_noise_onjts} ${v5_out_not_cond_base} ${use_objbase_v6} ${add_noise_onjts_single} ${only_cmb_finger} ${use_objbase_v7} ${v5_in_not_base_pos} ${v5_in_not_base} ${v5_in_without_glb}  ${finetune_with_cond} --test_tag ${test_tag} ${finetune_with_cond_rel} ${finetune_with_cond_jtsobj} --sel_basepts_idx ${sel_basepts_idx} ${use_same_noise_for_rep} --pert_type ${pert_type} ${phy_guided_sampling} ${use_anchors} ${use_arti_obj} --theta_dim ${theta_dim} --start_idx ${start_idx} ${use_reverse} --scale_obj ${scale_obj} ${resplit}