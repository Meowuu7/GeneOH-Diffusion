export overwrite=""
export pert_type="gaussian"
export phy_guided_sampling=""
export pred_basejtsrel_avgjts="--pred_basejtsrel_avgjts"
export pred_diff_noise="--pred_diff_noise"
export pred_joints_offset="--pred_joints_offset"
export predicted_info_fn=""
export predicted_info_fn_jts_only=""
export prev_test_tag=""
export real_basejtsrel_norm_stra="none"
export rep_type="obj_base_rel_dist_we_wj_latents"
export resplit=""
export resume_checkpoint=""





#### Model settings ####
export cuda="--cuda"
export data_dir=""
export data_root="data"
export dataset="motion_ours"
export debug=""
export deep_fuse_timeemb="--deep_fuse_timeemb"
export denoising_stra="rep"
export device=0
export diff_basejtse=""
export diff_basejtsrel="--diff_basejtsrel"
export diff_hand_params=""
export diff_joint_quants=""
export diff_jts=""
export diff_latents=""
export diff_realbasejtsrel=""
export diff_realbasejtsrel_to_joints=""
export diff_spatial="--diff_spatial"
export diffusion_steps=1000
export e_normalization_stra="cent"
export emb_trans_dec=""
export eval_batch_size=32
export eval_during_training=""
export eval_num_samples=1000
export eval_rep_times=3
export eval_split="test"
export finetune_cond_obj_feats_dim=3
export finetune_with_cond=""
export finetune_with_cond_jtsobj=""
export finetune_with_cond_rel=""
export grab_path=""
export grab_processed_dir="data/grab/source_data"
export in_eval=""
export input_text=""
export inst_normalization="--inst_normalization"
export inter_optim=""
export joint_quants_nn=2
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
export jts_pred_loss_coeff=20.0
export jts_sclae_stra="std"
export kl_weights=0.0
export lambda_fc=0.0
export lambda_rcxyz=0.0
export lambda_vel=0.0
export latent_dim=512
export layers=8
export local_rank=0
export log_interval=1000
export lr=0.0001
export lr_anneal_steps=0
export nn_base_pts=700
export noise_schedule="linear"
export not_add_noise=""
export not_canon_rep=""
export not_cond_base="--not_cond_base"
export not_diff_avgjts="--not_diff_avgjts"
export not_load_opt="--not_load_opt"
export not_pred_avg_jts="--not_pred_avg_jts"
export nprocs=1
export num_frames=60
export num_steps=600000000
export only_cmb_finger=""
export only_first_clip=""
export out_objbase_v5_bundle_out="--out_objbase_v5_bundle_out"

export add_noise_onjts="--add_noise_onjts"
export add_noise_onjts_single=""
export arch="trans_enc"
export augment=""
export basejtse_along_normal_loss_coeff=20.0
export basejtse_vt_normal_loss_coeff=20.0
export basejtsrel_pred_loss_coeff=20.0
export batch_size=10
export cad_model_fn=""
export cond_mask_prob=0.1
export const_noise=""
export corr_fn=""


export resume_diff=""
export rnd_noise=""
export save_interval=50000
export scale_obj=1
export seed=77
export sel_basepts_idx=-1
export seq_root=""
export set_attn_to_none="--set_attn_to_none"
export sigma_small="--sigma_small"
export single_frame_noise=""
export start_idx=0
export test_tag="rep_jts_hoi4d_rigid_toycar_t_400_st_idx_0_"
export theta_dim=45
export train_all_clips=""
export train_diff=""
export train_enc=""
export train_platform_type="NoPlatform"
export unconstrained="--unconstrained"
#### Model settings ####

#### Eval settings ####
export use_abs_jts_for_encoding=""
export use_abs_jts_for_encoding_obj_base=""
export use_abs_jts_pos=""
export use_anchors=""
export use_arctic=""
export use_canon_joints=""
export use_dec_rel_v2="--use_dec_rel_v2"
export use_hho=""
export use_interpolated_infos=""
export use_jts_pert_realbasejtsrel=""
export use_left=""
export use_objbase_out_v3=""
export use_objbase_out_v4=""
export use_objbase_out_v5="--use_objbase_out_v5"
export use_objbase_v2=""
export use_objbase_v3=""
export use_objbase_v4=""
export use_objbase_v5="--use_objbase_v5"
export use_objbase_v6=""
export use_objbase_v7=""
export use_ours_transformer_enc="--use_ours_transformer_enc"
export use_pose_pred=""
export use_predicted_infos=""
export use_reverse=""
export use_same_noise_for_rep=""
export use_sep_models="--use_sep_models"
export use_sigmoid=""
export use_t=400
export use_temporal_rep_v2=""
export use_vae=""
export use_var_sched="--use_var_sched"
export use_vox_data=""
export v5_in_not_base="--v5_in_not_base"
export v5_in_not_base_pos=""
export v5_in_without_glb="--v5_in_without_glb"
export v5_out_not_cond_base="--v5_out_not_cond_base"
export weight_decay=0.0
export window_size=60
export with_glb_info=""
export without_dec_pos_emb="--without_dec_pos_emb"
export wo_e_normalization=""
export wo_rel_normalization="--wo_rel_normalization"
export seq_root=""
export grab_path=""
#### Eval settings ####





#### Data and exp folders ####
export data_root="data"
export dataset="motion_ours"
export data_dir=""
# export single_seq_path="data/grab/source_data/test/14.npy"
export single_seq_path=""
export window_size=60
export use_arti_obj="--use_arti_obj"
#### Data and exp folders ####
export hoi4d_cad_model_root="data/hoi4d/HOI4D_CAD_Model_for_release"
export hoi4d_data_root="data/hoi4d/HOI_Processed_Data_Arti"
export hoi4d_category_name="Laptop"
export hoi4d_eval_st_idx=0
export hoi4d_eval_ed_idx=250
### this argument controls the index of the base part used to provide object points for denoising ###
### articulated objects considered in this project usually have two parts, so one can try 0 or 1 ###
export select_part_idx=0

################# Set to your paths #################
#### Data and exp folders ####
export save_dir="data/hoi4d/result"
# export grab_processed_dir="data/grab/source_data"

# export single_seq_path=data/grab/source_data/test/14.npy
# export save_dir=data/grab/result
# export grab_processed_dir=data/grab/source_data


################# Evaluation setting #################
# set `pert_type` to `gaussian` to add Gaussian noise
# set `pert_type` to `beta` to add noise from a beta distribution (GRAB (Beta) test set)`
export pert_type="gaussian"

################# Set to your model path #################
#### Model path ####
export model_path=ckpts/model.pt


################# Set to your favorate test tag #################
export test_tag="jts_hoi4d_t_400_test_"



export cuda_ids=4



# bash scripts/val/predict_hoi4d_arti_rndseed.sh


CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.predict_hoi4d --dataset motion_ours --save_dir ${save_dir}  --window_size ${window_size} ${unconstrained} ${inst_normalization}  --model_path ${model_path} --rep_type ${rep_type} --batch_size=${batch_size}  --denoising_stra ${denoising_stra} ${inter_optim} --seed ${seed} ${diff_jts} ${diff_basejtsrel} ${diff_basejtse}  ${use_sep_models} --jts_sclae_stra ${jts_sclae_stra} ${use_vae} ${use_sigmoid} ${train_enc} ${without_dec_pos_emb} ${pred_diff_noise} ${resume_diff} ${not_load_opt} ${deep_fuse_timeemb}  ${use_ours_transformer_enc} ${const_noise} ${set_attn_to_none} ${rnd_noise} ${wo_e_normalization} ${wo_rel_normalization} ${use_dec_rel_v2} ${pred_basejtsrel_avgjts} ${single_frame_noise} --use_t ${use_t} ${not_add_noise} ${not_cond_base}  ${not_pred_avg_jts} --latent_dim ${latent_dim} ${diff_spatial} --noise_schedule ${noise_schedule} ${pred_joints_offset}  ${not_diff_avgjts}  ${joint_std_v3}  ${joint_std_v2}  ${diff_latents}  ${use_canon_joints} ${use_var_sched} --e_normalization_stra ${e_normalization_stra} --real_basejtsrel_norm_stra ${real_basejtsrel_norm_stra} ${diff_realbasejtsrel}  ${diff_realbasejtsrel_to_joints} ${use_abs_jts_for_encoding} ${use_abs_jts_for_encoding_obj_base} ${use_abs_jts_pos} ${use_objbase_v2} ${use_objbase_out_v3} ${use_objbase_v4} ${use_objbase_out_v4} ${use_objbase_v5} ${use_objbase_out_v5} ${out_objbase_v5_bundle_out} --nn_base_pts ${nn_base_pts} ${add_noise_onjts} ${v5_out_not_cond_base} ${use_objbase_v6} ${add_noise_onjts_single} ${only_cmb_finger} ${use_objbase_v7} ${v5_in_not_base_pos} ${v5_in_not_base} ${v5_in_without_glb}  ${finetune_with_cond} --test_tag ${test_tag} ${finetune_with_cond_rel} ${finetune_with_cond_jtsobj} --sel_basepts_idx ${sel_basepts_idx} ${use_same_noise_for_rep} --pert_type ${pert_type} ${phy_guided_sampling} ${use_anchors} ${use_arti_obj} --theta_dim ${theta_dim} --start_idx ${start_idx} ${use_reverse} --select_part_idx ${select_part_idx} --hoi4d_category_name=${hoi4d_category_name} --hoi4d_data_root=${hoi4d_data_root} --hoi4d_eval_st_idx=${hoi4d_eval_st_idx} --hoi4d_eval_ed_idx=${hoi4d_eval_ed_idx} --hoi4d_cad_model_root=${hoi4d_cad_model_root}


