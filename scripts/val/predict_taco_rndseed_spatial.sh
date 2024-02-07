export PYTHONPATH=.






export use_t=100
export window_size=60
export unconstrained="--unconstrained"
export inst_normalization="--inst_normalization"
export rep_type="obj_base_rel_dist_we_wj_latents"

export denoising_stra="rep"
export inter_optim=""
export diff_jts=""
export diff_basejtse=""

export use_sep_models="--use_sep_models"
export jts_sclae_stra="std"

export batch_size=10
export use_vae=""
export use_sigmoid=""
export train_enc=""
export without_dec_pos_emb="--without_dec_pos_emb"
export pred_diff_noise="--pred_diff_noise"
export resume_diff=""
export not_load_opt="--not_load_opt"
export deep_fuse_timeemb="--deep_fuse_timeemb" 
export use_ours_transformer_enc="--use_ours_transformer_enc"
export const_noise=""
export set_attn_to_none="--set_attn_to_none"
export rnd_noise=""
export wo_e_normalization=""
export wo_rel_normalization="--wo_rel_normalization"
export use_dec_rel_v2="--use_dec_rel_v2"
export pred_basejtsrel_avgjts="--pred_basejtsrel_avgjts"
export single_frame_noise=""
export not_add_noise=""
export not_cond_base="--not_cond_base"
export not_pred_avg_jts="--not_pred_avg_jts"

export latent_dim=512
export diff_spatial="--diff_spatial"
export noise_schedule="linear"
export pred_joints_offset="--pred_joints_offset"
export not_diff_avgjts="--not_diff_avgjts"
export joint_std_v3="--joint_std_v3"
export joint_std_v2=""
export diff_latents=""
export use_canon_joints=""
export use_var_sched="--use_var_sched"
export e_normalization_stra="cent"
export real_basejtsrel_norm_stra="none"
export diff_realbasejtsrel_to_joints=""
export use_abs_jts_for_encoding=""
export use_abs_jts_for_encoding_obj_base=""
export use_abs_jts_pos=""
export use_objbase_v2=""
export use_objbase_out_v3=""
export use_objbase_v4=""
export use_objbase_out_v4=""
export use_objbase_v5="--use_objbase_v5"
export use_objbase_v6=""
export use_objbase_out_v5="--use_objbase_out_v5"
export out_objbase_v5_bundle_out="--out_objbase_v5_bundle_out"

export nn_base_pts=2000
export add_noise_onjts="--add_noise_onjts"
export v5_out_not_cond_base="--v5_out_not_cond_base"
export add_noise_onjts_single="" 
export only_cmb_finger="--only_cmb_finger"
export only_cmb_finger=""
export use_objbase_v7=""
export v5_in_not_base_pos=""

export v5_in_not_base="--v5_in_not_base"
export v5_in_without_glb="--v5_in_without_glb"
export finetune_with_cond=""
export finetune_with_cond_rel=""
export finetune_with_cond_jtsobj=""
export sel_basepts_idx=-1
export use_same_noise_for_rep=""
# export pert_type="uniform"
export pert_type="gaussian"
export phy_guided_sampling=""
export use_anchors=""
export use_arti_obj="--use_arti_obj"



export diff_hand_params=""
export diff_basejtsrel=""
export diff_realbasejtsrel="--diff_realbasejtsrel"
# export model_path=./ckpts/model000519000.pt 
export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_pred_diff_realbaesjtsrel_train_enc_with_diff_latents_prediffnoise_none_norm_rel_rel_to_jts_objbasev5in_objbasev5out_bundle_out_basepts700_innotcondbaseall_outnotcondbase_noisejtssingle_inwoglb_res_all_clips_aug_/model001039000.pt # 




export theta_dim=24

export start_idx="--start_idx 80"

export scale_obj=1



export single_seq_path="/data3/hlyang/results/test_data/20231105/20231105_010.pkl"
export test_tag="20231105_010_jts_spatial_t_100_hho_"


export single_seq_path="/data3/hlyang/results/test_data/20231105/20231105_020.pkl"
export test_tag="20231105_020_jts_spatial_t_100_hho_"


export single_seq_path="/data3/hlyang/results/test_data/20231103/20231103_020.pkl"
export test_tag="20231103_020_jts_spatial_t_100_hho_"


export use_arctic=""
export use_hho="--use_hho"
export use_left="--use_left"

export use_reverse=""

export seed=0


export save_dir="/data3/datasets/xueyi/hho_save_res"


export cuda_ids=2


# bash scripts/val/predict_taco_rndseed_spatial.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.predict_taco --dataset motion_ours --single_seq_path ${single_seq_path} --window_size ${window_size} ${unconstrained} ${inst_normalization}  --model_path ${model_path} --rep_type ${rep_type} --batch_size=${batch_size}  --denoising_stra ${denoising_stra} ${inter_optim} --save_dir ${save_dir} --seed ${seed} ${diff_jts} ${diff_basejtsrel} ${diff_basejtse}  ${use_sep_models} --jts_sclae_stra ${jts_sclae_stra} ${use_vae} ${use_sigmoid} ${train_enc} ${without_dec_pos_emb} ${pred_diff_noise} ${resume_diff} ${not_load_opt} ${deep_fuse_timeemb}  ${use_ours_transformer_enc} ${const_noise} ${set_attn_to_none} ${rnd_noise} ${wo_e_normalization} ${wo_rel_normalization} ${use_dec_rel_v2} ${pred_basejtsrel_avgjts} ${single_frame_noise} --use_t ${use_t} ${not_add_noise} ${not_cond_base}  ${not_pred_avg_jts} --latent_dim ${latent_dim} ${diff_spatial} --noise_schedule ${noise_schedule} ${pred_joints_offset}  ${not_diff_avgjts}  ${joint_std_v3}  ${joint_std_v2}  ${diff_latents}  ${use_canon_joints} ${use_var_sched} --e_normalization_stra ${e_normalization_stra} --real_basejtsrel_norm_stra ${real_basejtsrel_norm_stra} ${diff_realbasejtsrel}  ${diff_realbasejtsrel_to_joints} ${use_abs_jts_for_encoding} ${use_abs_jts_for_encoding_obj_base} ${use_abs_jts_pos} ${use_objbase_v2} ${use_objbase_out_v3} ${use_objbase_v4} ${use_objbase_out_v4} ${use_objbase_v5} ${use_objbase_out_v5} ${out_objbase_v5_bundle_out} --nn_base_pts ${nn_base_pts} ${add_noise_onjts} ${v5_out_not_cond_base} ${use_objbase_v6} ${add_noise_onjts_single} ${only_cmb_finger} ${use_objbase_v7} ${v5_in_not_base_pos} ${v5_in_not_base} ${v5_in_without_glb}  ${finetune_with_cond} --test_tag ${test_tag} ${finetune_with_cond_rel} ${finetune_with_cond_jtsobj} --sel_basepts_idx ${sel_basepts_idx} ${use_same_noise_for_rep} --pert_type ${pert_type} ${phy_guided_sampling} ${use_anchors} ${use_arti_obj} --theta_dim ${theta_dim} ${start_idx} ${use_reverse} --scale_obj ${scale_obj} ${use_arctic} ${use_left} ${use_hho}
