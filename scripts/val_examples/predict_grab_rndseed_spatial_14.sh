export PYTHONPATH=.

export cuda_ids=6
export cuda_ids=0
export cuda_ids=5


export save_dir="/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512"
# export save_dir="save/my_humanml_trans_enc_512_v2"

export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512_v2/model000000000.pt
export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/model000050000.pt
export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512/model000100000.pt

# export single_seq_path="/data1/xueyi/GRAB_processed/test_pert/50.npy"
export single_seq_path="/data1/xueyi/GRAB_processed/test/50.npy"
export single_seq_path="/data1/xueyi/GRAB_processed/train/1.npy"
export window_size=30




#### ==== unconstrained, with instance normalization, without input text ===== ####
export window_size=60
export unconstrained="--unconstrained"
export inst_normalization="--inst_normalization"
export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_unconstrainted_inst_norm_svi_500_/model000012500.pt
export input_text=""


### ==== unconstrianted, base pts rel representations here === ###
export rep_type="obj_base_rel_dist"
export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_/model000208500.pt
## ==== with data scaling ==== ##
export model_path=/home/xueyi/sim/motion-diffusion-model/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_scale_data_/model000015000.pt

export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_scale_data_/model000007000.pt
export single_seq_path="/data1/xueyi/GRAB_processed/test/50.npy"


export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_scale_data_/model000007000.pt

# ''' Normalization stra 4 ''' 
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_5_/model000031000.pt
# ''' Normalization stra 4 ''' 

# ''' Normalization stra 3 ''' 
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_4_/model000058500.pt
# ''' Normalization stra 3 ''' 

export cuda_ids=3
# CUDA_VISIBLE_DEVICES=${cuda_ids} python -m train.train_mdm --save_dir=${save_dir} --dataset motion_ours
export batch_size=2

export denoising_stra="rep"
export inter_optim=""




export cuda_ids=4
export denoising_stra="motion_to_rep"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_motion_to_rep_/model000003000.pt


### no itner optimization ###
export cuda_ids=5
export rep_type="obj_base_rel_dist"
export inter_optim=""
export denoising_stra="rep"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_/model000042500.pt
# model000350000.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_/model000350000.pt
export single_seq_path="/data1/xueyi/GRAB_processed/test/80.npy"


### no itner optimization, const noise for relative positions ###
# export cuda_ids=5
# export rep_type="obj_base_rel_dist"
# export inter_optim=""
# export denoising_stra="rep"
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_/model000064000.pt

# export inter_optim="--inter_optim"


# ### === no inter optimization ###
# export cuda_ids=7
# export rep_type="ambient_obj_base_rel_dist"
# export denoising_stra="rep"
# export inter_optim=""
# # export save_dir="/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_ambient_objbase_"
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_ambient_objbase_/model000003500.pt
# export batch_size=2


### no itner optimization ###
export cuda_ids=5
export rep_type="obj_base_rel_dist_we"
export inter_optim=""
export denoising_stra="rep"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_/model000024500.pt
export single_seq_path="/data1/xueyi/GRAB_processed/test/80.npy"


### no inter optimization ###
export cuda_ids=5
export rep_type="obj_base_rel_dist_we"
export inter_optim=""
export denoising_stra="rep"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_v2_/model000089000.pt
export single_seq_path="/data1/xueyi/GRAB_processed/test/80.npy"
export single_seq_path="/data1/xueyi/GRAB_processed/test/50.npy"
# export single_seq_path=/data1/xueyi/

export seed=11


### no inter optimization ### # 
export cuda_ids=2
export rep_type="obj_base_rel_dist_we_wj"
export inter_optim=""
export denoising_stra="rep"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_pnenc_/model000031000.pt
# export single_seq_path="/data1/xueyi/GRAB_processed/test/80.npy"
### === deccond === ###
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_/model000117000.pt
### === scaling === ###
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_scale_jts_/model000026000.pt
# ## with centralization ###
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_scale_jts_cent_/model000015000.pt
# ### === per frame norm === ##
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_/model000003000.pt
export single_seq_path="/data1/xueyi/GRAB_processed/test/50.npy"
# export single_seq_path=/data1/xueyi/



export cuda_ids=6
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_/model000147500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_/model000257500.pt



export window_size=30
export rep_type="obj_base_rel_dist_we_wj_latents"
export cuda_ids=2
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_/model000147500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_/model000063500.pt


export window_size=30
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_nsigmoid_scaled_loss_/model000018500.pt



export cuda_ids=0
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_nsigmoid_wrel_scaled_loss_/model000004000.pt


##### ===== model_path ===== #####
export cuda_ids=0
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_nsigmoid_wrel_scaled_loss_/model000004000.pt


##### ===== model_path, with e  ===== #####
export window_size=30
export cuda_ids=7
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_/model000003000.pt

export window_size=60
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_60_nsigmoid_wrel_wnergy_scaled_loss_lbsz_/model000006000.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_60_nsigmoid_wrel_wnergy_scaled_loss_lbsz_/model000015500.pt

# export window_size=30
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_/model000013000.pt


# export window_size=30
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_stdscalejts_/model000001500.pt




export cuda_ids=0
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_stdscalejts_/model000021000.pt

export window_size=30
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_lbsz_/model000026000.pt # jts only loss

export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_basejtsrelloss_/model000061500.pt


export window_size=60
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_objname_constrainted_inst_norm_svi_500_train_split_base_pts_rel_norm_data_stra_3_p_7_den_stra_rep_objbase_const_noise_bpts_wenergy_wjoints_deccond_perinst_norm_p1_latents_ws_60_nsigmoid_wrel_wnergy_scaled_loss_lbsz_/model000033500.pt

export window_size=30
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_lbsz_/model000027500.pt

export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bboxscale_/model000066500.pt




export window_size=60
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_baserelonlyloss_lbsz_/model000008000.pt

export cuda_ids=4

## diff options ##
export diff_jts="--diff_jts"
export diff_basejtsrel="--diff_basejtsrel"
export diff_basejtse="--diff_basejtse"


export use_sep_models=""
# export use_sep_models="--use_sep_models"




export cuda_ids=1
export use_sep_models="--use_sep_models"
export window_size=60
export diff_jts="--diff_jts"
export diff_basejtsrel=""
export diff_basejtse=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model__/model000001500.pt


export diff_jts=""
export diff_basejtsrel="--diff_basejtsrel"
export diff_basejtse=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_baserelonlyloss_lbsz_sep_model__/model000007000.pt



### ==== diff_jts, diff_basejtsrel, diff_basejtse ==== ###
export cuda_ids=0
export jts_sclae_stra="std"
export diff_jts="--diff_jts"
export diff_basejtsrel=""
export diff_basejtse=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_std_scale_lbsz_sep_model_/model000001500.pt


export jts_sclae_stra="bbox"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model__/model000003000.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_vae_kl_0001_/model000005000.pt
export use_vae="--use_vae"
export use_sigmoid=""


export use_vae=""
export jts_sclae_stra="std"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_std_scale_lbsz_sep_model_use_sigmoid_/model000000500.pt
export jts_sclae_stra="bbox"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_/model000000500.pt
export use_sigmoid="--use_sigmoid"


export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_/model000000500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_/model000006500.pt
# with diff; just for test
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_/model000008000.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_/model000008500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_/model000010000.pt

export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_/model000007000.pt
# export train_enc="--train_enc" ## # train_enc #
export train_enc="" ## train enc ## # trian_enc #

export without_dec_pos_emb="--without_dec_pos_emb"



export pred_diff_noise="--pred_diff_noise"
export resume_diff="--resume_diff"
export not_load_opt="--not_load_opt"
# export pred_diff_noise="--pred_diff_noise"
export deep_fuse_timeemb="--deep_fuse_timeemb"
export use_ours_transformer_enc="--use_ours_transformer_enc"

export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_/model000006500.pt



## predict noise 
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_/model000015000.pt
export seed=44
export seed=99
# export seed=11



export pred_diff_noise=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_/model000017500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_/model000040000.pt




# export cuda_ids=3
# export pred_diff_noise="--pred_diff_noise"
# export use_ours_transformer_enc=""
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_pred_noise_/model000018000.pt
# export use_ours_transformer_enc="--use_ours_transformer_enc"
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_/model000038000.pt



# 
export const_noise=""
export const_noise="--const_noise"



# /data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000014000.pt...


### ==== not use ours, use_vae, train_enc ==== ###
# export without_dec_pos_emb=""
# export use_ours_transformer_enc=""
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000014000.pt
# export use_vae="--use_vae"
# export train_enc="--train_enc" ## # train_enc #


# export pred_diff_noise=""
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_use_vae_real_train_enc_/model000000500.pt
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_use_vae_real_train_enc_/model000012500.pt
# export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_enc_use_vae_real_train_enc_/model000013500.pt


# andj ad 

export cuda_ids=5
export pred_diff_noise="--pred_diff_noise"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export use_vae="--use_vae" ## use_vae ##
export train_enc=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_use_vae_/model000013500.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_use_vae_/model000019000.pt
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_use_vae_/model000033000.pt


export set_attn_to_none=""

export cuda_ids=7
### const noise ###
export set_attn_to_none="--set_attn_to_none"
export const_noise="--const_noise"
# export const_noise=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_use_vae_attnmask_none_/model000034500.pt



export cuda_ids=6
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_nsigmoid_wrel_wnergy_scaled_loss_bboxscalejts_jtsonlyloss_bbox_scale_lbsz_sep_model_use_sigmoid_train_diff_wo_dec_pose_deepfusetimeemb_ours_transenc_prednoise_use_vae_attnmask_none_const_noise_/model000035000.pt




### tesst train enc for basejtsrel ###
export diff_jts=""
export diff_basejtsrel="--diff_basejtsrel"
export diff_basejtse=""
export set_attn_to_none="--set_attn_to_none"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export train_enc="--train_enc"
export train_diff=""
export use_vae=""
export const_noise=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_/model000000500.pt
### use vae for encoding
export use_vae="--use_vae"
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000002000.pt
### tesst train enc for basejtsrel ###
export cuda_ids=5 ## cuda_ids ###
export seed=11
export diff_jts=""
export diff_basejtsrel="--diff_basejtsrel"
export diff_basejtse=""
export set_attn_to_none="--set_attn_to_none"
export use_ours_transformer_enc="--use_ours_transformer_enc"
# export train_enc="--train_enc"
# export train_diff=""
export train_enc=""
export train_diff="--train_diff"
export pred_diff_noise="--pred_diff_noise"
export use_vae="--use_vae"
export const_noise=""
export model_path=/data1/xueyi/mdm/save/my_humanml_trans_enc_512_v2_w_60_bsz_64_norm_p1_latents_ws_30_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_diff_enc_use_vae_res_enc_balance_loss_weights_/model000055000.pt
export rnd_noise=""
export rnd_noise="--rnd_noise"




# export batch_size=10
export batch_size=10
# export batch_size=20
export diff_basejtsrel="--diff_basejtsrel"
export diff_jts=""
export diff_basejtse=""
export use_vae=""
export kl_weights=0.001 ## kl-weight ##
export jts_sclae_stra="bbox"
export train_enc="--train_enc"
export train_diff=""
export resume_checkpoint=""
export const_noise=""
export set_attn_to_none="--set_attn_to_none"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export not_load_opt="--not_load_opt"
export resume_diff=""
export const_noise=""
export use_sigmoid="--use_sigmoid"
export wo_e_normalization="--wo_e_normalization"
export wo_rel_normalization="--wo_rel_normalization"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae__/model000002500.pt
# /data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae__/model000002500.pt
export use_vae="--use_vae"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000002000.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000002500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_use_vae_/model000004500.pt

export use_vae=""
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae__/model000006500.pt



export use_vae="--use_vae"
export kl_weights=0.00 ## kl-weight ##
export jts_sclae_stra="bbox"
export train_enc="--train_enc"
export train_diff=""
export resume_checkpoint=""
export set_attn_to_none="--set_attn_to_none"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export not_load_opt="--not_load_opt"
export resume_diff=""
export const_noise=""
export use_sigmoid="--use_sigmoid"
export wo_e_normalization="--wo_e_normalization"
export wo_rel_normalization="--wo_rel_normalization" ## wihtout rel normalization but wie
export use_dec_rel_v2="--use_dec_rel_v2"
export pred_basejtsrel_avgjts="--pred_basejtsrel_avgjts"
export model_path="/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_/model000000500.pt"
## model-path; 
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_/model000001500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_/model000017000.pt


export train_enc="--train_enc"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_diff_enc_wodecpos_/model000017500.pt
export single_frame_noise="--single_frame_noise"




export cuda_ids=2
export diff_basejtsrel="--diff_basejtsrel"
export diff_jts=""
export diff_basejtse=""
export use_vae="--use_vae"
export kl_weights=0.00 ## kl-weight ##
export jts_sclae_stra="bbox"
export train_enc=""
export train_diff=""
export set_attn_to_none="--set_attn_to_none"
export use_ours_transformer_enc="--use_ours_transformer_enc"
export not_load_opt="--not_load_opt"
export resume_diff="" # not load diff
export const_noise=""
export use_sigmoid="--use_sigmoid"
export wo_e_normalization="--wo_e_normalization"
export wo_rel_normalization="--wo_rel_normalization" ## wihtout rel normalization but wie
export use_dec_rel_v2="--use_dec_rel_v2"
export pred_basejtsrel_avgjts="--pred_basejtsrel_avgjts"
export without_dec_pos_emb=""
export only_first_clip="--only_first_clip"
export single_frame_noise=""

export without_dec_pos_emb="--without_dec_pos_emb"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_diff_enc_wodecpos_/model000018500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_diff_enc_wodecpos_scale_latents_/model000019000.pt


export  use_t=400
# export  use_t=50 ## use_t; use_t; 
export  use_t=100 ## use_t; use_t; 
export  use_t=1 ## use_t; use_t; 

export  use_t=50 ## use_t; use_t; 
export not_add_noise=""
# export not_add_noise="--not_add_noise"
# export mod
export rnd_noise=""
# export rnd_noise="--rnd_noise"



## single seq path ##
# export single_seq_path="/data1/xueyi/GRAB_processed/train/1.npy"

# export train_enc="--train_enc"

# export use_vae="--use_vae"
# export kl_weights=0.00001 ## kl-weight ##
# export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_/model000000500.pt
# export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_/model000005500.pt
# export not_cond_base="--not_cond_base"


# export kl_weights=0.0000 ## kl-weight ##
# export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_train_enc_/model000004000.pt


export not_cond_base="--not_cond_base"
export use_dec_rel_v2="--use_dec_rel_v2"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_/model000013000.pt
export latent_dim=512
export diff_spatial="--diff_spatial"
export noise_schedule="linear"
# export pred_diff_noise=""
export  use_t=1
export pred_joints_offset="--pred_joints_offset"
export not_pred_avg_jts="--not_pred_avg_jts"

export  use_t=100
export  use_t=50
export jts_sclae_stra="std"
export noise_schedule="linear"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_/model000001500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_/model000002500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_/model000004500.pt
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_/model000008500.pt
## std scale for the joints ###
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_/model000008500.pt
export not_diff_avgjts=""




export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_notdiffavgjts_/model000004500.pt

### not predict avg jts ###
export not_diff_avgjts="--not_diff_avgjts"
export model_path=/data1/xueyi/mdm/save/trans_enc_512_rel_basejtsrelonly_lbsz_sep_model_use_sigmoid_train_enc_nusevae_dec_rel_v2_predavgjts_train_enc_diff_with_pos_enc_train_enc_wodecpos_scale_latents_not_cond_base_use_vae_npred_avg_jts_smaller_kl_diff_spatial_predoffset_stdscale_notdiffavgjts_/model000005500.pt
# export rnd_noise="--rnd_noise"
export  use_t=50


### joint std v2 ###
export joint_std_v2="--joint_std_v2"
export joint_std_v3=""

export rnd_noise="--rnd_noise"
export use_t=1000

# export  use_t=100
export seed=55
export seed=99
export single_seq_path="/data1/xueyi/GRAB_processed/test/80.npy"
# export single_seq_path="/data1/xueyi/GRAB_processed/test/50.npy"



## v3 std ### latent diffusions ###
# export resume_checkpoin
export diff_latents="--diff_latents"
export diff_spatial=""
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
export train_enc=""
export train_diff=""
export use_vae=""
export kl_weights=0.00
export use_sigmoid="--use_sigmoid" ## latents or not latents ##

### #####
export use_vae="--use_vae"
export kl_weights=0.00001
export  use_t=1000
export rnd_noise="--rnd_noise"

export use_vae=""
export kl_weights=0.00000


export  use_t=1000
export rnd_noise="--rnd_noise"
export diff_spatial="--diff_spatial"
export diff_latents=""
export pred_diff_noise="" 
export joint_std_v2=""
export batch_size=10
# export batch_size=80
export joint_std_v3="--joint_std_v3"
export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_notdiffavgjts_v3std_bsz_10_not_pred_diff_/model000004500.pt
export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_notdiffavgjts_v3std_bsz_10_not_pred_diff_/model000010500.pt


export seed=11
export  use_t=2
export rnd_noise=""


export seed=11
export  use_t=1000
export rnd_noise="--rnd_noise"

export seed=11
export  use_t=2
export rnd_noise=""

export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_notdiffavgjts_v3std_bsz_10_not_pred_diff_use_canon_jts_/model000003000.pt
export use_canon_joints="--use_canon_joints"

export use_var_sched=""



# export  use_t=1000
# export rnd_noise="--rnd_noise"
export pred_diff_noise="--pred_diff_noise" 
export diff_spatial="--diff_spatial"
export diff_latents=""
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
export use_var_sched="--use_var_sched"
export use_canon_joints=""
export use_t=100
# export use_t=1000
# export use_t=1
export seed=15
export rnd_noise=""
export cuda_ids=0

### diff_basejtsrel ###
export diff_jts=""
export diff_basejtsrel=""
export diff_basejtse="--diff_basejtse"
export pred_diff_noise="--pred_diff_noise" 
export diff_spatial="--diff_spatial"
export diff_latents=""
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
export use_var_sched="--use_var_sched"
export use_canon_joints=""
export use_t=1
export seed=15
export rnd_noise=""
export cuda_ids=0
export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_basejtse_/model000016500.pt


export e_normalization_stra="cent"
# diff spatial; diff latents; #
export diff_spatial=""
export diff_latents=""
# export cuda_ids=5
export seed=11
export diff_jts=""
export diff_basejtsrel=""
export diff_basejtse="--diff_basejtse"
export set_attn_to_none="--set_attn_to_none" ### none_attn
export use_ours_transformer_enc="--use_ours_transformer_enc"
export train_enc="" # 
export pred_diff_noise="--pred_diff_noise"
export deep_fuse_timeemb="--deep_fuse_timeemb" 
export train_diff="" # # use_vae 
export use_vae="--use_vae"
export jts_sclae_stra="std" ## eval
export const_noise=""

export rnd_noise="--rnd_noise" ### use rnd noise 
export rnd_noise="" # not use rnd noise #
export train_enc=""
export train_diff="--train_diff" 
export use_vae="" ### not use vae ###
export wo_e_normalization="--wo_e_normalization" # train enc without normalization ##  
export use_t=10


export real_basejtsrel_norm_stra="none"

export real_basejtsrel_norm_stra="mean"
export diff_realbasejtsrel="--diff_realbasejtsrel"
export use_canon_joints=""
export pred_diff_noise="--pred_diff_noise" # pred_diff_noise --> not rpedict diff nosie here ##
export diff_spatial="--diff_spatial"
export diff_basejtsrel="" ### diff_basejtse ###
export diff_jts=""
export diff_basejtse=""
export cuda_ids=2
# export save_dir="/data1/xueyi/mdm/save/predoffset_stdscale_notdiffavgjts_v3std_bsz_10_pred_diff_use_canon_jts_"
# export save_dir="/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_mean_for_norm_" ## pred bsejtse here ##
export joint_std_v2=""
export joint_std_v3="--joint_std_v3"
# export batch_size=10
# export batch_size=8
export e_normalization_stra="cent"
export wo_e_normalization=""
# export batch_size=80 ## jointsstdv3 ##

export pred_diff_noise=""
export train_enc=""
export train_enc="--train_enc"
export train_diff=""
export cuda_ids=5
export use_t=1
export real_basejtsrel_norm_stra="std"
# export real_basejtsrel_norm_stra="mean"
# export save_dir="/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_" ## pred bsejtse here ##
export joint_std_v2=""
export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_/model000002500.pt


export pred_diff_noise="--pred_diff_noise"
export cuda_ids=2
export train_enc="--train_enc" # train enc #
# export use_t=300
# export seed=99
# export seed=220
export use_sigmoid="--use_sigmoid"
# export model_path="/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_/model000004000.pt"
# export model_path="/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_prediffnoise_/model000086500.pt"
# export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_/model000093000.pt


# export diff_realbasejtsrel_to_joints="--diff_realbasejtsrel_to_joints"
# export diff_realbasejtsrel=""
# export pred_diff_noise="--pred_diff_noise" ## pred diff noise --- pred diff noise ##
# export use_sigmoid=""
# export train_enc=""
# export cuda_ids=6 # cuda_ids
# export use_t=100
export real_basejtsrel_norm_stra="none"


# export cuda_ids=4 
# export use_t=200
export use_abs_jts_for_encoding="--use_abs_jts_for_encoding"


# export use_abs_jts_for_encoding_obj_base="--use_abs_jts_for_encoding_obj_base"
# export use_abs_jts_for_encoding=""
# export model_path=/data1/xueyi/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_prediffnoise_none_norm_rel_rel_to_jts_use_absjtspos_absjtsforenc_objbase_/model000003000.pt
# given a skip
# upconv net #

# export use_t=500
# export use_t=300

# export cuda_ids=0 #  
# export seed=77 # 
# export nn_base_pts=700
# export cuda_ids=2
# export seed=101
# # export seed=220

export use_objbase_v2="--use_objbase_v2"
export use_abs_jts_for_encoding=""
export use_abs_jts_for_encoding_obj_base=""
export use_abs_jts_pos=""


# export seed=220
# export seed=31
# export seed=17
# # export seed=101
# # ### Latenets ###
# export use_t=200
export real_basejtsrel_norm_stra="none"
export diff_realbasejtsrel="--diff_realbasejtsrel"
export use_canon_joints=""
export pred_diff_noise="--pred_diff_noise"
export diff_spatial="--diff_spatial"
export diff_basejtsrel=""
export diff_jts=""
export diff_basejtse=""
export e_normalization_stra="cent"
export wo_e_normalization=""


#### Model settings and checkpoints ####
export finetune_with_cond=""
export finetune_with_cond_rel=""
export finetune_with_cond_jtsobj=""
export use_objbase_v7=""
export v5_in_without_glb="--v5_in_without_glb"
export add_noise_onjts="--add_noise_onjts"
export add_noise_onjts_single="" 
export v5_in_not_base="--v5_in_not_base"
export v5_in_not_base_pos=""
export use_objbase_out_v5="--use_objbase_out_v5"
export real_basejtsrel_norm_stra="none"
export use_objbase_v5="--use_objbase_v5"
export use_objbase_v6=""
export not_cond_base="--not_cond_base"
export v5_out_not_cond_base="--v5_out_not_cond_base"
export use_objbase_out_v3=""
export use_objbase_v2=""
export use_objbase_v4=""
export use_objbase_out_v4=""
export out_objbase_v5_bundle_out="--out_objbase_v5_bundle_out"
export train_enc=""
export use_sigmoid=""
export use_jts_pert_realbasejtsrel=""

export joint_std_v2=""
export joint_std_v3="--joint_std_v3"

export latent_dim=512

export diff_hand_params=""
export diff_basejtsrel=""
export diff_realbasejtsrel="--diff_realbasejtsrel"
export diff_realbasejtsrel_to_joints=""
export model_path=./ckpts/model_spatial.pt




export seed=33

export use_t=200
export scale_obj=1
export nn_base_pts=2000
export resplit=""
export use_arti_obj=""
export use_reverse=""
export phy_guided_sampling=""
export use_anchors=""

export start_idx="--start_idx 0"
export theta_dim=24
export only_cmb_finger=""
export cad_model_fn=""
export use_same_noise_for_rep=""
export sel_basepts_idx=-1
# export pert_type="uniform"
export pert_type="gaussian"
export use_var_sched="--use_var_sched"

export test_tag="jts_spatial_grab_t_200_test_"
export prev_test_tag="jts_grab_t_400_test_"


export single_seq_path=data/grab/source_data/test/14.npy
export save_dir=data/grab/result
export grab_processed_dir=data/grab/source_data

## cuda_idx ##


export cuda_ids=3

# bash scripts/val_examples/predict_grab_rndseed_spatial_14.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.predict_grab_spatial --dataset motion_ours --save_dir ${save_dir} --single_seq_path ${single_seq_path} --window_size ${window_size} ${unconstrained} ${inst_normalization}  --model_path ${model_path} --rep_type ${rep_type} --batch_size=${batch_size}  --denoising_stra ${denoising_stra} ${inter_optim} --seed ${seed} ${diff_jts} ${diff_basejtsrel} ${diff_basejtse}  ${use_sep_models} --jts_sclae_stra ${jts_sclae_stra} ${use_vae} ${use_sigmoid} ${train_enc} ${without_dec_pos_emb} ${pred_diff_noise} ${resume_diff} ${not_load_opt} ${deep_fuse_timeemb}  ${use_ours_transformer_enc} ${const_noise} ${set_attn_to_none} ${rnd_noise} ${wo_e_normalization} ${wo_rel_normalization} ${use_dec_rel_v2} ${pred_basejtsrel_avgjts} ${single_frame_noise} --use_t ${use_t} ${not_add_noise} ${not_cond_base}  ${not_pred_avg_jts} --latent_dim ${latent_dim} ${diff_spatial} --noise_schedule ${noise_schedule} ${pred_joints_offset}  ${not_diff_avgjts}  ${joint_std_v3}  ${joint_std_v2}  ${diff_latents}  ${use_canon_joints} ${use_var_sched} --e_normalization_stra ${e_normalization_stra} --real_basejtsrel_norm_stra ${real_basejtsrel_norm_stra} ${diff_realbasejtsrel}  ${diff_realbasejtsrel_to_joints} ${use_abs_jts_for_encoding} ${use_abs_jts_for_encoding_obj_base} ${use_abs_jts_pos} ${use_objbase_v2} ${use_objbase_out_v3} ${use_objbase_v4} ${use_objbase_out_v4} ${use_objbase_v5} ${use_objbase_out_v5} ${out_objbase_v5_bundle_out} --nn_base_pts ${nn_base_pts} ${add_noise_onjts} ${v5_out_not_cond_base} ${use_objbase_v6} ${add_noise_onjts_single} ${only_cmb_finger} ${use_objbase_v7} ${v5_in_not_base_pos} ${v5_in_not_base} ${v5_in_without_glb}  ${finetune_with_cond} --test_tag ${test_tag} ${finetune_with_cond_rel} ${finetune_with_cond_jtsobj} --sel_basepts_idx ${sel_basepts_idx} ${use_same_noise_for_rep} --pert_type ${pert_type} ${phy_guided_sampling} ${use_anchors} ${use_arti_obj} --theta_dim ${theta_dim} ${start_idx} ${use_reverse} --scale_obj ${scale_obj} ${resplit} --prev_test_tag=${prev_test_tag}


