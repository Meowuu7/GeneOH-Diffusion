export PYTHONPATH=.



# export single_seq_path="data/hoi4d/ToyCar/case3/merged_data.npy"
export test_tag="jts_spatial_hoi4d_t_200_test_"


export save_dir=data/hoi4d/result




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
#### Data and exp folders ####


################# Set to your paths #################
#### Data and exp folders ####
export save_dir="data/hoi4d/result"


export cuda_ids=0


# bash scripts/val/reconstruct_hoi4d_toycar_category.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_hoi4d_category --save_dir=${save_dir} --test_tag=${test_tag}  --hoi4d_category_name=${hoi4d_category_name} --hoi4d_data_root=${hoi4d_data_root} --hoi4d_eval_st_idx=${hoi4d_eval_st_idx} --hoi4d_eval_ed_idx=${hoi4d_eval_ed_idx} --hoi4d_cad_model_root=${hoi4d_cad_model_root} --select_part_idx=${select_part_idx} --use_arti_obj=${use_arti_obj}

# --single_seq_path=${single_seq_path}

