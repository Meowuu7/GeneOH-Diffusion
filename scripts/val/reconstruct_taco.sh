export PYTHONPATH=.

export cuda_ids=6







export single_seq_path="data/taco/source_data/20231104_017.pkl"
export test_tag="20231103_020_jts_spatial_t_100_hho_"


export save_dir="exp/taco/eval_save"




# bash scripts/val/reconstruct_taco.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_taco --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

