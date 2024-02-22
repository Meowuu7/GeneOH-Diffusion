export PYTHONPATH=.



export single_seq_path=./data/taco/source_data/20231104_017.pkl
export test_tag="20231104_017_jts_spatial_t_100_"


export save_dir=./data/taco/result



export cuda_ids=0

# val_examples/reconstruct_taco_20231104_017.sh
# bash scripts/val_examples/reconstruct_taco_20231104_017.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_taco --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

