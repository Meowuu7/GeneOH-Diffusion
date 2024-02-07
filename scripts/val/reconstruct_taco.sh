export PYTHONPATH=.

export cuda_ids=6



export single_seq_path="/data3/hlyang/results/test_data/20231105/20231105_020.pkl"
export test_tag="20231105_020_jts_spatial_t_100_hho_"


export single_seq_path="/data3/hlyang/results/test_data/20231103/20231103_020.pkl"
export test_tag="20231103_020_jts_spatial_t_100_hho_"


# export use_arctic=""
# export use_hho="--use_hho"
# export use_left="--use_left"

# export use_reverse=""

# export seed=0


export save_dir="/data3/datasets/xueyi/hho_save_res"




# bash scripts/val/reconstruct_taco.sh --save_dir=./data/taco/result --single_seq_path=./data/taco/source_data/20231105_020.pkl
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_taco --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

