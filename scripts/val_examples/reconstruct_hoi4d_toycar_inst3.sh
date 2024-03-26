export PYTHONPATH=.



export single_seq_path="data/hoi4d/ToyCar/case3/merged_data.npy"
export test_tag="jts_spatial_hoi4d_t_200_test_"


export save_dir=data/hoi4d/result



export cuda_ids=0


# bash scripts/val_examples/reconstruct_hoi4d_toycar_inst3.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_hoi4d --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

