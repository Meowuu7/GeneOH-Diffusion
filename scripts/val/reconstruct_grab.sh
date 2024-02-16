export PYTHONPATH=.

export cuda_ids=6



export save_dir="/data2/xueyi/eval_save/GRAB"

export test_tag=jts_spatial_grab_t_200_test_
export single_seq_path=/data1/xueyi/GRAB_processed/test/14.npy

# bash scripts/val/reconstruct_grab.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_grab --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

