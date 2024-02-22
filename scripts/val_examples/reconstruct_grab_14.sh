export PYTHONPATH=.



export single_seq_path=data/grab/source_data/test/14.npy
export test_tag="jts_spatial_grab_t_200_test_"


export save_dir=data/grab/result



export cuda_ids=0
# bash scripts/val_examples/reconstruct_grab_14.sh
CUDA_VISIBLE_DEVICES=${cuda_ids} python -m sample.reconstruct_data_grab --save_dir=${save_dir} --test_tag=${test_tag} --single_seq_path=${single_seq_path}

