# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

### add gp

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from train.training_loop_ours import TrainLoop as TrainLoop_Ours ### trainer ours ###
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset motion_ours
def main():
    args = train_args()
    fixseed(args.seed) # fixseed # 
    # train_platform_type, 
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args') # train platform

    if args.save_dir is None: # save dir was not specified #
        raise FileNotFoundError('save_dir was not specified.')
    # elif os.path.exists(args.save_dir) and not args.overwrite:
    #     raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    # elif not os.path.exists(args.save_dir):
        # os.makedirs(args.save_dir, exist_ok=True)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    ## === setup dist === ##
    dist_util.setup_dist(args.device)

    ## train mdm and dataest ##
    print("creating data loader...")
    # create data loaders # get dataset loader #
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, args=args)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev()) ## model-to-the-target-device ##
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    if args.dataset in ["motion_ours"] and args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist", "obj_base_rel_dist_we", "obj_base_rel_dist_we_wj", "obj_base_rel_dist_we_wj_latents"]:
        print(f"Start training loops for rep_type {args.rep_type}")
        TrainLoop_Ours(args, train_platform, model, diffusion, data).run_loop()
    else:
        TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
