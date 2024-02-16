from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, motion_ours_collate, motion_ours_singe_seq_collate, motion_ours_obj_base_rel_dist_collate
# from data_loaders.humanml.data.dataset import HumanML3D
import torch

def get_dataset_class(name, args=None):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses ## to pose ##
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "motion_ours": # motion ours 
        if len(args.single_seq_path) > 0 and not args.use_predicted_infos and not args.use_interpolated_infos:
            print(f"Using single frame dataset for evaluation purpose...")
            # from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V16
            if args.rep_type == "obj_base_rel_dist":
                from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V17 as my_data
            elif args.rep_type == "ambient_obj_base_rel_dist":
                from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V18 as my_data
            elif args.rep_type in[ "obj_base_rel_dist_we", "obj_base_rel_dist_we_wj", "obj_base_rel_dist_we_wj_latents"]:
                if args.use_arctic and args.use_pose_pred:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19_Arctic_from_Pred as my_data
                elif args.use_hho:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19_HHO as my_data
                elif args.use_arctic:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19_Arctic as my_data
                elif len(args.cad_model_fn) > 0:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19_Ours as my_data
                elif len(args.predicted_info_fn) > 0:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19_From_Evaluated_Info as my_data
                else:
                    from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V19 as my_data
            else:
                from data_loaders.humanml.data.dataset_ours_single_seq import GRAB_Dataset_V16 as my_data
            return my_data
        else:
            if args.rep_type == "obj_base_rel_dist":
                from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V17 as my_data
            elif args.rep_type == "ambient_obj_base_rel_dist":
                from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V18 as my_data
            elif args.rep_type in ["obj_base_rel_dist_we", "obj_base_rel_dist_we_wj", "obj_base_rel_dist_we_wj_latents"]:
                if args.use_arctic:
                    from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V19_ARCTIC as my_data
                elif args.use_vox_data: # use vox data here #
                    from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V20 as my_data
                elif args.use_predicted_infos: # train with predicted infos for test tim adaptation #
                    from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V21 as my_data
                elif args.use_interpolated_infos:
                    # GRAB_Dataset_V22
                    from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V22 as my_data
                else:
                    from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V19 as my_data
            else:
                from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V16 as my_data
            return my_data
            # from data_loaders.humanml.data.dataset_ours import GRAB_Dataset_V16
        # return GRAB_Dataset_V16
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', args=None):
    print(f"name: {name}, hml_mode: {hml_mode}")
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name in ["motion_ours"]:
        ## === single seq path === ##
        print(f"single_seq_path: {args.single_seq_path}, rep_type: {args.rep_type}")
        # motion_ours_obj_base_rel_dist_collate
        ## rep_type of the obj_base_pts rel_dist; ambient obj base rel dist ##
        if args.rep_type in ["obj_base_rel_dist", "ambient_obj_base_rel_dist", "obj_base_rel_dist_we", "obj_base_rel_dist_we_wj", "obj_base_rel_dist_we_wj_latents"]:
            return motion_ours_obj_base_rel_dist_collate
        else: # single_seq_path #
            if len(args.single_seq_path) > 0:
                return motion_ours_singe_seq_collate
            else:
                return motion_ours_collate
        # if len(args.single_seq_path) > 0:
        #     return motion_ours_singe_seq_collate
        # else:
        #     if args.rep_type == "obj_base_rel_dist":
        #         return motion_ours_obj_base_rel_dist_collate
        #     else:
        #         return motion_ours_collate
    else:
        return all_collate

## get dataset and datasset ### 
def get_dataset(name, num_frames, split='train', hml_mode='train', args=None):
    DATA = get_dataset_class(name, args=args)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name in ["motion_ours"]:
        # humanml_datawarper = HumanML3D(split=split, num_frames=num_frames, mode=hml_mode, load_vectorizer=True)
        # w_vectorizer = humanml_datawarper.w_vectorizer
        
        w_vectorizer = None
        # split = "val" ## add split, split here --> split --> split and split ##
        # data_path = "/data1/xueyi/GRAB_processed"
        data_path = args.grab_processed_dir
        # split, w_vectorizer, window_size=30, step_size=15, num_points=8000, args=None
        window_size = args.window_size
        # split=  "val"
        dataset = DATA(data_path, split=split, w_vectorizer=w_vectorizer, window_size=window_size, step_size=15, num_points=8000, args=args)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_only(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    dataset = get_dataset(name, num_frames, split, hml_mode, args=args)
    return dataset

# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset motion_ours
def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    dataset = get_dataset(name, num_frames, split, hml_mode, args=args)
    collate = get_collate_fn(name, hml_mode, args=args)
    
    if args is not None and name in ["motion_ours"] and len(args.single_seq_path) > 0:
        shuffle_loader = False
        drop_last = False
    else:
        shuffle_loader = True
        drop_last = True

    num_workers = 8 ## get data; get data loader ##
    num_workers = 16 # num_workers # ## num_workders #
    ### ==== create dataloader here ==== ###
    ### ==== create dataloader here ==== ###
    loader = DataLoader( # tag for each sequence
        dataset, batch_size=batch_size, shuffle=shuffle_loader,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    )

    return loader


# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset motion_ours
def get_dataset_loader_dist(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    dataset = get_dataset(name, num_frames, split, hml_mode, args=args)
    collate = get_collate_fn(name, hml_mode, args=args)
    
    if args is not None and name in ["motion_ours"] and len(args.single_seq_path) > 0:
        # shuffle_loader = False
        drop_last = False
    else:
        # shuffle_loader = True
        drop_last = True
        
    num_workers = 8 ## get data; get data loader ##
    num_workers = 16 # num_workers # ## num_workders #
    ### ==== create dataloader here ==== ###
    ### ==== create dataloader here ==== ###
        
    ''' dist sampler and loader '''
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, drop_last=drop_last, collate_fn=collate)

   
    # loader = DataLoader( # tag for each sequence
    #     dataset, batch_size=batch_size, shuffle=shuffle_loader,
    #     num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    # )

    return loader