
import numpy as np
from manopth.manolayer import ManoLayer
import os
import torch

def get_mano_models():
    mano_path = "/data1/sim/mano_models/mano/models" 
    rgt_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='right',
        mano_root=mano_path, # mano_root #
        ncomps=45,
        use_pca=False,
        # root_rot_mode='axisang',
        # joint_rot_mode='axisang'
    )
    
    lft_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='left',
        mano_root=mano_path, # mano_root #
        ncomps=45,
        use_pca=False,
        # root_rot_mode='axisang',
        # joint_rot_mode='axisang'
    )
    
    return rgt_mano_layer, lft_mano_layer


def get_embeddings():
    # /home/xueyi/sim/arctic/prepared_data_s01_box_grab_01_2.npy
    # /home/xueyi/sim/arctic/prepared_data_s01_box_grab_01_1.npy
    # /home/xueyi/sim/arctic/prepared_data_s01_laptop_use_01_1.npy
    # /home/xueyi/sim/arctic/prepared_data_s02_laptop_use_02_2.npy
    tags = [
        "s01_box_grab_01", "s01_laptop_use_01", "s02_laptop_use_02"
    ]
    
    sv_root = "/home/xueyi/sim/arctic/"
    
    tot_pose_diff_data = []
    tot_gaussian_diff_data = []
    tot_beta_diff_data = []
    ws = 60
    aug_pose = 0.5
    aug_pose_beta = 0.3
    for tag in tags:
        for cur_view in range(1, 9):
            cur_predicted_data_sv_fn = f"prepared_data_{tag}_{cur_view}.npy"
            cur_predicted_data_sv_fn = os.path.join(sv_root, cur_predicted_data_sv_fn)
            prepared_data = np.load(cur_predicted_data_sv_fn, allow_pickle=True).item()
    
            # prepared_data = np.load(prepared_data_fn, allow_pickle=True).item()
            # print(f"prepared_data: {prepared_data.keys()}")
            # tag = "targets"
            # tag = "pred"
            # rhand_verts = prepared_data[f"{tag}.mano.v3d.cam.r"].detach().cpu().numpy()
            # lhand_verts = prepared_data[f"{tag}.mano.v3d.cam.l"].detach().cpu().numpy()
            # pred_obj_verts = prepared_data[f'{tag}.object.v.cam'].detach().cpu().numpy() # pred.object.v.cam

            # get_mano_model
            # rgt_mano_model, lft_mano_layer = get_mano_models(ncomps=45, side='right') ## single path -> and then to the evaluation protocal #
            
            pred_tag = "pred"
            target_tag = "targets"
            pred_pose_r = prepared_data[f"{pred_tag}.mano.pose.r"]
            target_pose_r = prepared_data[f"{target_tag}.mano.pose.r"]
            
            pred_pose_r = pred_pose_r.contiguous().view(pred_pose_r.size(0), -1).contiguous() ###
            pred_pose_r = pred_pose_r[:, 3:]
            target_pose_r = target_pose_r[:, 3:]
            
            
            for i in range(0, pred_pose_r.size(0) - 60, ws):
                st_idx = i
                ed_idx = i + ws
                cur_w_pred_pose_r = pred_pose_r[st_idx: ed_idx, 3:]
                cur_w_target_pose_r = target_pose_r[st_idx: ed_idx, 3:]
                diff_cur_w_target_pred_pose_r = cur_w_pred_pose_r - cur_w_target_pose_r
                # diff_cur_w_target_pred_pose_
                tot_pose_diff_data.append(diff_cur_w_target_pred_pose_r.numpy())
                
                gaussian_rnd_noise = torch.randn_like(diff_cur_w_target_pred_pose_r) * aug_pose 
                gaussian_rnd_noise = gaussian_rnd_noise.numpy()
                
                tot_gaussian_diff_data.append(gaussian_rnd_noise)
                
                
                dist_beta = torch.distributions.beta.Beta(torch.tensor([8.]), torch.tensor([2.]))
                aug_pose_var = dist_beta.sample(diff_cur_w_target_pred_pose_r.size()).squeeze(-1) * aug_pose_beta
                
                # aug_pose_var = torch.randn_like(torch.from_numpy(diff_cur_w_target_pred_pose_r).float()) * aug_pose 
                aug_pose_var = aug_pose_var.numpy()
                
                tot_beta_diff_data.append(aug_pose_var)
                
    tot_pose_diff_data = np.stack(tot_pose_diff_data, axis=0)
    # tot_pose_diff_data = 
    tot_pose_diff_data = np.reshape(tot_pose_diff_data, (tot_pose_diff_data.shape[0], -1))
    np.save(f"tot_pose_diff_data.npy", tot_pose_diff_data)
    
    tot_gaussian_diff_data = np.stack(tot_gaussian_diff_data, axis=0)
    tot_gaussian_diff_data = np.reshape(tot_gaussian_diff_data, (tot_gaussian_diff_data.shape[0], -1))
    np.save(f"tot_gaussian_diff_data.npy", tot_gaussian_diff_data)
    
    
    tot_beta_diff_data = np.stack(tot_beta_diff_data, axis=0)
    tot_beta_diff_data = np.reshape(tot_beta_diff_data, (tot_beta_diff_data.shape[0], -1))
    np.save(f"tot_beta_diff_data.npy", tot_beta_diff_data)
    
    
    
    # pred pose r; target pose r #
    
    
    # trans_r = prepared_data[f'{tag}.mano.cam_t.r']
    # pose_r = prepared_data[f'{tag}.mano.pose.r']
    # pose_r = pose_r.contiguous().view(pose_r.size(0), -1).contiguous() ###

    # psoe_r #
    # gt_pose_r = prepared_data[f'targets.mano.pose.r']
    # pred_pose_r = prepared_data[f'{tag}.mano.pose.r']
    # pred_pose_r = pred_pose_r.contiguous().view(pose_r.size(0), -1).contiguous() ###
    # print(f"gt_pose_r: {gt_pose_r.shape}, pred_pose_r: {pred_pose_r.shape}")
    # pred_pose_pose = pred_pose_r[:, 3:]
    # gt_pose_pose = gt_pose_r[:, 3:]
    # diff_pred_gt_pose = torch.mean((pred_pose_pose - gt_pose_pose) ** 2)
    # # diff_pred_gt_pose = diff_pred_
    # print(diff_pred_gt_pose)


if __name__=='__main__':
    get_embeddings()
    
