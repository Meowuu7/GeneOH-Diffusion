
try:
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
    ps.set_screenshot_extension(".png")
except:
    pass

from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
sys.path.append("./manopth")
from manopth.manolayer import ManoLayer

color = [
(0,191/255.0,255/255.0),
    (186/255.0,85/255.0,211/255.0),
    (255/255.0,81/255.0,81/255.0),
    (92/255.0,122/255.0,234/255.0),
    (255/255.0,138/255.0,174/255.0),
    (77/255.0,150/255.0,255/255.0),
    (192/255.0,237/255.0,166/255.0)
    #
]

def seal(v, f):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype=np.int32)
    center = (v[circle_v_id, :]).mean(0)

    # sealed_mesh = copy.copy(mesh_to_seal)
    v = np.vstack([v, center])
    center_v_id = v.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i - 1], circle_v_id[i], center_v_id]
        f = np.vstack([f, new_faces])
    return v, f, center




def get_mano_model(ncomps=45, side='right', flat_hand_mean=False,):
    # ncomps = 45 # mano root #
    batch_size = 1
    mano_model = ManoLayer(
        mano_root='manopth/mano/models', use_pca=True if ncomps == 15 else False, ncomps=ncomps, flat_hand_mean=flat_hand_mean, side=side, center_idx=0)
    return mano_model




def vis_predicted_joints_hoi4d(predicted_info_fn, optimized_fn=None):
    mano_model = get_mano_model()
    faces = mano_model.th_faces.squeeze(0).numpy()

    # ws = ws
    # is_toch = False
    # predicted_info_data = np.load(predicted_info_fn, allow_pickle=True).item()
    if optimized_fn is not None:
        data = np.load(optimized_fn, allow_pickle=True).item()
        print(f"keys of optimized dict: {data.keys()}")
        if "optimized_out_hand_verts" in data:
            optimized_out_hand_verts = data["optimized_out_hand_verts"]
            if "optimized_out_hand_verts_before_contact_opt" in data:
                optimized_out_hand_verts_woopt = data["optimized_out_hand_verts_before_contact_opt"]
            else:
                optimized_out_hand_verts_woopt = optimized_out_hand_verts
            # optimized_out_hand_verts = data["optimized_out_hand_verts_ne"]
        elif "optimized_joints" in data:
            optimized_out_hand_verts = data["optimized_joints"]
        elif "hand_verts" in data:
            optimized_out_hand_verts = data["hand_verts"]  # [:ws]
            # optimized_out_hand_verts = data["bf_ct_verts"] # [:ws]
            # optimized_out_hand_verts_woopt = data["bf_ct_verts"]
            optimized_out_hand_verts_woopt = data["hand_verts"]
            # is_toch = True
        elif "hand_verts_tot" in data:
            optimized_out_hand_verts = data["hand_verts_tot"][:ws]
            optimized_out_hand_verts_woopt = optimized_out_hand_verts
            # is_toch = True

        if 'tot_base_pts_trans' in data:
            tot_base_pts_trans = data['tot_base_pts_trans']
        else:
            tot_base_pts_trans = None
    else:
        optimized_out_hand_verts = None
        optimized_tar_hand_verts = None
        tot_base_pts_trans = None

    data = np.load(predicted_info_fn, allow_pickle=True).item()

    print(f"data: {data.keys()}")
    try:
        targets = data['targets']
    except:
        targets = data['tot_gt_rhand_joints']
    # targets = data_jts['outputs']

    outputs = data['outputs']
    # outputs = data['tot_gt_rhand_joints'][0]
    if 'obj_verts' in data:
        obj_verts = data['obj_verts']
        obj_faces = data['obj_faces']
    elif 'tot_obj_pcs' in data:
        obj_verts = data['tot_obj_pcs'][0]
        if 'template_obj_fs' in data:
            obj_faces = data['template_obj_fs']  #
        else:
            obj_faces = None
    tot_base_pts = data["tot_base_pts"][0]  # total base points # bsz x nnbasepts x 3 #
    # tot_rhand_joints = data["tot_rhand_joints"][0]

    pert_verts = data['pert_verts']

    if 'tot_obj_rot' in data:
        tot_obj_rot = data['tot_obj_rot'][0]
        tot_obj_trans = data['tot_obj_transl'][0]
        obj_verts = np.matmul(obj_verts, tot_obj_rot) + tot_obj_trans.reshape(tot_obj_trans.shape[0], 1,
                                                                              3)  # ws x nn_obj x 3 #
        print(f"tot_obj_rot: {tot_obj_rot.shape}")
        # obj_verts = np.matmul( obj_verts, np.transpose(tot_obj_rot, (0, 2, 1))) + tot_obj_trans.reshape(tot_obj_trans.shape[0], 1,
        #                                                                       3)  # ws x nn_obj x 3 #

        outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_trans.reshape(tot_obj_trans.shape[0], 1,
                                                                          3)  # ws x nn_obj x 3 #

    # if toch_eval_fn is not None:
    #     optimized_out_hand_verts = tot_hand_verts[:60]

    jts_radius = 0.03378
    gray_color = (233 / 255., 241 / 255., 148 / 255.)
    # gray_color = (252/ 255.0, 249/ 255.0, 190/ 255.0)
    print(f"targets: {targets.shape}, outputs: {outputs.shape}")

    maxx_ws = 30
    maxx_ws = 100
    ws = 200
    maxx_ws = ws
    # with or wihout parameter smoothing and the effects here #
    skipp = 6
    skipp = 1
    iidx = 1
    tot_hand_verts_woopt = []
    print(f"maxx_ws: {maxx_ws}, optimized_out_hand_verts: {optimized_out_hand_verts.shape[0]}")
    for i_fr in range(0, min(maxx_ws, optimized_out_hand_verts.shape[0]), skipp):
        # cur_base_pts = tot_base_pts
        # # cur_rhand_joints = tot_rhand_joints[i_fr]

        if i_fr < obj_verts.shape[0]:
            cur_obj_verts = obj_verts[i_fr]
            cur_obj_faces = obj_faces

        if tot_base_pts_trans is not None:
            cur_fr_base_pts_trans = tot_base_pts_trans[i_fr]
            sel_idx = 540
            cur_fr_base_pts_trans_sel = cur_fr_base_pts_trans[sel_idx:sel_idx + 1]

        sealed_v, seald_f, center_wopt = seal(optimized_out_hand_verts[i_fr], faces)


        sealed_v, seald_f, center_woopt = seal(optimized_out_hand_verts_woopt[i_fr], faces)
        sealed_v_0 = sealed_v[0:1]
        hand_mesh = ps.register_surface_mesh(f"cur_hand_mesh_woopt", sealed_v,
                                                seald_f,
                                                color=color[0 % len(color)])

        tot_hand_verts_woopt.append(sealed_v)

        sealed_v, seald_f, center_woopt = seal(pert_verts[i_fr], faces)
        sealed_v = sealed_v - sealed_v[0:1] + sealed_v_0
        hand_mesh = ps.register_surface_mesh(f"cur_hand_gt_verts", sealed_v,
                                                seald_f,
                                                color=color[2 % len(color)])

        # sealed_v, seald_f, center_inputs = seal(optimized_out_hand_verts_woopt[i_fr], faces)
        sealed_v = sealed_v - 0.5 * np.reshape(center_woopt, (1, 3)) + 0.5 * np.reshape(center_wopt, (1, 3))


        if cur_obj_faces is not None:
            obj_mesh = ps.register_surface_mesh(f"cur_object", cur_obj_verts, cur_obj_faces,
                                            color=gray_color)
        else:
            obj_mesh = ps.register_point_cloud(f"cur_object", cur_obj_verts,
                                            color=gray_color)

        maxx_obj_verts = np.max(cur_obj_verts, axis=0)
        minn_obj_verts = np.min(cur_obj_verts, axis=0)
        print(f"maxx_obj_verts: {maxx_obj_verts}, minn_obj_verts: {minn_obj_verts}")
        iidx += 1
        if i_fr == 0: 
            ps.set_up_dir("y_up")
            ps.set_up_dir("neg_z_up")
            ps.set_up_dir("z_up")
            look_at_pos = (0.35, 0.0, 0.45)
            look_at_dir = (-1.0, 0.0, -0.7)
            look_at_pos = (0., 0.35, 0.45)
            look_at_dir = (0.0, -1.0, -0.7)
            look_at_pos = (0., -0.35, 0.45)
            look_at_dir = (0.0, 1.0, -0.7)
            ps.look_at(look_at_pos, look_at_dir)
            ps.show()
        ps.set_screenshot_extension(".jpg")
        ps.screenshot()
        ps.remove_all_structures()



# python visualize/vis_hoi4d_example_toycar_inst3.py
if __name__=='__main__':

    ws = 300
    ### configs ###
    tag = "jts_spatial_hoi4d_t_200_test_"
    seq_idx = 3
    ### configs ###

    
    ### get file paths ###
    seed = 0
    
    predicted_info_fn = f"data/hoi4d/result/ToyCar/predicted_infos_seq_{seq_idx}_seed_{seed}_tag_{tag}.npy"
    optimized_fn = f"data/hoi4d/result/ToyCar/optimized_infos_sv_dict_seq_{seq_idx}_seed_{seed}_tag_{tag}_dist_thres_0.01_with_proj_True.npy"


    vis_predicted_joints_hoi4d(predicted_info_fn, optimized_fn)
    
    