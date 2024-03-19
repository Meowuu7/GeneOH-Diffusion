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
    (0, 191 / 255.0, 255 / 255.0),
    (186 / 255.0, 85 / 255.0, 211 / 255.0),
    (255 / 255.0, 81 / 255.0, 81 / 255.0),
    (92 / 255.0, 122 / 255.0, 234 / 255.0),
    (255 / 255.0, 138 / 255.0, 174 / 255.0),
    (77 / 255.0, 150 / 255.0, 255 / 255.0),
    (192 / 255.0, 237 / 255.0, 166 / 255.0)
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


def get_mano_model(ncomps=45, side='right', flat_hand_mean=False, ):
    # ncomps = 45 # mano root #
    batch_size = 1
    mano_model = ManoLayer(
        mano_root='manopth/mano/models', use_pca=True if ncomps == 15 else False, ncomps=ncomps, flat_hand_mean=flat_hand_mean,
        side=side, center_idx=0)
    return mano_model


def vis_predicted_joints_taco(predicted_info_fn, optimized_fn=None, pkl_fn=None):
    mano_model = get_mano_model()
    faces = mano_model.th_faces.squeeze(0).numpy()

    # ws = ws
    # is_toch = False
    data = np.load(optimized_fn, allow_pickle=True).item()
    print(f"keys of optimized dict: {data.keys()}")

    if "optimized_out_hand_verts" in data:
        optimized_out_hand_verts = data["optimized_out_hand_verts"]
        if "optimized_out_hand_verts_before_contact_opt" in data:
            optimized_out_hand_verts_woopt = data["optimized_out_hand_verts_before_contact_opt"]
        else:
            optimized_out_hand_verts_woopt = optimized_out_hand_verts
    elif "optimized_joints" in data:
        optimized_out_hand_verts = data["optimized_joints"]
    elif "hand_verts" in data:
        optimized_out_hand_verts = data["hand_verts"]  # [:ws]
        # optimized_out_hand_verts = data["bf_ct_verts"] # [:ws]
        optimized_out_hand_verts_woopt = data["bf_ct_verts"]
        # is_toch = True
    elif "hand_verts_tot" in data:
        optimized_out_hand_verts = data["hand_verts_tot"][:ws]
        optimized_out_hand_verts_woopt = optimized_out_hand_verts

    data = np.load(predicted_info_fn, allow_pickle=True).item()

    print(f"data: {data.keys()}")
    outputs = data['outputs']
    # outputs = data['tot_gt_rhand_joints'][0]
    if 'obj_verts' in data:
        obj_verts = data['obj_verts']
        # obj_faces = data['obj_faces']
    elif 'tot_obj_pcs' in data:
        obj_verts = data['tot_obj_pcs'][0]

        # obj_faces = data['template_obj_fs']  #
    tot_base_pts = data["tot_base_pts"][0]  # total base points # bsz x nnbasepts x 3 #

    pert_verts = data['pert_verts']

    if pkl_fn is not None:
        import pickle as pkl
        data_dict = pkl.load(open(pkl_fn, 'rb'))
        # print(data_dict.keys())
        gray_color = (233 / 255., 241 / 255., 148 / 255.)
        for k in data_dict:
            data_val = data_dict[k]
            print(k, data_val.shape)

        pert_verts = data_dict['hand_verts']
        obj_faces = data_dict['obj_faces'] # [0]

    # verts = data['verts']

    # if pert_rhand_verts_fn is not None:
    #     pert_verts_sv_dict = np.load(pert_rhand_verts_fn, allow_pickle=True).item()
    #     pert_rhand_verts = pert_verts_sv_dict['pert_rhand_verts']
    #     pert_verts = pert_rhand_verts
    # # optimized_out_hand_verts_woopt = pert_verts

    print(f"obj_verts: {obj_verts.shape}")
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

    gray_color = (233 / 255., 241 / 255., 148 / 255.)

    ws = 400
    maxx_ws = ws
    skipp = 6
    skipp = 1
    iidx = 1
    # tot_hand_verts_woopt = []
    for i_fr in range(0, min(maxx_ws, optimized_out_hand_verts.shape[0]), skipp):

        cur_obj_verts = obj_verts[i_fr]
        cur_obj_faces = obj_faces

        print(f"cur_obj_verts: {cur_obj_verts.shape}, obj_faces: {obj_faces.shape}")

        sealed_v, seald_f, center_wopt = seal(optimized_out_hand_verts_woopt[i_fr], faces)
        sealed_v_0 = sealed_v[0:1]
        hand_mesh = ps.register_surface_mesh(f"cur_hand_mesh_wopt", sealed_v,
                                             seald_f,
                                             color=color[0 % len(color)])

        sealed_v, seald_f, center_woopt = seal(pert_verts[i_fr], faces)
        sealed_v = sealed_v - sealed_v[0:1] + sealed_v_0

        hand_mesh = ps.register_surface_mesh(f"cur_hand_ori_verts", sealed_v,
                                             seald_f,
                                             color=color[2 % len(color)])

        obj_mesh = ps.register_surface_mesh(f"cur_object", cur_obj_verts, cur_obj_faces,
                                            color=gray_color)

        iidx += 1
        if i_fr == 0:
            ps.set_up_dir("y_up")
            ps.set_up_dir("neg_z_up")
            ps.set_up_dir("z_up")

            look_at_pos = (0., -0.35, 0.45)
            look_at_dir = (0.0, 1.0, -0.7)
            # and also transform them for further
            ps.look_at(look_at_pos, look_at_dir)
            ps.show()

        ps.set_screenshot_extension(".jpg")
        ps.screenshot()
        ps.remove_all_structures()


if __name__ == '__main__':
    ### configs ###
    seed = 0
    t = 100
    seq_nm = "20231104_017"
    ntag = 4
    data_tag = ""
    ### configs ###

    ### get file paths ###
    pkl_fn = f"./data/taco/source_data/{seq_nm}.pkl"
    predicted_info_fn = f"./data/taco/result/predicted_infos_sv_dict_seed_{seed}_tag_{seq_nm}_jts_spatial_t_{t}_{data_tag}_st_0_multi_ntag_{ntag}.npy"
    optimized_fn = f"./data/taco/result/optimized_infos_sv_dict_seed_{seed}_tag_{seq_nm}_jts_spatial_t_{t}_{data_tag}_st_0_ntag_{ntag}.npy"
    ### get file paths ###


    vis_predicted_joints_taco(predicted_info_fn, optimized_fn, pkl_fn)