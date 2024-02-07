import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
        
    # 'st_idx': b['st_idx'],
        # 'ed_idx': b['ed_idx'],
    if 'st_idx' in notnone_batches[0]:
        st_idx_batch = [b['st_idx'] for b in notnone_batches]
        cond['y'].update({'st_idx': torch.tensor(st_idx_batch, dtype=torch.long)})
        
    if 'ed_idx' in notnone_batches[0]:
        st_idx_batch = [b['ed_idx'] for b in notnone_batches]
        cond['y'].update({'ed_idx': torch.tensor(st_idx_batch, dtype=torch.long)})
        
    # 'pert_verts': b['pert_verts'],
        # 'verts': b['verts'],
    if 'pert_verts' in notnone_batches[0]:
        pert_verts_batch = [b['pert_verts'] for b in notnone_batches]
        cond['y'].update({'pert_verts': torch.stack(pert_verts_batch, dim=0)})
    
    if 'verts' in notnone_batches[0]:
        verts_batch = [b['verts'] for b in notnone_batches]
        cond['y'].update({'verts': torch.stack(verts_batch, dim=0)})
        
    # 'avg_joints': b['avg_joints'],
    # 'std_joints': b['std_joints'],
    # ### avg_joints, std_joints ### #
    if 'avg_joints' in notnone_batches[0]:
        verts_batch = [b['avg_joints'] for b in notnone_batches]
        cond['y'].update({'avg_joints': torch.stack(verts_batch, dim=0)})
        
    if 'std_joints' in notnone_batches[0]: # 
        verts_batch = [b['std_joints'] for b in notnone_batches]
        cond['y'].update({'std_joints': torch.stack(verts_batch, dim=0)})
        
    # 'object_id': b['object_id'],
    # 'object_global_orient': b['object_global_orient'], 
    # 'object_transl': b['object_transl'],
    if 'object_global_orient' in notnone_batches[0]: # nnbsz x nnframes x 3 --> orientations #
        obj_global_orient_batch = [b['object_global_orient'] for b in notnone_batches]
        cond['y'].update({'object_global_orient': torch.stack(obj_global_orient_batch, dim=0)})
    
    if 'object_transl' in notnone_batches[0]: # nnbsz x nnframes x 3 --> orientations #
        obj_transl_batch = [b['object_transl'] for b in notnone_batches]
        cond['y'].update({'object_transl': torch.stack(obj_transl_batch, dim=0)})
        
    if 'object_id' in notnone_batches[0]: # nnbsz x nnframes x 3 --> orientations #
        # nn_bsz --> a one-D tensor here #
        object_id_batch = [b['object_id'] for b in notnone_batches]
        cond['y'].update({'object_id': torch.tensor(object_id_batch, dtype=torch.long)})
        
    # 'sampled_base_pts_nearest_obj_pc': b['sampled_base_pts_nearest_obj_pc'],
    #     'sampled_base_pts_nearest_obj_vns': b['sampled_base_pts_nearest_obj_vns'],
    
    
        
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True) # nf x nnjoints x 3 ->  #
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), 
        # 'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

# an adapter to our collate func
def motion_ours_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True) # nf x nnjoints x 3 ->  #
    adapted_batch = [{
        'inp': b[4].permute(1, 2, 0).contiguous(), 
        # 'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)




# An adapter for ours-single-seq collect function #
def motion_ours_singe_seq_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True) # nf x nnjoints x 3 ->  #
    # 'pert_verts': pert_rhand_verts, 
    #       'verts': rhand_verts,
    adapted_batch = [{
        'inp': b['motion'].permute(1, 2, 0).contiguous(), 
        # 'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b['caption'], #b[0]['caption']
        'tokens': b['tokens'],
        'lengths': b['m_length'],
        'st_idx': b['st_idx'],
        'ed_idx': b['ed_idx'],
        'pert_verts': b['pert_verts'],
        'verts': b['verts'],
        'avg_joints': b['avg_joints'],
        'std_joints': b['std_joints'],
        'object_id': b['object_id'],
        'object_global_orient': b['object_global_orient'], 
        'object_transl': b['object_transl'],
    } for b in batch]
    return collate(adapted_batch)




# # An adapter for ours-single-seq collect function #
# def motion_ours_singe_seq_collate_ambient_objbase(batch):
#     # batch.sort(key=lambda x: x[3], reverse=True) # nf x nnjoints x 3 ->  #
#     # 'pert_verts': pert_rhand_verts, 
#     #       'verts': rhand_verts,
#     adapted_batch = [{
#         'inp': b['motion'].permute(1, 2, 0).contiguous(), 
#         # 'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
#         'text': b['caption'], #b[0]['caption']
#         'tokens': b['tokens'],
#         'lengths': b['m_length'],
#         'st_idx': b['st_idx'],
#         'ed_idx': b['ed_idx'],
#         'pert_verts': b['pert_verts'],
#         'verts': b['verts'],
#         'avg_joints': b['avg_joints'],
#         'std_joints': b['std_joints'],
#         'object_id': b['object_id'],
#         'object_global_orient': b['object_global_orient'], 
#         'object_transl': b['object_transl'],
#         'sampled_base_pts_nearest_obj_pc': b['sampled_base_pts_nearest_obj_pc'],
#         'sampled_base_pts_nearest_obj_vns': b['sampled_base_pts_nearest_obj_vns'],
#     } for b in batch]
#     return collate(adapted_batch)




# An adapter for ours-single-seq collect function #
def motion_ours_obj_base_rel_dist_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True) # nf x nnjoints x 3 ->  #
    # 'pert_verts': pert_rhand_verts, 
    #       'verts': rhand_verts,
    
    ## motion ours, obj_base rel_dist, ... ##
    batch_dict = {}
    data_keys = batch[0].keys()

    for k in data_keys:
        if k == 'caption':
            batched_caption = [b[k] for b in batch]
            batch_dict.update({'text': batched_caption})
        elif k == 'lengths':
            batched_length = [b[k] for b in batch]
            max_length = max(batched_length)
            batched_length = torch.tensor(batched_length, dtype=torch.long)
            batch_dict.update({'lengths': batched_length})
            # maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
            masks = lengths_to_mask(batched_length, max_length)
            batch_dict.update({'mask': masks})
        elif k in ["object_id", 'st_idx', 'ed_idx']:
            batched_kdata = [b[k] for b in batch]
            batched_kdata = torch.tensor(batched_kdata, dtype=torch.long)
            batch_dict.update({k: batched_kdata})
        elif k in ["obj_verts", "obj_normals", "obj_faces"]:
            batched_kdata = [b[k] for b in batch]
            batch_dict.update({k: batched_kdata})
        else:
            batched_kdata = [b[k] for b in batch]
            batched_kdata = torch.stack(batched_kdata, dim=0)
            batch_dict.update({k: batched_kdata})
            
    return batch_dict
        
    
    # batch_dict = {
    #     k: [] 
    # }
    # for b in batch:
    #     for k in :
            
                
    
    # adapted_batch = [{
        
    #     ''
    #     'inp': b['motion'].permute(1, 2, 0).contiguous(), 
    #     # 'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
    #     'text': b['caption'], #b[0]['caption']
    #     'tokens': b['tokens'],
    #     'lengths': b['m_length'],
    #     'st_idx': b['st_idx'],
    #     'ed_idx': b['ed_idx'],
    #     'pert_verts': b['pert_verts'],
    #     'verts': b['verts'],
    #     'avg_joints': b['avg_joints'],
    #     'std_joints': b['std_joints'],
    #     'object_id': b['object_id'],
    #     'object_global_orient': b['object_global_orient'], 
    #     'object_transl': b['object_transl'],
    # } for b in batch]
    # return collate(adapted_batch)

