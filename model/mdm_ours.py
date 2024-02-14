import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
# from model.rotation2xyz import Rotation2xyz
import utils.model_utils as model_utils
from model.PointNet2 import PointnetPP
from model.DGCNN import PrimitiveNet

from utils.anchor_utils import masking_load_driver, anchor_load_driver ### load driver; masking load driver #

import os



class MDMV8(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        # mdm_ours #
        # self.args 
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # 
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        self.joints_feats_in_dim = 21 * 3
        
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        
        
        if self.diff_jts:
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        if self.diff_basejtsrel: ## basejtsrel ##
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = 21 * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ## Input process for joints ## ## joints_feats_in_dim -- 
            self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            
            ###### joints_feats_in_dim ######
            # self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            
            
            ## nf x ## diffbasejts rel ## 
            # inputs: bsz x nf x nnb x nn_b_in_feats # ## 
            
            if self.args.not_cond_base:
                self.rel_input_feats = 21 * ( 3)
            
            self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            
            ## and we can put one feature ahead of the transformer to learn information jointly ##

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### Encoding layer ###
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
                
            ###### joints_feats_in_dim ######
            self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            
            if self.args.use_dec_rel_v2:
                self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            else:
                # OutputProcessObjBaseRaw ## output process for basejtsrel #
                self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode 
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    # frre sample from the model? ##
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)
        
        # rt_dict = {
        #     "joint_seq_output": joint_seq_output,
        #     "basejtsrel_output": basejtsrel_output['dec_rel'],
        #     "joints_seq_latents": joints_seq_latents,
        #     "basejtsrel_seq_latents": basejtsrel_seq_latents,
            
        # }
        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat(
                [base_pts_disp_exp, base_normals_disp_exp, e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ##
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            ### calculate logvar, mean, and feats ###
            base_jts_e_feats_logvar = self.logvar_seqTransEncoder_e(base_jts_e_feats)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            base_jts_e_feats_var = torch.exp(base_jts_e_feats_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            base_jts_e_feats = self.reparameterization( base_jts_e_feats_mean, base_jts_e_feats_var)
            
            rt_dict['base_jts_e_feats'] = base_jts_e_feats
            rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
            
        if self.diff_basejtsrel:
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ### === Encode input joint sequences === ###
            # base_normals = x['base_normals'] # bsz x nnb x 3
            avg_joints_sequence = x['avg_joints_sequence'] # bsz x nnjoints x 3 ### -> for joint sequence #
            avg_joints_sequence_trans = avg_joints_sequence.unsqueeze(-1)
            avg_joints_feats = self.avg_joints_sequence_input_process(avg_joints_sequence_trans) ## 1 x bsz x dim ###
            
            # bsz x ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3 # bsz x ws x nnj x nnb x 3 #
            nnb = rel_base_pts_to_rhand_joints.size(3) # bsz x ws x nnj x nnb x 3
            
            # bsz x nf x nnj x nnb x 3 
            nnf, nnj = rel_base_pts_to_rhand_joints.size(1), rel_base_pts_to_rhand_joints.size(2)
            base_pts_exp_jts = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nnf, nnj, 1, 1).contiguous()
            base_normals_exp_jts = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nnf, nnj, 1, 1).contiguous()

            if self.args.not_cond_base:
                basejtsrel_enc_in_feats = rel_base_pts_to_rhand_joints
            else:
                basejtsrel_enc_in_feats = torch.cat(
                    [rel_base_pts_to_rhand_joints, base_pts_exp_jts, base_normals_exp_jts], dim=-1  #bsz x nf x nnj x nnb x nnfeats
                )
            basejtsrel_enc_in_feats = basejtsrel_enc_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            basejtsrel_enc_in_feats = basejtsrel_enc_in_feats.view(bsz, nframes, nnb, -1).contiguous()    
            # # transpose_rel_base_pts_to_ # bsz x ws x nnb x nnj x 3 --> bsz x ws x nnb x (nnj x 3) ### ---> rel positions
            # rel_base_pts_to_rhand_joints_exp = rel_base_pts_to_rhand_joints.permute(0, 1, 3, 2, 4).contiguous()
            # rel_base_pts_to_rhand_joints_exp = rel_base_pts_to_rhand_joints_exp.view(bsz, nframes, nnb, -1).contiguous()
            rel_base_pts_feats = self.input_process(basejtsrel_enc_in_feats)
            # sequence_pos_encoder
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(rel_base_pts_feats)
            # outputs rel base jts encoded latents ##
            # seqTransEncoder, logvar_seqTransEncoder
            # rel_base_pts_outputs_mean = self.basejtsrel_denoising_seqTransEncoder(rel_base_pts_feats_pos_embedding)
            # ### calculate logvar, mean, and feats ###
            # rel_base_pts_outputs_logvar = self.joint_sequence_logvar_seqTransEncoder(rel_base_pts_outputs)
            
            rel_base_pts_feats = torch.cat( # (seq_len + 1) x bsz x dim #
                [avg_joints_feats, rel_base_pts_feats_pos_embedding], dim=0
            )
            
            ## joints embedding for mean statistics and logvar statistics ##
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            
            if self.args.use_sigmoid and  (not self.args.kl_weights > 0.):
                rel_base_pts_outputs_mean = (torch.sigmoid(rel_base_pts_outputs_mean) - 0.5) * 2. # encode 
            
            ### calculate logvar, mean, and feats ###
            rel_base_pts_outputs_logvar = self.logvar_seqTransEncoder(rel_base_pts_feats)
            
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            rel_base_pts_outputs_var = torch.exp(rel_base_pts_outputs_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            rel_base_pts_outputs = self.reparameterization(rel_base_pts_outputs_mean, rel_base_pts_outputs_var)
            
            if self.args.single_frame_noise: # 
                rel_base_pts_outputs = rel_base_pts_outputs[0:1, ...] 
                rel_base_pts_outputs_mean = rel_base_pts_outputs_mean[0:1, ...] 
                rel_base_pts_outputs_logvar = rel_base_pts_outputs_logvar[0:1, ...] 
            
            if self.args.kl_weights > 0.:
                rt_dict['rel_base_pts_outputs'] = rel_base_pts_outputs
                rt_dict['rel_base_pts_outputs_mean'] = rel_base_pts_outputs_mean
                rt_dict['rel_base_pts_outputs_logvar'] = rel_base_pts_outputs_logvar
            else:
                rt_dict['rel_base_pts_outputs'] = rel_base_pts_outputs 
                rt_dict['rel_base_pts_outputs_mean'] = rel_base_pts_outputs_mean #
                rt_dict['rel_base_pts_outputs_logvar'] = rel_base_pts_outputs_logvar
                
            
        ## for construct rt_dict here ##
        # joint_seq_output = joint_seq_output # scale to [-1, 1]
        
        # rt_dict = {c
        #     'joint_seq_output': joint_seq_output,
        #     'rel_base_pts_outputs': rel_base_pts_outputs,
        #     'base_jts_e_feats': base_jts_e_feats,
        # }
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)






class MDMV9(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        # mdm_ours #
        # self.args 
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # 
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        self.joints_feats_in_dim = 21 * 3
        
        self.data_rep = "xyz"
        
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        
        
        if self.diff_jts:
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        if self.diff_basejtsrel: ## basejtsrel ##
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = 21 * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ## Input process for joints ## ## joints_feats_in_dim -- 
            self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            
            ###### joints_feats_in_dim ######
            # self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            
            
            ## nf x ## diffbasejts rel ## 
            # inputs: bsz x nf x nnb x nn_b_in_feats # ## 
            
            if self.args.not_cond_base:
                self.rel_input_feats = 21 * ( 3)
            
            self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            
            ## and we can put one feature ahead of the transformer to learn information jointly ##

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### Encoding layer ###
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
                
            ###### joints_feats_in_dim ######
            self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            
            if self.args.use_dec_rel_v2:
                self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            else:
                # OutputProcessObjBaseRaw ## output process for basejtsrel #
                self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode 
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    # frre sample from the model? ##
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        # base_normals # ## 
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat(
                [base_pts_disp_exp, base_normals_disp_exp, e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ##
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            ### calculate logvar, mean, and feats ###
            base_jts_e_feats_logvar = self.logvar_seqTransEncoder_e(base_jts_e_feats)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            base_jts_e_feats_var = torch.exp(base_jts_e_feats_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            base_jts_e_feats = self.reparameterization( base_jts_e_feats_mean, base_jts_e_feats_var)
            
            rt_dict['base_jts_e_feats'] = base_jts_e_feats
            rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
            
        if self.diff_basejtsrel:
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ### === Encode input joint sequences === ###
            # base_normals = x['base_normals'] # bsz x nnb x 3
            # avg_joints_sequence = x['avg_joints_sequence'] # bsz x nnjoints x 3 ### -> for joint sequence #
            avg_joints_sequence = x['pert_avg_joints_sequence']
            avg_joints_sequence_trans = avg_joints_sequence.unsqueeze(-1)
            avg_joints_feats = self.avg_joints_sequence_input_process(avg_joints_sequence_trans) ## 1 x bsz x dim ###
            
            # bsz x ws x nnj x nnb x 3 #
            rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3 # bsz x ws x nnj x nnb x 3 #
            nnb = rel_base_pts_to_rhand_joints.size(3) # bsz x ws x nnj x nnb x 3
            
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_rhand_joints']
            
            # bsz x nf x nnj x nnb x 3 
            nnf, nnj = rel_base_pts_to_rhand_joints.size(1), rel_base_pts_to_rhand_joints.size(2)
            base_pts_exp_jts = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nnf, nnj, 1, 1).contiguous()
            base_normals_exp_jts = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nnf, nnj, 1, 1).contiguous()

            if self.args.not_cond_base:
                basejtsrel_enc_in_feats = pert_rel_base_pts_to_rhand_joints
            else:
                basejtsrel_enc_in_feats = torch.cat(
                    [pert_rel_base_pts_to_rhand_joints, base_pts_exp_jts, base_normals_exp_jts], dim=-1  #bsz x nf x nnj x nnb x nnfeats
                )
            basejtsrel_enc_in_feats = basejtsrel_enc_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            basejtsrel_enc_in_feats = basejtsrel_enc_in_feats.view(bsz, nframes, nnb, -1).contiguous()    
            # # transpose_rel_base_pts_to_ # bsz x ws x nnb x nnj x 3 --> bsz x ws x nnb x (nnj x 3) ### ---> rel positions
            # rel_base_pts_to_rhand_joints_exp = rel_base_pts_to_rhand_joints.permute(0, 1, 3, 2, 4).contiguous()
            # rel_base_pts_to_rhand_joints_exp = rel_base_pts_to_rhand_joints_exp.view(bsz, nframes, nnb, -1).contiguous()
            rel_base_pts_feats = self.input_process(basejtsrel_enc_in_feats)
            # sequence_pos_encoder
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(rel_base_pts_feats)
            # outputs rel base jts encoded latents ##
            # seqTransEncoder, logvar_seqTransEncoder
            # rel_base_pts_outputs_mean = self.basejtsrel_denoising_seqTransEncoder(rel_base_pts_feats_pos_embedding)
            # ### calculate logvar, mean, and feats ###
            # rel_base_pts_outputs_logvar = self.joint_sequence_logvar_seqTransEncoder(rel_base_pts_outputs)
            
            rel_base_pts_feats = torch.cat( # (seq_len + 1) x bsz x dim #
                [avg_joints_feats, rel_base_pts_feats_pos_embedding], dim=0 ## jrel_base_pts_pos_embedding #
            )
            
            ## joints embedding for mean statistics and logvar statistics ##
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            
            
            if not self.args.without_dec_pos_emb: # without dec pos embedding
                avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs_mean[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs_mean = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            # if self.args.train_enc: # trian enc for seq latents ###
            #     basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
            #     if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
            #         basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
            #     basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            # elif self.args.pred_diff_noise:
            #     basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            # else:
            #     basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            
            
            rt_dict['basejtsrel_output'] = basejtsrel_output['dec_rel']
            rt_dict['avg_jts_outputs'] = avg_jts_outputs
            
            
            # # if self.args.use_sigmoid and  (not self.args.kl_weights > 0.):
            # #     rel_base_pts_outputs_mean = (torch.sigmoid(rel_base_pts_outputs_mean) - 0.5) * 2. # encode 
            
            # # ### calculate logvar, mean, and feats ###
            # # rel_base_pts_outputs_logvar = self.logvar_seqTransEncoder(rel_base_pts_feats)
            
            # # # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            # # rel_base_pts_outputs_var = torch.exp(rel_base_pts_outputs_logvar) # seq x bs x d --> encodeing and decoding
            # # ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            # # rel_base_pts_outputs = self.reparameterization(rel_base_pts_outputs_mean, rel_base_pts_outputs_var)
            
            # # if self.args.single_frame_noise: # 
            # #     rel_base_pts_outputs = rel_base_pts_outputs[0:1, ...] 
            # #     rel_base_pts_outputs_mean = rel_base_pts_outputs_mean[0:1, ...] 
            # #     rel_base_pts_outputs_logvar = rel_base_pts_outputs_logvar[0:1, ...] 
            
            # if self.args.kl_weights > 0.:
            #     rt_dict['rel_base_pts_outputs'] = rel_base_pts_outputs
            #     rt_dict['rel_base_pts_outputs_mean'] = rel_base_pts_outputs_mean
            #     rt_dict['rel_base_pts_outputs_logvar'] = rel_base_pts_outputs_logvar
            # else:
            #     rt_dict['rel_base_pts_outputs'] = rel_base_pts_outputs 
            #     rt_dict['rel_base_pts_outputs_mean'] = rel_base_pts_outputs_mean #
            #     rt_dict['rel_base_pts_outputs_logvar'] = rel_base_pts_outputs_logvar
                
            
        ## for construct rt_dict here ##
        # joint_seq_output = joint_seq_output # scale to [-1, 1]
        
        # rt_dict = {c
        #     'joint_seq_output': joint_seq_output,
        #     'rel_base_pts_outputs': rel_base_pts_outputs,
        #     'base_jts_e_feats': base_jts_e_feats,
        # }
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)



### MDM 10 ###
class MDMV10(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        self.diff_realbasejtsrel = self.args.diff_realbasejtsrel
        self.diff_realbasejtsrel_to_joints = self.args.diff_realbasejtsrel_to_joints
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        self.use_anchors = self.args.use_anchors
        
        if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
            # anchor_load_driver, masking_load_driver #
            inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
            fvi, aw, _, _ = anchor_load_driver(inpath)
            self.face_vertex_index = torch.from_numpy(fvi).long()
            self.anchor_weight = torch.from_numpy(aw).float()
            
            anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
            palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
            hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
            # self.hand_palm_vertex_mask for hand palm mask #
            self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
            self.nn_anchors = int(self.hand_palm_vertex_mask.float().sum()) #### number of anchors here ###
        
        
        # self.joints_feats_in_dim = 21 * 3
        # joints feats in dim #
        
        self.nn_keypoints = 21
        if self.args.use_anchors:
            # self.nn_keypoints = self.nn_anchors # nn_anchors #
            self.nn_keypoints = 32 # nn_anchors #
        
        self.joints_feats_in_dim = self.nn_keypoints * 3
        self.data_rep = "xyz"
        
        
        if self.diff_jts:
            
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        
        # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
        if self.diff_realbasejtsrel_to_joints: # feature for each joint point? --> for the denoising purpose #
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            layernorm = True
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            if self.args.use_abs_jts_for_encoding_obj_base:
                self.rel_input_feats = 21 * (3)
                # layernorm = False
                self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                # self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            # elif self.args.use
            else:        
                if self.args.use_objbase_v2:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                elif self.args.use_objbase_v3:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV3(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                else:
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            if self.args.use_abs_jts_for_encoding: # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
                self.real_basejtsrel_to_joints_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            self.real_basejtsrel_to_joints_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ### # InputProcessObjBaseV2
            real_basejtsrel_to_joints_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout # 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_to_joints_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_to_joints_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_to_joints_sequence_pos_encoder)
            
            self.real_basejtsrel_to_joints_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_to_joints_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            self.real_basejtsrel_to_joints_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
        
        
        if self.diff_realbasejtsrel:
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            # self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            # self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim)
            
            self.rel_input_feats = self.nn_keypoints *  (3 + 3 + 3) 
            
            layernorm = True
            if self.args.use_objbase_v2:
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, glb_feats_trans=True)
            elif self.args.use_objbase_v4: # use_objbase_out_v4
                self.rel_input_feats = (self.args.nn_base_pts * (3 + 3 + 3)) # current joint positions # how to keep the dimension
                self.real_basejtsrel_input_process = InputProcessObjBaseV4(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v5: # use_objbase_v5, 
                if self.args.v5_in_not_base:
                    self.rel_input_feats = (self.nn_keypoints * 3) 
                elif self.args.v5_in_not_base_pos:
                    self.rel_input_feats = 3 + (self.nn_keypoints * 3) 
                else:
                    self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV5(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, without_glb=self.args.v5_in_without_glb)
            elif self.args.use_objbase_v6: # real_basejtsrel_input_process
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3) + 3
                self.real_basejtsrel_input_process = InputProcessObjBaseV6(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v7:
                # InputProcessObjBaseV7
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV7(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            else:
                self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            
            self.real_basejtsrel_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ###
            real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout #  # dropout # # dropout 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_sequence_pos_encoder)
            
            self.real_basejtsrel_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
            print(f"not_cond_base: {self.args.not_cond_base}, latent_dim: {self.latent_dim}")
            
            
            if self.args.use_jts_pert_realbasejtsrel:
                print(f"use_jts_pert_realbasejtsrel!!!!!!")
                self.real_basejtsrel_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3)
            else:
                if self.args.use_objbase_out_v3:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV3(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v4:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV4(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v5: # use_objbase_v5, use_objbase_out_v5
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV5(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base, out_objbase_v5_bundle_out=self.args.out_objbase_v5_bundle_out, v5_out_not_cond_base=self.args.v5_out_not_cond_base, nn_keypoints=self.nn_keypoints)
                else:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # OutputProcessCond
        
        if self.diff_basejtsrel:
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = self.nn_keypoints * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            if self.args.with_glb_info:
                # InputProcessWithGlbInfo
                self.joints_offset_input_process = InputProcessWithGlbInfo(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            else:
                self.joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)

     
            if self.args.not_cond_base:
                self.rel_input_feats = self.nn_keypoints * ( 3)
            # self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### Encoding layer ###
            # logvar_seqTransEncoder_e, logvar_seqTransEncoder # logvar_seqTranEncoder
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise: # add to attention network # 
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            ###### joints_feats_in_dim ###### # a linear transformation net with weights and bias set to zero #
            self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3) # output avgjts sequence 
            # OutputProcessCond
            self.joint_offset_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3)
            
            if self.args.use_dec_rel_v2:
                self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            else:
                # OutputProcessObjBaseRaw ## output process for basejtsrel #
                self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = self.nn_keypoints * (3 + 3 + 1 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            
    def set_bn_to_eval(self):
        if self.args.use_objbase_v6: # real_basejtsrel_input_process
            try:
                self.real_basejtsrel_input_process.pnpp_conv_net.set_bn_no_training()
            except:
                pass

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode # ours float
            clip_model)  # Actually this line is unnecessary since clip by default already on float16 ### ours 

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled
    
    def decode_realbasejtsrel_from_objbasefeats(self, objbasefeats, input_data):
        real_dec_basejtsrel = self.real_basejtsrel_output_process(
                objbasefeats, input_data
            )
        # real_dec_basejtsrel -> decoded realtive positions #
        real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous()
        return real_dec_basejtsrel
    
    def denoising_realbasejtsrel_objbasefeats(self, pert_obj_base_pts_feats, timesteps):
        if self.args.deep_fuse_timeemb:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(pert_obj_base_pts_feats.size(0), 1, 1).contiguous()
            real_basejtsrel_seq_latents = pert_obj_base_pts_feats + real_basejtsrel_time_emb
            
            if self.args.use_ours_transformer_enc:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
            else: # seq des
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
        else:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_seq_latents = torch.cat(
                [real_basejtsrel_time_emb, pert_obj_base_pts_feats], dim=0
            )
            
            if self.args.use_ours_transformer_enc: ## mdm ours ##
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
            else:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                
        # bsz, nframes, nnb, nnj, 3 --> 
        # 
        # real_dec_basejtsrel = self.real_basejtsrel_output_process(
        #     real_basejtsrel_seq_latents, x
        # )
        return real_basejtsrel_seq_latents

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        # base_normals # ## 
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            e_disp_rel_to_base_along_normals = x['pert_e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['pert_e_disp_rel_to_base_vt_normals']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat( # along normals; # vt normals #
                [base_pts_disp_exp, base_normals_disp_exp, e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ## # 
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            #### base_jts_e_feats, base_jts_e_feats_mean ####
            # ## us basejtsefeats for denoising directly ##
            base_jts_e_feats = base_jts_e_feats_mean
            if not self.args.without_dec_pos_emb: ## use positional encoding ##
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)

            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            # ### sequence latents ###
            # if self.args.train_enc: # trian enc for seq latents ###
            #     base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            ##### output_process_e -> output energies #####
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
            # rt_dict['base_jts_e_feats'] = base_jts_e_feats
            # rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            # rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        else:
            diff_basejtse_dict = {}
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
        
        if self.args.diff_realbasejtsrel_to_joints:  # nframes x nnbase x nnjts x (base pts + base normals + 3) 2) point feature for each point; point feature for; condition on the noisy input for the denoised information
            # real_basejtsrel_to_joints_input_process, real_basejtsrel_to_joints_sequence_pos_encoder, real_basejtsrel_to_joints_seqTransEncoder
            # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_joints_for_jts_pred'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_joints_for_jts_pred'] # bsz x nf x nnj x nnb x 3 
            
            
            ## use_abs_jts_pos --> obj jts pos for the encodingj
            if self.args.use_abs_jts_pos: ## bsz x nf x nnj x nnb x 3 ## ---> abs jts pos ##
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1)
                
            # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
            if self.args.use_abs_jts_for_encoding:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                abs_jts = pert_rel_base_pts_to_rhand_joints[..., 0, :]
                abs_jts = abs_jts.permute(0, 2, 3, 1).contiguous()
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(abs_jts)
            elif  self.args.use_abs_jts_for_encoding_obj_base:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints[:, :, :, 0:1, :]
                # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], pert_rel_base_pts_to_rhand_joints], dim=-1
                # )
                obj_base_in_feats = pert_rel_base_pts_to_rhand_joints
                # --> tnrasform the input feature dim to 21 * 3 here for encoding # 
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, 1, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
            else:
                if self.args.use_objbase_v2:
                    # bsz x nf x nnj x nnb x 3
                    pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                        [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                    )
                    # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                    obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
                
            # ### real_basejtsrel_to_joints_input_process --> real_basejtsrel_to_joints_input_process --> for the joints and input process ### # obj_base_encoded_feats
            # obj_base_encoded_feats 
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_to_joints_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_to_joints_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()
                real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                else: # seq des
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_seq_latents = torch.cat(
                    [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            joints_offset_output =  self.real_basejtsrel_to_joints_output_process(real_basejtsrel_seq_latents)
                
            joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            if self.args.diff_basejtsrel:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output_from_rel': joints_offset_output
                }
            else:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output': joints_offset_output
                }
                
        else:
            diff_basejtsrel_to_joints_dict = {}
            
            
        if self.diff_realbasejtsrel: # real_dec_basejtsrel
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_rhand_joints'] # bsz x nf x nnj x nnb x 3 
            
            if self.args.use_objbase_v2:
                # bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            elif self.args.use_objbase_v4:
                # use_objbase_v4: # use_objbase_out_v4
                exp_normed_base_pts = normed_base_pts.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                exp_base_normals = base_normals.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                obj_base_in_feats = torch.cat(
                    [pert_rel_base_pts_to_rhand_joints, exp_normed_base_pts, exp_base_normals], dim=-1 # bsz x nf x nnj x nnb x (3 + 3 + 3) # -> exp_base_normals
                )
                obj_base_in_feats = obj_base_in_feats.view(bsz, nf, nnj, -1).contiguous()
            elif self.args.use_objbase_v5: # use_objbase_v5, use_objbase_out_v5
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                if self.args.v5_in_not_base:
                    obj_base_in_feats = torch.cat(
                        [ pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                elif self.args.v5_in_not_base_pos:
                    obj_base_in_feats = torch.cat(
                        [base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
            elif self.args.use_objbase_v6 or self.args.use_objbase_v7:
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            else:
                obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                    [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                )
                # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            
            # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
            #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
            # )
            # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
            # obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            if self.args.use_objbase_v6:
                normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1) # and repeat for the base pts #
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats, normed_base_pts_exp) 
            else:
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats) # nf x bsz x feat_dim # nf x bsz x nnbasepts x feats_dim #
            # obj_base_encoded_feats 
            
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            
            if self.args.train_enc:
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                real_dec_basejtsrel = self.real_basejtsrel_output_process(
                    obj_base_pts_feats, x
                )
                real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                if self.args.use_objbase_out_v3:
                    real_dec_basejtsrel = real_dec_basejtsrel
                else:
                    real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            else:
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    # print(f"timesteps: {timesteps.size()}, obj_base_pts_feats: {obj_base_pts_feats.size()}")
                    
                    if self.args.use_objbase_v5:
                        cur_timesteps = timesteps.unsqueeze(1).repeat(1, nnb).view(-1)
                    else:
                        cur_timesteps = timesteps
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(cur_timesteps)
                    real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()  
                    real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                    
                    if self.args.use_ours_transformer_enc:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                    else: # seq des
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
                    real_basejtsrel_seq_latents = torch.cat(
                        [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                if self.args.use_jts_pert_realbasejtsrel:
                    joints_offset_output =  self.real_basejtsrel_output_process(real_basejtsrel_seq_latents)
                    joints_offset_output = joints_offset_output.permute(0, 3, 1, 2) # bsz x nf x nnj x 3
                    real_dec_basejtsrel = joints_offset_output.unsqueeze(-2).repeat(1, 1, 1, nnb, 1)
                    # real_dec_basejtsrel = joints_offset_output
                else:
                    real_dec_basejtsrel = self.real_basejtsrel_output_process(
                        real_basejtsrel_seq_latents, x
                    )
                    # real_dec_basejtsrel -> decoded realtive positions #
                    real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                    if self.args.use_objbase_out_v3 or self.args.use_objbase_out_v4 or self.args.use_objbase_out_v5:
                        real_dec_basejtsrel = real_dec_basejtsrel
                    else:
                        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            diff_realbasejtsrel_out_dict = {
                'real_dec_basejtsrel': real_dec_basejtsrel,
                'obj_base_pts_feats': obj_base_pts_feats,
            }
        else:
            diff_realbasejtsrel_out_dict = {}
                        
            
        # relative joints encoder; obj pos encoder; obj pos encoder; # penetrations, depth, --- how to use depth for guidance -> and also penetrations # object penetrations #
        if self.diff_basejtsrel:
            # joints_offset_sequence --> x['pert_joints_offset_sequence']
            joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
            joints_offset_sequence = joints_offset_sequence.permute(0, 2, 3, 1).contiguous()
            joints_offset_feats = self.joints_offset_input_process(joints_offset_sequence) # nf x bsz x dim
            
            # rel_base_pts_feats = self.input_process(basejtsrel_enc_in_feats)
            # sequence_pos_encoder
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(joints_offset_feats)
            
            # print(f"joints_offset_feats: {joints_offset_feats.size()}, rel_base_pts_feats_pos_embedding: {rel_base_pts_feats_pos_embedding.size()}")
            # outputs rel base jts encoded latents ##
            # seqTransEncoder, logvar_seqTransEncoder
            # rel_base_pts_outputs_mean = self.basejtsrel_denoising_seqTransEncoder(rel_base_pts_feats_pos_embedding)
            # ### calculate logvar, mean, and feats ###
            # rel_base_pts_outputs_logvar = self.joint_sequence_logvar_seqTransEncoder(rel_base_pts_outputs)
            
            if self.args.not_diff_avgjts: # not use diff avgjts ##
                rel_base_pts_feats = rel_base_pts_feats_pos_embedding
                # seqTransEncoder, logvar_seqTransEncoder #
                rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
                # print(f"rel_base_pts_outputs_mean 1: {rel_base_pts_outputs_mean.size()}")
                if not self.args.without_dec_pos_emb: # without dec pos embedding
                    # avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                    other_rel_base_pts_outputs = rel_base_pts_outputs_mean # 
                    other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                    rel_base_pts_outputs_mean = other_rel_base_pts_outputs
                    
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                    basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                    # print(f"basejtsrel_seq_latents: {basejtsrel_seq_latents.size()}, rel_base_pts_outputs_mean: {rel_base_pts_outputs_mean.size()}, basejtsrel_time_emb: {basejtsrel_time_emb.size()}")
                    if self.args.use_ours_transformer_enc:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_seq_latents = torch.cat(
                        [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_seq_latents_pred_feats
                # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
                other_basejtsrel_seq_latents = basejtsrel_seq_latents # [1:, ...]
                
                joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
                
                joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
                
                # print(f"joints_offset_output in MDM: {joints_offset_output.size()}, joints_offset_sequence: {joints_offset_sequence.size()}, other_basejtsrel_seq_latents: {other_basejtsrel_seq_latents.size()}")
                
                diff_basejtsrel_dict = {
                    'joints_offset_output': joints_offset_output
                }
                
                # rt_dict['joints_offset_output'] = joints_offset_output
            else:
                avg_joints_sequence = x['pert_avg_joints_sequence']
                avg_joints_sequence_trans = avg_joints_sequence.unsqueeze(-1)
                avg_joints_feats = self.avg_joints_sequence_input_process(avg_joints_sequence_trans) ## 1 x bsz x dim ###
            
                rel_base_pts_feats = torch.cat( # (seq_len + 1) x bsz x dim #
                    [avg_joints_feats, rel_base_pts_feats_pos_embedding], dim=0 ## jrel_base_pts_pos_embedding #
                )
                
                ## joints embedding for mean statistics and logvar statistics ##
                # seqTransEncoder, logvar_seqTransEncoder #
                rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
                
                
                if not self.args.without_dec_pos_emb: # without dec pos embedding
                    avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                    other_rel_base_pts_outputs = rel_base_pts_outputs_mean[1: , ...]
                    other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                    rel_base_pts_outputs_mean = torch.cat(
                        [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                    )
                
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                    basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                    
                    if self.args.use_ours_transformer_enc:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_seq_latents = torch.cat(
                        [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                # basejtsrel_seq_latents_pred_feats
                avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
                other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
                
                avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
                avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
                # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
                joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
                
                joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
                
                rt_dict['joints_offset_output'] = joints_offset_output
                rt_dict['avg_jts_outputs'] = avg_jts_outputs
            
        else:
            diff_basejtsrel_dict = {}    
            
            
        rt_dict = {}
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict) ### rt_dict and diff_basejtse
        rt_dict.update(diff_basejtsrel_to_joints_dict)
        rt_dict.update(diff_realbasejtsrel_out_dict) ### diff 
        
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)



### MDM 10 ###
class MDMV14(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        self.diff_realbasejtsrel = self.args.diff_realbasejtsrel
        self.diff_realbasejtsrel_to_joints = self.args.diff_realbasejtsrel_to_joints
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        self.use_anchors = self.args.use_anchors
        
        if self.use_anchors: # use anchors # anchor_load_driver, masking_load_driver #
            # anchor_load_driver, masking_load_driver #
            inpath = "/home/xueyi/sim/CPF/assets" # contact potential field; assets # ##
            fvi, aw, _, _ = anchor_load_driver(inpath)
            self.face_vertex_index = torch.from_numpy(fvi).long()
            self.anchor_weight = torch.from_numpy(aw).float()
            
            anchor_path = os.path.join("/home/xueyi/sim/CPF/assets", "anchor")
            palm_path = os.path.join("/home/xueyi/sim/CPF/assets", "hand_palm_full.txt")
            hand_region_assignment, hand_palm_vertex_mask = masking_load_driver(anchor_path, palm_path)
            # self.hand_palm_vertex_mask for hand palm mask #
            self.hand_palm_vertex_mask = torch.from_numpy(hand_palm_vertex_mask).bool() ## the mask for hand palm to get hand anchors #
            self.nn_anchors = int(self.hand_palm_vertex_mask.float().sum()) #### number of anchors here ###
        
        
        # self.joints_feats_in_dim = 21 * 3
        # joints feats in dim #
        
        self.nn_keypoints = 21
        if self.args.use_anchors:
            # self.nn_keypoints = self.nn_anchors # nn_anchors #
            self.nn_keypoints = 32 # nn_anchors #
        
        self.joints_feats_in_dim = self.nn_keypoints * 3
        self.data_rep = "xyz"
        
        
        if self.diff_jts:
            
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        
        # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
        if self.diff_realbasejtsrel_to_joints: # feature for each joint point? --> for the denoising purpose #
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            layernorm = True
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            if self.args.use_abs_jts_for_encoding_obj_base:
                self.rel_input_feats = 21 * (3)
                # layernorm = False
                self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                # self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            # elif self.args.use
            else:        
                if self.args.use_objbase_v2:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                elif self.args.use_objbase_v3:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV3(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                else:
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            if self.args.use_abs_jts_for_encoding: # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
                self.real_basejtsrel_to_joints_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            self.real_basejtsrel_to_joints_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ### # InputProcessObjBaseV2
            real_basejtsrel_to_joints_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout # 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_to_joints_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_to_joints_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_to_joints_sequence_pos_encoder)
            
            self.real_basejtsrel_to_joints_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_to_joints_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            self.real_basejtsrel_to_joints_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
        
        
        ## diff real base jts r el ##
        if self.diff_realbasejtsrel: # realtive base pts to joints #
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            # self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            # self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim)
            
            self.rel_input_feats = self.nn_keypoints *  (3 + 3 + 3) 
            
            layernorm = True
            if self.args.use_objbase_v2:
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, glb_feats_trans=True)
            elif self.args.use_objbase_v4: # use_objbase_out_v4
                self.rel_input_feats = (self.args.nn_base_pts * (3 + 3 + 3)) # current joint positions # how to keep the dimension
                self.real_basejtsrel_input_process = InputProcessObjBaseV4(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v5: # use_objbase_v5, 
                if self.args.v5_in_not_base:
                    self.rel_input_feats = (self.nn_keypoints * 3) 
                elif self.args.v5_in_not_base_pos:
                    self.rel_input_feats = 3 + (self.nn_keypoints * 3) 
                else:
                    self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV5(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, without_glb=self.args.v5_in_without_glb)
            elif self.args.use_objbase_v6: # real_basejtsrel_input_process
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3) + 3
                self.real_basejtsrel_input_process = InputProcessObjBaseV6(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v7:
                # InputProcessObjBaseV7
                self.rel_input_feats = 3 + 3 + (self.nn_keypoints * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV7(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            else:
                self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            
            self.real_basejtsrel_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ###
            real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout #  # dropout # # dropout 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_sequence_pos_encoder)
            
            self.real_basejtsrel_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
            print(f"not_cond_base: {self.args.not_cond_base}, latent_dim: {self.latent_dim}")
            
            
            if self.args.use_jts_pert_realbasejtsrel:
                print(f"use_jts_pert_realbasejtsrel!!!!!!")
                self.real_basejtsrel_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3)
            else:
                if self.args.use_objbase_out_v3:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV3(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v4:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV4(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v5: # use_objbase_v5, use_objbase_out_v5
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV5(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base, out_objbase_v5_bundle_out=self.args.out_objbase_v5_bundle_out, v5_out_not_cond_base=self.args.v5_out_not_cond_base, nn_keypoints=self.nn_keypoints)
                else:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # OutputProcessCond
        
        if self.diff_basejtsrel:
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = self.nn_keypoints * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            self.rel_input_feats = 3 + 3 + 24
            
            # InputProcessParams(data_rep, input_feats, latent_dim) # 
            # latent_dim, data_rep, rel_input_feats # 
            # # params_input_process --> input: ws x bsz x input_feats; output: ws x bsz x latent_dim #
            self.params_input_process = InputProcessParams(self.data_rep, self.rel_input_feats, self.latent_dim)
            
            # # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            # self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            # if self.args.with_glb_info:
            #     # InputProcessWithGlbInfo
            #     self.joints_offset_input_process = InputProcessWithGlbInfo(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # else:
            #     self.joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)

     
            # if self.args.not_cond_base: # not cond 
            #     self.rel_input_feats = self.nn_keypoints * ( 3)
            # # self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            # ### Encoding layer ###
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder # logvar_seqTranEncoder
            # seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                 nhead=self.num_heads,
            #                                                 dim_feedforward=self.ff_size,
            #                                                 dropout=self.dropout,
            #                                                 activation=self.activation)

            # self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
            #                                             num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            # if self.args.const_noise: # add to attention network # 
            #     # 1) max pool latents over the sequence
            #     # 2) transform the pooled latnets via the linear layer
            #     self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
            #         nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
            #         nn.Linear(self.latent_dim * 2, self.latent_dim)
            #     )
                
            # OutputProcessParams(data_rep, input_feats, latent_dim, njoints, nfeats)
            self.params_output_process = OutputProcessParams(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3)
            
            # ###### joints_feats_in_dim ###### # a linear transformation net with weights and bias set to zero #
            # # self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3) # output avgjts sequence 
            # # # OutputProcessCond
            # self.joint_offset_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, self.nn_keypoints, 3)
            
            # if self.args.use_dec_rel_v2:
            #     self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # else:
            #     # OutputProcessObjBaseRaw ## output process for basejtsrel #
            #     self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            #     ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = self.nn_keypoints * (3 + 3 + 1 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            
    def set_bn_to_eval(self):
        if self.args.use_objbase_v6: # real_basejtsrel_input_process
            try:
                self.real_basejtsrel_input_process.pnpp_conv_net.set_bn_no_training()
            except:
                pass

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode # ours float
            clip_model)  # Actually this line is unnecessary since clip by default already on float16 ### ours 

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled
    
    def decode_realbasejtsrel_from_objbasefeats(self, objbasefeats, input_data):
        real_dec_basejtsrel = self.real_basejtsrel_output_process(
                objbasefeats, input_data
            )
        # real_dec_basejtsrel -> decoded realtive positions #
        real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous()
        return real_dec_basejtsrel
    
    def denoising_realbasejtsrel_objbasefeats(self, pert_obj_base_pts_feats, timesteps):
        if self.args.deep_fuse_timeemb:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(pert_obj_base_pts_feats.size(0), 1, 1).contiguous()
            real_basejtsrel_seq_latents = pert_obj_base_pts_feats + real_basejtsrel_time_emb
            
            if self.args.use_ours_transformer_enc:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
            else: # seq des
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
        else:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_seq_latents = torch.cat(
                [real_basejtsrel_time_emb, pert_obj_base_pts_feats], dim=0
            )
            
            if self.args.use_ours_transformer_enc: ## mdm ours ##
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
            else:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                
        # bsz, nframes, nnb, nnj, 3 --> 
        # 
        # real_dec_basejtsrel = self.real_basejtsrel_output_process(
        #     real_basejtsrel_seq_latents, x
        # )
        return real_basejtsrel_seq_latents

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        # base_normals # ## 
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            e_disp_rel_to_base_along_normals = x['pert_e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['pert_e_disp_rel_to_base_vt_normals']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat( # along normals; # vt normals #
                [base_pts_disp_exp, base_normals_disp_exp, e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ## # 
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            #### base_jts_e_feats, base_jts_e_feats_mean ####
            # ## us basejtsefeats for denoising directly ##
            base_jts_e_feats = base_jts_e_feats_mean
            if not self.args.without_dec_pos_emb: ## use positional encoding ##
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)

            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            # ### sequence latents ###
            # if self.args.train_enc: # trian enc for seq latents ###
            #     base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            ##### output_process_e -> output energies #####
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
            # rt_dict['base_jts_e_feats'] = base_jts_e_feats
            # rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            # rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        else:
            diff_basejtse_dict = {}
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
        
        if self.args.diff_realbasejtsrel_to_joints:  # nframes x nnbase x nnjts x (base pts + base normals + 3) 2) point feature for each point; point feature for; condition on the noisy input for the denoised information
            # real_basejtsrel_to_joints_input_process, real_basejtsrel_to_joints_sequence_pos_encoder, real_basejtsrel_to_joints_seqTransEncoder
            # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_joints_for_jts_pred'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_joints_for_jts_pred'] # bsz x nf x nnj x nnb x 3 
            
            
            ## use_abs_jts_pos --> obj jts pos for the encodingj
            if self.args.use_abs_jts_pos: ## bsz x nf x nnj x nnb x 3 ## ---> abs jts pos ##
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1)
                
            # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
            if self.args.use_abs_jts_for_encoding:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                abs_jts = pert_rel_base_pts_to_rhand_joints[..., 0, :]
                abs_jts = abs_jts.permute(0, 2, 3, 1).contiguous()
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(abs_jts)
            elif  self.args.use_abs_jts_for_encoding_obj_base:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints[:, :, :, 0:1, :]
                # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], pert_rel_base_pts_to_rhand_joints], dim=-1
                # )
                obj_base_in_feats = pert_rel_base_pts_to_rhand_joints
                # --> tnrasform the input feature dim to 21 * 3 here for encoding # 
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, 1, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
            else:
                if self.args.use_objbase_v2:
                    # bsz x nf x nnj x nnb x 3
                    pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                        [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                    )
                    # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                    obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
                
            # ### real_basejtsrel_to_joints_input_process --> real_basejtsrel_to_joints_input_process --> for the joints and input process ### # obj_base_encoded_feats
            # obj_base_encoded_feats 
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_to_joints_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_to_joints_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()
                real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                else: # seq des
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_seq_latents = torch.cat(
                    [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            joints_offset_output =  self.real_basejtsrel_to_joints_output_process(real_basejtsrel_seq_latents)
                
            joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            if self.args.diff_basejtsrel:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output_from_rel': joints_offset_output
                }
            else:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output': joints_offset_output
                }
                
        else:
            diff_basejtsrel_to_joints_dict = {}
            
            
        if self.diff_realbasejtsrel: # real_dec_basejtsrel
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_rhand_joints'] # bsz x nf x nnj x nnb x 3 
            
            if self.args.use_objbase_v2:
                # bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            elif self.args.use_objbase_v4:
                # use_objbase_v4: # use_objbase_out_v4
                exp_normed_base_pts = normed_base_pts.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                exp_base_normals = base_normals.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                obj_base_in_feats = torch.cat(
                    [pert_rel_base_pts_to_rhand_joints, exp_normed_base_pts, exp_base_normals], dim=-1 # bsz x nf x nnj x nnb x (3 + 3 + 3) # -> exp_base_normals
                )
                obj_base_in_feats = obj_base_in_feats.view(bsz, nf, nnj, -1).contiguous()
            elif self.args.use_objbase_v5: # use_objbase_v5, use_objbase_out_v5
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                if self.args.v5_in_not_base:
                    obj_base_in_feats = torch.cat(
                        [ pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                elif self.args.v5_in_not_base_pos:
                    obj_base_in_feats = torch.cat(
                        [base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
            elif self.args.use_objbase_v6 or self.args.use_objbase_v7:
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            else:
                obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                    [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                )
                # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            
            # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
            #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
            # )
            # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
            # obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            if self.args.use_objbase_v6:
                normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1) # and repeat for the base pts #
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats, normed_base_pts_exp) 
            else:
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats) # nf x bsz x feat_dim # nf x bsz x nnbasepts x feats_dim #
            # obj_base_encoded_feats 
            
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            
            if self.args.train_enc:
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                real_dec_basejtsrel = self.real_basejtsrel_output_process(
                    obj_base_pts_feats, x
                )
                real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                if self.args.use_objbase_out_v3:
                    real_dec_basejtsrel = real_dec_basejtsrel
                else:
                    real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            else:
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    # print(f"timesteps: {timesteps.size()}, obj_base_pts_feats: {obj_base_pts_feats.size()}")
                    
                    if self.args.use_objbase_v5:
                        cur_timesteps = timesteps.unsqueeze(1).repeat(1, nnb).view(-1)
                    else:
                        cur_timesteps = timesteps
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(cur_timesteps)
                    real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()  
                    real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                    
                    if self.args.use_ours_transformer_enc:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                    else: # seq des
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
                    real_basejtsrel_seq_latents = torch.cat(
                        [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                if self.args.use_jts_pert_realbasejtsrel:
                    joints_offset_output =  self.real_basejtsrel_output_process(real_basejtsrel_seq_latents)
                    joints_offset_output = joints_offset_output.permute(0, 3, 1, 2) # bsz x nf x nnj x 3
                    real_dec_basejtsrel = joints_offset_output.unsqueeze(-2).repeat(1, 1, 1, nnb, 1)
                    # real_dec_basejtsrel = joints_offset_output
                else:
                    real_dec_basejtsrel = self.real_basejtsrel_output_process(
                        real_basejtsrel_seq_latents, x
                    )
                    # real_dec_basejtsrel -> decoded realtive positions #
                    real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                    if self.args.use_objbase_out_v3 or self.args.use_objbase_out_v4 or self.args.use_objbase_out_v5:
                        real_dec_basejtsrel = real_dec_basejtsrel
                    else:
                        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            diff_realbasejtsrel_out_dict = {
                'real_dec_basejtsrel': real_dec_basejtsrel,
                'obj_base_pts_feats': obj_base_pts_feats,
            }
        else:
            diff_realbasejtsrel_out_dict = {}
                        
            
        # relative joints encoder; obj pos encoder; obj pos encoder; # penetrations, depth, --- how to use depth for guidance -> and also penetrations # object penetrations #
        if self.diff_basejtsrel:
            
            hand_params = x['pert_rhand_params']
            
            # params_input_process
            hand_params_feats = self.params_input_process(hand_params)
            
            
            # # joints_offset_sequence --> x['pert_joints_offset_sequence']
            # joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
            # joints_offset_sequence = joints_offset_sequence.permute(0, 2, 3, 1).contiguous()
            # joints_offset_feats = self.joints_offset_input_process(joints_offset_sequence) # nf x bsz x dim
            
            # rel_base_pts_feats = self.input_process(basejtsrel_enc_in_feats)
            # sequence_pos_encoder
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(hand_params_feats)
            
            # print(f"joints_offset_feats: {joints_offset_feats.size()}, rel_base_pts_feats_pos_embedding: {rel_base_pts_feats_pos_embedding.size()}")
            # outputs rel base jts encoded latents ##
            # seqTransEncoder, logvar_seqTransEncoder
            # rel_base_pts_outputs_mean = self.basejtsrel_denoising_seqTransEncoder(rel_base_pts_feats_pos_embedding)
            # ### calculate logvar, mean, and feats ###
            # rel_base_pts_outputs_logvar = self.joint_sequence_logvar_seqTransEncoder(rel_base_pts_outputs)
            
            # if self.args.not_diff_avgjts: # not use diff avgjts ##
            rel_base_pts_feats = rel_base_pts_feats_pos_embedding
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            # print(f"rel_base_pts_outputs_mean 1: {rel_base_pts_outputs_mean.size()}")
            if not self.args.without_dec_pos_emb: # without dec pos embedding
                # avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs_mean # 
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs_mean = other_rel_base_pts_outputs
                
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                # print(f"basejtsrel_seq_latents: {basejtsrel_seq_latents.size()}, rel_base_pts_outputs_mean: {rel_base_pts_outputs_mean.size()}, basejtsrel_time_emb: {basejtsrel_time_emb.size()}")
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_seq_latents_pred_feats
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents # [1:, ...]
            
            # joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
            
            params_output = self.params_output_process(other_basejtsrel_seq_latents)
            
            params_output = params_output.permute(1, 0, 2).contiguous()
            # joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            # print(f"joints_offset_output in MDM: {joints_offset_output.size()}, joints_offset_sequence: {joints_offset_sequence.size()}, other_basejtsrel_seq_latents: {other_basejtsrel_seq_latents.size()}")
            
            diff_basejtsrel_dict = {
                'params_output': params_output
            }
                
                # rt_dict['joints_offset_output'] = joints_offset_output
            # else:
            #     avg_joints_sequence = x['pert_avg_joints_sequence']
            #     avg_joints_sequence_trans = avg_joints_sequence.unsqueeze(-1)
            #     avg_joints_feats = self.avg_joints_sequence_input_process(avg_joints_sequence_trans) ## 1 x bsz x dim ###
            
            #     rel_base_pts_feats = torch.cat( # (seq_len + 1) x bsz x dim #
            #         [avg_joints_feats, rel_base_pts_feats_pos_embedding], dim=0 ## jrel_base_pts_pos_embedding #
            #     )
                
            #     ## joints embedding for mean statistics and logvar statistics ##
            #     # seqTransEncoder, logvar_seqTransEncoder #
            #     rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
                
                
            #     if not self.args.without_dec_pos_emb: # without dec pos embedding
            #         avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
            #         other_rel_base_pts_outputs = rel_base_pts_outputs_mean[1: , ...]
            #         other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
            #         rel_base_pts_outputs_mean = torch.cat(
            #             [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
            #         )
                
            #     if self.args.deep_fuse_timeemb:
            #         ## denoising process ###
            #         ## GET joints seq time embeddings ### ### embed time stamps ###
            #         basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            #         basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
            #         basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                    
            #         if self.args.use_ours_transformer_enc:
            #             basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
            #             basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
            #         else:
            #             basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            #     else:
            #         ## denoising process ###
            #         ## GET joints seq time embeddings ### ### embed time stamps ###
            #         basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            #         basejtsrel_seq_latents = torch.cat(
            #             [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
            #         )
                    
            #         if self.args.use_ours_transformer_enc: ## mdm ours ##
            #             basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
            #             basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
            #         else:
            #             basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
            #     # basejtsrel_seq_latents_pred_feats
            #     avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            #     other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
                
            #     avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            #     avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            #     # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            #     # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            #     joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
                
            #     joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
                
            #     rt_dict['joints_offset_output'] = joints_offset_output
            #     rt_dict['avg_jts_outputs'] = avg_jts_outputs
            
        else:
            diff_basejtsrel_dict = {}    
            
            
        rt_dict = {}
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict) ### rt_dict and diff_basejtse
        rt_dict.update(diff_basejtsrel_to_joints_dict)
        rt_dict.update(diff_realbasejtsrel_out_dict) ### diff 
        
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)



### MDM 10 ###
class MDMV13(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        self.diff_realbasejtsrel = self.args.diff_realbasejtsrel
        self.diff_realbasejtsrel_to_joints = self.args.diff_realbasejtsrel_to_joints
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        self.joints_feats_in_dim = 21 * 3
        
        self.data_rep = "xyz"
        
        
        if self.diff_jts:
            
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        
        if self.diff_realbasejtsrel_to_joints: # feature for each joint point? --> for the denoising purpose #
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            layernorm = True
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            if self.args.use_abs_jts_for_encoding_obj_base:
                self.rel_input_feats = 21 * (3)
                # layernorm = False
                self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                # self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            # elif self.args.use
            else:        
                if self.args.use_objbase_v2:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                elif self.args.use_objbase_v3:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV3(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                else:
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            if self.args.use_abs_jts_for_encoding: # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
                self.real_basejtsrel_to_joints_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            self.real_basejtsrel_to_joints_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ### # InputProcessObjBaseV2
            real_basejtsrel_to_joints_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout # 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_to_joints_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_to_joints_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_to_joints_sequence_pos_encoder)
            
            self.real_basejtsrel_to_joints_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_to_joints_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            self.real_basejtsrel_to_joints_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
        
        
        if self.diff_realbasejtsrel:
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            # self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim)
            
            layernorm = True
            if self.args.use_objbase_v2:
                self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, glb_feats_trans=True)
            elif self.args.use_objbase_v4: # use_objbase_out_v4
                self.rel_input_feats = (self.args.nn_base_pts * (3 + 3 + 3)) # current joint positions # how to keep the dimension
                self.real_basejtsrel_input_process = InputProcessObjBaseV4(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v5: # use_objbase_v5, 
                if self.args.v5_in_not_base:
                    self.rel_input_feats = (21 * 3) 
                elif self.args.v5_in_not_base_pos:
                    self.rel_input_feats = 3 + (21 * 3) 
                else:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV5(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, without_glb=self.args.v5_in_without_glb)
            elif self.args.use_objbase_v6: # real_basejtsrel_input_process
                self.rel_input_feats = 3 + 3 + (21 * 3) + 3
                self.real_basejtsrel_input_process = InputProcessObjBaseV6(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v7:
                # InputProcessObjBaseV7
                self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV7(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            else:
                self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            
            self.real_basejtsrel_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ###
            real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout #  # dropout # # dropout 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_sequence_pos_encoder)
            
            self.real_basejtsrel_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
            print(f"not_cond_base: {self.args.not_cond_base}, latent_dim: {self.latent_dim}")
            
            
            if self.args.use_jts_pert_realbasejtsrel:
                print(f"use_jts_pert_realbasejtsrel!!!!!!")
                self.real_basejtsrel_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            else:
                if self.args.use_objbase_out_v3:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV3(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v4:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV4(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v5: # use_objbase_v5, use_objbase_out_v5
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV5(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base, out_objbase_v5_bundle_out=self.args.out_objbase_v5_bundle_out, v5_out_not_cond_base=self.args.v5_out_not_cond_base)
                else:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # OutputProcessCond
        
        if self.diff_basejtsrel: # baes jts rel #
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = 21 * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            
            # # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            # self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            # minn_dist, qu
            if self.args.use_temporal_rep_v2:
                self.joints_quants_in_dim = 21 * (3 + 1 + 1 + 1) # quants # 
            else:
                self.joints_quants_in_dim = 21 * (1 + 1 + 1) # quants # 
            self.joints_quants_input_process = InputProcess(self.data_rep, self.joints_quants_in_dim, self.latent_dim)
            # self.joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)

     
            if self.args.not_cond_base:
                self.rel_input_feats = 21 * ( 3)
            # self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            # self.emb_trans_dec = emb_trans_dec

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            # ### Encoding layer ###
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder # logvar_seqTranEncoder
            # seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                 nhead=self.num_heads,
            #                                                 dim_feedforward=self.ff_size,
            #                                                 dropout=self.dropout,
            #                                                 activation=self.activation)

            # self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
            #                                             num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            # if self.args.use_ours_transformer_enc: # our transformer encoder #
            self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                hidden_size=self.latent_dim,
                fc_size=self.ff_size,
                num_heads=self.num_heads,
                layer_norm=True,
                num_layers=self.num_layers,
                dropout_rate=0.2,
                re_zero=True,
                memory_efficient=False,
            )
            # else:
            #     seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            #     self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                     num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            # if self.args.const_noise: # add to attention network # 
            #     # 1) max pool latents over the sequence
            #     # 2) transform the pooled latnets via the linear layer
            #     self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
            #         nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
            #         nn.Linear(self.latent_dim * 2, self.latent_dim)
            #     )
            
            ###### joints_feats_in_dim ###### # a linear transformation net with weights and bias set to zero #
            # self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3) # output avgjts sequence 
            # OutputProcessCond
            # self.joint_offset_output_process = OutputProcess(self.data_rep, self.joints_quants_in_dim, self.latent_dim, 21, 3)
            
            # self.joint_quants_output_process = OutputProcess(self.data_rep, self.joints_quants_in_dim, self.latent_dim, 21, 3)
            # self.joint_dist_output_process, self.joint_disp_along_normals_output_process, self.joint_disp_vt_normals_output_process #
            self.joint_dist_output_process = OutputProcess(self.data_rep, 21, self.latent_dim, 21, 1)
            self.joint_disp_along_normals_output_process = OutputProcess(self.data_rep, 21, self.latent_dim, 21, 1)
            self.joint_disp_vt_normals_output_process = OutputProcess(self.data_rep, 21, self.latent_dim, 21, 1)
            
            # if self.args.use_dec_rel_v2:
            #     self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # else:
            #     # OutputProcessObjBaseRaw ## output process for basejtsrel #
            #     self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1 + 3)  + 3 # 
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    # diff disp #
    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            
    def set_bn_to_eval(self):
        if self.args.use_objbase_v6: # real_basejtsrel_input_process
            try:
                self.real_basejtsrel_input_process.pnpp_conv_net.set_bn_no_training()
            except:
                pass

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode # ours float
            clip_model)  # Actually this line is unnecessary since clip by default already on float16 ### ours 

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            # bsz x nn b x nnj 
        #     'dec_d': dec_d,
        #   'rel_vel_dec': rel_vel_dec
            dec_d  = dec_d.contiguous().permute(0, 1, 3, 2).contiguous()
            rel_vel_dec_in_feats = rel_vel_dec_in_feats.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
                'dec_d': dec_d, # dec_d, rel_vel_dec # 
                'rel_vel_dec': rel_vel_dec
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled
    
    def decode_realbasejtsrel_from_objbasefeats(self, objbasefeats, input_data):
        real_dec_basejtsrel = self.real_basejtsrel_output_process(
                objbasefeats, input_data
            )
        # real_dec_basejtsrel -> decoded realtive positions #
        real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous()
        return real_dec_basejtsrel
    
    def pred_joint_quants_from_latent_feats(self, denoised_latents):
        # joints_offset_output =  self.joint_quants_output_process(denoised_latents)
            
        # joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
        
        
        other_basejtsrel_seq_latents = denoised_latents # baesjts_seq_latents #
            
        # # self.joint_dist_output_process, self.joint_disp_along_normals_output_process, self.joint_disp_vt_normals_output_process #
        
        joints_dist_output = self.joint_dist_output_process(other_basejtsrel_seq_latents)
        joints_dist_output = joints_dist_output.permute(0, 3, 1, 2)
        
        joints_disp_along_normals_output = self.joint_disp_along_normals_output_process(other_basejtsrel_seq_latents)
        joints_disp_along_normals_output = joints_disp_along_normals_output.permute(0, 3, 1, 2)
        
        joints_disp_vt_normals_output = self.joint_disp_vt_normals_output_process(other_basejtsrel_seq_latents)
        joints_disp_vt_normals_output = joints_disp_vt_normals_output.permute(0, 3, 1, 2)
        
        # joints_offset_output =  self.joint_quants_output_process(other_basejtsrel_seq_latents)
        
        # joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
        
        joints_offset_output = torch.cat(
            [joints_dist_output, joints_disp_along_normals_output, joints_disp_vt_normals_output], dim=-1
        )
        # print(f"joints_offset_output in MDM: {joints_offset_output.size()}, joints_offset_sequence: {joints_offset_sequence.size()}, other_basejtsrel_seq_latents: {other_basejtsrel_seq_latents.size()}")
            
        
        return joints_offset_output
            
    
    def denoising_joint_quants_feats(self, pert_joints_quant_feats, timesteps):
        # if self.args.deep_fuse_timeemb:
        ## denoising process ### # pert_joints_quant_feats, timesteps #
        ## GET joints seq time embeddings ### ### embed time stamps ### # basejts rel denoising quantities # 
        basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
        basejtsrel_time_emb = basejtsrel_time_emb.repeat(pert_joints_quant_feats.size(0), 1, 1).contiguous()
        basejtsrel_seq_latents = pert_joints_quant_feats + basejtsrel_time_emb ### seq_len x bsz x dim ##
        if self.args.use_ours_transformer_enc: # pert_joints_qunat_feats: seq_len x bsz x dim --> 
            basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
            basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
        else:
            basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)

        return basejtsrel_seq_latents

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        rt_dict = {}
           
            
        # relative joints encoder; obj pos encoder; obj pos encoder; # penetrations, depth, --- how to use depth for guidance -> and also penetrations # object penetrations #
        if self.diff_basejtsrel:
            # joints_offset_sequence --> x['pert_joints_offset_sequence']
            joints_quants = x['joints_quants']
            # joints_offset_sequence: bsz x (nf - 1) x nnjoints x joints_in_feats_dim #
            # joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
            joints_quants = joints_quants.permute(0, 2, 3, 1).contiguous() # bsz x nf x nnj x 3 --> (bsz x nf x (nnj x 3)) # bsz x nnj x 3 x nf #
            joints_offset_feats = self.joints_quants_input_process(joints_quants) # nf x bsz x dim # joints_offset_feats 
            
            ## 
            # rel_base_pts_feats = self.input_process(basejtsrel_enc_in_feats)
            # sequence_pos_encoder # rel_base_pts_feats_pos_: nf x bsz x dim # #
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(joints_offset_feats)
            
            
            # if self.args.not_diff_avgjts: # not use diff avgjts ##
            rel_base_pts_feats = rel_base_pts_feats_pos_embedding
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            
            if self.args.use_sigmoid:
                rel_base_pts_outputs_mean = (torch.sigmoid(rel_base_pts_outputs_mean) - 0.5) * 2.
            
            
            # if self.args.train_enc: # other latent  # and we do nto need denoised latents here #
            other_basejtsrel_seq_latents = rel_base_pts_outputs_mean # baesjts_seq_latents #
            
            # # self.joint_dist_output_process, self.joint_disp_along_normals_output_process, self.joint_disp_vt_normals_output_process #
            
            joints_dist_output = self.joint_dist_output_process(other_basejtsrel_seq_latents)
            joints_dist_output = joints_dist_output.permute(0, 3, 1, 2)
            
            joints_disp_along_normals_output = self.joint_disp_along_normals_output_process(other_basejtsrel_seq_latents)
            joints_disp_along_normals_output = joints_disp_along_normals_output.permute(0, 3, 1, 2)
            
            joints_disp_vt_normals_output = self.joint_disp_vt_normals_output_process(other_basejtsrel_seq_latents)
            joints_disp_vt_normals_output = joints_disp_vt_normals_output.permute(0, 3, 1, 2)
            
            # joints_offset_output =  self.joint_quants_output_process(other_basejtsrel_seq_latents)
            
            # joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            joints_offset_output = torch.cat(
                [joints_dist_output, joints_disp_along_normals_output, joints_disp_vt_normals_output], dim=-1
            )
            # print(f"joints_offset_output in MDM: {joints_offset_output.size()}, joints_offset_sequence: {joints_offset_sequence.size()}, other_basejtsrel_seq_latents: {other_basejtsrel_seq_latents.size()}")
            
            diff_basejtsrel_dict = {
                'joints_quants_output': joints_offset_output,
                'joints_quants_latents': rel_base_pts_outputs_mean,
            }

        else:
            diff_basejtsrel_dict = {}    
            
            
        rt_dict = {}
        rt_dict.update(diff_basejtsrel_dict)
        # rt_dict.update(diff_basejtse_dict) ### rt_dict and diff_basejtse
        # rt_dict.update(diff_basejtsrel_to_joints_dict)
        # rt_dict.update(diff_realbasejtsrel_out_dict) ### diff 
        
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)



### MDM 10 ###
class MDMV12(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        self.diff_realbasejtsrel = self.args.diff_realbasejtsrel
        self.diff_realbasejtsrel_to_joints = self.args.diff_realbasejtsrel_to_joints
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # 
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        self.joints_feats_in_dim = 21 * 3
        
        self.data_rep = "xyz"
        
        
        if self.diff_jts:
            
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        
        # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
        if self.diff_realbasejtsrel_to_joints: # feature for each joint point? --> for the denoising purpose #
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            layernorm = True
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            if self.args.use_abs_jts_for_encoding_obj_base:
                self.rel_input_feats = 21 * (3)
                # layernorm = False
                self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                # self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            # elif self.args.use
            else:        
                if self.args.use_objbase_v2:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                elif self.args.use_objbase_v3:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBaseV3(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
                else:
                    self.real_basejtsrel_to_joints_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            if self.args.use_abs_jts_for_encoding: # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
                self.real_basejtsrel_to_joints_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            self.real_basejtsrel_to_joints_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ### # InputProcessObjBaseV2
            real_basejtsrel_to_joints_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout # 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_to_joints_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_to_joints_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_to_joints_sequence_pos_encoder)
            
            self.real_basejtsrel_to_joints_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_to_joints_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_to_joints_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_to_joints_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            self.real_basejtsrel_to_joints_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
        
        
        if self.diff_realbasejtsrel:
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            self.rel_input_feats = 21 * (3 + 3 + 3) # base pts, normals, the relative positions 
            # self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim)
            
            layernorm = True
            if self.args.use_objbase_v2:
                self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV2(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, glb_feats_trans=True)
            elif self.args.use_objbase_v4: # use_objbase_out_v4
                self.rel_input_feats = (self.args.nn_base_pts * (3 + 3 + 3)) # current joint positions # how to keep the dimension
                self.real_basejtsrel_input_process = InputProcessObjBaseV4(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v5: # use_objbase_v5, 
                if self.args.v5_in_not_base:
                    self.rel_input_feats = (21 * 3) 
                elif self.args.v5_in_not_base_pos:
                    self.rel_input_feats = 3 + (21 * 3) 
                else:
                    self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV5(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, without_glb=self.args.v5_in_without_glb)
            elif self.args.use_objbase_v6: # real_basejtsrel_input_process
                self.rel_input_feats = 3 + 3 + (21 * 3) + 3
                self.real_basejtsrel_input_process = InputProcessObjBaseV6(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            elif self.args.use_objbase_v7:
                # InputProcessObjBaseV7
                self.rel_input_feats = 3 + 3 + (21 * 3)
                self.real_basejtsrel_input_process = InputProcessObjBaseV7(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            else:
                self.real_basejtsrel_input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm)
            
            
            self.real_basejtsrel_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            ### Encoding layer ###
            real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # latent dim # nn_heads # ff_size # dropout #  # dropout # # dropout 
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.real_basejtsrel_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_seqTransEncoderLayer, # basejtsrel_seqTrans
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.real_basejtsrel_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_sequence_pos_encoder)
            
            self.real_basejtsrel_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.real_basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                real_basejtsrel_denoising_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.real_basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(real_basejtsrel_denoising_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
            print(f"not_cond_base: {self.args.not_cond_base}, latent_dim: {self.latent_dim}")
            
            
            if self.args.use_jts_pert_realbasejtsrel:
                print(f"use_jts_pert_realbasejtsrel!!!!!!")
                self.real_basejtsrel_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            else:
                if self.args.use_objbase_out_v3:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV3(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v4:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV4(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                elif self.args.use_objbase_out_v5: # use_objbase_v5, use_objbase_out_v5
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV5(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base, out_objbase_v5_bundle_out=self.args.out_objbase_v5_bundle_out, v5_out_not_cond_base=self.args.v5_out_not_cond_base)
                else:
                    self.real_basejtsrel_output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            # OutputProcessCond
        
        if self.diff_basejtsrel: ## basejtsrel ##
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = 21 * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            # cond_real_basejtsrel_input_process, cond_real_basejtsrel_pos_encoder, cond_real_basejtsrel_seqTransEncoderLayer, cond_trans_linear_layer #
            # Conditional strategy 1: use the relative position embeddings for guidance (cannot use the origianl weights for finetuning) #
            layernorm = True
            # [finetune_with_cond_rel, finetune_with_cond_jtsobj]
            if self.args.finetune_with_cond_rel:
                if self.args.use_objbase_v5: # use_objbase_v5 # 
                    if self.args.v5_in_not_base:
                        self.rel_input_feats = (21 * 3) 
                    elif self.args.v5_in_not_base_pos:
                        self.rel_input_feats = 3 + (21 * 3) 
                    else: # as an additional conditions for the input and denoising #
                        self.rel_input_feats = 3 + 3 + (21 * 3)
                    self.cond_real_basejtsrel_input_process = InputProcessObjBaseV5(self.data_rep, self.rel_input_feats, self.latent_dim, layernorm=layernorm, without_glb=self.args.v5_in_without_glb, only_with_glb=True)
                else:
                    raise ValueError(f"Must use objbase_v5 currently, others have not been implemented yet.")
                self.cond_real_basejtsrel_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                cond_real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.cond_real_basejtsrel_seqTransEncoderLayer = nn.TransformerEncoder(cond_real_basejtsrel_seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
            elif self.args.finetune_with_cond_jtsobj: # cond_obj_trans_layer
                # InputProcessObjV6(self, data_rep, input_feats, latent_dim, layernorm=True)
                # cond_obj_input_layer, cond_obj_trans_layer, cond_joints_offset_input_process, cond_sequence_pos_encoder, cond_jtsobj_seqTransEncoder # 
                # TODO: cond_obj_trans_layer : cond_joints_offset_input_process <- joints_offset_input_process; cond_sequence_pos_encoder <- sequence_pos_encoder; cond_seqTransEncoder <- seqTransEncoder
                self.cond_obj_input_feats = 3 
                # if self.args.finetune_cond_obj_feats_dim == 3:
                #     self.cond_obj_input_feats = 3 
                self.cond_obj_input_feats = self.args.finetune_cond_obj_feats_dim # finetune cond with obj feats dim #
                self.cond_obj_input_layer = InputProcessObjV6(self.data_rep, self.cond_obj_input_feats, self.latent_dim, layernorm=layernorm)
                # TODO: remember to set this layer to zero! #
                self.cond_obj_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
                # hand_embedding + zero_trans_layer(obj_cond_embedding)
                self.cond_joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
                self.cond_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                cond_jtsobj_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
                self.cond_seqTransEncoder = nn.TransformerEncoder(cond_jtsobj_seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            else:
                raise ValueError(f"Either ``finetune_with_cond_rel'' or ``finetune_with_cond_jtsobj'' should be activated!")
                
            # TODO: remember to initialize it to zero! #
            self.cond_trans_linear_layer = nn.Linear(self.latent_dim, self.latent_dim)
            
            
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            # TODO: should set those weights to frozen --> joints offset input process... 
            self.joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)

     
            if self.args.not_cond_base:
                self.rel_input_feats = 21 * ( 3)
            # self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### Encoding layer ###
            # logvar_seqTransEncoder_e, logvar_seqTransEncoder # logvar_seqTranEncoder
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ### # TimestepEmbedder -> embedding times #
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # # seq_len x bsz x dim 
            # if self.args.const_noise: # add to attention network # add j
            #     # 1) max pool latents over the sequence
            #     # 2) transform the pooled latnets via the linear layer
            #     self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
            #         nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
            #         nn.Linear(self.latent_dim * 2, self.latent_dim)
            #     )
            ###### joints_feats_in_dim ###### # a linear transformation net with weights and bias set to zero #
            self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3) # output avgjts sequence 
            # OutputProcessCond
            self.joint_offset_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            
            if self.args.use_dec_rel_v2:
                self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base, finetune_with_cond=self.args.finetune_with_cond)
            else:
                # OutputProcessObjBaseRaw ## output process for basejtsrel #
                self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            # self.input_feats_e = 21 * (3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1 + 3 + 3 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            # if self.args.const_noise:
            #     # 1) max pool latents over the sequence
            #     # 2) transform the pooled latnets via the linear layer
            #     self.basejtse_denoising_seqTransEncoder = nn.Sequential(
            #         nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
            #         nn.Linear(self.latent_dim * 2, self.latent_dim)
            #     )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            
    def set_bn_to_eval(self):
        if self.args.use_objbase_v6: # real_basejtsrel_input_process
            try:
                self.real_basejtsrel_input_process.pnpp_conv_net.set_bn_no_training()
            except:
                pass

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode 
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats[1:, ...]
        
        avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        basejtsrel_dec_out = {
            'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': basejtsrel_output['dec_rel'],
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_outputs"]
            
            if rel_base_pts_outputs.size(0) == 1 and self.args.single_frame_noise:
                rel_base_pts_outputs = rel_base_pts_outputs.repeat(self.args.window_size + 1, 1, 1)
            
            if not self.args.without_dec_pos_emb: # without 
                avg_jts_inputs = rel_base_pts_outputs[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs[1: , ...]
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs = torch.cat(
                    [avg_jts_inputs, other_rel_base_pts_outputs], dim=0
                )
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs + basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                    
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                basejtsrel_seq_latents = input_latent_feats["rel_base_pts_outputs_enc"]
                if basejtsrel_seq_latents.size(0) == 1 and self.args.single_frame_noise:
                    basejtsrel_seq_latents = basejtsrel_seq_latents.repeat(self.args.window_size + 1, 1, 1)
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            elif self.args.pred_diff_noise:
                basejtsrel_seq_latents_pred_feats = input_latent_feats["rel_base_pts_outputs"] - basejtsrel_seq_latents
            else:
                basejtsrel_seq_latents_pred_feats = basejtsrel_seq_latents
            
            # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
            ### GET joints seq output ###
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            # basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
            # basejtsrel_seq_latents = torch.cat(
            #     [basejtsrel_time_emb, rel_base_pts_outputs], dim=0
            # )
            
            
            # basejtsrel_seq_latents_pred_feats
            avg_jts_seq_latents = basejtsrel_seq_latents_pred_feats[0:1, ...]
            other_basejtsrel_seq_latents = basejtsrel_seq_latents_pred_feats[1:, ...]
            
            avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "basejtsrel_output": basejtsrel_output['dec_rel'],
                "basejtsrel_seq_latents": basejtsrel_seq_latents,
                "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_d = basejtsrel_output['dec_d']
            rel_vel_dec = basejtsrel_output['rel_vel_dec']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_d = dec_d.contiguous().permute(0, 1, 3, 2).contiguous()
            rel_vel_dec = rel_vel_dec.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
                'dec_d': dec_d,
                'rel_vel_dec': rel_vel_dec,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled
    
    def decode_realbasejtsrel_from_objbasefeats(self, objbasefeats, input_data):
        real_dec_basejtsrel = self.real_basejtsrel_output_process(
                objbasefeats, input_data
            )
        # real_dec_basejtsrel -> decoded realtive positions #
        real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous()
        return real_dec_basejtsrel
    
    def denoising_realbasejtsrel_objbasefeats(self, pert_obj_base_pts_feats, timesteps):
        if self.args.deep_fuse_timeemb:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(pert_obj_base_pts_feats.size(0), 1, 1).contiguous()
            real_basejtsrel_seq_latents = pert_obj_base_pts_feats + real_basejtsrel_time_emb
            
            if self.args.use_ours_transformer_enc:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
            else: # seq
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
        else:
            ## denoising process ###
            ## GET joints seq time embeddings ### ### embed time stamps ###
            real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
            real_basejtsrel_seq_latents = torch.cat(
                [real_basejtsrel_time_emb, pert_obj_base_pts_feats], dim=0
            )
            
            if self.args.use_ours_transformer_enc: ## mdm ours ##
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
            else:
                real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                
        # bsz, nframes, nnb, nnj, 3 --> 
        # real_dec_basejtsrel = self.real_basejtsrel_output_process(
        #     real_basejtsrel_seq_latents, x
        # )
        return real_basejtsrel_seq_latents

    def get_cond_parameters(self):
        # [finetune_with_cond_rel, finetune_with_cond_jtsobj]
        # cond_real_basejtsrel_input_process, cond_real_basejtsrel_pos_encoder, cond_real_basejtsrel_seqTransEncoderLayer, cond_trans_linear_layer #
        # finetune_with_cond_jtsobj : cond_joints_offset_input_process <- joints_offset_input_process; cond_sequence_pos_encoder <- sequence_pos_encoder; cond_seqTransEncoder <- seqTransEncoder
        # cond_obj_input_layer, cond_obj_trans_layer, cond_joints_offset_input_process, cond_sequence_pos_encoder, cond_seqTransEncoder # 
        if self.args.diff_basejtsrel:
            # TODO: finetune_with_cond_jtsobj : cond_joints_offset_input_process <- joints_offset_input_process; cond_sequence_pos_encoder <- sequence_pos_encoder; cond_seqTransEncoder <- seqTransEncoder
            if self.args.finetune_with_cond_rel:
                params = list(self.cond_real_basejtsrel_input_process.parameters()) + list(self.cond_real_basejtsrel_pos_encoder.parameters()) + list(self.cond_real_basejtsrel_seqTransEncoderLayer.parameters()) + list(self.cond_trans_linear_layer.parameters())
            elif self.args.finetune_with_cond_jtsobj:
                params = list(self.cond_joints_offset_input_process.parameters()) + list(self.cond_sequence_pos_encoder.parameters()) + list(self.cond_seqTransEncoder.parameters()) + list(self.cond_obj_trans_layer.parameters()) + list(self.cond_obj_input_layer.parameters())
            else:
                raise ValueError(f"Either ``finetune_with_cond_rel'' or ``finetune_with_cond_jtsobj'' should be activated !")
        else:
            raise ValueError(f"Must use diff_basejtsrel currently, others have not been implemented yet.")
        return params
    
    def set_trans_linear_layer_to_zero(self):
        torch.nn.init.zeros_(self.cond_trans_linear_layer.weight)
        torch.nn.init.zeros_(self.cond_trans_linear_layer.bias)
        # finetune_with_cond_jtsobj: # cond_obj_trans_layer
        if self.args.finetune_with_cond_jtsobj:
            torch.nn.init.zeros_(self.cond_obj_trans_layer.weight)
            torch.nn.init.zeros_(self.cond_obj_trans_layer.bias)
  
    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        # base_normals # ## 
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            e_disp_rel_to_base_along_normals = x['pert_e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['pert_e_disp_rel_to_base_vt_normals']
            obj_pts_disp = x['obj_pts_disp']
            vel_obj_pts_to_hand_pts = x['vel_obj_pts_to_hand_pts']
            disp_dist = x['disp_dist']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            obj_pts_disp_exp = obj_pts_disp.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat( # along normals; # vt normals #
                [base_pts_disp_exp, base_normals_disp_exp, obj_pts_disp_exp, vel_obj_pts_to_hand_pts, disp_dist,  e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ## # 
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            #### base_jts_e_feats, base_jts_e_feats_mean ####
            # ## us basejtsefeats for denoising directly ##
            base_jts_e_feats = base_jts_e_feats_mean
            if not self.args.without_dec_pos_emb: ## use positional encoding ##
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)

            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            # ### sequence latents ###
            # if self.args.train_enc: # trian enc for seq latents ###
            #     base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            ##### output_process_e -> output energies #####
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_d = base_jts_e_output['dec_d']
            rel_vel_dec = base_jts_e_output['rel_vel_dec']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous() # bsz x (ws - 1) x nnj x nnb
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
                'dec_d': dec_d,
                'rel_vel_dec': rel_vel_dec,
            }
            # rt_dict['base_jts_e_feats'] = base_jts_e_feats
            # rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            # rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        else:
            diff_basejtse_dict = {}
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
        
        if self.args.diff_realbasejtsrel_to_joints:  # nframes x nnbase x nnjts x (base pts + base normals + 3) 2) point feature for each point; point feature for; condition on the noisy input for the denoised information
            # real_basejtsrel_to_joints_input_process, real_basejtsrel_to_joints_sequence_pos_encoder, real_basejtsrel_to_joints_seqTransEncoder
            # real_basejtsrel_to_joints_embed_timestep, real_basejtsrel_to_joints_sequence_pos_denoising_encoder, real_basejtsrel_to_joints_denoising_seqTransEncoder, real_basejtsrel_to_joints_output_process
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_joints_for_jts_pred'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_joints_for_jts_pred'] # bsz x nf x nnj x nnb x 3 
            
            
            ## use_abs_jts_pos --> obj jts pos for the encodingj
            if self.args.use_abs_jts_pos: ## bsz x nf x nnj x nnb x 3 ## ---> abs jts pos ##
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1)
                
            # use_abs_jts_for_encoding, real_basejtsrel_to_joints_input_process
            if self.args.use_abs_jts_for_encoding:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                abs_jts = pert_rel_base_pts_to_rhand_joints[..., 0, :]
                abs_jts = abs_jts.permute(0, 2, 3, 1).contiguous()
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(abs_jts)
            elif  self.args.use_abs_jts_for_encoding_obj_base:
                if not self.args.use_abs_jts_pos:
                    pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints + normed_base_pts.unsqueeze(1).unsqueeze(1) # pert_rel_base_pts_to_rhand_joints: bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints = pert_rel_base_pts_to_rhand_joints[:, :, :, 0:1, :]
                # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1)[:, :, :, 0:1, :], pert_rel_base_pts_to_rhand_joints], dim=-1
                # )
                obj_base_in_feats = pert_rel_base_pts_to_rhand_joints
                # --> tnrasform the input feature dim to 21 * 3 here for encoding # 
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, 1, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
            else:
                if self.args.use_objbase_v2:
                    # bsz x nf x nnj x nnb x 3
                    pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                        [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                    )
                    # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                    obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
                obj_base_encoded_feats = self.real_basejtsrel_to_joints_input_process(obj_base_in_feats) # nf x bsz x feat_dim #
                
            # ### real_basejtsrel_to_joints_input_process --> real_basejtsrel_to_joints_input_process --> for the joints and input process ### # obj_base_encoded_feats
            # obj_base_encoded_feats 
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_to_joints_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_to_joints_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()
                real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                
                if self.args.use_ours_transformer_enc:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                else: # seq des
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                real_basejtsrel_time_emb = self.real_basejtsrel_to_joints_embed_timestep(timesteps)
                real_basejtsrel_seq_latents = torch.cat(
                    [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    real_basejtsrel_seq_latents = self.real_basejtsrel_to_joints_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            joints_offset_output =  self.real_basejtsrel_to_joints_output_process(real_basejtsrel_seq_latents)
                
            joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            if self.args.diff_basejtsrel:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output_from_rel': joints_offset_output
                }
            else:
                diff_basejtsrel_to_joints_dict = {
                    'joints_offset_output': joints_offset_output
                }
                
        else:
            diff_basejtsrel_to_joints_dict = {}
            
            
        if self.diff_realbasejtsrel: # real_dec_basejtsrel
            # real_basejtsrel_input_process, real_basejtsrel_sequence_pos_encoder, real_basejtsrel_seqTransEncoder, real_basejtsrel_embed_timestep, real_basejtsrel_sequence_pos_denoising_encoder, real_basejtsrel_denoising_seqTransEncoder
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].size()[:4]
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_rhand_joints'] # bsz x nf x nnj x nnb x 3 
            
            if self.args.use_objbase_v2:
                # bsz x nf x nnj x nnb x 3
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            elif self.args.use_objbase_v4:
                # use_objbase_v4: # use_objbase_out_v4
                exp_normed_base_pts = normed_base_pts.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                exp_base_normals = base_normals.unsqueeze(1).unsqueeze(2).repeat(1, nf, nnj, 1, 1).contiguous()
                obj_base_in_feats = torch.cat(
                    [pert_rel_base_pts_to_rhand_joints, exp_normed_base_pts, exp_base_normals], dim=-1 # bsz x nf x nnj x nnb x (3 + 3 + 3) # -> exp_base_normals
                )
                obj_base_in_feats = obj_base_in_feats.view(bsz, nf, nnj, -1).contiguous()
            elif self.args.use_objbase_v5: # use_objbase_v5, use_objbase_out_v5
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                if self.args.v5_in_not_base:
                    obj_base_in_feats = torch.cat(
                        [ pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                elif self.args.v5_in_not_base_pos:
                    obj_base_in_feats = torch.cat(
                        [base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else:
                    obj_base_in_feats = torch.cat(
                        [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
            elif self.args.use_objbase_v6 or self.args.use_objbase_v7:
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                obj_base_in_feats = torch.cat(
                    [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                )
            else:
                obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
                    [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
                )
                # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
                obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            
            # obj_base_in_feats = torch.cat( # bsz x nf x nnj x nnb x (3 + 3 + 3)
            #     [normed_base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nf, nnj, 1, 1), pert_rel_base_pts_to_rhand_joints], dim=-1
            # )
            # print(f"obj_base_in_feats: {obj_base_in_feats.size()}, bsz: {bsz}, nf: {nf}, nnj: {nnj}, nnb: {nnb}")
            # obj_base_in_feats = obj_base_in_feats.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous() #
            if self.args.use_objbase_v6:
                normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1) # and repeat for the base pts #
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats, normed_base_pts_exp) 
            else:
                obj_base_encoded_feats = self.real_basejtsrel_input_process(obj_base_in_feats) # nf x bsz x feat_dim # nf x bsz x nnbasepts x feats_dim #
            # obj_base_encoded_feats 
            
            obj_base_pts_feats_pos_embedding = self.real_basejtsrel_sequence_pos_encoder(obj_base_encoded_feats)
            obj_base_pts_feats = self.real_basejtsrel_seqTransEncoder(obj_base_pts_feats_pos_embedding)
            
            if self.args.use_sigmoid:
                obj_base_pts_feats = (torch.sigmoid(obj_base_pts_feats) - 0.5) * 2.
            
            if self.args.train_enc:
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                real_dec_basejtsrel = self.real_basejtsrel_output_process(
                    obj_base_pts_feats, x
                )
                real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                if self.args.use_objbase_out_v3:
                    real_dec_basejtsrel = real_dec_basejtsrel
                else:
                    real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            else:
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    # print(f"timesteps: {timesteps.size()}, obj_base_pts_feats: {obj_base_pts_feats.size()}")
                    
                    if self.args.use_objbase_v5:
                        cur_timesteps = timesteps.unsqueeze(1).repeat(1, nnb).view(-1)
                    else:
                        cur_timesteps = timesteps
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(cur_timesteps)
                    real_basejtsrel_time_emb = real_basejtsrel_time_emb.repeat(obj_base_pts_feats.size(0), 1, 1).contiguous()  
                    real_basejtsrel_seq_latents = obj_base_pts_feats + real_basejtsrel_time_emb
                    
                    if self.args.use_ours_transformer_enc:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)
                    else: # seq des
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    real_basejtsrel_time_emb = self.real_basejtsrel_embed_timestep(timesteps)
                    real_basejtsrel_seq_latents = torch.cat(
                        [real_basejtsrel_time_emb, obj_base_pts_feats], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(real_basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        real_basejtsrel_seq_latents = real_basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        real_basejtsrel_seq_latents = self.real_basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_seq
                # bsz, nframes, nnb, nnj, 3 --> 
                # 
                if self.args.use_jts_pert_realbasejtsrel:
                    joints_offset_output =  self.real_basejtsrel_output_process(real_basejtsrel_seq_latents)
                    joints_offset_output = joints_offset_output.permute(0, 3, 1, 2) # bsz x nf x nnj x 3
                    real_dec_basejtsrel = joints_offset_output.unsqueeze(-2).repeat(1, 1, 1, nnb, 1)
                    # real_dec_basejtsrel = joints_offset_output
                else:
                    real_dec_basejtsrel = self.real_basejtsrel_output_process(
                        real_basejtsrel_seq_latents, x
                    )
                    # real_dec_basejtsrel -> decoded realtive positions #
                    real_dec_basejtsrel = real_dec_basejtsrel['dec_rel']
                    if self.args.use_objbase_out_v3 or self.args.use_objbase_out_v4 or self.args.use_objbase_out_v5:
                        real_dec_basejtsrel = real_dec_basejtsrel
                    else:
                        real_dec_basejtsrel = real_dec_basejtsrel.permute(0, 1, 3, 2, 4).contiguous() # bsz x nf x nnj x nnb x 3 
            diff_realbasejtsrel_out_dict = {
                'real_dec_basejtsrel': real_dec_basejtsrel,
                'obj_base_pts_feats': obj_base_pts_feats,
            }
        else:
            diff_realbasejtsrel_out_dict = {}
                        
            
            
        if self.diff_basejtsrel:
            bsz, nf, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].size()[:4]
            # cond_real_basejtsrel_input_process, cond_real_basejtsrel_pos_encoder, cond_real_basejtsrel_seqTransEncoderLayer, cond_trans_linear_layer #
            normed_base_pts = x['normed_base_pts']
            base_normals = x['base_normals']
            pert_rel_base_pts_to_rhand_joints = x['pert_rel_base_pts_to_rhand_joints'] # bsz x nf x nnj x nnb x 3 
            
            # conditional strategies and representations for denoising #
            # conditional strategies 
            # [finetune_with_cond_rel, finetune_with_cond_jtsobj] # 
            # cond_real_basejtsrel_input_process, cond_real_basejtsrel_pos_encoder, cond_real_basejtsrel_seqTransEncoderLayer, cond_trans_linear_layer #
            # finetune_with_cond_jtsobj : cond_joints_offset_input_process <- joints_offset_input_process; cond_sequence_pos_encoder <- sequence_pos_encoder; cond_seqTransEncoder <- seqTransEncoder
            
            if self.args.finetune_with_cond_rel: # finetune with cond rel # finetune with cond  # finetune with 
                if self.args.use_objbase_v5: # use_objbase_v5, use_objbase_out_v5
                    pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                    if self.args.v5_in_not_base:
                        obj_base_in_feats = torch.cat(
                            [ pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                        )
                    elif self.args.v5_in_not_base_pos:
                        obj_base_in_feats = torch.cat(
                            [base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                        )
                    else:
                        obj_base_in_feats = torch.cat(
                            [normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1), base_normals.unsqueeze(1).repeat(1, nf, 1, 1), pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                        )
                else:
                    raise ValueError(f"Must use objbase_v5 currently, others have not been implemented yet.")
                
                # print(f'obj_base_in_feats: {obj_base_in_feats.size()}')
                obj_base_encoded_feats = self.cond_real_basejtsrel_input_process(obj_base_in_feats) 
                obj_base_pts_feats_pos_embedding = self.cond_real_basejtsrel_pos_encoder(obj_base_encoded_feats)
                # print(f"obj_base_pts_feats_pos_embedding: {obj_base_pts_feats_pos_embedding.size()}")
                obj_base_pts_feats = self.cond_real_basejtsrel_seqTransEncoderLayer(obj_base_pts_feats_pos_embedding) # seq x bsz x latent_dim
            elif self.args.finetune_with_cond_jtsobj:
                # cond_obj_input_layer, cond_obj_trans_layer, cond_joints_offset_input_process, cond_sequence_pos_encoder, cond_seqTransEncoder # 
                cond_joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
                cond_joints_offset_sequence = cond_joints_offset_sequence.permute(0, 2, 3, 1).contiguous()
                cond_joints_offset_feats = self.cond_joints_offset_input_process(cond_joints_offset_sequence) # nf x bsz x dim
                pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                
                # 
                if self.cond_obj_input_feats == 3:
                    normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1).contiguous()
                elif self.cond_obj_input_feats == 6:
                    normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1).contiguous()
                    normed_base_normals_exp = base_normals.unsqueeze(1).repeat(1, nf, 1, 1).contiguous()
                    normed_base_pts_exp = torch.cat(
                        [normed_base_pts_exp, normed_base_normals_exp], dim=-1 ## bsz x nf x nn_base_pts x (3 + 3)
                    )
                elif self.cond_obj_input_feats == (6 + 21 * 3): # 63 + 6 -> 69
                    normed_base_pts_exp = normed_base_pts.unsqueeze(1).repeat(1, nf, 1, 1).contiguous()
                    normed_base_normals_exp = base_normals.unsqueeze(1).repeat(1, nf, 1, 1).contiguous()
                    pert_rel_base_pts_to_rhand_joints_exp = pert_rel_base_pts_to_rhand_joints.transpose(-2, -3).contiguous().view(bsz, nf, nnb, -1).contiguous()
                    normed_base_pts_exp = torch.cat(
                        [normed_base_pts_exp, normed_base_normals_exp, pert_rel_base_pts_to_rhand_joints_exp], dim=-1
                    )
                else: # cond_obj_input_feats --> cond_obj_input_feats # finetune_cond_obj_feats_dim
                    raise ValueError(f"Unrecognized cond_obj_input_feats: {self.cond_obj_input_feats}")
                
                cond_obj_feats = self.cond_obj_input_layer(normed_base_pts_exp) # nf x bsz x feats_dim #
                cond_obj_feats = self.cond_obj_trans_layer(cond_obj_feats)
                cond_jtsobj_feats = cond_joints_offset_feats + cond_obj_feats
                cond_jtsobj_feats = self.cond_sequence_pos_encoder(cond_jtsobj_feats)
                obj_base_pts_feats = self.cond_seqTransEncoder(cond_jtsobj_feats)
            else:
                raise ValueError(f"Either ``finetune_with_cond_rel'' or ``finetune_with_cond_jtsobj'' should be activated! ")
                
                
            obj_base_pts_feats = self.cond_trans_linear_layer(obj_base_pts_feats)
            # print(f"obj_base_pts_feats: {obj_base_pts_feats.size()}")
            
            
            ### ==== Encoding joints ==== ###
            joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
            joints_offset_sequence = joints_offset_sequence.permute(0, 2, 3, 1).contiguous()
            joints_offset_feats = self.joints_offset_input_process(joints_offset_sequence) # nf x bsz x dim
            
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(joints_offset_feats)
            
            rel_base_pts_feats = rel_base_pts_feats_pos_embedding
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            ### ==== Encoding joints ==== ###
            
            
            ### ==== fuse conditional embeddings with joints embeddigns ==== ###
            rel_base_pts_outputs_mean = rel_base_pts_outputs_mean + obj_base_pts_feats
            ### ==== fuse conditional embeddings with joints embeddigns ==== ###
            # print(f"rel_base_pts_outputs_mean: {rel_base_pts_outputs_mean.size()}")
            
            
            ### ==== denoise latent features ==== ###
            if not self.args.without_dec_pos_emb: # without dec pos embedding
                # avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                other_rel_base_pts_outputs = rel_base_pts_outputs_mean # 
                other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                rel_base_pts_outputs_mean = other_rel_base_pts_outputs
                
            if self.args.deep_fuse_timeemb:
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb
                if self.args.use_ours_transformer_enc:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                basejtsrel_seq_latents = torch.cat(
                    [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                else:
                    basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            ### ==== denoise latent features ==== ###
            
            other_basejtsrel_seq_latents = basejtsrel_seq_latents # [1:, ...]
            
            joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
            
            joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
            diff_basejtsrel_dict = {
                'joints_offset_output': joints_offset_output
            }
            
            
        else:
            diff_basejtsrel_dict = {}    
            
            
        rt_dict = {}
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict) ### rt_dict and diff_basejtse
        rt_dict.update(diff_basejtsrel_to_joints_dict)
        rt_dict.update(diff_realbasejtsrel_out_dict) ### diff 
        
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)




class MDMV11(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        # mdm_ours #
        # self.args 
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 
        
        ### GET args ###
        self.args = kargs.get('args', None)
        
        ### GET the diff. suit ###
        self.diff_jts = self.args.diff_jts
        self.diff_basejtsrel = self.args.diff_basejtsrel
        self.diff_basejtse = self.args.diff_basejtse
        ### GET the diff. suit ###
        
        
        self.arch = arch
        ## ==== gru_emb_dim ==== ## # gru emb dim #
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        # 
        
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        self.joints_feats_in_dim = 21 * 3
        
        self.data_rep = "xyz"
        
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        
        
        if self.diff_jts:
            ## Input process for joints ##
            self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            # self.joint_sequence_input_process = InputProcessObjBase(self.data_rep, 3, self.latent_dim)
            # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.joint_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init") ## transformer encoder layer ## UNet 
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### logvar for the encoding laeyer and 
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            ## Joint sequence transformer encoder layer ## # sequence encoder #
            self.joint_sequence_logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads, # num_heads 
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.joint_sequence_seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru':
            #     print("GRU init")
            #     self.joint_sequence_gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            ### joint sequence embed timestep ## ## timestep
            self.joint_sequence_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            # self.joint_sequence_output_process = OutputProcess(self.data_rep, self.latent_dim)
            # (self, data_rep, input_feats, latent_dim, njoints, nfeats):
            
            #### ====== joint sequence denoising block ====== ####
            ## seqTransEncoder ##
            self.joint_sequence_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.joint_sequence_pos_encoder)
            
            self.joint_sequence_denoising_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            if self.args.use_ours_transformer_enc:
                self.joint_sequence_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            
            # refinement for predicted joints # --> not in the paradigm of generation #
            # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)

            # self.joint_sequence_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            #### ====== joint sequence denoiisng block ====== ####
            ### Output process ### output proces for joint sequence ### # output proces --> datarep, joints feats in dim, latent dim ##
            ###### joints_feats_in_dim ######
            self.joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            # self.joint_sequence_output_process = OutputProcessCond(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            ###### ======= Construct joint sequence encoder, communicator, and decoder ======== #######
        
        if self.diff_basejtsrel: ## basejtsrel ##
            # treate them as textures of signals to model # # base pts -> dec on base pts features --> 
            # latent space denoising and feature decoding --> a little bit concern about the feature decoding process #
            # TODO: add base_pts and base_normals to the base points -rel-to- rhand joints encoding process #
            self.rel_input_feats = 21 * (3 + 3 + 3) # relative positions from base pts to rhand joints ##
            
            
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ## Input process for joints ## ## joints_feats_in_dim -- 
            self.avg_joints_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            
            self.joints_offset_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # self.joint_sequence_input_process = InputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim)
            # # # InputProcessObjBase(self, data_rep, input_feats, latent_dim)
            
            ###### joints_feats_in_dim ######
            # self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            # OutputProcessCond
            
            
            ## nf x ## diffbasejts rel ## 
            # inputs: bsz x nf x nnb x nn_b_in_feats # ## 
            
            if self.args.not_cond_base:
                self.rel_input_feats = 21 * ( 3)
            
            # self.input_process = InputProcessObjBase(self.data_rep, self.rel_input_feats+self.gru_emb_dim, self.latent_dim)
            
            
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            
            ## and we can put one feature ahead of the transformer to learn information jointly ##

            ### Encoding layer ###
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            ### Encoding layer ###
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder
            seqTransEncoderLayer_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer_logvar,
                                                        num_layers=self.num_layers)
            
            ### timesteps embedding layer ###
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process # # baseptsrel #
            self.basejtsrel_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            
            self.sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtsrel_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                )
            else:
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.basejtsrel_denoising_seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                                num_layers=self.num_layers)
                
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtsrel_glb_denoising_latents_trans_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
                
            ###### joints_feats_in_dim ######
            self.avg_joint_sequence_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3) # output avgjts sequence 
            # OutputProcessCond
            
            self.joint_offset_output_process = OutputProcess(self.data_rep, self.joints_feats_in_dim, self.latent_dim, 21, 3)
            
            if self.args.use_dec_rel_v2:
                self.output_process = OutputProcessObjBaseRawV2(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
            else:
                # OutputProcessObjBaseRaw ## output process for basejtsrel #
                self.output_process = OutputProcessObjBaseRaw(self.data_rep, self.latent_dim, not_cond_base=self.args.not_cond_base)
                ##### ==== input process, communications, output process for rel, dists ==== #####
            
        if self.diff_basejtse:
            ### input process obj base ###
            # construct input_process_e # 
            # self.input_feats_e = 21 * (3 + 3 + 3 + 1 + 1)
            self.input_feats_e = 21 * (3 + 3 + 1 + 1)
            self.input_process_e = InputProcessObjBase(self.data_rep, self.input_feats_e+self.gru_emb_dim, self.latent_dim)
            
            self.sequence_pos_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            self.emb_trans_dec = emb_trans_dec
            # # single layer transformers # ## predict relative position for each base point?  # existing model 
            # if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_e = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e,
                                                        num_layers=self.num_layers)
            
            print("TRANS_ENC init")
            # logvar_seqTransEncoder_e, 
            seqTransEncoderLayer_e_logvar = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.logvar_seqTransEncoder_e = nn.TransformerEncoder(seqTransEncoderLayer_e_logvar,
                                                        num_layers=self.num_layers)
            
            # 
            # elif self.arch == 'trans_dec':
            #     print("TRANS_DEC init")
            #     seqTransDecoderLayer_e = nn.TransformerDecoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=activation)
            #     self.seqTransDecoder_e = nn.TransformerDecoder(seqTransDecoderLayer_e,
            #                                                 num_layers=self.num_layers)
            # elif self.arch == 'gru': ## arch ##
            #     print("GRU init")
            #     self.gru_e = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            # else:
            #     raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
            # tiemstep # # timestep embedding e # Embed timestep e #
            self.embed_timestep_e = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            self.sequence_pos_denoising_encoder_e = PositionalEncoding(self.latent_dim, self.dropout)
            # basejtsrel_denoising_embed_timestep, basejtsrel_denoising_seqTransEncoder, output_process #
            self.basejtse_denoising_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_e)
            
            # /data1/sim/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_/model000073000.pt # 
            # /data1/sim/mdm/save/predoffset_stdscale_bsz_10_pred_diff_realbaesjtsrel_nonorm_std_for_norm_train_enc_with_diff_latents_/model000073000.pt # nonorm std
            
            if self.args.use_ours_transformer_enc: # our transformer encoder #
                self.basejtse_denoising_seqTransEncoder = model_utils.TransformerEncoder(
                    hidden_size=self.latent_dim,
                    fc_size=self.ff_size,
                    num_heads=self.num_heads,
                    layer_norm=True,
                    num_layers=self.num_layers,
                    dropout_rate=0.2,
                    re_zero=True,
                    memory_efficient=False,
                ) ### basejtse_denoising_seqTransEncoder ###
            else:
                basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
                                                                num_layers=self.num_layers)

            # basejtse_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
            #                                                     nhead=self.num_heads,
            #                                                     dim_feedforward=self.ff_size,
            #                                                     dropout=self.dropout,
            #                                                     activation=self.activation)
            
            # self.basejtse_denoising_seqTransEncoder = nn.TransformerEncoder(basejtse_seqTransEncoderLayer,
            #                                                 num_layers=self.num_layers)
            
            # seq_len x bsz x dim 
            if self.args.const_noise:
                # 1) max pool latents over the sequence
                # 2) transform the pooled latnets via the linear layer
                self.basejtse_denoising_seqTransEncoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 2), nn.ReLU(),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
        
            # self.output_process_e = OutputProcessObjBaseV3(self.data_rep, self.latent_dim)
            self.output_process_e = OutputProcessObjBaseERaw(self.data_rep, self.latent_dim)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def set_enc_to_eval(self):
        # jts: joint_sequence_input_process, joint_sequence_pos_encoder, joint_sequence_seqTransEncoder, joint_sequence_logvar_seqTransEncoder
        # basejtsrel: input_process, sequence_pos_encoder, seqTransEncoder, logvar_seqTransEncoder, 
        # basejtse: input_process_e, sequence_pos_encoder_e, seqTransEncoder_e, logvar_seqTransEncoder_e, 
        if self.diff_jts:
            self.joint_sequence_input_process.eval()
            self.joint_sequence_pos_encoder.eval()
            self.joint_sequence_seqTransEncoder.eval()
            self.joint_sequence_logvar_seqTransEncoder.eval()
        if self.diff_basejtse:
            self.input_process_e.eval()
            self.sequence_pos_encoder_e.eval()
            self.seqTransEncoder_e.eval()
            self.logvar_seqTransEncoder_e.eval()
        if self.diff_basejtsrel:
            self.input_process.eval()
            self.sequence_pos_encoder.eval()
            self.seqTransEncoder.eval() # seqTransEncoder, logvar_seqTransEncoder
            self.logvar_seqTransEncoder.eval() 
            

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights( # encode 
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    # frre sample from the model? ##
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts #
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'motion_ours'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else: ## 
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    
    def dec_jts_only_fr_latents(self, latents_feats):
        joint_seq_output = self.joint_sequence_output_process(latents_feats)  # [bs, njoints, nfeats, nframes]
        # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
        joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
        
        ## joints seq outputs ##
        diff_jts_dict = {
            "joint_seq_output": joint_seq_output,
            "joints_seq_latents": latents_feats,
        }
        return diff_jts_dict
    
    def dec_basejtsrel_only_fr_latents(self, latent_feats, x):
        # basejtsrel_seq_latents_pred_feats
        # avg_jts_seq_latents = latent_feats[0:1, ...]
        other_basejtsrel_seq_latents = latent_feats # [1:, ...]
        
        # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
        # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
        # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
        # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
        
        joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
            
        joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
        basejtsrel_dec_out = {
            # 'avg_jts_outputs': avg_jts_outputs,
            'basejtsrel_output': joints_offset_output,
        }
        return basejtsrel_dec_out

    # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps):
    def dec_latents_to_joints_with_t(self, input_latent_feats, x, timesteps):
        # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        # joints_seq_latents: seq x bs x d --> perturbed joitns_seq_latents \in [-1, 1] ##
        # def dec_latents_to_joints_with_t(self, joints_seq_latents, timesteps): ## timesteps #
        ## positional encoding for denoising ##
        # rt_dict = {
            # 'joint_seq_output': joint_seq_output,
            # 'rel_base_pts_outputs': rel_base_pts_outputs,
        # }
        rt_dict = {}
        if self.diff_jts:
            ####### input latent feats #######
            joints_seq_latents = input_latent_feats["joints_seq_latents"]
            if not self.args.without_dec_pos_emb:
                joints_seq_latents = self.joint_sequence_denoising_pos_encoder(joints_seq_latents)
                
            # ### GET joints seq time embeddings ### ### embed time stamps ###
            # joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
            # joints_seq_latents = torch.cat(
            #     [joints_seq_time_emb, joints_seq_latents], dim=0
            # )
            # joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:] # seq x bs x d
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_time_emb = joints_seq_time_emb.repeat(joints_seq_latents.size(0), 1, 1).contiguous()
                joints_seq_latents = joints_seq_latents + joints_seq_time_emb
                
                if self.args.use_ours_transformer_enc:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)
            else:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                joints_seq_time_emb = self.joint_sequence_embed_timestep(timesteps)
                joints_seq_latents = torch.cat(
                    [joints_seq_time_emb, joints_seq_latents], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## mdm ours ##
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    joints_seq_latents = joints_seq_latents.permute(1, 0, 2)[1:]
                else:
                    joints_seq_latents = self.joint_sequence_denoising_seqTransEncoder(joints_seq_latents)[1:]
                
            # joints_seq_latents: seq_len x bsz x latent_dim #
            if self.args.const_noise:
                seq_len = joints_seq_latents.size(0)
                # if self.args.const_noise:
                joints_seq_latents, _ = torch.max(joints_seq_latents, dim=0, keepdim=True)
                joints_seq_latents = self.glb_denoising_latents_trans_layer(joints_seq_latents) # seq_len x bsz x latent_dim
                joints_seq_latents = joints_seq_latents.repeat(seq_len, 1, 1).contiguous()
                
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                joints_seq_latents = input_latent_feats["joints_seq_latents_enc"]
            
            
            # bsz x ws x nnj x 3 #
            joint_seq_output = self.joint_sequence_output_process(joints_seq_latents)  # [bs, njoints, nfeats, nframes]
            # bsz x ws x nnj x nnfeats # --> joints_seq_outputs #
            joint_seq_output = joint_seq_output.permute(0, 3, 1, 2).contiguous()
            
            diff_jts_dict = {
                "joint_seq_output": joint_seq_output,
                "joints_seq_latents": joints_seq_latents,
            }
        else:
            diff_jts_dict = {}
            
        if self.diff_basejtsrel:
            rel_base_pts_outputs = input_latent_feats["rel_base_pts_latents"]
            
            if self.args.train_enc:
                other_basejtsrel_seq_latents = input_latent_feats["rel_base_pts_latents_enc"]
            else:
                # else:
                if not self.args.without_dec_pos_emb: # without dec pos embedding
                    # avg_jts_inputs = rel_base_pts_outputs_mean[0:1, ...]
                    other_rel_base_pts_outputs = rel_base_pts_outputs
                    other_rel_base_pts_outputs = self.sequence_pos_denoising_encoder(other_rel_base_pts_outputs)
                    rel_base_pts_outputs_mean = other_rel_base_pts_outputs
                else:
                    rel_base_pts_outputs_mean = rel_base_pts_outputs
                    
                if self.args.deep_fuse_timeemb:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_time_emb = basejtsrel_time_emb.repeat(rel_base_pts_outputs_mean.size(0), 1, 1).contiguous()
                    basejtsrel_seq_latents = rel_base_pts_outputs_mean + basejtsrel_time_emb ### time embeddings and relbaseptsoutputs 
                    
                    if self.args.use_ours_transformer_enc:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)
                else:
                    ## denoising process ###
                    ## GET joints seq time embeddings ### ### embed time stamps ###
                    basejtsrel_time_emb = self.basejtsrel_denoising_embed_timestep(timesteps)
                    basejtsrel_seq_latents = torch.cat(
                        [basejtsrel_time_emb, rel_base_pts_outputs_mean], dim=0
                    )
                    
                    if self.args.use_ours_transformer_enc: ## mdm ours ##
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                        basejtsrel_seq_latents = basejtsrel_seq_latents.permute(1, 0, 2)[1:]
                    else:
                        basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
                # basejtsrel_seq_latents_pred_feats
                # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
                other_basejtsrel_seq_latents = basejtsrel_seq_latents # [1:, ...] # 
                
            
            
            joints_offset_output =  self.joint_offset_output_process(other_basejtsrel_seq_latents)
            
            joints_offset_output = joints_offset_output.permute(0, 3, 1, 2)
            
                
            #### 
            # avg_jts_seq_latents = basejtsrel_seq_latents[0:1, ...]
            # other_basejtsrel_seq_latents = basejtsrel_seq_latents[1:, ...]
            
            # avg_jts_outputs = self.avg_joint_sequence_output_process(avg_jts_seq_latents)  # bsz x njoints x nfeats x 1
            # avg_jts_outputs = avg_jts_outputs.squeeze(-1) # bsz x nnjoints x 1 --> avg joints here #
            # # basejtsrel_seq_latents = self.basejtsrel_denoising_seqTransEncoder(basejtsrel_seq_latents)[1:]
            # basejtsrel_output = self.output_process(other_basejtsrel_seq_latents, x)
            diff_basejtsrel_dict = {
                "joints_offset_output": joints_offset_output, # bsz x seq x nnjts x 3 #
                'joints_denoised_latents': other_basejtsrel_seq_latents, # seq x bsz x dim # 
                # "basejtsrel_seq_latents": basejtsrel_seq_latents,
                # "avg_jts_outputs": avg_jts_outputs,
            }
        else:
            diff_basejtsrel_dict = {}
        
        if self.diff_basejtse:
            # e_disp_rel_to_base_along_normals = input_latent_feats['e_disp_rel_to_base_along_normals']
            # e_disp_rel_to_baes_vt_normals = input_latent_feats['e_disp_rel_to_baes_vt_normals'] 
            base_jts_e_feats = input_latent_feats['base_jts_e_feats'] # seq x bs x d --> e feats 
            
            if not self.args.without_dec_pos_emb:
                # rel_base_pts_outputs = self.sequence_pos_denoising_encoder(rel_base_pts_outputs)
                base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            
            
            if self.args.deep_fuse_timeemb:
                ## denoising process ###
                ## GET joints seq time embeddings ### ### embed time stamps ###
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_time_emb = base_jts_e_time_emb.repeat(base_jts_e_feats.size(0), 1, 1).contiguous()
                base_jts_e_feats = base_jts_e_feats + base_jts_e_time_emb
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)
            else:
                #### Decode e_along_normals and e_vt_normals ####
                base_jts_e_time_emb = self.embed_timestep_e(timesteps)
                base_jts_e_feats = torch.cat(
                    [base_jts_e_time_emb, base_jts_e_feats], dim=0
                )
                
                if self.args.use_ours_transformer_enc: ## transformer encoder ##
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats.permute(1, 0, 2), set_attn_to_none=self.args.set_attn_to_none) 
                    base_jts_e_feats = base_jts_e_feats.permute(1, 0, 2)[1:]
                else:
                    base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
                
            ### sequence latents ###
            if self.args.train_enc: # trian enc for seq latents ###
                base_jts_e_feats = input_latent_feats["base_jts_e_feats_enc"]
            
            # base_jts_e_feats = self.sequence_pos_denoising_encoder_e(base_jts_e_feats)
            #### Decode e_along_normals and e_vt_normals ####
            ### bae jts e embeddings feats ###
            # base_jts_e_time_emb = self.embed_timestep_e(timesteps)
            # base_jts_e_feats = torch.cat(
            #     [base_jts_e_time_emb, base_jts_e_feats], dim=0
            # )
            # base_jts_e_feats = self.basejtse_denoising_seqTransEncoder(base_jts_e_feats)[1:]
            
            
            base_jts_e_output = self.output_process_e(base_jts_e_feats, x)
            # dec_e_along_normals: bsz x (ws - 1) x nnb x nnj # 
            dec_e_along_normals = base_jts_e_output['dec_e_along_normals']
            dec_e_vt_normals = base_jts_e_output['dec_e_vt_normals']
            dec_e_along_normals = dec_e_along_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            dec_e_vt_normals = dec_e_vt_normals.contiguous().permute(0, 1, 3, 2).contiguous()
            #### Decode e_along_normals and e_vt_normals ####
            diff_basejtse_dict = {
                'dec_e_along_normals': dec_e_along_normals,
                'dec_e_vt_normals': dec_e_vt_normals, 
                'base_jts_e_feats': base_jts_e_feats,
            }
        else:
            diff_basejtse_dict = {}
    
        rt_dict = {}
        rt_dict.update(diff_jts_dict)
        rt_dict.update(diff_basejtsrel_dict)
        rt_dict.update(diff_basejtse_dict)

        ### rt_dict --> rt_dict of joints, rel ###
        return rt_dict
        
        # return joint_seq_output, joints_seq_latents
        
    def reparameterization(self, val_mean, val_var):
        val_noise = torch.randn_like(val_mean)
        val_sampled = val_mean + val_noise * val_var ### sample the value 
        if self.args.rnd_noise:
            val_sampled = val_noise
        return val_sampled

    def forward(self, x, timesteps):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # joint_sequence_input_process; joint_sequence_pos_encoder; joint_sequence_seqTransEncoder; joint_sequence_seqTransDecoder; joint_sequence_embed_timestep; joint_sequence_output_process #
        # bsz, nframes, nnj = x['pert_rhand_joints'].shape[:3]
        # pert_rhand_joints = x['pert_rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        bsz, nframes, nnj = x['rhand_joints'].shape[:3]
        pert_rhand_joints = x['rhand_joints'] # bsz x ws x nnj x 3 # bsz x nnj x 3 x ws
        
        base_pts = x['base_pts'] ### bsz x nnb x 3 ###
        base_normals = x['base_normals'] ### bsz x nnb x 3 ### --> base normals ###
        
        # base_normals # ## 
        
        rt_dict = {}
        
        ## # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
        if self.diff_basejtse:
            ### Embed physicss quantities ###
            # e_disp_rel_to_base_along_normals: bsz x (ws - 1) x nnj x nnb #
            # e_disp_rel_to_baes_vt_normals: bsz x (ws - 1) x nnj x nnb #
            e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
            e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
            
            nnb = base_pts.size(1)
            disp_ws = e_disp_rel_to_base_along_normals.size(1) ### --> base normals ###
            base_pts_disp_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            base_normals_disp_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, disp_ws, nnj, 1, 1).contiguous()
            # bsz x (ws - 1) x nnj x nnb x (3 + 3 + 1 + 1)
            base_pts_normals_e_in_feats = torch.cat(
                [base_pts_disp_exp, base_normals_disp_exp, e_disp_rel_to_base_along_normals.unsqueeze(-1), e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1 
            )
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.permute(0, 1, 3, 2, 4).contiguous()
            # bsz x (ws - 1) x nnb x (nnj x (xxx feats_dim))
            base_pts_normals_e_in_feats = base_pts_normals_e_in_feats.view(bsz, disp_ws, nnb, -1).contiguous()
            
            ## input process ##
            base_jts_e_feats = self.input_process_e(base_pts_normals_e_in_feats)
            base_jts_e_feats = self.sequence_pos_encoder_e(base_jts_e_feats)
            
            ## seq transformation for e ##
            base_jts_e_feats_mean = self.seqTransEncoder_e(base_jts_e_feats) ## mean, mdm_ours ##
            # print(f"base_jts_e_feats: {base_jts_e_feats.size()}")
            ### Embed physicss quantities ###
            
            ### calculate logvar, mean, and feats ###
            base_jts_e_feats_logvar = self.logvar_seqTransEncoder_e(base_jts_e_feats)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            base_jts_e_feats_var = torch.exp(base_jts_e_feats_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            base_jts_e_feats = self.reparameterization( base_jts_e_feats_mean, base_jts_e_feats_var)
            
            rt_dict['base_jts_e_feats'] = base_jts_e_feats
            rt_dict['base_jts_e_feats_mean'] = base_jts_e_feats_mean
            rt_dict['base_jts_e_feats_logvar'] = base_jts_e_feats_logvar # log_var #
        
        
        if self.diff_jts:
            # base_pts_normal
            ### InputProcess ###
            pert_rhand_joints_trans = pert_rhand_joints.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x ws #
            rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints_trans) #  [seqlen, bs, d]
            ### InputProcessObjBase ###
            # rhand_joints_feats = self.joint_sequence_input_process(pert_rhand_joints)
            ### === Encode input joint sequences === ###
            # bs, njoints, nfeats, nframes = x.shape
            # rhand_joints_emb = self.joint_sequence_embed_timestep(timesteps)  # [1, bs, d]
            # if self.arch == 'trans_enc':
            xseq = rhand_joints_feats # [seqlen+1, bs, d]
            xseq = self.joint_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            joint_seq_output_mean = self.joint_sequence_seqTransEncoder(xseq) # [1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            ### calculate logvar, mean, and feats ###
            joint_seq_output_logvar = self.joint_sequence_logvar_seqTransEncoder(xseq)
            # # logvar_seqTransEncoder_e, logvar_seqTransEncoder, joint_sequence_logvar_seqTransEncoder # #
            joint_seq_output_var = torch.exp(joint_seq_output_logvar) # seq x bs x d --> encodeing and decoding
            ## base_jts_e_feats: seqlen x bs x d --> val latents ##
            joint_seq_output = self.reparameterization(joint_seq_output_mean, joint_seq_output_var)
            
            rt_dict['joint_seq_output'] = joint_seq_output
            # rt_dict['joint_seq_output'] = joint_seq_output_mean
            rt_dict['joint_seq_output_mean'] = joint_seq_output_mean
            rt_dict['joint_seq_output_logvar'] = joint_seq_output_logvar
            
        if self.diff_basejtsrel:
            # self.avg_joints_sequence_input_process, self.avg_joint_sequence_output_process #
            ### === Encode input joint sequences === ###
            # base_normals = x['base_normals'] # bsz x nnb x 3
            # avg_joints_sequence = x['avg_joints_sequence'] # bsz x nnjoints x 3 ### -> for joint sequence #
            
            joints_offset_sequence = x['pert_joints_offset_sequence'] # bsz x nf x nnj x 3
            joints_offset_sequence = joints_offset_sequence.permute(0, 2, 3, 1).contiguous()
            joints_offset_feats = self.joints_offset_input_process(joints_offset_sequence) # nf x bsz x dim
            
            
            rel_base_pts_feats_pos_embedding = self.sequence_pos_encoder(joints_offset_feats)
            # outputs rel base jts encoded latents ##
            # seqTransEncoder, logvar_seqTransEncoder
            # rel_base_pts_outputs_mean = self.basejtsrel_denoising_seqTransEncoder(rel_base_pts_feats_pos_embedding)
            # ### calculate logvar, mean, and feats ###
            # rel_base_pts_outputs_logvar = self.joint_sequence_logvar_seqTransEncoder(rel_base_pts_outputs)
            
            # if self.args.not_diff_avgjts: # not use diff avgjts ##
            rel_base_pts_feats = rel_base_pts_feats_pos_embedding
            # seqTransEncoder, logvar_seqTransEncoder #
            rel_base_pts_outputs_mean = self.seqTransEncoder(rel_base_pts_feats)
            
            if self.args.use_sigmoid and ((not self.args.use_vae) or self.args.kl_weights == 0.):
                rel_base_pts_outputs = (torch.sigmoid(rel_base_pts_outputs_mean) - 0.5) * 2.
                rt_dict['rel_base_pts_latents'] = rel_base_pts_outputs
            elif self.args.use_vae:
                rel_base_ptS_outputs_logvar = self.logvar_seqTransEncoder(rel_base_pts_feats)
                rel_base_pts_outputs_var = torch.exp(rel_base_ptS_outputs_logvar)
                
                rel_base_pts_outputs = self.reparameterization(rel_base_pts_outputs_mean, rel_base_pts_outputs_var)
                rt_dict['rel_base_pts_latents'] = rel_base_pts_outputs
                rt_dict['rel_base_pts_latents_mean'] = rel_base_pts_outputs_mean
                rt_dict['rel_base_pts_latents_logvar'] = rel_base_ptS_outputs_logvar
            else:
                rel_base_pts_outputs = rel_base_pts_outputs_mean
                rt_dict['rel_base_pts_latents'] = rel_base_pts_outputs #
                
            
            # rt_dict['rel_base_pts_latents'] = rel_base_pts_outputs # rel_base_pts_latents: seq x bsz x dim #
                

        ## for construct rt_dict here ##
        # joint_seq_output = joint_seq_output # scale to [-1, 1]
        
        # rt_dict = {c
        #     'joint_seq_output': joint_seq_output,
        #     'rel_base_pts_outputs': rel_base_pts_outputs,
        #     'base_jts_e_feats': base_jts_e_feats,
        # }
        
        return rt_dict
        

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)



### LayerNorm layer ###
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
       x = (x - x.mean(dim=self.dim, keepdim=True)) / torch.sqrt(x.var(dim=self.dim, keepdim=True)+self.eps)
       return x

### PointNet block ###
class STPointNetBlock(nn.Module):
    def __init__(self, input_dim, layer_dims, layernorm=True, global_feat=False,
                 transposed=False):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.layernorm = layernorm
        self.global_feat = global_feat
        self.transposed = transposed
        self.activation = nn.ReLU()

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, input_dim)

        self.conv_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx], layer_dims[idx + 1], 1))
        if layernorm:
            self.ln = LayerNorm(dim=1)

        if not global_feat:
            self.last_conv = nn.Conv1d(layer_dims[-1]*2, layer_dims[-1]*2, 1)

    def forward(self, x):
      # bsz x nnframes x nnpts x nfeats 
        if self.transposed:
            batch_size, window_size, num_points = x.size(0), x.size(2), x.size(3)
            x = x.view(batch_size, self.input_dim, window_size*num_points)
        else:
            batch_size, window_size, num_points = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 3, 1, 2).view(batch_size, self.input_dim, window_size*num_points)

        x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False: ## bsz x feats_dim x ws x nn_pts ##
            local_features = x.view(batch_size, -1, window_size, num_points)

        ## use conv layers as activations ##
        for idx in range(1, len(self.conv_layers) - 1):
            x = self.activation(self.conv_layers[idx](x))
            # 

        ## conv layers ##
        x = self.conv_layers[-1](x)

        x = x.view(-1, self.layer_dims[-1], window_size, num_points)
        x = torch.max(x, 3)[0] ## global communication --> 

        if self.global_feat:
            if self.layernorm:
                return self.ln(x)
            return x

        x = x.view(-1, self.layer_dims[-1], window_size, 1).repeat(1, 1, 1, num_points)

        x = torch.cat((x, local_features), dim=1)
        x = x.view(batch_size, -1, window_size*num_points) + self.last_conv(x.view(batch_size, -1, window_size*num_points))
        x = x.view(batch_size, -1, window_size, num_points)
        
        if self.layernorm:
            return self.ln(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)



class InputProcessObjV7(nn.Module): # inputobjbase
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        # self.pnpp_conv_net = PointnetPP(input_feats)
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(),  # input_feats --- dimension of input feats #
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),  # 
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # self.pts_glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    # load the trainable copy and use the trainable copy for joints encoding #
    def forward(self, x_pos): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3)
        bsz, nf, nnb = x_pos.size()[:3] # n
        
        # bsz x nf x nnb x nnfeats #
        # x_exp = x.view(bsz * nf, nnb, -1).contiguous()
        # x_pos_exp = x_po
        x_pos_exp = x_pos.view(bsz * nf, nnb, -1).contiguous() # x_posexp for the 
        
        x_pts_emb = self.pts_feats_encoding_net(x_pos_exp)
        x_glb_emb, _ = torch.max(x_pts_emb, dim=-2) #
        
        
        x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # (bsz x nf) x latent_dim as the global embedding #
        x_glb_emb = x_glb_emb.view(bsz, nf, -1).contiguous().permute(1, 0, 2) # nf x bsz x latent_dim #
        
        return x_glb_emb


# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjV6(nn.Module): # inputobjbase  # InputProcessObjV6(self, data_rep, input_feats, latent_dim, layernorm=True)
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        # self.pnpp_conv_net = PointnetPP(input_feats)
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.pts_glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
    # load the trainable copy and use the trainable copy for joints encoding #
    def forward(self, x_pos): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3)
        bsz, nf, nnb = x_pos.size()[:3] # n
        
        # bsz x nf x nnb x nnfeats 
        # x_exp = x.view(bsz * nf, nnb, -1).contiguous()
        x_pos_exp = x_pos.view(bsz * nf, nnb, -1).contiguous()
        
        x_pts_emb = self.pts_feats_encoding_net(x_pos_exp)
        x_glb_emb, _ = torch.max(x_pts_emb, dim=-2) #
        
        
        x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # (bsz x nf) x latent_dim as the global embedding #
        x_glb_emb = x_glb_emb.view(bsz, nf, -1).contiguous().permute(1, 0, 2) # nf x bsz x latent_dim #
        
        return x_glb_emb


# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObj(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        ## pnpp 
        ### pnpp_conv_net  #### for base points encoding here ####
        self.pnpp_conv_net = PointnetPP(input_feats) # for base points encoding #
        
        # self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
        # self.glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )

        # self.pts_glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    def forward(self, x, x_pos): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3)
        bsz, nf, nnb = x.size()[:3] # n
        
        # bsz x nf x nnb x nnfeats 
        x_exp = x.view(bsz * nf, nnb, -1).contiguous()
        x_pos_exp = x_pos.view(bsz * nf, nnb, -1).contiguous()
        
        # pnpp conv net #
        x_pts_emb, x_pos_exp = self.pnpp_conv_net(None, x_pos_exp)
        # x_pts_emb, x_pos_exp = self.pnpp_conv_net(x_exp, x_pos_exp)
        
        x_pts_emb = x_pts_emb.view(bsz, nf, nnb, -1)
        
        # print(f"x_pts_emb: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) 
        # print(f"x_pts_emb after permutation: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # 
        
        return x_pts_emb
    

# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBase(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        
        ## input process obj base ##
        self.embedding_pn_blk = nn.Sequential( # nnb --> 21
          STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
          STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
          STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
          STPointNetBlock(256, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=True, global_feat=True, layernorm=layernorm)
        )
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        # #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        # #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        # #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(input_feats, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=True, layernorm=layernorm)
        # )
        # latent_dim #### basepts basenormals and the same vectors as well so it is not very friendly to use such representations right? 
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # x_emb: bsz x latent_dim x nn_frames ## x_emb ##
        # x: bsz x nnframes x nnjoints x nfeats #
        x_emb = self.embedding_pn_blk(
          x # input_feats ## input feats ##
        )
        # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        x_emb  = x_emb.permute(2, 0, 1).contiguous() # 
        return x_emb
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d]
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError


# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV3(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(256, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=True, global_feat=True, layernorm=layernorm)
        # )
        
        # self.embedding_pn_blk = nn.Linear(self.input_feats, self.latent_dim)
        
        ## input process obj base ##
        self.embedding_pn_blk = nn.Sequential( # nnb --> 21
          STPointNetBlock(input_feats, [self.latent_dim // 2, self.latent_dim // 2, self.latent_dim // 2], transposed=False, layernorm=layernorm),
        #   STPointNetBlock(self.latent_dim, [self.latent_dim // 2, self.latent_dim // 2, self.latent_dim // 2], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
          STPointNetBlock(self.latent_dim, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=True, global_feat=True, layernorm=layernorm)
        )
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        # #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        # #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        # #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(input_feats, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=True, layernorm=layernorm)
        # )
        # latent_dim #### basepts basenormals and the same vectors as well so it is not very friendly to use such representations right? 
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # x_emb: bsz x latent_dim x nn_frames ## x_emb ##
        # x: bsz x nnframes x nnjoints x nfeats # bsz x nnf x 1 x nndim 
        # x_emb = self.embedding_pn_blk(
        #   x # input_feats ## input feats ##
        # )
        # # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        # x_emb  = x_emb.squeeze(-2).permute(1, 0, 2).contiguous() # 
        x_emb = self.embedding_pn_blk(
          x # input_feats ## input feats ##
        )
        # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        x_emb  = x_emb.permute(2, 0, 1).contiguous() # 
        return x_emb
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d]
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError



# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV7(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        # args.dgcnn_out_dim = 128
        #     args.dgcnn_in_feat_dim = 6 if self.args.input_normal else 3
        class dummy:
            def __init__(self):
                self.dgcnn_out_dim = latent_dim
                self.dgcnn_in_feat_dim = input_feats
                self.dgcnn_layers = 3
                self.backbone = 'DGCNN'
        dgcnn_opt = dummy()
        
        self.dgcnn_conv_net = PrimitiveNet(dgcnn_opt)
        
        # self.pnpp_conv_net = PointnetPP(input_feats)
        
        # self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
        # self.glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )

        # self.pts_glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3)
        bsz, nf, nnb = x.size()[:3] # n
        
        # bsz x nf x nnb x nnfeats 
        x_exp = x.view(bsz * nf, nnb, -1).contiguous()
        
        x_pts_emb = self.dgcnn_conv_net(x_exp, x_exp)
        
        # x_pos_exp = x_pos.view(bsz * nf, nnb, -1).contiguous()
        
        
        
        # x_pts_emb, x_pos_exp = self.pnpp_conv_net(x_exp, x_pos_exp)
        
        x_pts_emb = x_pts_emb.view(bsz, nf, nnb, -1)
        
        # print(f"x_pts_emb: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) 
        # print(f"x_pts_emb after permutation: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # 
        # print(f"x_pts_emb after view: {x_pts_emb.size()}")
        
        # x_pts_emb = self.pts_feats_encoding_net( # input # noisy --- too noisy #
        #     x
        # )
        # x_glb_emb, _ = torch.max(x_pts_emb, dim=2, keepdim=True)
        # x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # bsz x nf x 1 x latnet_dim
        # x_pts_emb = torch.cat(
        #     [x_pts_emb, x_glb_emb.repeat(1, 1, nnb, 1)], dim=-1
        # )
        # x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb) # bsz x nf x nn_base_pts x latent_dim #
        # x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) # nf x bsz x nn_base_pts x latent_dim #
        
        # x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # nf x (bsz x nn_base_pts) x latent_dim #
        
        return x_pts_emb
  

# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV6(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        self.pnpp_conv_net = PointnetPP(input_feats)
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.pts_glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    def forward(self, x, x_pos): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3)
        bsz, nf, nnb = x.size()[:3] # n
        
        # bsz x nf x nnb x nnfeats 
        x_exp = x.view(bsz * nf, nnb, -1).contiguous()
        x_pos_exp = x_pos.view(bsz * nf, nnb, -1).contiguous()
        
        
        
        x_pts_emb, x_pos_exp = self.pnpp_conv_net(x_exp, x_pos_exp)
        
        x_pts_emb = x_pts_emb.view(bsz, nf, nnb, -1)
        
        # print(f"x_pts_emb: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) 
        # print(f"x_pts_emb after permutation: {x_pts_emb.size()}")
        
        x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # 
        # print(f"x_pts_emb after view: {x_pts_emb.size()}")
        
        # x_pts_emb = self.pts_feats_encoding_net( # input # noisy --- too noisy #
        #     x
        # )
        # x_glb_emb, _ = torch.max(x_pts_emb, dim=2, keepdim=True)
        # x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # bsz x nf x 1 x latnet_dim
        # x_pts_emb = torch.cat(
        #     [x_pts_emb, x_glb_emb.repeat(1, 1, nnb, 1)], dim=-1
        # )
        # x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb) # bsz x nf x nn_base_pts x latent_dim #
        # x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) # nf x bsz x nn_base_pts x latent_dim #
        
        # x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # nf x (bsz x nn_base_pts) x latent_dim #
        
        return x_pts_emb
    

## hand sequence; object shape
# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV5(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.without_glb = without_glb
        self.only_with_glb = only_with_glb
        
        if self.without_glb:
            self.pts_glb_feats_encoding_net = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.pts_glb_feats_encoding_net = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        bsz, nf, nnb = x.size()[:3]
        
        if self.only_with_glb:
            x_pts_emb = self.pts_feats_encoding_net( # input # noisy --- too noisy #
                x
            )
            x_glb_emb, _ = torch.max(x_pts_emb, dim=2, keepdim=True)
            x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # bsz x nf x latent_dim 
            x_glb_emb = x_glb_emb.squeeze(-2)
            x_pts_emb = x_glb_emb.permute(1, 0, 2).contiguous()
        else:
            x_pts_emb = self.pts_feats_encoding_net( # input # noisy --- too noisy #
                x
            )
            if not self.without_glb:
                x_glb_emb, _ = torch.max(x_pts_emb, dim=2, keepdim=True)
                x_glb_emb = self.glb_feats_encoding_net(x_glb_emb) # bsz x nf x 1 x latnet_dim
                x_pts_emb = torch.cat( # 1 
                    [x_pts_emb, x_glb_emb.repeat(1, 1, nnb, 1)], dim=-1
                )
            x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb) # bsz x nf x nn_base_pts x latent_dim #
            x_pts_emb = x_pts_emb.permute(1, 0, 2, 3) # nf x bsz x nn_base_pts x latent_dim #
            
            x_pts_emb = x_pts_emb.contiguous().view(x_pts_emb.size(0), bsz * nnb, -1).contiguous() # nf x (bsz x nn_base_pts) x latent_dim #
        
        return x_pts_emb
    


# # ## x_pts_emb
# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV4(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(256, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=True, global_feat=True, layernorm=layernorm)
        # )
        
        # self.embedding_pn_blk = nn.Linear(self.input_feats, self.latent_dim)
        
        ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        # #   STPointNetBlock(input_feats, [self.latent_dim // 2, self.latent_dim // 2, self.latent_dim // 2], transposed=False, layernorm=layernorm),
        # #   STPointNetBlock(self.latent_dim, [self.latent_dim // 2, self.latent_dim // 2, self.latent_dim // 2], transposed=True, layernorm=layernorm),
        # #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(self.latent_dim, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=False, layernorm=layernorm)
        # )
        
        # embedding pn blk # 
        self.embedding_pn_blk = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        # #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        # #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        # #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(input_feats, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=True, layernorm=layernorm)
        # )
        # latent_dim #### basepts basenormals and the same vectors as well so it is not very friendly to use such representations right? 
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # x_emb: bsz x latent_dim x nn_frames ## x_emb ##
        # x: bsz x nnframes x nnjoints x nfeats # bsz x nnf x 1 x nndim 
        # x_emb = self.embedding_pn_blk(
        #   x # input_feats ## input feats ##
        # )
        # # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        # x_emb  = x_emb.squeeze(-2).permute(1, 0, 2).contiguous() # 
        
        # bae pts..? 
        # bsz x nf x nnj x (3 + nnb x (3 + 3)) 
        bsz, nf, nnj = x.size()[:3]
        # print(f"input_feats: {self.input_feats}, latent_dim: {self.latent_dim}, x: {x.size()}")
        x_emb = self.embedding_pn_blk( # bsz x nf x nnj x latent_dim #
          x # input_feats ## input feats ##
        ) # bsz x nf x nnj
        # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        x_emb = x_emb.view(bsz, nf * nnj, -1).contiguous()
        x_emb = x_emb.permute(1, 0, 2)
        # x_emb  = x_emb.permute(2, 0, 1).contiguous() #  # 
        return x_emb
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d]
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError


# InputProcessObjBase(self, data_rep, input_feats, latent_dim)
class InputProcessObjBaseV2(nn.Module): # inputobjbase 
    def __init__(self, data_rep, input_feats, latent_dim, layernorm=True, glb_feats_trans=False): 
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.glb_feats_trans = glb_feats_trans
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(256, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=True, global_feat=True, layernorm=layernorm)
        # )
        
        # self.embedding_pn_blk = nn.Linear(self.input_feats, self.latent_dim)
        
        ## input process obj base ##
        self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
          STPointNetBlock(input_feats, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=True, layernorm=layernorm)
        )
        
        if self.glb_feats_trans:
            self.glb_feats_trans_blk = nn.Linear(self.latent_dim, self.latent_dim)
        
        # ## input process obj base ##
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        # #   STPointNetBlock(input_feats, [32, 32, 32], transposed=False, layernorm=layernorm),
        # #   STPointNetBlock(64, [64, 64, 64], transposed=True, layernorm=layernorm),
        # #   STPointNetBlock(128, [128, 128, 128], transposed=True, layernorm=layernorm),
        #   STPointNetBlock(input_feats, [self.latent_dim, self.latent_dim, self.latent_dim], transposed=False, global_feat=True, layernorm=layernorm)
        # )
        # latent_dim #### basepts basenormals and the same vectors as well so it is not very friendly to use such representations right? 
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # x_emb: bsz x latent_dim x nn_frames ## x_emb ##
        # x: bsz x nnframes x nnjoints x nfeats # bsz x nnf x 1 x nndim 
        # x_emb = self.embedding_pn_blk(
        #   x # input_feats ## input feats ##
        # )
        # # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        # x_emb  = x_emb.squeeze(-2).permute(1, 0, 2).contiguous() # 
        x_emb = self.embedding_pn_blk(
          x # input_feats ## input feats ##
        )
        # x_emb: bsz x latent_dim x nnf -> nnf x bsz x latent_dim #
        x_emb  = x_emb.permute(2, 0, 1).contiguous() # 
        
        if self.glb_feats_trans:
            x_emb = self.glb_feats_trans_blk(x_emb)
        
        return x_emb
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d]
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError



class OutputProcessObjBaseRawV5(nn.Module):
    def __init__(self, data_rep, latent_dim, not_cond_base=False, out_objbase_v5_bundle_out=False, v5_out_not_cond_base=False, nn_keypoints=21):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        self.v5_out_not_cond_base = v5_out_not_cond_base
        
        if self.not_cond_base:
            self.rel_dec_cond_dim = self.latent_dim
            self.dist_dec_cond_dim = self.latent_dim
        else:
            self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
            self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        # self.use_anchors = use_anchors
        self.nn_keypoints = nn_keypoints
        # if self.use_anchors:
        #     self.nn_keypoints = 
        
        # self.rel_dec_blk = nn.Sequential(
        #     nn.Linear(self.rel_dec_cond_dim,  3,),
        # )
        
        self.out_objbase_v5_bundle_out = out_objbase_v5_bundle_out
        
        if self.out_objbase_v5_bundle_out:
            if self.v5_out_not_cond_base:
                self.rel_dec_blk = nn.Sequential(
                    nn.Linear(self.latent_dim,  self.latent_dim // 2), nn.ReLU(),
                    nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
                )
            else:
                self.rel_dec_blk = nn.Sequential(
                    nn.Linear(self.latent_dim + 3 + 3,  self.latent_dim // 2), nn.ReLU(),
                    nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
                )
        else:
            self.rel_dec_blk = nn.Sequential(
                nn.Linear(self.rel_dec_cond_dim,  3,),
            )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # )
        self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
          self.dist_dec_cond_dim, 1 * self.nn_keypoints
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x): # output 
        # nframes, bs, d = output.shape
        
        # bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4] # pert_rel_base_pts_to_rhand_joints
        bsz, nframes, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].shape[:4] # bsz x nf x nnj x nnb x 3  # nf x nnb x 3 --> noisy input for denoised values #
        # forward the samole # base_pts, base_normals, # 
        # base_pts = x['base_pts'] # bsz x nnb x 3
        base_pts = x['normed_base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        ## 
        # output: bsz x nf x nnj x latent_dim
        
        output = output.view(nframes, bsz, nnb, -1) # nframes x bsz x nnb x latent_dim 
        output = output.permute(1, 0, 2, 3) # bsz x nnf x nnb x latent_dim ### for the output_dim #
        
        if self.out_objbase_v5_bundle_out:
            if self.v5_out_not_cond_base:
                output_exp = output
            else: # otuptu_exp for rel_dec_blk
                base_pts_exp = base_pts.unsqueeze(1).repeat(1, nframes, 1, 1)
                base_normals_exp = base_normals.unsqueeze(1).repeat(1, nframes, 1, 1)
                output_exp = torch.cat( # with input noisy data # ############### denoised latents for each base pts ###
                    [output, base_pts_exp, base_normals_exp], dim=-1
                )
            dec_rel = self.rel_dec_blk(output_exp)
            dec_rel = dec_rel.view(bsz, nframes, nnb, nnj, 3).permute(0, 1, 3, 2, 4).contiguous()
        else:
            # output = output.permute(1, 0, 2)
            # output = output.view(bsz, nframes, nnj, -1).contiguous() # bsz x nf x nnj x (decoded_latent_dim) # 
            output = output.unsqueeze(2).repeat(1, 1, nnj, 1, 1).contiguous()
            # bsz x nnframes x d #  # 
            # output = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1).contiguous()
            base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
            base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
            # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
            
            # if self.not_cond_base:
            #     output_exp = output
            # else:
            output_exp = torch.cat( # with input noisy data
                [output, base_pts_exp, base_normals_exp, x['pert_rel_base_pts_to_rhand_joints']], dim=-1
            )
            dec_rel = self.rel_dec_blk(output_exp) # bsz x nnframes x nnb x (21 * 3) --> decoded relative positions #
            dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
        #   'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        



class OutputProcessObjBaseRawV4(nn.Module):
    def __init__(self, data_rep, latent_dim, not_cond_base=False):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        if self.not_cond_base:
            self.rel_dec_cond_dim = self.latent_dim
            self.dist_dec_cond_dim = self.latent_dim
        else:
            self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
            self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        
        self.rel_dec_blk = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  3,),
        )
        # self.rel_dec_blk = nn.Sequential(
        #     # nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
        #     # nn.Linear(512, 1024,), nn.ReLU(),
        #     # nn.Linear(1024, 512,), nn.ReLU(),
        #     # nn.Linear(512,  3 * 21,),
            
        #     # nn.Linear(, 512,), nn.ReLU(),
        #     # nn.Linear(512, 1024,), nn.ReLU(),
        #     # nn.Linear(1024, 512,), nn.ReLU(),
        #     nn.Linear(self.rel_dec_cond_dim,  3 * 21,),
        # )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # )
        self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
          self.dist_dec_cond_dim, 1 * 21
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x): # output 
        # nframes, bs, d = output.shape
        
        # bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4] # pert_rel_base_pts_to_rhand_joints
        bsz, nframes, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].shape[:4] # bsz x nf x nnj x nnb x 3  # nf x nnb x 3 --> noisy input for denoised values #
        # forward the samole 
        # base_pts = x['base_pts'] # bsz x nnb x 3
        base_pts = x['normed_base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        ## 
        # output: bsz x nf x nnj x latent_dim
        output = output.permute(1, 0, 2)
        output = output.view(bsz, nframes, nnj, -1).contiguous() # bsz x nf x nnj x (decoded_latent_dim) # 
        output = output.unsqueeze(-2).repeat(1, 1, 1, nnb, 1).contiguous()
        # bsz x nnframes x d #  # 
        # output = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1).contiguous()
        base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
        
        # if self.not_cond_base:
        #     output_exp = output
        # else:
        output_exp = torch.cat( # with input noisy data
            [output, base_pts_exp, base_normals_exp, x['pert_rel_base_pts_to_rhand_joints']], dim=-1
        )
        dec_rel = self.rel_dec_blk(output_exp) # bsz x nnframes x nnb x (21 * 3) --> decoded relative positions #
        dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
        #   'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output




class OutputProcessObjBaseRawV3(nn.Module):
    def __init__(self, data_rep, latent_dim, not_cond_base=False):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        if self.not_cond_base:
            self.rel_dec_cond_dim = self.latent_dim
            self.dist_dec_cond_dim = self.latent_dim
        else:
            self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
            self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        
        self.rel_dec_blk = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  3,),
        )
        # self.rel_dec_blk = nn.Sequential(
        #     # nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
        #     # nn.Linear(512, 1024,), nn.ReLU(),
        #     # nn.Linear(1024, 512,), nn.ReLU(),
        #     # nn.Linear(512,  3 * 21,),
            
        #     # nn.Linear(, 512,), nn.ReLU(),
        #     # nn.Linear(512, 1024,), nn.ReLU(),
        #     # nn.Linear(1024, 512,), nn.ReLU(),
        #     nn.Linear(self.rel_dec_cond_dim,  3 * 21,),
        # )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # )
        self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
          self.dist_dec_cond_dim, 1 * 21
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x): # output 
        nframes, bs, d = output.shape
        
        # bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4] # pert_rel_base_pts_to_rhand_joints
        bsz, nframes, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].shape[:4] # bsz x nf x nnj x nnb x 3 
        # forward the samole 
        # base_pts = x['base_pts'] # bsz x nnb x 3
        base_pts = x['normed_base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        ## 
        
        # bsz x nnframes x d #  # 
        
        output = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1).contiguous()
        base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
        
        # if self.not_cond_base:
        #     output_exp = output
        # else:
        output_exp = torch.cat( # with input noisy data
            [output, base_pts_exp, base_normals_exp, x['pert_rel_base_pts_to_rhand_joints']], dim=-1
        )
        dec_rel = self.rel_dec_blk(output_exp) # bsz x nnframes x nnb x (21 * 3) --> decoded relative positions #
        dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
        #   'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output


class OutputProcessObjBaseRawV2(nn.Module):
    def __init__(self, data_rep, latent_dim, not_cond_base=False, finetune_with_cond=False):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        if self.not_cond_base:
            self.rel_dec_cond_dim = self.latent_dim
            self.dist_dec_cond_dim = self.latent_dim
        else:
            self.rel_dec_cond_dim = self.latent_dim + 3 + 3
            self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        ### for multiple settings###
        # if self.not_cond_base:
        
        # self.rel_dec_cond_dim = self.latent_dim
        # self.dist_dec_cond_dim = self.latent_dim
        
        # else:
        #     self.rel_dec_cond_dim = self.latent_dim + 3 + 3
        #     self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        self.rel_dec_blk = nn.Sequential(
                nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
                nn.Linear(512, 1024,), nn.ReLU(),
                nn.Linear(1024, 512,), nn.ReLU(),
                nn.Linear(512,  3 * 21,),
            )
        
        # if finetune_with_cond:
        #     self.rel_dec_blk = nn.Sequential(
        #         nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
        #         nn.Linear(512, 1024,), nn.ReLU(),
        #         nn.Linear(1024, 512,), nn.ReLU(),
        #         nn.Linear(512,  3 * 21,),
        #     )
        # else:
        #     self.rel_dec_blk = nn.Sequential(
        #         # nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
        #         # nn.Linear(512, 1024,), nn.ReLU(),
        #         # nn.Linear(1024, 512,), nn.ReLU(),
        #         # nn.Linear(512,  3 * 21,),
                
        #         # nn.Linear(, 512,), nn.ReLU(),
        #         # nn.Linear(512, 1024,), nn.ReLU(),
        #         # nn.Linear(1024, 512,), nn.ReLU(),
        #         nn.Linear(self.rel_dec_cond_dim,  3 * 21,),
        #     )
        
        
        # self.rel_dec_blk = nn.Sequential(
        #     nn.Linear(self.rel_dec_cond_dim, 512,), nn.ReLU(),
        #     nn.Linear(512, 1024,), nn.ReLU(),
        #     nn.Linear(1024, 512,), nn.ReLU(),
        #     nn.Linear(512,  3 * 21,),
            
        #     # nn.Linear(, 512,), nn.ReLU(),
        #     # nn.Linear(512, 1024,), nn.ReLU(),
        #     # nn.Linear(1024, 512,), nn.ReLU(),
        #     # nn.Linear(self.rel_dec_cond_dim,  3 * 21,),
        # )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # )
        
        # self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
          self.dist_dec_cond_dim, 1 * 21
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x): # output 
        nframes, bs, d = output.shape
        
        bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4]
        # forward the samole 
        # base_pts = x['base_pts'] # bsz x nnb x 3
        base_pts = x['normed_base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        # bsz x nnframes x d # 
        output = output.permute(1, 0, 2).contiguous().unsqueeze(2).repeat(1, 1, nnb, 1).contiguous()
        base_pts_exp = base_pts.unsqueeze(1).repeat(1, nframes, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).repeat(1, nframes, 1, 1)
        # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
        
        if self.not_cond_base:
            output_exp = output
        else:
            output_exp = torch.cat(
                [output, base_pts_exp, base_normals_exp], dim=-1
            )
        dec_rel = self.rel_dec_blk(output_exp) # bsz x nnframes x nnb x (21 * 3) --> decoded relative positions #
        dec_rel = dec_rel.contiguous().view(bsz, nframes, nnb, nnj, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        # bsz, nframes, nnb, nnj, 3
        # output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1)
        # base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # # obj_base_feats = torch.cat(
        # #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # # )
        # rel_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints], dim=-1
        # )
        # dist_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        # dec_rel = self.rel_dec_blk(rel_dec_in_feats)
        # dec_dist = self.dist_dec_blk(dist_dec_in_feats)
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
        #   'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output




class OutputProcessObjBaseRaw(nn.Module):
    def __init__(self, data_rep, latent_dim, not_cond_base=False):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        self.rel_dec_cond_dim = self.latent_dim + 3 + 3
        self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
          self.rel_dec_cond_dim, 3 * 21
        )
        self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
          self.dist_dec_cond_dim, 1 * 21
        )
        self.not_cond_base = not_cond_base
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x):
        nframes, bs, d = output.shape
        
        bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4]
        # forward the samole 
        base_pts = x['base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        # bsz x nnframes x d # 
        output = output.permute(1, 0, 2).contiguous().unsqueeze(2).repeat(1, 1, nnb, 1).contiguous()
        base_pts_exp = base_pts.unsqueeze(1).repeat(1, nframes, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).repeat(1, nframes, 1, 1)
        # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
        
        if self.not_cond_base:
            output_exp = output
        else:
            output_exp = torch.cat(
                [output, base_pts_exp, base_normals_exp], dim=-1
            )
        dec_rel = self.rel_dec_blk(output_exp) # bsz x nnframes x nnb x (21 * 3) --> decoded relative positions #
        dec_rel = dec_rel.contiguous().view(bsz, nframes, nnb, nnj, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1)
        # base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # # obj_base_feats = torch.cat(
        # #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # # )
        # rel_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints], dim=-1
        # )
        # dist_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        # dec_rel = self.rel_dec_blk(rel_dec_in_feats)
        # dec_dist = self.dist_dec_blk(dist_dec_in_feats)
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
        #   'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output




class OutputProcessObjBase(nn.Module):
    def __init__(self, data_rep, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
        self.dist_dec_cond_dim = self.latent_dim + 3 + 3 + 1
        self.rel_dec_blk = nn.Linear(
          self.rel_dec_cond_dim, 3
        )
        self.dist_dec_blk = nn.Linear(
          self.dist_dec_cond_dim, 1
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x):
        nframes, bs, d = output.shape
        
        bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4]
        # forward the samole 
        base_pts = x['base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1)
        base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # obj_base_feats = torch.cat(
        #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        rel_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints], dim=-1
        )
        dist_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, base_normals_exp, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        )
        dec_rel = self.rel_dec_blk(rel_dec_in_feats)
        dec_dist = self.dist_dec_blk(dist_dec_in_feats)
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
          'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output


class OutputProcessObjBaseV2(nn.Module):
    def __init__(self, data_rep, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.nfeats = nfeats
        self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3 + 3 ## base pts, obj pts, obj normals, joints pts #
        self.dist_dec_cond_dim = self.latent_dim + 3 + 3 + 3 + 1
        self.rel_dec_blk = nn.Linear(
          self.rel_dec_cond_dim, 3
        )
        self.dist_dec_blk = nn.Linear(
          self.dist_dec_cond_dim, 1
        )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, x):
        nframes, bs, d = output.shape
        
        bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4]
        # forward the samole 
        ## bsz x nnb x 3 ##
        sampled_base_pts_nearest_obj_pc = x['sampled_base_pts_nearest_obj_pc']
        sampled_base_pts_nearest_obj_vns = x['sampled_base_pts_nearest_obj_vns']
        base_pts = x['base_pts'] # bsz x nnb x 3
        # base_normals = x['base_normals'] # bsz x nnb x 3
        rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        
        
        output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1)
        base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        sampled_base_pts_nearest_obj_pc_exp = sampled_base_pts_nearest_obj_pc.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        sampled_base_pts_nearest_obj_vns_exp = sampled_base_pts_nearest_obj_vns.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # obj_base_feats = torch.cat(
        #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        rel_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, sampled_base_pts_nearest_obj_pc_exp, sampled_base_pts_nearest_obj_vns_exp, rel_base_pts_to_rhand_joints], dim=-1
        )
        dist_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, sampled_base_pts_nearest_obj_pc_exp, sampled_base_pts_nearest_obj_vns_exp, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        )
        dec_rel = self.rel_dec_blk(rel_dec_in_feats)
        dec_dist = self.dist_dec_blk(dist_dec_in_feats)
        
        # decoded rel, decoded distances #
        out = {
          'dec_rel': dec_rel,
          'dec_dist': dec_dist.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output


class OutputProcessObjBaseV3(nn.Module):
    def __init__(self, data_rep, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        # 
        # self.input_feats = input_feats # object base pts base normals # object base pts base normals #
        self.latent_dim = latent_dim
        # self.njoints = njoints # 
        # self.nfeats = nfeats # 
        self.e_along_normal_dec_cond_dim = self.latent_dim + 3 + 3 + 3 + 1 ## base pts, obj pts, obj normals, joints pts #
        self.e_vt_normal_dec_cond_dim = self.latent_dim + 3 + 3 + 3 + 1
        self.e_along_normal_dec_blk = nn.Linear(
          self.e_along_normal_dec_cond_dim, 1
        )
        self.e_vt_normal_dec_blk = nn.Linear(
          self.e_vt_normal_dec_cond_dim, 1
        )
    
    ### forward for basepts v3 ###
    def forward(self, output, x):
        nframes, bs, d = output.shape
        
        # inputs for the 
        bsz, nframes, nnj, nnb = x['e_disp_rel_to_base_along_normals'].shape[:4]
        # forward the samole 
        ## bsz x nnb x 3 ##
        e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
        e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
        base_pts = x['base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        
        
        output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1)
        base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        
        # sampled_base_pts_nearest_obj_pc_exp = sampled_base_pts_nearest_obj_pc.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # sampled_base_pts_nearest_obj_vns_exp = sampled_base_pts_nearest_obj_vns.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # obj_base_feats = torch.cat(
        #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        e_along_normals_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, base_normals_exp,  rel_base_pts_to_rhand_joints[:, :-1], e_disp_rel_to_base_along_normals.unsqueeze(-1)], dim=-1
        )
        e_vt_normals_dec_in_feats = torch.cat(
          [output_exp, base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints[:, :-1], e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1
        )
        dec_e_along_normals = self.e_along_normal_dec_blk(e_along_normals_dec_in_feats)
        dec_e_vt_normals = self.e_vt_normal_dec_blk(e_vt_normals_dec_in_feats)
        
        # decoded rel, decoded distances #
        out = {
          'dec_e_along_normals': dec_e_along_normals.squeeze(-1),
          'dec_e_vt_normals': dec_e_vt_normals.squeeze(-1),
        }
        return out ## output
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output




class OutputProcessObjBaseERaw(nn.Module):
    def __init__(self, data_rep, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        # self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.njoints = njoints # 
        # self.nfeats = nfeats # 
        # base pts, base normals -> otuptu should be a distan
        self.e_along_normal_dec_cond_dim = self.latent_dim + 3 + 3 # + 3 + 1 ## obj pts, obj normals, joints pts #
        self.e_vt_normal_dec_cond_dim = self.latent_dim + 3 + 3 # + 3 + 1
        self.d_dec_cond_dim = self.latent_dim + 3 + 3  # obj pts obj normal
        self.rel_vel_dec_cond_dim = self.latent_dim + 3 + 3
        self.e_along_normal_dec_blk = nn.Linear(
          self.e_along_normal_dec_cond_dim, 21
        )
        self.e_vt_normal_dec_blk = nn.Linear(
          self.e_vt_normal_dec_cond_dim, 21
        )
        self.d_dec_blk = nn.Linear(
          self.d_dec_cond_dim, 21
        )
        self.rel_vel_dec_blk = nn.Linear(
          self.rel_vel_dec_cond_dim, 21
        )
    
    ### forward for basepts v3 ###
    def forward(self, output, x):
        nframes, bs, d = output.shape
        
        # inputs for the 
        bsz, nframes, nnj, nnb = x['e_disp_rel_to_base_along_normals'].shape[:4]
        # forward the samole 
        ## bsz x nnb x 3 ##
        # e_disp_rel_to_base_along_normals = x['e_disp_rel_to_base_along_normals']
        # e_disp_rel_to_baes_vt_normals = x['e_disp_rel_to_baes_vt_normals']
        base_pts = x['base_pts'] # bsz x nnb x 3
        base_normals = x['base_normals'] # bsz x nnb x 3
        # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        
        # print(f"output: {output.size()}")
        output_exp = output.permute(1, 0, 2).contiguous().unsqueeze(2).repeat(1, 1, nnb, 1)
        base_pts_exp = base_pts.unsqueeze(1).repeat(1, nframes, 1, 1)
        base_normals_exp = base_normals.unsqueeze(1).repeat(1, nframes, 1, 1)
        
        e_along_normals_dec_in_feats = torch.cat(
            [output_exp, base_pts_exp, base_normals_exp], dim=-1 # bsz x nnf x nnb x (latent_dim + 3 + 3)
        )
        e_vt_normals_dec_in_feats = torch.cat(
            [output_exp, base_pts_exp, base_normals_exp], dim=-1 # the same dims
        )
        # bsz x nnf x nnb x 21 #
        dec_e_along_normals = self.e_along_normal_dec_blk(e_along_normals_dec_in_feats)
        # bsz x nnf x nnb x 21 #
        dec_e_vt_normals = self.e_vt_normal_dec_blk(e_vt_normals_dec_in_feats)
        
        d_dec_in_feats = e_vt_normals_dec_in_feats.clone()
        rel_vel_dec_in_feats = e_vt_normals_dec_in_feats.clone()
        dec_d = self.d_dec_blk(d_dec_in_feats)
        rel_vel_dec = self.rel_vel_dec_blk(rel_vel_dec_in_feats)
        
        
        out = {
          'dec_e_along_normals': dec_e_along_normals, ## along normals ##
          'dec_e_vt_normals': dec_e_vt_normals, ## vt normals ##
          'dec_d': dec_d,
          'rel_vel_dec': rel_vel_dec
        }
        
        # sampled_base_pts_nearest_obj_pc_exp = sampled_base_pts_nearest_obj_pc.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # sampled_base_pts_nearest_obj_vns_exp = sampled_base_pts_nearest_obj_vns.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # obj_base_feats = torch.cat(
        #   [base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints, dist_base_pts_to_rhand_joints.unsqueeze(-1)], dim=-1
        # )
        # e_along_normals_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp,  rel_base_pts_to_rhand_joints[:, :-1], e_disp_rel_to_base_along_normals.unsqueeze(-1)], dim=-1
        # )
        # e_vt_normals_dec_in_feats = torch.cat(
        #   [output_exp, base_pts_exp, base_normals_exp, rel_base_pts_to_rhand_joints[:, :-1], e_disp_rel_to_baes_vt_normals.unsqueeze(-1)], dim=-1
        # )
        # dec_e_along_normals = self.e_along_normal_dec_blk(e_along_normals_dec_in_feats)
        # dec_e_vt_normals = self.e_vt_normal_dec_blk(e_vt_normals_dec_in_feats)
        
        # # decoded rel, decoded distances #
        # out = {
        #   'dec_e_along_normals': dec_e_along_normals.squeeze(-1),
        #   'dec_e_vt_normals': dec_e_vt_normals.squeeze(-1),
        # }
        return out ## output
        




# key points #
class InputProcessWithGlbInfo(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.pts_embedding_layer = nn.Linear(3, self.latent_dim)
        self.glb_embedding_layer = nn.Linear(self.latent_dim, self.latent_dim)
        # self.pts_out_embedding_layer = nn.Sequential(
        #     nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim)
        # )

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape # nnjoints and nnfeats 
        
        x = x.permute(3, 0, 1, 2).contiguous()
        x_pts_emb = self.pts_embedding_layer(x)
        x_glb_emb, _ = torch.max(x_pts_emb, dim=-2, keepdim=False)
        # x_glb_emb = x_glb_emb.repeat(1, 1, njoints, 1).contiguous()
        # x_pts_glb_emb = torch.cat(
        #     [x_pts_emb, x_glb_emb], dim=-1
        # )
        # x_pts_emb = self.pts_out_embedding_layer(
        #     x_pts_glb_emb # x_pts_emb: nframes x bsz x nn_joints x n_feats #
        # )
        
        x_glb_emb = self.glb_embedding_layer(x_glb_emb) # nframes x bsz x latent_dim # glb embedding layer #
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        return x_glb_emb

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d] # seqlen x bs x d
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError



# key points #
# InputProcessParams(data_rep, input_feats, latent_dim) #
# x_embedding_feats
class InputProcessParams(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        #### params embedding layer ####
        self.params_embedding_layer = nn.Linear(self.input_feats, self.latent_dim) # input_feats x latent_dim #
        #### params embedding layer ####
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x): # x: bsz x ws x latent_dim #
        x = x.permute(1, 0, 2).contiguous()
        x_embedding_feats = self.params_embedding_layer(x)
        return x_embedding_feats # ws x bsz x latnet_dim -> the features # 
        
        # bs, njoints, nfeats, nframes = x.shape # nnjoints and nnfeats 
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     x = self.poseEmbedding(x)  # [seqlen, bs, d] # seqlen x bs x d
        #     return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError



# key points #
class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape # nnjoints and nnfeats 
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d] # seqlen x bs x d
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


# decode whole joints together #
# decode joints #
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats ## to output input_feats for the joints positions decoding ## ## 
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel': # rot_vel 
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        # bsz x 
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output




# decode whole joints together #
# decode joints #
# OutputProcessParams(data_rep, input_feats, latent_dim, njoints, nfeats)
class OutputProcessParams(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats ## to output input_feats for the joints positions decoding ## ## 
        
        # self.output_process = nn.Sequential(
        #     nn.Linear
        # )
        self.transl_dim = 3 
        self.rot_dim = 3 
        self.theta_dim = 24
        self.transl_out = nn.Linear(self.latent_dim, self.transl_dim)
        self.rot_out = nn.Linear(self.latent_dim, self.rot_dim)
        self.theta_out = nn.Linear(self.latent_dim, self.theta_dim) # linear layers for transl, rot, and theta #
        
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel': # rot_vel 
            # self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        
        # output: nfrmes x bsz x d # 
        nframes, bs, d = output.shape
        transl_out = self.transl_out(output)
        rot_out = self.rot_out(output)
        theta_out = self.theta_out(output)
        
        params_out = torch.cat(
            [transl_out, rot_out, theta_out], dim=-1
        )
        return params_out
        
        # nframes, bs, d = output.shape
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        # # bsz x 
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        # return output



# decode whole joints together #
# decode joints #
class OutputProcessCond(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        self.condposefinal = nn.Linear(self.latent_dim + nfeats, nfeats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, cond):
        # cond: bsz x nf x nnj x 3 #
        nnj = cond.size(2)
        nframes, bs, d = output.shape
        output = output.permute(1, 0, 2).contiguous() # bsz x nnframes x d
        output_exp = output.unsqueeze(2).repeat(1, 1, nnj, 1).contiguous()
        cat_output = torch.cat(
            [output_exp, cond], dim=-1 ## bsz x nframes x nnj x (d + 3) #
        )
        output = self.condposefinal(cat_output) # bsz x nframes x nnj x 3 #
        output = output.permute(0, 2, 3, 1).contiguous() # bsz x nnj x 3 x nframes #
        
        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #     output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        
        # # bsz x 
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output



class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output