# import sonnet as snt
# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_layers
# import tensorflow.compat.v1 as tf
# from tensorflow.python.framework import function
# import tensorflow_probability as tfp

import numpy as np
import torch.nn as nn
# import layer_utils
import torch
# import data_utils_torch as data_utils
import math ## 

# from options.options import opt



### 
class TransformerEncoder(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 memory_efficient=False,
                 ):
        super(TransformerEncoder, self).__init__()
        ## hidden size, fc size, ##
        ## hidden size, fc size, num heads, layer_norm, num_layers, dropout_rate, 
        self.hidden_size = hidden_size
        self.fc_size = fc_size ## fc_size ##
        self.num_heads = num_heads ## num_heads ##
        # self.num_heads = 1
        self.layer_norm = layer_norm
        self.num_layers = num_layers ## num_layers ##
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.memory_efficient = memory_efficient

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
        if self.dropout_rate: # dropout rate
            self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers): ## dropout rate, kdim, vdim, 
            cur_atten_layer = nn.MultiheadAttention( ## hidden_size, hidden_size ##
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size, batch_first=True)
            self.attention_layers.append(cur_atten_layer)
            if self.layer_norm: ## layernorm ##
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.dropout_layers.append(cur_dropout_layer)

        ### Attention layer and related modules ###
        self.fc_layers = nn.ModuleList()
        if self.layer_norm:
            self.fc_layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.fc_re_zero_vars = nn.ParameterList()
        if self.dropout_rate:
            self.fc_dropout_layers = nn.ModuleList() # dropout layers
        for i in range(self.num_layers):
            cur_fc_layer = nn.Linear(in_features=self.hidden_size, out_features=self.fc_size, bias=True)
            cur_fc_layer_2 = nn.Linear(in_features=self.fc_size, out_features=self.hidden_size, bias=True)
            self.fc_layers.append(nn.Sequential(*[cur_fc_layer, cur_fc_layer_2]))
            if self.layer_norm: # layer norm
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.fc_layer_norm_layers.append(cur_layer_norm)
            if self.re_zero: # re_zero_var 
                cur_re_zero_var = torch.nn.Parameter(
                    torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.fc_re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate:
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.fc_dropout_layers.append(cur_dropout_layer)

        if self.layer_norm:
            self.out_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, set_attn_to_none=False):
        ### padding 
        # bsz x seq_length x embedding_dim #
        bsz, seq_length = inputs.size(0), inputs.size(1)
        
        if set_attn_to_none:
            atten_mask = None
        else:
            atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri ### elements in the upper triangle are set to 1.0 ###
            atten_mask = torch.from_numpy(atten_mask).float() # .cuda()
            atten_mask = atten_mask.to(inputs.device)
            atten_mask = atten_mask > 0.5 ## the bool format
            
        
        # if inputs_mask is None:
        #     encoder_padding = layer_utils.embedding_to_padding(inputs) # bsz x n_vertices
        # else:
        #     encoder_padding = inputs_mask # inputs_mask: bsz x n_vertices
        # bsz = inputs.size(0)
        # seq_length = inputs.size(1)
        # ## attention masksingle direction ## need 
        # # encoder_self_attention_bias = layer_utils.attention_bias_ignore_padding(encoder_padding)
        # # encoder_self_attention_mask = layer_utils.attention_mask(encoder_padding)
        # encoder_self_attention_mask = layer_utils.attention_mask_single_direction(encoder_padding)
        # # print(f"in vertex model forwarding function, encoder_self_attention_mask: {encoder_self_attention_mask.size()}, inputs: {inputs.size()}")
        # encoder_self_attention_mask = encoder_self_attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # encoder_self_attention_mask = encoder_self_attention_mask.contiguous().view(bsz * self.num_heads, seq_length, seq_length).contiguous()
        # seq_length = inputs.size(1)
        x = inputs ## bsz x seq_length x # bsz x seq x seq for the mask #

        # zeros padding layer ## remember to add that! # zero padding layer #
        # atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # mask; 
        # atten_mask = torch.from_numpy(atten_mask).float().cuda() ## mask single direction 
        ## encode for each 
        for i in range(self.num_layers):
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res) ## res, res ## ## layernorm layers ##
            # print(f"before attention {i}/{self.num_layers}, res: {res.size()}")
            # res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask) 
            ## attentiion layers ### ## self-attention ## ## memory, q, k, v ## bsz x seq x latnetdim --> bsz x seq x seq for frame-frame weights ###
            ### bsz x seq x seq --> weights ### initialize something to zero for modeling controls ###
            res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            # print(f"after attention {i}/{self.num_layers}, res: {res.size()}")
            if self.re_zero:
                res = res * self.re_zero_vars[i]
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            res = x.clone()
            if self.layer_norm:
                res = self.fc_layer_norm_layers[i](res) # fc norm #
            res = self.fc_layers[i](res)
            if self.re_zero:
                res = res * self.fc_re_zero_vars[i]
            if self.dropout_rate:
                res = self.fc_dropout_layers[i](res)
            x = x + res
        if self.layer_norm:
            x = self.out_layer_norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 re_zero=True,
                 with_seq_context=False
                 ):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.with_seq_context = with_seq_context
        # self.context_window = opt.model.context_window
        self.atten_mask = None
        self.context_atten_mask = None
        # self.prefix_key_len = opt.model.prefix_key_len ## can add prefix key values for prefix queries ##
        # self.prefix_value_len = opt.model.prefix_value_len ## can add prefix key values for prefix queries ##
        # self.prefix_value_len = value length #

        ### Attention layer and related modules ###
        self.attention_layers = nn.ModuleList()
        if self.layer_norm:
            self.layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.re_zero_vars = nn.ParameterList()
        if self.dropout_rate:
            self.dropout_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            cur_atten_layer = nn.MultiheadAttention(
                self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
                batch_first=True)
            self.attention_layers.append(cur_atten_layer)
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate: ## dropout
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.dropout_layers.append(cur_dropout_layer)
        
        if self.with_seq_context:
            ##### attention, re_zero, dropout layers for the context attention layers #####
            self.context_attention_layers = nn.ModuleList()
            if self.layer_norm:
                self.context_norm_layers = nn.ModuleList()
            if self.re_zero:
                self.context_re_zero_vars = nn.ParameterList()
            if self.dropout_rate:
                self.context_dropout_layers = nn.ModuleList()
            for i in range(self.num_layers):
                cur_atten_layer = nn.MultiheadAttention(
                    self.hidden_size, self.num_heads, dropout=0.0, bias=True, kdim=self.hidden_size, vdim=self.hidden_size,
                    batch_first=True)
                self.context_attention_layers.append(cur_atten_layer)
                if self.layer_norm:
                    cur_layer_norm = nn.LayerNorm(self.hidden_size)
                    self.context_norm_layers.append(cur_layer_norm)
                if self.re_zero:
                    cur_re_zero_var = torch.nn.Parameter(torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                    self.context_re_zero_vars.append(cur_re_zero_var)
                if self.dropout_rate:
                    cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                    # dropout layers
                    self.context_dropout_layers.append(cur_dropout_layer)

        ### Attention layer and related modules ###
        self.fc_layers = nn.ModuleList()
        if self.layer_norm:
            self.fc_layer_norm_layers = nn.ModuleList()
        if self.re_zero:
            self.fc_re_zero_vars = nn.ParameterList()
            # self.fc_re_zero_vars = nn.ModuleList()
        if self.dropout_rate:
            self.fc_dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cur_fc_layer = nn.Linear(in_features=self.hidden_size, out_features=self.fc_size, bias=True)
            cur_fc_layer_2 = nn.Linear(in_features=self.fc_size, out_features=self.hidden_size, bias=True)
            self.fc_layers.append(nn.Sequential(*[cur_fc_layer, cur_fc_layer_2]))
            if self.layer_norm:
                cur_layer_norm = nn.LayerNorm(self.hidden_size)
                self.fc_layer_norm_layers.append(cur_layer_norm)
            if self.re_zero:
                cur_re_zero_var = torch.nn.Parameter(
                    torch.zeros(size=(1,), dtype=torch.float32, requires_grad=True), requires_grad=True)
                self.fc_re_zero_vars.append(cur_re_zero_var)
            if self.dropout_rate: ## dropout rate ##
                cur_dropout_layer = nn.Dropout(p=self.dropout_rate)
                self.fc_dropout_layers.append(cur_dropout_layer)

        if self.layer_norm:
            self.out_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, sequential_context_embeddings=None):
        seq_length = inputs.size(1)
        bsz = inputs.size(0)
        # #### ## sequential context embeddings for the embedding --> bsz x seq_length x feat_dim ####
        # TODO: mask for inputs can be set to 1) None, then a fully-attention setting, 2) self-mask setting #
        # ### sequential context mask should be set to a self-mask setting -> each self element can attend to self and before information ### #
        atten_mask = None ## mask for inputs
        
        if sequential_context_embeddings is not None:
            # sequential_context_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri ## triangle mask ##
            ## 
            # sequential_context_mask = np.tri(inputs.size(1), sequential_context_embeddings.size(1), -1.0, dtype=np.float32).T # tri
            # 1 x 30 --> no mask !
            sequential_context_mask = np.tri(sequential_context_embeddings.size(1), inputs.size(1), -1.0, dtype=np.float32).T # tri
            sequential_context_mask = torch.from_numpy(sequential_context_mask).float() # .cuda()
            sequential_context_mask = sequential_context_mask.to(inputs.device)
            sequential_context_mask = sequential_context_mask > 0.5
        
        # # print(f"inputs: {inputs.size()}") ####
        # if self.training:
        #     if self.atten_mask is None: ## seq length ##
        #         atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        #         # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        #         atten_mask = torch.from_numpy(atten_mask).float().cuda()
        #         self.atten_mask = atten_mask
        #     else:
        #         atten_mask = self.atten_mask
        # else: ### atten_mask 
        #     atten_mask = np.tri(seq_length, seq_length, -1.0, dtype=np.float32).T # tri
        #     # atten_mask = np.tri(seq_length, seq_length, 0.0, dtype=np.float32)
        #     atten_mask = torch.from_numpy(atten_mask).float().cuda()

        # context_window 
        # if self.context_window > 0 and sequential_context_embeddings is None:
        #     # ##### add global context embeddings to embedding vectors ##### #
        #     # inputs = inputs[:, 0:1] + inputs # add the contextual information to inputs # not add...
        #     # if opt.model.debug:
        #     #     print(f"Using context window {self.context_window} for decoding...")
        #     if self.training:
        #         if self.context_atten_mask is None:
        #             context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #             context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #             self.context_atten_mask = context_atten_mask
        #         else:
        #             context_atten_mask = self.context_atten_mask
        #     else:
        #         context_atten_mask = np.tri(seq_length, seq_length, -1.0 * float(self.context_window), dtype=np.float32)
        #         context_atten_mask = torch.from_numpy(context_atten_mask).float().cuda()
        #     atten_mask = context_atten_mask + atten_mask
        # # context attention mask
        # atten_mask = (atten_mask > 0.5)

        # if len(atten_mask.size()) == 2:
        #     atten_mask[: self.prefix_key_len, ] = False
        # else:
        #     atten_mask[:, : self.prefix_key_len] = False

        
        # print(atten_mask)

        # if sequential_context_embeddings is not None:
        #     context_length = sequential_context_embeddings.size(1)
        #     # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)

        #     if sequential_context_mask is None:
        #       sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)
        #     else: # 
        #       sequential_context_padding = 1. - sequential_context_mask.float() # sequential context mask?
        #       # sequential_context_padding = layer_utils.embedding_to_padding(sequential_context_embeddings)
        #     # j
        #     sequential_context_atten_mask = layer_utils.attention_mask_single_direction(sequential_context_padding, other_len=seq_length)
        #     # print(f"in decoder's forward function, sequential_context_padding: {sequential_context_padding.size()}, sequential_context_atten_mask: {sequential_context_atten_mask.size()}")
        #     sequential_context_atten_mask = sequential_context_atten_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        #     sequential_context_atten_mask = sequential_context_atten_mask.contiguous().view(bsz * self.num_heads, seq_length, context_length).contiguous()
        
        x = inputs

        for i in range(self.num_layers):
            res = x.clone()
            if self.layer_norm:
                res = self.layer_norm_layers[i](res) # # self attention; all self attention; sequential 
            res, _ = self.attention_layers[i](res, res, res, attn_mask=atten_mask)
            if self.re_zero:
                res = res * self.re_zero_vars[i].unsqueeze(0).unsqueeze(0)
            if self.dropout_rate:
                res = self.dropout_layers[i](res)
            x = x + res

            # if we use sequential context embeddings
            if sequential_context_embeddings is not None:
                # for sequential context embedding
                res = x.clone()
                # then layer_norm, attention layer, re_zero layer and the dropout layer
                if self.layer_norm:
                    res = self.context_norm_layers[i](res) ## need sequential masks! res can only attent to former sequential contexts ## ## 
                res, _ = self.context_attention_layers[i](res, sequential_context_embeddings, sequential_context_embeddings, attn_mask=sequential_context_mask)
                if self.re_zero:
                    res = res * self.context_re_zero_vars[i].unsqueeze(0).unsqueeze(0)
                if self.dropout_rate:
                    res = self.context_dropout_layers[i](res)
                x = x + res
            

            res = x.clone()
            if self.layer_norm:
                res = self.fc_layer_norm_layers[i](res)
            res = self.fc_layers[i](res)
            if self.re_zero:
                res = res * self.fc_re_zero_vars[i]
            if self.dropout_rate: # dropout layers # fc_dropout_layers
                res = self.fc_dropout_layers[i](res)
            x = x + res
        if self.layer_norm:
            x = self.out_layer_norm(x)
        # x = x[:, self.prefix_key_len - 1: ]
        return x

