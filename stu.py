from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .components.encoder import Encoder, GraphEncoder, SEncoder
from .components.decoder import Decoder, TxpCnn, DGCDecoder, SDecoder
from .components.positional import PositionalEncoding
from .components.utils import subsequent_mask
from .components.embedding import LinearEmbedding
from .components.dropblock import DropBlock
from .components.non_ar_qgb import NonAutoRegression
from .components.DynamicGCf import GCfEncoder
from .components.utils import AddAndNorm, make_mlp
from .components.attention import RefPosMultiHeadAttention
from .components.DynamicSpaTemGF import SpaTemGCfEncoder
from .components.GroupNet import PastEncoder
from .utils import outputActivation


class STUNET(nn.Module):

    def __init__(self, cfg, dataset):
        super(STUNET, self).__init__()
        # dataset params
        self.input_length = dataset.historical_length
        self.output_length = dataset.future_length
        self.num_maneuvers = dataset.num_maneuvers
        self.input_dims = dataset.num_features
        self._max_num_agents = dataset.max_num_agents
        self._max_len_edge = dataset.max_vertical_distance
        # hyper params
        self.device = cfg.device
        self.feature_dims = cfg.attention_dims #64
        self.hidden_dims = cfg.feedforward_dims
        self.num_spatial_heads = cfg.num_spatial_heads
        self.num_temporal_heads = cfg.num_temporal_heads
        self.num_decoder_heads = cfg.num_decoder_heads
        self.num_spatial_encoders = cfg.num_spatial_encoders
        self.num_temporal_encoders = cfg.num_temporal_encoders
        self.num_decoders = cfg.num_decoders
        self.attention_dropout = cfg.attention_dropout if 1. > cfg.attention_dropout > 0. else None
        self.input_dropout = cfg.input_dropout if 1. > cfg.input_dropout > 0. else None
        self.feature_dropout = cfg.feature_dropout if 1. > cfg.feature_dropout > 0. else None
        # network structure
        self._cfg_len_edge = cfg.max_len_edge
        assert cfg.num_agents <= self._max_num_agents, 'invalid number of agents'
        self._num_agents = cfg.num_agents
        self._enable_maneuver_ = cfg.maneuver
        self._multiple_agents_ = cfg.multi_agents
        self.num_spatial_time = cfg.num_spatial_time
        self.use_nll = cfg.use_nll
        self._mult_traj = cfg.mult_traj
        self.use_true_man = cfg.use_true_man
        self.use_hard_man = cfg.use_hard_man
        # input dropout block
        self.dropout_block = DropBlock(dropout_rate=self.input_dropout) if not (
                self.input_dropout is None) else nn.Identity()

        # pre-encodings
        self.dynamic_encoder = nn.Linear(self.input_dims, self.feature_dims, bias=False)
        self.dynamic_encoder_ref = nn.Linear(2, self.feature_dims, bias=False)
        self.track_encoder = nn.Linear(2, self.feature_dims, bias=False)
        self.pos_feature_encoder = nn.Linear(2, self.feature_dims, bias=False)
        self.pos_enc = PositionalEncoding(self.feature_dims, self.feature_dropout)

        # spatial encodings
        # self.static_weight_encoder = nn.ModuleList(nn.Sequential(nn.Linear(2, self.feature_dims, bias=False),
        #                                                          nn.PReLU(),
        #                                                          LinearEmbedding(self.feature_dims,
        #                                                                          self.num_spatial_heads)) for _ in
        #                                            range(self.num_spatial_time))
        self.static_weight_encoder = nn.ModuleList(nn.Sequential(nn.Linear(2, self.feature_dims, bias=False),
                                                                 nn.PReLU(),
                                                                 LinearEmbedding(self.feature_dims,
                                                                                 self.num_spatial_heads)
                                                                 ) for _ in
                                                   range(self.num_spatial_time))
        # self.static_weight_encoder = nn.ModuleList(nn.Sequential(nn.Linear(2, self.feature_dims)) for _ in
        #                                            range(self.num_spatial_time))
        self.spatial_encoder = nn.ModuleList(GraphEncoder(self.num_spatial_encoders, self.feature_dims,
                                                          self.num_spatial_heads, self.hidden_dims,
                                                          self.attention_dropout)
                                             for _ in range(self.num_spatial_time)) 
        self.group_encoder = PastEncoder(cfg, self.feature_dims)

        self.spatial_de = nn.ModuleList(nn.Sequential(nn.Linear(self.feature_dims, self.feature_dims)) for _ in
                                        range(self.num_spatial_time))
        # time encodings
        self.temporal_encoder = nn.ModuleList(
            (SEncoder(self.num_temporal_encoders, self.feature_dims, self.num_decoder_heads,
                     self.hidden_dims, self.attention_dropout)) for _ in range(self.num_spatial_time))
        self.temporal_token = nn.Parameter(torch.empty(1, 1, self.feature_dims))  # [1, 1, C]
        self.tzfc = nn.Linear(self.feature_dims*2,self.feature_dims)
        # maneuver classification
        if self._enable_maneuver_:
            assert self.num_maneuvers > 0, 'invalid maneuvers'
            self.classifier = nn.Linear(self.feature_dims, self.num_maneuvers, bias=False)
            if self._multiple_agents_:
                self.maneuver_encoder = nn.Linear(self.num_maneuvers, self.feature_dims)
            else:
                # self.maneuver_encoder = nn.Embedding(self.num_maneuvers,)
                self.maneuver_encoder = nn.Linear(self.num_maneuvers, self.feature_dims)
        if self.use_nll:
            self.generator = nn.Linear(self.feature_dims, 5, bias=False)
        else:
            self.generator = nn.Linear(self.feature_dims, 2, bias=False)
        # track decodings
        self.QGB = NonAutoRegression(feature_dims=self.feature_dims,
                                     pred_seq_len=self.output_length, d_model=self.feature_dims)
        self.nast_random = cfg.nast_random
        self.decoder = SDecoder(self.num_decoders, self.feature_dims, self.num_decoder_heads, self.hidden_dims,
                               self.attention_dropout)
        # masks
        self._subsequent_masks_ = nn.Parameter(subsequent_mask(self.output_length),
                                               requires_grad=False)  # [1, T', T']

        # initialize embeddings and norm layers
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight)
        nn.init.normal_(self.temporal_token, 0.0, 0.02)
        nn.init.normal_(self.dynamic_encoder.weight, 0.0, 0.02)
        nn.init.normal_(self.track_encoder.weight, 0.0, 0.02)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor,
                truth: Optional[torch.Tensor] = None,
                maneuvers: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        '''
        Params:
        input tensor can expand one extra dimension if predicts multiple agents
            observed: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Input_features] (delta-x,delta-y, ...)
            observed_xy: 4d Tensor, shape = [Batch_size, Obs_len, Num_max_agents, 2] (x,y)
            mask: 3d Tensor, shape = [Batch_size, Obs_len, Num_max_agents] (0|1)
            truth: 3d/4d Tensor, shape = [Batch_size, Pred_len, Input_features] / [Batch_size, Num_max_agents, Pred_len, Input_features]
                   or NoneType, if not training
            maneuvers: 2d/3d Tensor, shape = [Batch_size, Num_max_agents] / [Batch_size, Num_max_agents, Num_max_agents]
                       or NoneType, not using maneuvers
            multimodal: boolean, flag used in evaluation
            softmaneuver: boolean, flag used in evaluation
        Outputs:
            track: 3d/4d Tensor, shape = [Batch_size, Pred_len, 2] / [Batch_size, Num_max_agents, Pred_len, 2] (delta-x,delta-y)
            track_xy: 3d/4d Tensor, shape = [Batch_size, Pred_len, 2] / [Batch_size, Num_max_agents, Pred_len, 2] (x,y)
                      or 4d/5d Tensor, shape = [Num_maneuvers, Batch_size, Pred_len, 2] / [Num_maneuvers, Batch_size, Num_max_agents, Pred_len, 2] (x,y) if using multimodal and maneuvers
            maneuver_prob: 2d/3d Tensor, shape = [Batch_size, Num_maneuvers] / [Batch_size, Num_max_agents, Num_maneuvers] (range in [0,1])
                           or NoneType, if not using maneuvers
        '''
        """ print("===================================================")
        print("this is student network.")
        print("===================================================") """
        # input dropout
        # observed[..., :2] = observed_xy.transpose(1, 2)
        # observed = self.dropout_block(observed)
        # generate distance matrix
        # print(observed.shape)
        observed_xy = observed[..., :2].transpose(1, 2)
        observed_xy_reshaped = observed_xy.contiguous().view(-1, self._num_agents, 2)  # [B*T, N, 2]
        dist_mat = torch.cdist(observed_xy_reshaped, observed_xy_reshaped, p=2)  # [B*T, N, N]
        # generate adjacent matrix
        mask_mat = torch.bmm(mask.view(-1, self._num_agents, 1),
                             mask.view(-1, 1, self._num_agents)).bool()  # [B*T, N, N]
        Adj = (mask_mat & (dist_mat <= self._cfg_len_edge) & (dist_mat >= 0)).view(-1, self.input_length,
                                                                                   self._num_agents,
                                                                                   self._num_agents)  # [B, T, N, N]
        rel_pos = observed_xy.unsqueeze(2) - observed_xy.unsqueeze(3)  # [B, T, N, N, 2]

        def group_encoding(dynamics: torch.Tensor, num: int) -> torch.Tensor:
            '''
            Params:
                dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
            Outputs:
                spatial_dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
            '''
            spatial_bias = self.static_weight_encoder[num](rel_pos).view(-1,
                                                                         self._num_agents, self._num_agents,
                                                                         self.num_spatial_heads)  # [B*T, N, N, H]
            spatial_inputs = dynamics.transpose(1, 2).contiguous().view(-1, self._num_agents,
                                                                        self.feature_dims)  # [B*T, N, C]
            batch_size = dynamics.shape[0]
            agent_num = dynamics.shape[1]
            #print(batch_size) # 43
            
            spatial_dynamics = self.group_encoder(dynamics, batch_size, agent_num)
            return spatial_dynamics.view(-1, self.input_length, self._num_agents, self.feature_dims).transpose(1,
                                                                                                               2).contiguous()  # [B, N, T, C]

        def spatial_encoding(dynamics: torch.Tensor, num: int) -> torch.Tensor:
            '''
            Params:
                dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
            Outputs:
                spatial_dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
            '''
            spatial_bias = self.static_weight_encoder[num](rel_pos).view(-1,
                                                                         self._num_agents, self._num_agents,
                                                                         self.num_spatial_heads)  # [B*T, N, N, H]
            spatial_inputs = dynamics.transpose(1, 2).contiguous().view(-1, self._num_agents,
                                                                        self.feature_dims)  # [B*T, N, C]
            # spatial_inputs_xy = self.dynamic_encoder_ref(observed_xy_reshaped)
            # spatial_inputs = torch.cat((spatial_inputs, spatial_inputs_xy), dim=-1)
            spatial_dynamics = self.spatial_encoder[num](spatial_inputs, spatial_inputs,
                                                         Adj.view(-1, self._num_agents, self._num_agents),
                                                         spatial_bias)  # [B*T, N, C]
            spatial_dynamics = self.spatial_de[num](spatial_dynamics)
            return spatial_dynamics.view(-1, self.input_length, self._num_agents, self.feature_dims).transpose(1,
                                                                                                               2).contiguous() 
        # temporal encoding (TF)
        def temporal_encoding(dynamics: torch.Tensor, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
            '''
            Params:
                dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
            Outputs:
                temporal_dynamics: 3d Tensor, shape = [Batch_size, Obe_len, Embedded_features]
                temporal_token: 2d Tensor, shape = [Batch_size, Embedded_features]
            '''
            temporal_inputs = dynamics.view(-1, self.input_length, self.feature_dims)  # [B*N, T, C]
            temporal_inputs = self.pos_enc(
                torch.cat([self.temporal_token.repeat(temporal_inputs.shape[0], 1, 1), temporal_inputs],
                          dim=1))  # [B*N, T+1, C]
            temporal_outputs = self.temporal_encoder[num](temporal_inputs, None)  # [B*N, T+1, C]
            temporal_outputs = temporal_outputs.view(-1, self._num_agents, self.input_length + 1,
                                                     self.feature_dims)  # [B, N, T+1, C]
            temporal_token, temporal_dynamics = temporal_outputs[:, :, 0, :], temporal_outputs[:, :, 1:, :]
            return temporal_dynamics, temporal_token

        # dynamic encoding
        #print(observed.shape)
        dynamics = self.dynamic_encoder(observed)  # [B, N, T, C]
        #print("dynamics")
        #print(dynamics.shape) # 86,20,16,64
        action_gr_1 = group_encoding(dynamics, 0)
        action_gr_2 = group_encoding(action_gr_1, 1)
        action_gr = torch.cat((action_gr_1, action_gr_2), dim=3)
        action_gr = self.tzfc(action_gr)
        # print("action")
        # print(action.shape) #原来的86,20,16,64 新改的 32,20,16,64
        action_s_1 = spatial_encoding(dynamics, 0)
        action_s_2 = spatial_encoding(action_s_1, 1)
        action_s = torch.cat((action_s_1, action_s_2), dim=3)
        action_s = self.tzfc(action_s)

        action = torch.cat((action_gr, action_s), dim=3)
        action = self.tzfc(action)
        
        action_temporal, token = temporal_encoding(action, 0)
        #action_temporal_1, token_1 = temporal_encoding(dynamics, 0)
        #action_temporal_2, token_2 = temporal_encoding(action_temporal_1, 1)
        #token=token_2
        # action_temporal = torch.cat((action_temporal_1, action_temporal_2), dim=3)
        # action_temporal = self.tzfc(action_temporal)

        # action_temporal = torch.cat((action, action_temporal), dim=3)
        # action_temporal = self.tzfc(action_temporal)

        #print(action_temporal.shape) #32,20,16,64
        #print(token.shape) #32,20,64
        for i in range(1, self.num_spatial_time):
            now_action_temporal = action_temporal
            action_gr = group_encoding(action_temporal, i)
            action_s = spatial_encoding(action_temporal, i)
            action = torch.cat((action_gr, action_s), dim=3)
            action = self.tzfc(action)

            action_temporal, token = temporal_encoding(action, i)
            #print(now_action_temporal.shape) #32,20,16,64 差了4倍 加一个FC层，128,2,16,64
            #print(action_temporal.shape) #8,20,16,64 #128,20,16,64
            action_temporal = now_action_temporal + action_temporal 
        
        #     --------------------------decoder----------------------
        memory = action_temporal[:, 0, :, :]  # [B,1, T, C]
        # maneuver classification
        if self._mult_traj and (not self.training):
            m_logits = self.classifier(token[:, 0, :])
            maneuver_prob = F.log_softmax(self.classifier(token[:, 0, :]), dim=-1)  # [B, M]
            tracks = []
            for i in range(9):
                maneuver_idx = torch.zeros_like(maneuver_prob)
                maneuver_idx[:, i] = 1
                maneuver_bias = self.maneuver_encoder(maneuver_idx)  # [B,C] or [B, N, C] if network.multi_agents
                hist_embed = self.track_encoder(observed[:, 0:1, :, 0:2])
                # query = self.QGB(hist_embed, memory.unsqueeze(1),
                #                  self.output_length, maneuver_bias).squeeze()
                query = self.QGB(hist_embed, memory.unsqueeze(1),
                         self.output_length).squeeze() + maneuver_bias.unsqueeze(1)
                tgt_mask = self._subsequent_masks_
                output = self.decoder(memory, query, None,
                                      tgt_mask)  # [B, T', C] or [B*N, T', C] if network.multi_agents
                track = self.generator(output)
                # generate true track
                if self.use_nll:
                    track = outputActivation(track)
                tracks.append(track)
            return tracks, maneuver_prob, action_temporal, token, m_logits, output
        if self._enable_maneuver_:
            m_logits = self.classifier(token[:, 0, :])
            maneuver_prob = F.log_softmax(m_logits, dim=-1)  # [B, M]
            maneuver_idx = maneuver_prob.exp()  # [B, ] or [B, N] if network.multi_agents
            if self.training and self.use_true_man:
                maneuver_bias = self.maneuver_encoder(maneuvers)  # [B,C] or [B, N, C] if network.multi_agents
            else:
                if self.use_hard_man:
                    maneuver_bias = torch.zeros_like(maneuver_idx).to(maneuver_idx.device)
                    maneuver_bias.scatter_(-1, maneuver_idx.argmax(-1).unsqueeze(1), 1)
                    maneuver_bias = self.maneuver_encoder(maneuver_bias)  # [B,C] or [B, N, C] if network.multi_agents
                else:
                    maneuver_bias = self.maneuver_encoder(maneuver_idx)
        else:
            maneuver_prob = None
        hist_embed = self.track_encoder(observed[:, 0:1, :, 0:2])
        query = self.QGB(hist_embed, memory.unsqueeze(1),
                         self.output_length).squeeze() + maneuver_bias.unsqueeze(1)
        tgt_mask = self._subsequent_masks_
        output = self.decoder(memory, query, None,
                              tgt_mask)  # [B, T', C] or [B*N, T', C] if network.multi_agents
        track = self.generator(output)
        # generate true track
        if self.use_nll:
            track = outputActivation(track)
        return track, maneuver_prob, action_temporal, token, m_logits, output