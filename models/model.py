from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np
from numpy.core.fromnumeric import repeat
import torch
import torch as th
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


class TemporalSegmentation(nn.Module):
    def __init__(self, kernelInfo, input_encoding_size, temporal_length, clip_number):
        super(TemporalSegmentation, self).__init__()

        self.kernel_low = kernelInfo[0]
        self.kernel_high = kernelInfo[1]
        self.kernel_interval = kernelInfo[2]
        self.temporal_length = temporal_length
        self.input_encoding_size = input_encoding_size
        self.kernel_number = (
            self.kernel_high-self.kernel_low)/self.kernel_interval+1
        self.kernel_list = []
        ts = []
        # 16 Kernels/Anchors = [3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123]
        for i in range(1, int(self.kernel_number)+1):
            kernel_size = self.kernel_low+(i-1)*self.kernel_interval
            self.kernel_list.append(kernel_size)
            ts.append(nn.Conv1d(self.input_encoding_size, 3, kernel_size))
        
        self.anchor_l = torch.FloatTensor(self.kernel_list)
        self.anchor_l = nn.parameter.Parameter(repeat(self.anchor_l, 'k -> b k t', b=1, t=self.temporal_length))
        # for kernel in 
        self.anchor_c = torch.from_numpy(np.arange(1, self.temporal_length+1)).float()
        self.anchor_c = nn.parameter.Parameter(repeat(self.anchor_c, 't -> b k t', b=1, k=int(self.kernel_number)))

        self.ts = nn.ModuleList(ts)
        self.proposal_score_fn = nn.Sigmoid()
        self.lenth_offset_fn = nn.Tanh()
        self.center_offset_fn = nn.Tanh()

    def forward(self, x):
        """"
        x: Batch X Temporal_length X Feature_dim
        """
        device_id='cpu'
        if x.is_cuda:
            device_id = 'cuda:{}'.format(x.get_device())
        B = x.size(0)
        tc_output = torch.zeros(
            (B, int(self.kernel_number), 3, self.temporal_length), device=device_id)

        x = x.transpose(1, 2)
        # Temporal Convolution
        for i, kernel in enumerate(self.ts):
            kernel_size = self.kernel_list[i]
            tc_output[:, i, :, int((kernel_size-1)/2):self.temporal_length -
                      int((kernel_size-1)/2)] = kernel(x)
        anchor_proposals = self.proposal_score_fn(tc_output[:, :, 0, :])
        # TODO: Why multiplication of 0.1
        # B x K x L
        length_offset = self.lenth_offset_fn(tc_output[:, :, 1, :])
        center_offset = self.center_offset_fn(tc_output[:, :, 2, :])

        self.tc_boundary = torch.zeros(
            (B, 2, int(self.kernel_number), self.temporal_length), device=device_id)

        length = self.anchor_l * torch.exp(length_offset)
        center = self.anchor_c + center_offset * self.anchor_l
        self.tc_boundary[:, 0, :, :] = center - length / 2
        self.tc_boundary[:, 1, :, :] = center + length / 2
        return anchor_proposals, self.tc_boundary


class ProcNet(nn.Module):
    def __init__(self, input_encoding_size=512, rnn_size=512, clip_number=16,
                 kernelInfo=[3, 123, 8], frames_per_video=500, mp_scale=(8, 5), video_feat=512):
        super(ProcNet, self).__init__()
        """
        input_encoding_size: size of image feature (resnet34)
        rnn_size: bi-lstm hidden size
        clip number : maximum number of clips per video
        kernelinfo: tuple of three elements=
                    - smallest kernel size for temporal conv
                    - largest kernel size for temporal conv
                    - kernel size interval for temporal conv
        mp_scale_h: proposal score max pooling kernel height
        mp_scale_w: proposal score max pooling kernel width

        """

        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.kernel_low = kernelInfo[0]
        self.kernel_high = kernelInfo[1]
        self.kernel_interval = kernelInfo[2]
        self.clip_number = clip_number
        self.frames_per_video = frames_per_video
        self.kernel_list = []
        self.mp_scale_h = mp_scale[0]
        self.mp_scale_w = mp_scale[1]

        self.core1 = nn.LSTM(input_size=video_feat,
                             hidden_size=self.rnn_size,
                             batch_first=True,
                             bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(3*self.input_encoding_size, self.input_encoding_size),
            # nn.BatchNorm1d(self.input_encoding_size),
            nn.ReLU(),
        )
        self.temporal_segment = TemporalSegmentation(
            [self.kernel_low, self.kernel_high, self.kernel_interval], self.input_encoding_size, self.frames_per_video, clip_number)

        self.score_maxpool = nn.MaxPool2d(
            (self.mp_scale_h, self.mp_scale_w), stride=(self.mp_scale_h, self.mp_scale_w), return_indices=True)

        self.clip_prop_encoding = self.frames_per_video * \
            ((self.kernel_high-self.kernel_low)/self.kernel_interval+1)
        self.clip_prop_encoding = int(self.clip_prop_encoding/self.mp_scale_h/self.mp_scale_w)
        # 2 Extra Embedding for Start/End Token
        self.loc_embedding = nn.Embedding(
            self.clip_prop_encoding+1, self.clip_prop_encoding)
        self.core2 = nn.LSTMCell(
            input_size=self.clip_prop_encoding*3, hidden_size=self.clip_prop_encoding)

        self.dropout = nn.Dropout(p=0.5)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.seq_fc = nn.Linear(in_features=self.clip_prop_encoding, out_features=self.clip_prop_encoding+1)

        # Transform Masked and Averaged Features for Sequential Input ready
        self.vis_feat_linear = nn.Linear(self.input_encoding_size, self.clip_prop_encoding)

    def get_nearest_segments(self, maxpool_boundaries, gt_segments, device_id):
        B, N, C = gt_segments.size()
        assert B == 1, "Batch Size should be 1 only."
        gt_prop_segments = torch.empty(N, dtype=torch.long, device=device_id).fill_(self.clip_prop_encoding)
        _, _, k, l = maxpool_boundaries.size()
        maxpool_boundaries_local = maxpool_boundaries[0].view(2, k*l)
        for i in range(N):
            gt = gt_segments[0][i]
            gt = repeat(gt, 'c -> c t', t=k*l)
            dist = (maxpool_boundaries_local[1, :] - gt[1, :])**2 + \
                (maxpool_boundaries_local[0, :] - gt[0, :])**2
            near_index = torch.argmin(dist).item()
            gt_prop_segments[i] = near_index

        return gt_prop_segments

    def train_seq_forward(self, vis_feats, maxpool_proposals, maxpool_boundaries, gt_segments, device_id):
        # TODO: Constrain the Model to work only for BatchSize = 1
        maxpool_boundaries_detached = maxpool_boundaries.detach().cpu()
        gt_seg_enc = self.get_nearest_segments(
                maxpool_boundaries_detached, gt_segments.detach().cpu(), device_id)

        _, K, L = maxpool_proposals.size()
        # Sequential Prediction
        lstm2_output = []
        N = gt_segments.size(1) if gt_segments is not None else self.clip_number
        for t in range(N):
            # t = 0 is for start Token
            emp_input = self.clip_prop_encoding if t==0 else gt_seg_enc[t-1]
            emb = self.loc_embedding(torch.empty(1, dtype=torch.long, device=device_id).fill_(emp_input))
            _input = torch.cat((maxpool_proposals.view(1,-1), emb), dim=-1)
            # TODO: Concatenate Masked Segment/Image Features
            if t == 0:
                masked_feat = self.vis_feat_linear(torch.mean(vis_feats, dim=1))
                _input = torch.cat((_input, masked_feat), dim=-1)
                pred_t_h, pred_t_c = self.core2(_input) # Initial State is Zero
            else:
                row = gt_seg_enc[t-1] // L
                col = gt_seg_enc[t-1] % L
                start_ = max(0, torch.floor(maxpool_boundaries_detached[0, 0,row, col]).long().item())
                end_ = min(L, torch.ceil(maxpool_boundaries_detached[0, 1,row, col]).long().item())
                mask = torch.empty_like(vis_feats).fill_(0.)
                mask[:, start_:end_, :] = 1
                masked_feat = self.vis_feat_linear(torch.mean(vis_feats*mask, dim=1))
                _input = torch.cat((_input, masked_feat), dim=-1)
                pred_t_h, pred_t_c = self.core2(_input, (pred_t_h, pred_t_c))
            pred_t = self.seq_fc(self.dropout(pred_t_h))
            
            lstm2_output.append(pred_t.squeeze(0))
        
        return gt_seg_enc, torch.stack(lstm2_output, dim=0)


    def val_seq_forward(self, vis_feats, maxpool_proposals, maxpool_boundaries, device_id):
        # Sequential Prediction
        _, K, L = maxpool_proposals.size()
        lstm2_output = []
        prev_proposal_index = [self.clip_prop_encoding for t in range(self.clip_number)]
        no_further_processing = False
        for t in range(self.clip_number):
            if no_further_processing:
                lstm2_output.append(torch.empty_like(pred_t.squeeze(0)).fill_(self.clip_prop_encoding))
                continue
            # t = 0 is for start Token
            emp_input = self.clip_prop_encoding if t==0 else prev_proposal_index[t-1]
            emb = self.loc_embedding(torch.empty(1, dtype=torch.long, device=device_id).fill_(emp_input))
            _input = torch.cat((maxpool_proposals.view(1,-1), emb), dim=-1)
            # TODO: Concatenate Masked Segment/Image Features
            if t == 0:
                masked_feat = self.vis_feat_linear(torch.mean(vis_feats, dim=1))
                _input = torch.cat((_input, masked_feat), dim=-1)
                pred_t_h, pred_t_c = self.core2(_input) # Initial State is Zero
            else:
                row = prev_proposal_index[t-1] // L
                col = prev_proposal_index[t-1] % L
                start_ = max(0, torch.floor(maxpool_boundaries[0, 0,row, col]).long().item())
                end_ = min(L, torch.ceil(maxpool_boundaries[0, 1,row, col]).long().item())
                mask = torch.empty_like(vis_feats).fill_(0.)
                mask[:, start_:end_, :] = 1.
                masked_feat = self.vis_feat_linear(torch.mean(vis_feats*mask, dim=1))
                _input = torch.cat((_input, masked_feat), dim=-1)
                pred_t_h, pred_t_c = self.core2(_input, (pred_t_h, pred_t_c))
            pred_t = self.seq_fc(self.dropout(pred_t_h))
            
            next_segment_index = torch.argmax(pred_t)
            prev_proposal_index[t] = next_segment_index
            lstm2_output.append(pred_t.squeeze(0))
            # Break if Start/End token is predicted.
            if next_segment_index == self.clip_prop_encoding:
                no_further_processing = True

        return torch.LongTensor(prev_proposal_index).to(device_id), torch.stack(lstm2_output, dim=0)
        

    def forward(self, input_x, gt_segments):
        device_id='cpu'
        if input_x.is_cuda:
            device_id = 'cuda:{}'.format(input_x.get_device())

        # Context aware video encoding
        feat_context, _ = self.core1(input_x)
        x = torch.cat((input_x, feat_context), dim=-1)
        x = self.mlp(x)

        # Procedure Segment Proposal
        proposals, boundaries = self.temporal_segment(x)
        maxpool_proposals, indices = self.score_maxpool(proposals)
        B, K_h, L_w = maxpool_proposals.size()
        maxpool_boundaries = torch.gather(torch.flatten(
            boundaries, 2), 2, torch.flatten(repeat(indices, 'b k t -> b s k t', s=2), 2))
        maxpool_boundaries = maxpool_boundaries.view(B, 2, K_h, L_w)

        # Sequential Module
        if self.training:
            gt_seg_enc, lstm2_output = self.train_seq_forward(input_x, maxpool_proposals, maxpool_boundaries, gt_segments, device_id)
            return proposals, boundaries, gt_seg_enc, lstm2_output
        else:
            pred_seg_index, lstm2_output = self.val_seq_forward(input_x, maxpool_proposals, maxpool_boundaries, device_id)
            return maxpool_proposals, maxpool_boundaries, pred_seg_index, lstm2_output


if __name__ == '__main__':
    model = ProcNet(input_encoding_size=512, rnn_size=512, clip_number=16,
                    kernelInfo=[3, 123, 8], frames_per_video=500, mp_scale=(8,5),
                    video_feat=512)
    model.cuda()
    model.train()
    inp = torch.randn(1, 500, 512).float().cuda()
    segments = torch.randn(1,10,2).float().cuda()
    out, out2, out3, out4 = model(inp, segments)
    print(out.size(), out2.size(), out3.size(), out4.size())

    model.eval()
    with torch.no_grad():
        out, out2, out3, out4 = model(inp, segments)
        print(out.size(), out2.size(), out3.size(), out4.size())
    del model, inp, out, out2, out3, out4
    torch.cuda.empty_cache()
