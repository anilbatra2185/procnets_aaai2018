from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np
from numpy.core.fromnumeric import repeat
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


class ProcNetLoss(nn.Module):
    def __init__(self, samples=200, kernelInfo=[3, 123, 8]):
        super(ProcNetLoss, self).__init__()

        self.ktl = kernelInfo[0]
        self.ktu = kernelInfo[1]
        self.kts = kernelInfo[2]
        self.train_sample = samples
        self.pos_iou_thresh = 0.8
        self.neg_iou_thresh = 0.2
        self.L1Loss = nn.SmoothL1Loss()
        self.bce = nn.BCELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss()

    def compute_iou(self, c1, c2):
        intersection = max(0, min(c1[1], c2[1])-max(c1[0], c2[0]))
        if intersection == 0:
            return 0
        else:
            union = max(c1[1], c2[1]) - min(c1[0], c2[0])
            return intersection/union

    def find_pos_neg_samples(self, boundaries, gt_boundaries):
        """
        boundaries - 2 x K x L - 2 x 16 x 500
        gt_boundaries - 2 x N, N = number of segments in a video, which will vary per video
        """
        pos_mask = None
        neg_mask = None
        # print(boundaries.size(), gt_boundaries.size())
        _, K, L = boundaries.size()
        rand_order_i = torch.rand(K)
        _, ind_i = torch.sort(rand_order_i)
        rand_order_j = torch.rand(L)
        _, ind_j = torch.sort(rand_order_j)
        # Get Number of segments
        seg_num = gt_boundaries.size(1)
        pos_mask = torch.zeros(K, L)
        neg_mask = torch.zeros(K, L)
        pos_segments = torch.zeros(2, K, L)

        positive_samples = 0
        negative_samples = 0
        for ii in range(K):
            for jj in range(L):
                i = ind_i[ii]
                j = ind_j[jj]
                clip_low = boundaries[0][i][j]
                clip_high = boundaries[1][i][j]
                pos_flag = 0
                neg_flag = 1
                if clip_low > 0 and clip_high <= L:
                    rand_order_seg = torch.rand(seg_num)
                    _, ind_seg = torch.sort(rand_order_seg)
                    for kk in range(seg_num):
                        k = ind_seg[kk]
                        clip_iou = self.compute_iou(
                            c1=(clip_low, clip_high), c2=gt_boundaries[k])
                        if clip_iou > self.pos_iou_thresh:
                            pos_segments[0, i, j] = gt_boundaries[k][0]
                            pos_segments[1, i, j] = gt_boundaries[k][1]
                            pos_flag = 1
                            break
                        elif clip_iou > self.neg_iou_thresh:
                            neg_flag = 0
                    if pos_flag == 1:
                        positive_samples += 1
                        pos_mask[i, j] = 1
                    elif neg_flag == 1:
                        negative_samples += 1
                        neg_mask[i, j] = 1
                if positive_samples >= self.train_sample/2 and negative_samples >= self.train_sample/2:
                    break
            if positive_samples >= self.train_sample/2 and negative_samples >= self.train_sample/2:
                break

        return pos_mask.unsqueeze(0), neg_mask.unsqueeze(0), pos_segments.unsqueeze(0)

    def forward(self, input_segments, gt_segments):
        device_id='cpu'
        if gt_segments.is_cuda:
            device_id = 'cuda:{}'.format(gt_segments.get_device())

        prop_scores = input_segments[0]
        prop_boundary = input_segments[1]
        gt_segment_enc = input_segments[2].to(device_id)
        sequential_segment_index = input_segments[3]

        pos_mask, neg_mask, pos_gt_segments = self.find_pos_neg_samples(boundaries=prop_boundary.detach(
        ).cpu().clone().squeeze(0), gt_boundaries=gt_segments.detach().cpu().clone().squeeze(0))

        pos_mask = pos_mask.to(device_id)
        neg_mask = neg_mask.to(device_id)
        pos_gt_segments = pos_gt_segments.to(device_id)
        
        B, K, L = prop_scores.size()
        binary_target = torch.zeros(B, K, L).float().to(device_id)
        binary_target[pos_mask == 1] = 1
        mask = pos_mask + neg_mask

        loss_classification = self.bce(prop_scores, binary_target)
        # Consider only positive and negative training samples, ignore other
        loss_classification = torch.div(
            torch.sum(loss_classification * mask), torch.sum(mask))
        # Boundary regression for positive samples.
        positive_prop_boundary = prop_boundary * pos_mask
        boundary_loss = self.L1Loss(positive_prop_boundary, pos_gt_segments)
        # Cross Entroy loss on next segment index prediction by sequential lstm
        sequential_loss = self.ce(sequential_segment_index, gt_segment_enc)
        # Equal weights give good results as per the paper, mentioned in Loss Function paragraph
        loss = loss_classification + boundary_loss + sequential_loss

        return loss
