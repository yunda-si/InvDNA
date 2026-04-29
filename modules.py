# -*- coding = utf-8 -*-
"""
Created on 2025/7/14
author: yunda_si@ucac.ac.cn
"""

import torch
from torch import nn
from torch.nn import RMSNorm
import math
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint


class FeedForwardNetwork(nn.Module):

    def __init__(
                self,
                in_channels,
                dropout_p=0.1,
                scalen=4,
                ):
        super(FeedForwardNetwork, self).__init__()

        hidden_channels = scalen*in_channels

        self.norm = nn.RMSNorm(in_channels)
        self.left = nn.Linear(in_channels, hidden_channels, bias=False)
        self.right = nn.Linear(in_channels, hidden_channels, bias=False)
        self.linear_ff = nn.Linear(hidden_channels, in_channels, bias=False)

        self.dropout_module = nn.Dropout(dropout_p)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.norm(x)

        left = self.left(x)
        right = self.right(x)
        x = self.linear_ff(self.act(left) * right)

        x = residual + x#self.dropout_module(x)
        return x


class InputEmbedder(nn.Module):
    def __init__(self,
                 pair_channel,
                 nums_aa=5,
                 atom_channel=256,
                 nums_posclass=65,
                 ):
        super(InputEmbedder, self).__init__()

        self.embd_pos = nn.Embedding(nums_posclass, pair_channel)
        self.embed_seq = nn.Embedding(nums_aa, atom_channel)

    def forward(self, dna):

        idx = self.relpos(dna['sel_idx']).to(dna['dna_coords'].device)
        idx = self.embd_pos(idx).unsqueeze(0)
        seq_init = self.embed_seq(dna[f'masked_seq'].unsqueeze(0))

        return idx, seq_init

    def relpos(self, idx, min_dis=-32, max_dis=32):
        num_class = max_dis - min_dis + 1

        idx = idx.unsqueeze(0) - idx.unsqueeze(1)
        idx = torch.clamp(idx, min_dis, max_dis) - min_dis


        return idx



class SeqHead(nn.Module):

    def __init__(self,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 dropout_p,
                 dropout_p2d,
                 split_size,
                 residue_window,
                 seq_window,
                 no_bins_seq=5,
                 num_atom=28,
                 tied_attn=False,
                 with_column_attn=False,
                 ):
        super(SeqHead, self).__init__()

        self.atom_channel = atom_channel
        self.atom_nhead = atom_nhead
        self.pair_channel = pair_channel
        self.pair_nhead = pair_nhead
        self.dropout_p = dropout_p
        self.dropout_p2d = dropout_p2d
        self.split_size = split_size
        self.residue_window = residue_window
        self.seq_window = seq_window
        self.tied_attn = tied_attn
        self.with_column_attn = with_column_attn

        self.out_seq = nn.Sequential(nn.RMSNorm(atom_channel),
                                     nn.Linear(atom_channel, no_bins_seq, bias=False))


    def forward(self, seq):

        pred_seq = self.out_seq(seq[:,5:6])
        return pred_seq
