# -*- coding = utf-8 -*-
"""
Created on 2025/7/14
author: yunda_si@ucac.ac.cn
"""


import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from modules import InputEmbedder, SeqHead
from structure_module import StructureModule
from utils.seq_utils import seq2atom, localrigids
from utils.rigid_utils import rot_to_quat

class InvDNA(nn.Module):

    def __init__(self,
                 blocks_structure_decoder,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 num_structure_recycle=8,
                 num_atom=28,
                 dropout_p=0.0,
                 dropout_p2d=0.0,
                 split_size=None,
                 split_atom=None,
                 residue_window=(-1,-1),
                 ):

        super(RFold, self).__init__()

        self.dropout_p = dropout_p
        self.num_atom = num_atom
        self.pair_channel = pair_channel
        self.num_structure_recycle = num_structure_recycle
        self.split_size = split_size

        self.embed_struc = nn.Embedding(num_atom + 1, atom_channel)

        self.input_embedder = InputEmbedder(pair_channel=pair_channel,
                                            nums_aa=6,
                                            atom_channel=atom_channel)

        self.structure_block = StructureModule(atom_channel=atom_channel,
                                               pair_channel=pair_channel,
                                               atom_nhead=atom_nhead,
                                               block_num=blocks_structure_decoder,
                                               dropout_p=dropout_p,
                                               num_atom=num_atom,
                                               split_atom=split_atom)

        self.seq_block = SeqHead(
                               atom_channel=atom_channel,
                               atom_nhead=atom_nhead,
                               pair_channel=pair_channel,
                               pair_nhead=pair_nhead,
                               dropout_p=dropout_p,
                               dropout_p2d=dropout_p2d,
                               split_size=split_size,
                               residue_window=residue_window,
                               no_bins_seq=5,
                               seq_window=(-1, -1))

    def forward(self, dna):

        pred_coords = []
        pred_seqs = []
        pred_quaternion = []
        pair_idx, seq_init = self.input_embedder(dna)
        inp_seq = dna['masked_seq']

        coords = torch.permute(dna['dna_coords'], [0, 2, 1, 3]).float()
        basis,_ = localrigids(torch.transpose(coords, 1, 2), dna['masked_seq'])
        quat = rot_to_quat(basis)
        b, a, l, c = coords.shape


        struc_emb = torch.repeat_interleave(seq2atom(inp_seq, 'dna').unsqueeze(0),b, 0)
        struc_emb = self.embed_struc(struc_emb).float() + seq_init.unsqueeze(1)

        coords, struc_emb, quat = checkpoint(self.structure_block, struc_emb, pair_idx, coords, quat, use_reentrant=False)

        pred_coords.append(coords * 10)
        pred_quaternion.append(quat)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            pred_seq = self.seq_block(struc_emb[0:1])
        pred_seqs.append(pred_seq)

        pred_coords = torch.permute(torch.stack(pred_coords), [0, 1, 3, 2, 4]) # S*B*L*A*3
        pred_seqs = torch.stack(pred_seqs)
        pred_quaternion = torch.stack(pred_quaternion)

        return {
                'pred_coords': pred_coords,
                'pred_seqs': pred_seqs,
                'pred_quaternion': pred_quaternion,
                }