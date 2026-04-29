# -*- coding = utf-8 -*-
"""
Created on 2025/7/14
author: yunda_si@ucac.ac.cn
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from utils.rigid_utils import quat_to_rot, quat_multiply_by_vec

class StructureModuleTransition(nn.Module):

    def __init__(
                self,
                in_channels,
                dropout_p=0.1,
                scalen=4,
                ):
        super().__init__()

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

        x = residual + self.dropout_module(x)
        return x


class IPA(nn.Module):

    def __init__(self,
                 pair_channel=128,
                 atom_channel=128,
                 num_atom=27,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 ):

        super(IPA, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.q_seq = nn.Linear(atom_channel, atom_channel, bias=False)
        self.k_seq = nn.Linear(atom_channel, atom_channel, bias=False)
        # self.v_seq = nn.Linear(atom_channel, atom_channel, bias=False)

        self.qp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.kp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.vp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)

        self.norm_pair = nn.RMSNorm(pair_channel)
        self.trans_pair = nn.Parameter((torch.rand(num_atom, pair_channel, atom_head) - 0.5) * 2 / (pair_channel ** 0.5))
        self.head_weights = nn.Parameter(torch.zeros(self.nhead))
        self.ipa_point_weights_init_(self.head_weights)

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)

    @staticmethod
    def ipa_point_weights_init_(weights):
        with torch.no_grad():
            softplus_inverse_1 = 0.541324854612918
            weights.fill_(softplus_inverse_1)

    def forward(self, seq, pair, coords, basis, atom_idx=None):
        residue_seq = seq

        seq = self.ln_seq_in(seq)
        if atom_idx == None:
            pair_bias = torch.einsum('bijc, ach -> bhaij', self.norm_pair(pair), self.trans_pair)
        else:
            pair_bias = torch.einsum('bijc, ach -> bhaij', self.norm_pair(pair), self.trans_pair[atom_idx])

        batch, num_atom, num_residue, _ = seq.shape

        q = self.q_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)
        k = self.k_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)
        # v = self.v_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)

        # q = self.rope(q.transpose(2,3)).transpose(2,3)
        # k = self.rope(k.transpose(2, 3)).transpose(2, 3)

        q = q * self.scale_q
        q_attn_weights = torch.einsum('baihc,bajhc -> bhaij', q, k)

        qp = self.qp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        kp = self.kp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        vp = self.vp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)

        qp = torch.einsum('balhpi, blji -> bhalpj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        kp = torch.einsum('balhpi, blji -> bhalpj', kp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        vp = torch.einsum('balhpi, blji -> bhalpj', vp, basis) + coords.unsqueeze(1).unsqueeze(-2)

        qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                              kp ** 2).unsqueeze(
            -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * self.scale_qp * (-0.5)
        for i in range(3):
            head_weights = head_weights.unsqueeze(-1)

        qp_attn_weights = qp_attn_weights * head_weights

        attn = pair_bias + q_attn_weights + qp_attn_weights
        attn = attn * (3 ** -0.5)
        attn = attn.softmax(-1)

        o3 = torch.einsum('bhaij, bhajpc -> bhaipc', attn, vp) - coords.unsqueeze(1).unsqueeze(-2)
        o3 = torch.einsum('bhalpi, blij -> balhpj', o3, basis)
        o3 = torch.flatten(o3, -3, -1)

        seq = self.linear_out(torch.cat((o3,), dim=-1))
        seq = residue_seq + self.dropout_module(seq)

        return seq


class IPAATOM(nn.Module):

    def __init__(self,
                 atom_channel=128,
                 pair_channel=128,
                 num_atom=26,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 ):
        super(IPAATOM, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.q_seq = nn.Linear(atom_channel, atom_channel, bias=False)
        self.k_seq = nn.Linear(atom_channel, atom_channel, bias=False)

        self.qp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.kp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.vp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)

        self.head_weights = nn.Parameter(torch.zeros(self.nhead))
        self.ipa_point_weights_init_(self.head_weights)

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)

    def ipa_point_weights_init_(self, weights):
        with torch.no_grad():
            softplus_inverse_1 = 0.541324854612918
            weights.fill_(softplus_inverse_1)

    def forward(self, seq, coords, basis):
        residue_seq = seq

        coords = torch.permute(coords, [0, 2, 1, 3]) #blac

        seq = self.ln_seq_in(seq)
        batch, num_atom, num_residue, _ = seq.shape

        q = self.q_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)
        k = self.k_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)

        q = q * self.scale_q
        q_attn_weights = torch.einsum('bilhc,bjlhc -> bhlij', q, k)

        qp = self.qp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        kp = self.kp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        vp = self.vp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)

        qp = torch.einsum('balhpi, blji -> bhlapj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        kp = torch.einsum('balhpi, blji -> bhlapj', kp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        vp = torch.einsum('balhpi, blji -> bhlapj', vp, basis) + coords.unsqueeze(1).unsqueeze(-2)

        qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                              kp ** 2).unsqueeze(
            -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * self.scale_qp * (-0.5)
        for i in range(3):
            head_weights = head_weights.unsqueeze(-1)

        qp_attn_weights = qp_attn_weights * head_weights

        attn = q_attn_weights + qp_attn_weights
        attn = attn * (2 ** -0.5)
        attn = attn.softmax(-1)

        o3 = torch.einsum('bhlij, bhljpc -> bhlipc', attn, vp) - coords.unsqueeze(1).unsqueeze(-2)
        o3 = torch.einsum('bhlapi, blij -> balhpj', o3, basis)
        o3 = torch.flatten(o3, -3, -1)

        seq = self.linear_out(torch.cat((o3,), dim=-1))
        seq = residue_seq + self.dropout_module(seq)

        return seq


class StructureBlock(nn.Module):
    def __init__(self,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=28,
                 dropout_p=0.0,
                 points=8,
                 eps=1e-7,
                 split_atom=None
                 ):

        super(StructureBlock, self).__init__()

        self.num_atom = num_atom
        self.split_atom = split_atom

        self.ipa = IPA(pair_channel=pair_channel,
                       atom_channel=atom_channel,
                       num_atom=num_atom,
                       atom_head=atom_nhead,
                       points=points,
                       eps=eps,
                       dropout_p=dropout_p
                       )

        self.ipa_atom = IPAATOM(atom_channel=atom_channel,
                                pair_channel=pair_channel,
                                num_atom=num_atom,
                                atom_head=atom_nhead,
                                points=points,
                                eps=eps,
                                dropout_p=dropout_p
                                )

        self.transition = StructureModuleTransition(in_channels=atom_channel,
                                                    dropout_p=dropout_p)

    def forward(self, struc_emb, pair, coords, basis):

        b_seq = struc_emb.shape[0]

        if self.split_atom != None:
            full_out = []
            for atom_idx in torch.split(torch.arange(self.num_atom), self.split_atom):
                x = self.ipa(struc_emb[:, atom_idx], pair, coords[:, atom_idx], basis, atom_idx=atom_idx)
                full_out.append(x)
            struc_emb = torch.concat(full_out, dim=1)
        else:
            struc_emb = self.ipa(struc_emb, pair, coords, basis)

        struc_emb = self.ipa_atom(struc_emb, coords, basis)

        struc_emb = self.transition(struc_emb)

        return struc_emb


class StructureModule(nn.Module):

    def __init__(self,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=28,
                 block_num=8,
                 using_flash=True,
                 dropout_p=0.0,
                 split_atom=None,
                 split_size=None
                 ):

        super(StructureModule, self).__init__()

        self.dropout_p = dropout_p
        self.using_flash = using_flash
        self.split_atom = split_atom
        self.split_size = split_size

        self.structure_block = self._make_atom_encoder(block_num, atom_channel, pair_channel, atom_nhead, num_atom)
        self.ln_seq_out = nn.RMSNorm(atom_channel)

        self.coords_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                        nn.RMSNorm(atom_channel),
                                        nn.GELU())

        self.weights_coords = nn.Parameter((torch.rand(num_atom, atom_channel, 3) - 0.5) * 2 / (atom_channel ** 0.5))

        self.basis_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                       nn.RMSNorm(atom_channel),
                                       nn.GELU(),
                                       nn.Linear(atom_channel, 3, bias=False),
                                       )

    def _make_atom_encoder(self, block_num, atom_channel, pair_channel, atom_nhead, num_atom):

        layers = []
        for index in range(block_num):
            layer = StructureBlock(atom_channel=atom_channel,
                                   pair_channel=pair_channel,
                                   atom_nhead=atom_nhead,
                                   num_atom=num_atom,
                                   dropout_p=self.dropout_p,
                                   split_atom=self.split_atom)

            layers.append(('struc_block' + str(index), layer))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, struc_emb, pair, ori_coords, quat):

        quat = quat.float().detach()
        basis = quat_to_rot(quat)

        for idx_layer, layer in enumerate(self.structure_block):
            struc_emb = layer(struc_emb, pair, ori_coords, basis)

        seq = self.ln_seq_out(struc_emb.float())
        coords = torch.einsum('bali, aij -> balj', self.coords_out(seq), self.weights_coords)
        coords = ori_coords + torch.einsum('bali, blji -> balj', coords, basis)#*((1-update_atom).

        quat_update = self.basis_out(seq[:, 3, :, :])
        new_quat = quat + quat_multiply_by_vec(quat, quat_update)
        new_quat = new_quat / torch.linalg.norm(new_quat, dim=-1, keepdim=True)

        return coords, struc_emb, new_quat


