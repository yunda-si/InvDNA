# -*- coding = utf-8 -*-
"""
Created on 2025/7/18
author: yunda_si@ucac.ac.cn
"""

import random
from model import InvDNA
import torch
from dataset import MonomerDataset
import warnings
from config import get_cfg
from utils import to_pdb
import os
import numpy as np
import np.residue_constants as resc
import argparse

def predicted(weight_file, device, total_feature):

    #### model
    cfg = get_cfg(1)
    model = InvDNA(blocks_structure_decoder=cfg.model.blocks_structure_decoder,
                  atom_channel=cfg.model.atom_channel,
                  atom_nhead=cfg.model.atom_nhead,
                  pair_channel=cfg.model.pair_channel,
                  pair_nhead=cfg.model.pair_nhead,
                  num_structure_recycle=cfg.model.num_structure_recycle,
                  dropout_p=cfg.model.dropout_p,
                  dropout_p2d=cfg.model.dropout_p2d,
                  num_atom=cfg.model.num_atom,
                  residue_window=cfg.model.residue_window,
                  )
    checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
    checkpoint = {i: j for i, j in checkpoint.items() if 'rope' not in i}
    
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    #### model
    for key, value in total_feature.items():
        if type(value) is torch.Tensor:
            total_feature[key] = value.to(device)

    preds = model(total_feature)

    for key, value in total_feature.items():
        if type(value) == torch.Tensor:
            total_feature[key] = value.cpu()

    total_feature['sel_idx'] = total_feature['sel_idx'].cpu().numpy()
    pred_coords = preds['pred_coords'][0,0].cpu()

    unrelaxed_protein = to_pdb.to_pdb(pred_coords, [int(i) for i in torch.max(preds['pred_seqs'].cpu()[0].squeeze(), dim=-1).indices])
    with open(total_feature['save_pdb_file'], 'w') as fp:
        fp.write(unrelaxed_protein)

    with open(total_feature['save_seq_file'], 'w') as fp:
        fp.write('>pred\n')
        fp.write(''.join([resc.rev_restype1_order[int(i.item())] for i in torch.max(preds['pred_seqs'].cpu()[0].squeeze(), dim=-1).indices]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='InvDNA')
    parser.add_argument('-seq_file', '--seq_file',
                        default="/mnt/raid5/yunda_si/datasets/dna_design/test_set_dna/single_seq/RXl2mEM7Uqxu_0.fasta",
                        type=str,
                        help='fasta format, contains "ATCGX"',
                        required=False)
    parser.add_argument('-pdb_file', '--pdb_file',
                        default="/mnt/raid5/yunda_si/datasets/dna_design/test_set_dna/single_pdb/RXl2mEM7Uqxu_0.pdb",
                        type=str,
                        help='pdb format',
                        required=False)
    parser.add_argument('-save_path', '--save_path',
                        default='./',
                        type=str,
                        help='path to the save directory',
                        required=False)
    parser.add_argument('-weight', '--weight_file',
                        default="/mnt/raid5/yunda_si/PythonProjects/DNAdesign/model_dna_design/full_20_seq.pt",
                        type=str,
                        help='model weights')
    parser.add_argument('-noise_sd', '--noise_sd',
                        default=0.02,
                        type=int,
                        help='sd of noises add to backbone',
                        required=False)
    parser.add_argument('-sample', '--sample_atoms',
                        default=True,
                        type=bool,
                        help='diversity output',
                        required=False)
    parser.add_argument('-device', '--device',
                        default='cuda:0',
                        type=str,
                        help='device to run the model')
    parser.add_argument('-seed', '--random_seed',
                        default=42,
                        type=int,
                        help='random seed',
                        required=False)
    args = parser.parse_args()

    seq_file = args.seq_file
    pdb_file = args.pdb_file
    save_path = args.save_path
    weight_file = args.weight_file
    noise_sd = args.noise_sd
    sample = args.sample_atoms
    device = torch.device(args.device)
    seed = args.random_seed

    random.seed(seed)
    torch.manual_seed(seed)


    Monomer = MonomerDataset(bb_atom_types=['OP1', 'OP2', "C2'", "C4'", "C5'", "O3'", "O4'", "O5'", "P", "C1'", "C3'"],
                             atom_type_num=28,
                             noise_sd=noise_sd,
                             sample=sample)

    features = Monomer.get_feature(seq_file, pdb_file)
    features['save_seq_file'] = os.path.join(save_path, 'seq.fasta')
    features['save_pdb_file'] = os.path.join(save_path, 'structure.pdb')

    predicted(weight_file, device, features)




