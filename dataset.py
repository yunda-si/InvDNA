# -*- coding = utf-8 -*-
"""
Created on 2025/7/14
author: yunda_si@ucac.ac.cn
"""

import os.path
from torch.utils.data import Dataset
import numpy as np
import torch
import np.residue_constants as resc
from Bio.PDB import MMCIFParser, PDBParser

cifparser = MMCIFParser()
pdbparser = PDBParser()

class MonomerDataset(Dataset):

    def __init__(self,
                 bb_atom_types=None,
                 atom_type_num=28,
                 sample=False,
                 noise_sd=0.02,
                 ):

        self.bb_atom_types = bb_atom_types
        self.atom_type_num = atom_type_num
        self.sample = sample
        self.noise_sd = noise_sd

    def get_feature(self, seq_file, pdb_file):

        seq = open(seq_file).readlines()[-1].strip()
        seq_np = [resc.restype1_order[i] for i in seq]

        seq_length = len(seq)
        sel_idx = torch.arange(seq_length)

        monomer_feature = {'sel_idx': sel_idx,
                           'seq_length': seq_length,
                           'masked_seq': torch.from_numpy(np.array(seq_np)).long(),
                           }

        if pdb_file.endswith('.cif'):
            structure = cifparser.get_structure(' ', pdb_file)
        else:
            structure = pdbparser.get_structure(' ', pdb_file)

        chain_structure = [j for j in [i for i in structure][0]][0]
        atom_mask, atom_coords = self.get_coord(chain_structure)


        if self.sample == True:
            sel_bb_atoms = torch.zeros(seq_length, self.atom_type_num).long()#np.random.choice(a=self.bb_atom_types, replace=False, size=(np.random.randint(1,len(self.bb_atom_types))))
            for i in range(len(sel_idx)):
                res_bb_atoms = np.random.choice(a=self.bb_atom_types, replace=False, size=(np.random.randint(0,len(self.bb_atom_types))))
                res_bb_atoms = [i for i in res_bb_atoms] + ["P", "C3'", "C1'"]
                res_bb_atoms = set(res_bb_atoms)
                res_bb_atoms = [resc.atom_types.index(i) for i in res_bb_atoms]
                sel_bb_atoms[i, res_bb_atoms] = 1
        else:
            sel_bb_atoms = torch.zeros(seq_length, self.atom_type_num).long()
            for i in range(len(sel_idx)):
                res_bb_atoms = [resc.atom_types.index(i) for i in self.bb_atom_types]
                sel_bb_atoms[i, res_bb_atoms] = 1


        peptide_coords = [self.buildpeptide(atom_coords, sel_bb_atoms*atom_mask)]
        peptide_coords = torch.stack(peptide_coords)
        peptide_coords = peptide_coords / 10
        monomer_feature['dna_coords'] = peptide_coords

        return monomer_feature


    def buildpeptide(self, atom_coords, bb_atom_idx):

        ori_pep_coords = torch.repeat_interleave(atom_coords[:,:1,:], dim=1, repeats=atom_coords.shape[1]).float()
        ori_pep_coords[bb_atom_idx==1] = atom_coords[bb_atom_idx==1]
        pep_coords = ori_pep_coords - torch.mean(ori_pep_coords[:,:3,:], dim=(0,1))

        pep_coords += torch.normal(0,self.noise_sd, pep_coords.shape)
        pep_coords -= torch.mean(pep_coords[:,:3,:], dim=(0,1))

        return pep_coords


    def get_coord(self, chain):
        num_residues = len(chain)
        atom_coords = np.zeros((num_residues, self.atom_type_num, 3))
        atom_mask = np.zeros((num_residues, self.atom_type_num))  # 0 means mask


        for residue_idx, residue in enumerate(chain):

            if residue.get_resname() not in resc.restype_3to1:
                continue
            for atom in residue:
                if atom.name not in resc.residue_atoms[residue.get_resname()]:
                    continue

                atom_coords[residue_idx, resc.atom_order[atom.name]] = atom.coord
                atom_mask[residue_idx, resc.atom_order[atom.name]] = 1.0

        atom_mask = torch.tensor(atom_mask).float()
        atom_coords = torch.tensor(atom_coords).float()

        return atom_mask, atom_coords
