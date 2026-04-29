#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:11:56 2024

@author: yunda_si
"""

from ml_collections import config_dict


def get_cfg(world_size):
    config = config_dict.ConfigDict()

    config.weight_file = None
    config.init_seed = 42
    config.world_size = world_size

    config.training = config_dict.ConfigDict()

    config.model = config_dict.ConfigDict()
    config.model.blocks_structure_decoder = 12
    config.model.atom_channel = 384
    config.model.atom_nhead = 12
    config.model.pair_channel = 384
    config.model.pair_nhead = 12
    config.model.num_structure_recycle = 8
    config.model.num_atom = 28
    config.model.dropout_p = 0.15
    config.model.dropout_p2d = 0.15
    config.model.residue_window = (-1, -1)
    config.model.seq_window = (-1, -1)

    return config

































