# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:18:53 2023

@author: alesk
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing

from global_values import TPM_PATH, METADATA_PATH, JOINED_DATA_PATH

allowed_tissues = ['young_seedling', 'young_seed','seed_seed','mature_seed',
                   'mature_leaf', 'mature_seedling', 'mature_root', 'mature_flower',
                   'senescence_senescence_reproductive', 'senescence_senescence_green']

if __name__ == '__main__':
    
    tpm_table = pd.read_table(TPM_PATH)
    metadata_table = pd.read_table(METADATA_PATH)
    
    dict_list = []
    
    gene_id_col = None
        
    for column in tpm_table :
        if column != 'gene_id' :
            experiment_id = column[:-4]
            experiment_metadata = metadata_table.loc[metadata_table['SRR_accession'] == experiment_id]
            
            if experiment_metadata['tissue_super'].values in allowed_tissues:
                experiment_data = tpm_table[column].values
                
                curr_dict = {}
                curr_dict['tissue_super'] = experiment_metadata['tissue_super'].values[0]
                for gene_id, value in zip(gene_id_col, experiment_data) :
                    curr_dict[gene_id] = value
                dict_list.append(curr_dict)                    
        else :
            gene_id_col = tpm_table[column].values
        
    joined_tpm = pd.DataFrame(dict_list)
    
    joined_tpm.to_csv(JOINED_DATA_PATH, sep="\t")