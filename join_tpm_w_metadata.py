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

import global_values as gv

allowed_tissues = ['young_seedling', 'seed',
                   'mature_leaf', 'mature_root', 'mature_flower']

def process_data(tpm_table_T, metadata_table, save_path, group_tissues=False, secondary_perturbation=False) :
    metadata_cut = metadata_table[['perturbation_group', 'tissue_super', 'SRAStudy', 'perturbation_group_2_1']]
    
    metadata_cut = metadata_cut.rename(columns={"perturbation_group_2_1": "secondary_perturbation"})
    metadata_cut = metadata_cut.rename(columns={"SRAStudy": "sra_study"})
    
    metadata_cut = metadata_cut[metadata_cut['perturbation_group'] != 'unknown']
    metadata_cut = metadata_cut[metadata_cut['secondary_perturbation'] != 'unknown']
    
    if group_tissues: 
        #metadata_cut['tissue_super'] = metadata_cut['tissue_super'].apply(lambda x: 'senescence' if 'senescence' in x else x)
        metadata_cut['tissue_super'] = metadata_cut['tissue_super'].apply(lambda x: 'seed' if 'seed' in x and 'seedling' not in x else x)
        
        metadata_cut = metadata_cut[metadata_cut['tissue_super'].isin(allowed_tissues)]
        
    if secondary_perturbation :
        metadata_cut['secondary_perturbation'] = np.where(metadata_cut['secondary_perturbation'] == metadata_cut['secondary_perturbation'],
                                                      metadata_cut['secondary_perturbation'],
                                                      metadata_cut['perturbation_group'])
        
        metadata_cut['secondary_perturbation'] = metadata_cut['secondary_perturbation'].apply(
                                            lambda x: 'detachment stress' if 'detachment stress' in x else x)
        
        metadata_cut['secondary_perturbation'] = metadata_cut['secondary_perturbation'].apply(lambda x: 'control' if x == 'unstressed' else x)
        
    else :
        metadata_cut.drop("secondary_perturbation", axis=1, inplace=True)
        
    metadata_cut.dropna(inplace=True)
    metadata_cut['perturbation_group'] = metadata_cut['perturbation_group'].apply(lambda x: 'control' if x == 'unstressed' else x)
    
    joined_table = tpm_table_T.join(metadata_cut)
    
    joined_table.dropna(inplace=True)
    
    joined_table.to_csv(save_path, sep="\t")
    
    
    batch_count = joined_table['sra_study'].value_counts()
    
    """
    chosen_batches = []
    i = 0
    for batch in batch_count.index :
        chosen_batches.append(batch)
        i += 1
        
        if i > 40 :
            break
    
    joined_table = joined_table[joined_table['sra_study'].isin(chosen_batches)]
    """
    """
    joined_table[['perturbation_group', 'tissue_super', 'sra_study']].to_csv('data/metadata_limma_test.tsv', sep="\t")
    joined_table = joined_table[joined_table.columns.difference(['perturbation_group', 'tissue_super', 'sra_study'])]
    joined_table = joined_table.apply(lambda x: np.log1p(x))
    
    joined_table.to_csv('data/limma_log1p_data.tsv', sep="\t")
    """
if __name__ == '__main__':
    
    datasets = [
        #{'path': gv.JOINED_DATA, 'group': False, 'extend': False},
        {'path': gv.GROUPED_DATA, 'group': True, 'extend': False},
        #{'path': gv.EXTENDED_DATA, 'group': False, 'extend': True},
        #{'path': gv.EXTENDED_GROUPED_DATA, 'group': True, 'extend': True}
        #{'path': './data/harmony_joined.tsv', 'group': True, 'extend': False},
        #{'path': './data/vae_joined.tsv', 'group': True, 'extend': False},
        ]
    
    #tpm_table = pd.read_table('./data/harmony.tsv')
    tpm_table = pd.read_table(gv.TPM_PATH)
    #tpm_table = pd.read_table('./data/vae_transformed.tsv')
    metadata_table = pd.read_table(gv.METADATA_PATH)
    
    #tpm_table.set_index('SRR_accession', inplace=True)
    tpm_table.set_index('gene_id', inplace=True)
    metadata_table.set_index('SRR_accession', inplace=True)
    
    tpm_table_T = tpm_table.T
    
    for dataset in datasets :
        process_data(tpm_table_T.copy(),
                     metadata_table.copy(),
                     save_path=dataset['path'],
                     group_tissues=dataset['group'],
                     secondary_perturbation=dataset['extend'])
        

