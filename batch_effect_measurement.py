# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:30:13 2024

@author: alesk
"""
import random
import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import pandas as pd

from load_tpm_data import NonPriorData
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import global_values as gv
from scipy.spatial.distance import pdist, squareform


def measure_batch_effect(tpm_table, categories, categories_count, category_name, categories2, categories_count2, category_name2) :
    
    print('Calculating distances.')
    
    distances = pdist(tpm_table.values, metric='cosine')
    dist_matrix = squareform(distances)
    
    print('Calculated distances.')
    
    category_logs = {}
    category_logs2 = {}
    j = 0
    for i in range(0, len(dist_matrix)) :
        category = categories.iloc[i][category_name]
        category2 = categories2.iloc[i][category_name2]
        
        if j % 100 == 0 :
            print(j)
        j += 1
        if category == category :
        
            if category not in category_logs :
                category_size = categories_count[category]
                category_logs[category] = {'size' : category_size, 'sum' : 0}
            
        
            idx = np.argpartition(dist_matrix[i, :], category_logs[category]['size'] + 1, axis=0)
            
            curr_sum = 0
            for closest in idx[:category_logs[category]['size'] + 1] :
                if i != closest and categories.iloc[closest][category_name] == category :
                    curr_sum += 1
                    
            curr_sum /= category_logs[category]['size']
            category_logs[category]['sum'] += curr_sum
        
        if category2 == category2 :
        
            if category2 not in category_logs2 :
                category_size2 = categories_count2[category2]
                category_logs2[category2] = {'size' : category_size2, 'sum' : 0}
            
        
            idx = np.argpartition(dist_matrix[i, :], category_logs2[category2]['size'] + 1, axis=0)
            
            curr_sum = 0
            for closest in idx[:category_logs2[category2]['size'] + 1] :
                if i != closest and categories2.iloc[closest][category_name2] == category2 :
                    curr_sum += 1
                    
            curr_sum /= category_logs2[category2]['size']
            category_logs2[category2]['sum'] += curr_sum
            
         
        
    for category in category_logs :
        category_logs[category]['sum'] /= category_logs[category]['size']

    for category2 in category_logs2 :
        category_logs2[category2]['sum'] /= category_logs2[category2]['size']    


    return category_logs, category_logs2


if __name__ == '__main__':
    
    measurements = [
            #{'tpm_path' : gv.GROUPED_DATA, 'T' : False, 'meta_path' : None},
            #{'tpm_path' : './data/athaliana_annotated.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'},
            #{'tpm_path' : './data/combat.tsv', 'T' : True, 'meta_path' : './data/metadata_T.tsv'},
            #{'tpm_path' : './data/combat_seq.tsv', 'T' : True, 'meta_path' : './data/metadata_T.tsv'},
            #{'tpm_path' : './data/vae_transformed.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'},
            #{'tpm_path' : './data/vae_latent.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'}, 
            #{'tpm_path' : './data/vae_cov_transformed.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'},
            #{'tpm_path' : './data/vae_cov_latent.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'}, 
            {'tpm_path' : './data/vae_cov_transformed_smol.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv'},
            ]
    f = open('./data/BE_measures_cosine.txt', 'a')
    chosen_rank = 2
    for measurement in measurements :
        tpm_table = pd.read_table(measurement['tpm_path'], index_col=0)
        #tpm_table.dropna(inplace=True)
        
        if measurement['T'] :
            tpm_table = tpm_table.T
        
        #if measurement['meta_path'] is not None:
        #    metadata_table = pd.read_table(measurement['meta_path'], index_col=0)
            
        #    tpm_table = tpm_table.join(metadata_table, how='inner')
        
        """
        batch_count = tpm_table['sra_study'].value_counts()
        chosen_batches = []

        for batch in batch_count.index :
            batch_data = tpm_table[tpm_table['sra_study'] == batch]
            
            if 'control' not in batch_data['perturbation_group'].unique() or 'chemical stress' not in batch_data['perturbation_group'].unique():
                continue
            
            chosen_batches.append(batch)
        
        tpm_table = tpm_table[tpm_table['sra_study'].isin(chosen_batches)]
        tpm_table = tpm_table[tpm_table['perturbation_group'].isin(['control', 'chemical stress'])]
        """
        """
        rank_file = open(gv.CKN_GENE_RANKS, 'r')
        genes_list = []

        for line in rank_file:
            gene_rank = int(line.split('\t')[1])
            if gene_rank <= chosen_rank :
                genes_list.append(line.split('\t')[0])
        """            
        
        if measurement['meta_path'] == None :
            perturbation_count = tpm_table['perturbation_group'].value_counts()
            perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
            
            tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
            tissue_count = tpm_table['tissue_super'].value_counts()
        else :
            metadata_table = pd.read_table(measurement['meta_path'], index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
            tpm_table = tpm_table.join(metadata_table, how='inner')
            
            perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
            perturbation_count = perturbations['perturbation_group'].value_counts()
            
            tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
            tissue_count = tissues['tissue_super'].value_counts()
            
        if 'sra_study' in tpm_table.columns :
            tpm_table.drop("sra_study", axis=1, inplace=True)
          
        if 'tissue_super' in tpm_table.columns :
            tpm_table.drop("tissue_super", axis=1, inplace=True)
            
        if 'perturbation_group' in tpm_table.columns :
            tpm_table.drop("perturbation_group", axis=1, inplace=True)
          
        if 'secondary_perturbation' in tpm_table.columns :
            tpm_table.drop("secondary_perturbation", axis=1, inplace=True)
    
        #tpm_table = tpm_table.drop(columns=[col for col in tpm_table if col.split('.')[0] not in genes_list])
        
        print('Measuring BE metric on ' + measurement['tpm_path'])
    
        perturbation_logs, tissue_logs = measure_batch_effect(tpm_table,
                                                              perturbations, perturbation_count, 'perturbation_group', 
                                                              tissues, tissue_count, 'tissue_super')
        print(measurement)
        print(perturbation_logs)
        print(tissue_logs)
        f.write(measurement['tpm_path'] + ':\n' + str(perturbation_logs)+ '\n' + str(tissue_logs) + '\n\n')

    f.close()