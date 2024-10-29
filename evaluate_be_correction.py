# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:05:16 2024

@author: alesk
"""

import pandas as pd

from batch_effect_measurement import measure_batch_effect
import global_values as gv

if __name__ == '__main__':
    
    measurements = [
            {'tpm_path' : gv.GROUPED_DATA, 'meta_path' : None, 'display_name' : 'Unprocessed data'},
            {'tpm_path' : 'data/prop_gene_exp_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'Proportional gene expression'},
            {'tpm_path' : 'data/combat_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'ComBat'},
            {'tpm_path' : 'data/combat-seq_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'ComBat-Seq'},
            {'tpm_path' : 'data/limma_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'Limma library method'},
            {'tpm_path' : 'data/scvi_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'scVI VAE'},
            {'tpm_path' : 'data/scvi_w_cov_adjusted.tsv', 'meta_path' : gv.METADATA_PROC_PATH, 'display_name' : 'scVI VAE w/ covariates'},
        ]
    f = open('./data/BE_measures.txt', 'a')
    chosen_rank = 2
    for measurement in measurements :
        tpm_table = pd.read_table(measurement['tpm_path'], index_col=0)
        tpm_table.dropna(inplace=True)
        
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
        
        rank_file = open(gv.CKN_GENE_RANKS, 'r')
        genes_list = []

        for line in rank_file:
            gene_rank = int(line.split('\t')[1])
            if gene_rank <= chosen_rank :
                genes_list.append(line.split('\t')[0])
                      
        
        if measurement['meta_path'] == None :
            perturbation_count = tpm_table['perturbation_group'].value_counts()
            perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
            
            tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
            tissue_count = tpm_table['tissue_super'].value_counts()
        else :
            metadata_table = pd.read_table(measurement['meta_path'], index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
            tpm_table = metadata_table.join(tpm_table, how='inner')
            
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
        f.write(measurement['display_name'] + ':\n' + str(perturbation_logs)+ '\n' + str(tissue_logs) + '\n\n')

    f.close()