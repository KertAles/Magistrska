# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:53:28 2024

@author: alesk
"""

import pandas as pd

if __name__ == '__main__':
    tpm = pd.read_table('./data/vae_cov_transformed.tsv', sep='\t', index_col=0)
    metadata_table = pd.read_table('./data/metadata_T.tsv', index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
    tpm = tpm.join(metadata_table, how='inner')
    
    batches = list(tpm['sra_study'].unique())
    
    fitting_batches = []
    for batch in batches :
        batch_tpm = tpm[tpm['sra_study'] == batch]
        
        perturbations = list(batch_tpm['perturbation_group'].unique())
        
        if 'environmental stress' in perturbations :
            fitting_batches.append(batch)
            
            
    filtered_tpm = tpm[tpm['sra_study'].isin(fitting_batches)]
    filtered_tpm = filtered_tpm[filtered_tpm['perturbation_group'].isin(['control', 'environmental stress'])]
    
    tissues = filtered_tpm['tissue_super'].value_counts()
    perturbations = filtered_tpm['perturbation_group'].value_counts()
    
    
    filtered_tpm.to_csv('./data/vae_cov_transformed_filtered.tsv', sep="\t")