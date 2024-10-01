# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:20:10 2024

@author: alesk
"""

import pandas as pd

import global_values as gv

tpm_table = pd.read_table(gv.GROUPED_DATA, sep='\t', index_col=0)
#metadata_table = pd.read_table(gv.GROUPED_DATA, sep='\t', index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
metadata_table = tpm_table[['perturbation_group', 'tissue_super', 'sra_study']]

tpm_table = tpm_table.drop(['perturbation_group', 'tissue_super', 'sra_study'], axis=1)

keep_columns = []
rename_cols = {}

for col in tpm_table.columns :
    gene = col.split('.')[0]
    
    if gene not in keep_columns :
        tpm_table[col] = tpm_table[[column for column in tpm_table.columns if gene in column]].mean(axis=1)
        
        keep_columns.append(gene)
        rename_cols[col] = gene
 
tpm_table = tpm_table.rename(columns=rename_cols)
tpm_table = tpm_table[keep_columns]


tpm_table = metadata_table.join(tpm_table, how='inner')


tpm_table.to_csv('data/grouped_tpm_avg.tsv', sep="\t")