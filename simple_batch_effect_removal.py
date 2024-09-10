# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:30:20 2024

@author: alesk
"""

import pandas as pd
import qnorm
import numpy as np

import openTSNE
from data_dim_reduction_plotting import plot
import global_values as gv



#tpm_table = pd.read_table(gv.GROUPED_DATA)
#tpm_table.set_index('SRR_accession', inplace=True)

tpm_table = pd.read_table('./data/limma_test.tsv', index_col=0)
metadata_table = pd.read_table('./data/metadata_T.tsv')

#tpm_table.drop_duplicates('SRR_accession', inplace=True)
#metadata_table.drop_duplicates('SRR_accession', inplace=True)

metadata_table.set_index('SRR_accession', inplace=True)
#tpm_table.set_index(0, inplace=True)

file = open('./data/relevant_studies.txt')

batches = []

for line in file :
  batches.append(line[:-1])
  
  
metadata_table = metadata_table[metadata_table['study_accession'].isin(batches)]
metadata_table = metadata_table.rename(columns={"study_accession": "sra_study"})
  
tpm_table = metadata_table.join(tpm_table)

tpm_table.dropna(inplace=True)


batches = tpm_table['sra_study'].unique()
metadata =  tpm_table[['perturbation_group', 'tissue_super', 'sra_study']]
tpm_table = tpm_table[tpm_table.columns.difference(['perturbation_group', 'tissue_super', 'sra_study'])]


#for batch in batches :
#    tpm_table[metadata['sra_study'] == batch] = qnorm.quantile_normalize(tpm_table[metadata['sra_study'] == batch] , axis=0)
    
    

tpm_table = metadata.join(tpm_table)



tpm_table = pd.read_table(gv.VAE_GROUPED_DATA)
tpm_table.set_index('SRR_accession', inplace=True)
 
perturbations = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
tissues = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
sra_studies = tpm_table.loc[:, tpm_table.columns == "sra_study"]

tpm_table.drop("tissue_super", axis=1, inplace=True)
tpm_table.drop("perturbation_group", axis=1, inplace=True)
tpm_table.drop("sra_study", axis=1, inplace=True)

#perturbations['perturbation_group'] = perturbations['perturbation_group'].apply(lambda x: 'nan' if x !=x else x)
#tissues['tissue_super'] = tissues['tissue_super'].apply(lambda x: 'nan' if x !=x else x)


data_raw = tpm_table.values
data_log = data_raw
#data_log = np.log1p(data_raw + 1e-6)
nans = np.isnan(data_log).sum(axis=1)
print(np.isnan(data_log).sum())    
    
embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=8,
    random_state=3,
    ).fit(data_log)
    
colors = ['grey', 'red', 'gold', 'lime', 'teal',
              'blue', 'crimson', 'fuchsia', 'chocolate', 'yellow',
              'darkgreen', 'cyan', 'royalblue', 'tomato', 'skyblue',
              'thistle', 'indigo', 'olive', 'green', 'black', 'violet']
    
plot(embedding_pca_cosine, perturbations['perturbation_group'], tissues['tissue_super'])

plot(embedding_pca_cosine, sra_studies['sra_study'], perturbations['perturbation_group'], draw_legend=False)