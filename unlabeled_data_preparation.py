# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:40:28 2024

@author: alesk
"""


import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import anndata
import global_values as gv



allowed_tissues = ['young_seedling', 'seed',
                   'mature_leaf', 'mature_root', 'mature_flower']

allowed_strats = ['RNA-Seq']
allowed_selecs = ['cDNA', 'RANDOM', 'PolyA', 'Oligo-dT']

#tpm_table = pd.read_table('./data/athaliana_trimmed_2.tsv')
metadata_table = pd.read_table('./data/athaliana_metadata.tsv')
metadata_table_2 = pd.read_table(gv.METADATA_PATH)


#tpm_table.drop_duplicates('SRR_accession', inplace=True)
metadata_table.drop_duplicates('SRR_accession', inplace=True)

metadata_table.set_index('SRR_accession', inplace=True)
metadata_table_2.set_index('SRR_accession', inplace=True)


metadata_table = metadata_table[['study_accession', 'experiment_library_strategy', 'experiment_library_selection', 'experiment_instrument_model']]
metadata_table = metadata_table.rename(columns={"study_accession": "sra_study"})


metadata_table_2 = metadata_table_2[metadata_table_2['perturbation_group'] != 'unknown']

metadata_table_2['tissue_super'] = metadata_table_2['tissue_super'].apply(lambda x: 'senescence' if 'senescence' in x else x)
metadata_table_2['tissue_super'] = metadata_table_2['tissue_super'].apply(lambda x: 'seed' if 'seed' in x and 'seedling' not in x else x)
    
metadata_table_2 = metadata_table_2[metadata_table_2['tissue_super'].isin(allowed_tissues)]
metadata_table_2['perturbation_group'] = metadata_table_2['perturbation_group'].apply(lambda x: 'control' if x == 'unstressed' else x)


metadata_table_2 = metadata_table_2[['perturbation_group', 'tissue_super']]
#tpm_table.set_index('SRR_accession', inplace=True)
#tpm_table = metadata_table.join(tpm_table)
metadata_table = metadata_table.join(metadata_table_2)

metadata_table = metadata_table[metadata_table['experiment_library_strategy'].isin(allowed_strats)]
metadata_table = metadata_table[metadata_table['experiment_library_selection'].isin(allowed_selecs)]

model_count = metadata_table['experiment_instrument_model'].value_counts()
chosen_models = []
"""
for count, model in zip(model_count, model_count.index)  :
    if count > 100 :
        chosen_models.append(model)

metadata_table = metadata_table[metadata_table['experiment_instrument_model'].isin(chosen_models)]
"""

batch_count = metadata_table['sra_study'].value_counts()
chosen_batches = []
"""
for count, batch in zip(batch_count, batch_count.index)  :
    if count > 10 :
        chosen_batches.append(batch)

metadata_table = metadata_table[metadata_table['sra_study'].isin(chosen_batches)]
"""

metadata_table.to_csv('./data/metadata_T.tsv', sep="\t")


"""
tpm_table = pd.read_table('./data/limma_test.tsv')
metadata_table = pd.read_table('./data/metadata_T.tsv')

#tpm_table.drop_duplicates('SRR_accession', inplace=True)
#metadata_table.drop_duplicates('SRR_accession', inplace=True)

metadata_table.set_index('SRR_accession', inplace=True)
tpm_table.set_index('SRR_accession', inplace=True)
tpm_table = metadata_table.join(tpm_table)


tpm_table[['perturbation_group', 'tissue_super', 'study_accession']].to_csv('data/metadata_T.tsv', sep="\t")
tpm_table[tpm_table.columns.difference(['perturbation_group', 'tissue_super', 'study_accession'])].T.to_csv('./data/athaliana_trimmed_3.tsv', sep="\t")

"""
"""
print(tpm_table.isin([np.inf, -np.inf]).sum().sum())
print(tpm_table.isnull().sum().sum())

X = tpm_table.iloc[0:,1:]
obs = pd.DataFrame(tpm_table['study_accession'])
var = pd.DataFrame(tpm_table.columns[1:])

studies = list(obs['study_accession'].value_counts().index)

file = open('relevant_studies.txt', 'w')

for study in studies :
    file.write(study + '\n')

file.close()

var = var.rename(columns={0: "gene_id"})
var.set_index('gene_id', inplace=True)


adata = anndata.AnnData(X = X,
                        obs = obs,
                        var = var)

adata
#with open("./data/athaliana_trimmed_2.tsv") as your_data:
#    adata = anndata.read_csv(your_data, delimiter='\t')


"""

"""
tpm_table.set_index('SRR_accession', inplace=True)
#tpm_table.set_index('gene_id', inplace=True)
metadata_table.set_index('SRR_accession', inplace=True)

metadata_table = metadata_table[['study_accession']]
tpm_table.index
metadata_table.index
#tpm_table_T = tpm_table.T

joined_table = tpm_table.join(metadata_table)

batch_count = joined_table['study_accession'].value_counts()

chosen_batches = []

for count, batch in zip(batch_count, batch_count.index)  :
    if count > 10 :
        chosen_batches.append(batch)


joined_table = joined_table[joined_table['study_accession'].isin(chosen_batches)]


joined_table[['study_accession']].to_csv('data/metadata_T.tsv', sep="\t")
joined_table[joined_table.columns.difference(['study_accession'])].T.to_csv('data/athaliana_trimmed_T.tsv', sep="\t")

    
    
"""
    
    
        