# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:15:56 2024

@author: alesk
"""

from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

from os.path import abspath, dirname, join

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import manifold
from sklearn import preprocessing

import seaborn as sns

import matplotlib.pyplot as plt

import openTSNE

from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization


import global_values as gv


meta_table = pd.read_csv('./data/athaliana_metadata.tsv', sep='\t', index_col=0)

strats = meta_table['experiment_library_strategy'].value_counts()
selecs = meta_table['experiment_library_selection'].value_counts()
models = meta_table['experiment_instrument_model'].value_counts()



tpm_table = pd.read_csv(gv.GROUPED_DATA, sep="\t", index_col=0)

tpm_table = meta_table.join(tpm_table, how='inner')

strats = tpm_table['experiment_library_strategy'].value_counts()
selecs = tpm_table['experiment_library_selection'].value_counts()
models = tpm_table['experiment_instrument_model'].value_counts()

tpm_table[['perturbation_group', 'tissue_super', 'sra_study']].to_csv('./data/metadata_proc.tsv', sep='\t')



tpm_table = pd.read_csv(gv.GROUPED_DATA, sep="\t", index_col=0)
genes_list = ['sra_study']
ckn_file = open('./data/CKN_gene_ranks.txt', 'r')
gene_positions = {}
i = 0
for line in ckn_file :
    gene_id = line.split('\t')[0]
    rank = int(line.split('\t')[1])
    
    if rank <= 2 and gene_id[:2] == 'AT' and '|' not in gene_id:
        gene_positions[gene_id] = i
        genes_list.append(gene_id)
        i += 1
        

tpm_table = tpm_table.drop(columns=[col for col in tpm_table if col.split('.')[0] not in genes_list])

for scaling, ScaleFunction in [
    ('power-scaled', PowerTransformer()),
    ('quantile-scaled', RobustScaler(with_centering=False)),
    ('robust-scaled', RobustScaler()),
    ('quantile-transformed', QuantileTransformer()), 
    ('quantile-transformed-guassian', QuantileTransformer(output_distribution='normal'))
    ]:
    print(f'Scaling using {scaling}')
    with open(f"./data/{scaling}.tsv", "w") as out:
        out.write("SRR_accession\t" + "\t".join(tpm_table.columns.difference(['sra_study'])) + "\n")
        for group, group_df in tpm_table.groupby("sra_study"):
                group_df = group_df[tpm_table.columns.difference(['sra_study'])]
                # fit scaler to the entire study
                all_group_data = group_df.values.flatten().reshape(-1, 1)
                scaler = ScaleFunction.fit(all_group_data)
        
                group_df = group_df.T # apply per sample, so can save all genes per row
                for col in group_df.columns:
                    x = group_df[col].values.reshape(-1, 1)
                    rescaled = scaler.transform(x)
                    out.write(col + "\t" + "\t".join(np.char.mod('%f', rescaled.reshape(-1)))+"\n")
            