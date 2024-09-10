# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:49:22 2024

@author: alesk
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
from load_tpm_data import NonPriorData

from data_dim_reduction_plotting import plot

import global_values as gv
#metadata_table = pd.read_table(gv.METADATA_PATH)
#batch_count = metadata_table['SRAStudy'].value_counts()

tpm_table = pd.read_table(gv.JOINED_DATA)
tpm_table.set_index('SRR_accession', inplace=True)

idx_list = tpm_table.index[tpm_table['sra_study'] == "SRP151817"].tolist()

batch_dataset = tpm_table.iloc[idx_list]


batch_dataset.drop(batch_dataset.columns[0], axis=1, inplace=True)

perturbations = batch_dataset.loc[:, batch_dataset.columns == "perturbation_group"]
tissues = batch_dataset.loc[:, batch_dataset.columns == "tissue_super"]

batch_dataset.drop("tissue_super", axis=1, inplace=True)
batch_dataset.drop("perturbation_group", axis=1, inplace=True)
batch_dataset.drop("sra_study", axis=1, inplace=True)

data_raw = batch_dataset

data_raw = data_raw.values
#perturbations = perturbation_raw.values

data_log = np.log10(data_raw + 1)

import openTSNE
from openTSNE import affinity
from openTSNE import initialization, TSNEEmbedding


affinities = affinity.Multiscale(
    data_log,
    perplexities=[10, 27],
    metric="cosine",
    n_jobs=8,
    random_state=3,
)

init = initialization.pca(data_log, random_state=42)

embedding = TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",
    n_jobs=8
)
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

plot(embedding, perturbations['perturbation_group'], tissues['tissue_super'])