# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:48:31 2023

@author: alesk
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import manifold
from sklearn import preprocessing

import seaborn as sns

from global_values import JOINED_DATA_PATH

tpm_table = pd.read_table(JOINED_DATA_PATH)
#tpm_table.to_csv('joined_tpm.tsv', sep="\t") 

tpm_table.drop("idx", axis=1, inplace=True)

perturbation_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
tissue_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]

tpm_table.drop("tissue_super", axis=1, inplace=True)
tpm_table.drop("perturbation_group", axis=1, inplace=True)


data_raw = tpm_table


data_raw = data_raw.values
perturbation = perturbation_raw.values
tissue = tissue_raw.values

data_log = np.log10(data_raw + 1e-6)

label_tissue = preprocessing.LabelEncoder()
label_perturbation = preprocessing.LabelEncoder()

tissue_transformed = label_tissue.fit_transform(tissue_raw)
perturbation_transformed = label_perturbation.fit_transform(perturbation_raw)

tsne = manifold.TSNE()

data_transformed = tsne.fit_transform(data_log)

tissue_df = pd.concat([pd.DataFrame(data_transformed), tissue_raw], axis=1)
perturbation_df = pd.concat([pd.DataFrame(data_transformed), perturbation_raw], axis=1)

sns.pairplot(tissue_df, hue="tissue_super", height=16)

sns.pairplot(perturbation_df, hue="perturbation_group", height=16)

