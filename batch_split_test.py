# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:23:24 2024

@author: alesk
"""
from os.path import abspath, dirname, join

import scipy.sparse as sp
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import manifold
from sklearn import preprocessing

import seaborn as sns
import umap
import umap.plot

import matplotlib.pyplot as plt

import openTSNE
from data_dim_reduction_plotting import plot
import global_values as gv
from sklearn.model_selection import train_test_split


tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)
#tpm_table.set_index('', inplace=True)

batch_count = tpm_table['sra_study'].value_counts()

chosen_batches = []

for batch in batch_count.index :
    batch_data = tpm_table[tpm_table['sra_study'] == batch]
    
    #if 'control' not in batch_data['perturbation_group'].unique() or 'mutant' not in batch_data['perturbation_group'].unique():
    #    continue
    
    chosen_batches.append(batch)

num_batches = len(chosen_batches)
train, testval = train_test_split(chosen_batches, test_size=int(num_batches * 0.25))

batch_data = tpm_table[tpm_table['sra_study'].isin(train)]
train_n = batch_data.value_counts().sum()
    
test, val = train_test_split(testval, test_size=int((num_batches * 0.25) / 2.0))

batch_data = tpm_table[tpm_table['sra_study'].isin(test)]
test_n = batch_data.value_counts().sum()
    
batch_data = tpm_table[tpm_table['sra_study'].isin(val)]
val_n = batch_data.value_counts().sum()
    

f = open("train_batches.txt", "w")
for batch in train :
    f.write(batch + '\n')
f.close()

f = open("test_batches.txt", "w")
for batch in test :
    f.write(batch + '\n')
f.close()

f = open("val_batches.txt", "w")
for batch in val :
    f.write(batch + '\n')
f.close()