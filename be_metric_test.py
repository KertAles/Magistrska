# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:55:36 2024

@author: alesk
"""

import scib
from os.path import abspath, dirname, join

import scipy.sparse as sp
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import manifold
from sklearn import preprocessing

import anndata
import seaborn as sns

import matplotlib.pyplot as plt

import openTSNE

import scanpy as sc
from scipy import sparse
import scvi

from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

import global_values as gv


tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)

studies = tpm_table['sra_study'].value_counts()
perturbations = tpm_table['perturbation_group'].value_counts()
tissues = tpm_table['tissue_super'].value_counts()

tpm_table.dropna(inplace=True)
X = tpm_table.iloc[0:,:-3]
obs = pd.DataFrame(tpm_table[['sra_study', 'perturbation_group', 'tissue_super']])
var = pd.DataFrame(tpm_table.columns[:-3])

var = var.rename(columns={0: "gene_id"})
var.set_index('gene_id', inplace=True)


adata_raw = anndata.AnnData(X = X,
                        obs = obs,
                        var = var)

sc.pp.pca(adata_raw)
sc.pp.neighbors(adata_raw)

tpm_table = pd.read_table('./data/pycombat_transformed.tsv', index_col=0)

tpm_table.dropna(inplace=True)
"""
metadata_table = pd.read_table('./data/metadata_proc.tsv', index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
tpm_table = metadata_table.join(tpm_table, how='inner')

perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
perturbation_count = perturbations['perturbation_group'].value_counts()

tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
tissue_count = tissues['tissue_super'].value_counts()
"""

X = tpm_table.iloc[0:,:-3]
obs = pd.DataFrame(tpm_table[['sra_study', 'perturbation_group', 'tissue_super']])
var = pd.DataFrame(tpm_table.columns[:-3])

var = var.rename(columns={0: "gene_id"})
var.set_index('gene_id', inplace=True)

adata_vae = anndata.AnnData(X = X,
                        obs = obs,
                        var = var)

sc.pp.pca(adata_vae)
sc.pp.neighbors(adata_vae)
batch_key = 'sra_study'

hvg_overlap = scib.metrics.hvg_overlap(adata_raw, adata_vae, batch_key)
silhouette = scib.metrics.silhouette(adata_raw, batch_key, embed='X_pca')
silhouette2 = scib.metrics.silhouette(adata_vae, batch_key, embed='X_pca')
pcr_comp = scib.metrics.pcr_comparison(adata_raw, adata_vae, batch_key)