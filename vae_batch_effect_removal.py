# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:05:18 2024

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

import anndata
import seaborn as sns

import matplotlib.pyplot as plt

import openTSNE
from data_dim_reduction_plotting import plot
import global_values as gv

import scanpy as sc
import scvi

from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

if __name__ == '__main__':
    """
    tpm_table = pd.read_table(gv. )
    tpm_table.set_index('SRR_accession', inplace=True)

    #tpm_table.drop("SRR_accession", axis=1, inplace=True)
    tpm_table.drop("sra_study", axis=1, inplace=True)
    tpm_table.drop("perturbation_group", axis=1, inplace=True)
    tpm_table.drop("tissue_super", axis=1, inplace=True)
    """
    
    
    adata_raw = sc.read_h5ad("./data/anndata.h5ad")
    adata_raw.layers["logcounts"] = adata_raw.X
    
    sc.pp.highly_variable_genes(adata_raw)
    
    adata_scvi = adata_raw[:, adata_raw.var["highly_variable"]].copy()

    
    scvi.model.SCVI.setup_anndata(adata_scvi, layer="logcounts", batch_key='sra_study')
    
    
    model_scvi = scvi.model.SCVI(adata_scvi)
    
    model_scvi.view_anndata_setup()
    
    max_epochs_scvi = np.min([round((20000 / adata_raw.n_obs) * 400), 400])
    
    model_scvi.train(max_epochs=max_epochs_scvi)
    
    adata_scvi.obsm["X_scVI"] = model_scvi.get_normalized_expression()
    
    latent = adata_scvi.obsm["X_scVI"]
    
    sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
    sc.tl.umap(adata_scvi)
    
    sc.pl.umap(adata_scvi, color=['perturbation', 'sra_study'], wspace=1)