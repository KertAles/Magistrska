# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:41:48 2024

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

def cell_type_counts(datasets, field: str = "labels"):
    cell_counts = pd.DataFrame(
        [pd.Series.value_counts(ds.obs[field]) for ds in datasets]
    )
    cell_counts.index = [ds.uns["name"] for ds in datasets]
    cell_counts = cell_counts.T.fillna(0).astype(int).sort_index()

    styler = cell_counts.style
    styler = styler.format(lambda x: "" if x == 0 else x)
    styler = styler.set_properties(**{"width": "10em"})

    return styler


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def select_genes(
    data,
    threshold=0,
    atleast=1,
    yoffset=0.02,
    xoffset=2,
    decay=2.5,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[detected.index[detected]] > threshold, np.log2(data[detected.index[detected]]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def get_colors_for(adata):
    """Get pretty colors for each class."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#7f7f7f",  # This is the grey one
              "#e377c2", "#bcbd22", "#17becf",

              "#0000A6", "#63FFAC", "#004D43", "#8FB0FF"]

    colors = dict(zip(adata['perturbation_group'].value_counts().sort_values(ascending=False).index, colors))

    colors["Other"] = "#7f7f7f"

    assert all(l in colors for l in (adata['perturbation_group'].unique()))

    return colors


from openTSNE import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

if __name__ == '__main__':
    tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)
    #tpm_table.set_index('SRR_accession', inplace=True)
    
    batch_count = tpm_table['sra_study'].value_counts()
    
    required_perturbations = ['chemical stress', 'environmental stress', 'mechanical stress', 'mutant']
    required_tissues = ['mature_flower', 'mature_leaf', 'mature_root', 'seed', 'senescence', 'young_seedling']
    
    chosen_batches = []
    other_batches = []
    i = 0
    for batch in batch_count.index :
        batch_data = tpm_table[tpm_table['sra_study'] == batch]
        
        if 'control' not in batch_data['perturbation_group'].unique() :
            continue
        """
        for perturbation in batch_data['perturbation_group'].unique() :
            if perturbation in required_perturbations :
                if batch not in chosen_batches : 
                    chosen_batches.append(batch)
                required_perturbations.remove(perturbation)
                
        for tissue in batch_data['tissue_super'].unique() :
            if tissue in required_tissues :
                if batch not in chosen_batches : 
                    chosen_batches.append(batch)
                required_tissues.remove(tissue)
        
        if batch not in chosen_batches :
            other_batches.append(batch)
        """
        chosen_batches.append(batch)
        
        #if i == 20:
        #    break
        i+=1
    df_list = []
    pert_list = []
    tiss_list = []
    sra_list = []
    first_group = tpm_table[tpm_table['sra_study'].isin(chosen_batches)]
    
    for chosen_batch in chosen_batches :
        batch_data = tpm_table[tpm_table['sra_study'] == chosen_batch]
        
        pert_list.append(batch_data.loc[:, batch_data.columns == "perturbation_group"])
        tiss_list.append(batch_data.loc[:, batch_data.columns == "tissue_super"])
        sra_list.append(batch_data.loc[:, batch_data.columns == "sra_study"])
        
        batch_data.drop("tissue_super", axis=1, inplace=True)
        batch_data.drop("sra_study", axis=1, inplace=True)
        batch_control = batch_data[batch_data['perturbation_group'] == 'control']
        
        batch_data.drop("perturbation_group", axis=1, inplace=True)
        batch_control.drop("perturbation_group", axis=1, inplace=True)
        
        batch_control = batch_control.mean(axis=0)
        
        batch_data = batch_data.div(batch_control + 1e-6)
        
        df_list.append(batch_data)
    
    first_group = pd.concat(df_list)
    perturbations_first = pd.concat(pert_list)
    tissues_first = pd.concat(tiss_list)
    sra_first = pd.concat(sra_list)
    
    #proportional_data = pd.concat([first_group, perturbations_first, tissues_first, sra_first], axis=1)
    #proportional_data.to_csv(gv.PROPORTIONAL_DATA_CONTROLS, sep="\t")
    
    #first_group = first_group.drop(perturbations_first[perturbations_first['perturbation_group'] == 'control'].index)
    #tissues_first = tissues_first.drop(perturbations_first[perturbations_first['perturbation_group'] == 'control'].index)
    #sra_first = sra_first.drop(perturbations_first[perturbations_first['perturbation_group'] == 'control'].index)
    #perturbations_first = perturbations_first.drop(perturbations_first[perturbations_first['perturbation_group'] == 'control'].index)

    #proportional_data = pd.concat([first_group, perturbations_first, tissues_first, sra_first], axis=1)
    #proportional_data.to_csv(gv.PROPORTIONAL_DATA_NO_CONTROLS, sep="\t")
    
    """
    chosen_rank = 0
    
    rank_file = open(gv.CKN_GENE_RANKS, 'r')
    genes_in_lcc = []

    for line in rank_file:
        gene_rank = int(line.split('\t')[1])
        if gene_rank <= chosen_rank :
            genes_in_lcc.append(line.split('\t')[0])
    
    first_group = first_group.drop(columns=[col for col in first_group if col.split('.')[0] not in genes_in_lcc])
    """
    first_group_raw = first_group.values
    first_group_raw = np.log1p(first_group_raw)
    #first_group_raw -= np.mean(first_group_raw, axis=0)
    #first_group_raw /= np.std(first_group_raw, axis=0) + 1e-6

    
    
    
    embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=8,
    random_state=3,
    ).fit(first_group_raw)
    
    colors = ['grey', 'red', 'gold', 'lime', 'teal',
              'blue', 'indigo', 'fuchsia', 'chocolate', 'yellow',
              'darkgreen', 'cyan', 'royalblue', 'tomato', 'skyblue',
              'thistle', 'crimson', 'olive']
    
    plot(embedding_pca_cosine, perturbations_first['perturbation_group'],
                     tissues_first['tissue_super'], draw_legend=True)
    plot(embedding_pca_cosine, tissues_first['tissue_super'], perturbations_first['perturbation_group'],
                      draw_legend=True)
    #perturbations_first = first_group.loc[:, first_group.columns == "perturbation_group"]
    #tissues_first = first_group.loc[:, first_group.columns == "tissue_super"]
    
    
    
    #first_group.drop("perturbation_group", axis=1, inplace=True)
    #first_group.drop("tissue_super", axis=1, inplace=True)
    #first_group.drop("sra_study", axis=1, inplace=True)
    """
    first_group_copy = first_group.copy()
    
    max_values_per_genes = first_group.max(axis=0)
    active_genes = max_values_per_genes[max_values_per_genes.gt(5)].index
    
    first_group = first_group.loc[:, active_genes]
    
    gene_mask = select_genes(first_group, n=3000, threshold=0)
    
    first_group = first_group.loc[:, gene_mask]
    
    max_values_per_genes2 = first_group.max(axis=0)   
 
    
    first_group_raw = first_group.values
    first_group_raw = np.log1p(first_group_raw)
    first_group_raw -= np.mean(first_group_raw, axis=0)
    first_group_raw /= np.std(first_group_raw, axis=0)
    
    
    U, S, V = np.linalg.svd(first_group_raw, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    first_group_pca = np.dot(U, np.diag(S))
    first_group_pca = first_group_pca[:, np.argsort(S)[::-1]][:, :50]
    
    
    affinities = affinity.Multiscale(
        first_group_pca,
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=3,
    )
    init = initialization.pca(first_group_pca, random_state=42)
    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
    )
    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
    
    first_group_tsne = embedding
    
    plot(first_group_tsne, perturbations_first['perturbation_group'],
                     tissues_first['tissue_super'], draw_legend=True)
    
    
    ijk = 0
    for batch in other_batches :
        batch_data = tpm_table[tpm_table['sra_study'] == batch]
        
        perturbations_batch = batch_data.loc[:, batch_data.columns == "perturbation_group"]
        tissues_batch = batch_data.loc[:, batch_data.columns == "tissue_super"]

        batch_data.drop("tissue_super", axis=1, inplace=True)
        batch_data.drop("sra_study", axis=1, inplace=True)
        
        batch_control = batch_data[batch_data['perturbation_group'] == 'control']
        
        batch_data.drop("perturbation_group", axis=1, inplace=True)
        batch_control.drop("perturbation_group", axis=1, inplace=True)
        
        batch_control = batch_control.mean(axis=0)
        
        batch_data = batch_data.div(batch_control + 1e-6)
        
        #batch_data = batch_data[first_group.columns]
        
        max_values_per_genes_batch = batch_data.max(axis=0)
        active_genes_batch = max_values_per_genes_batch[max_values_per_genes_batch.gt(5)].index
        
        first_group = first_group_copy.loc[:, active_genes_batch]
        batch_data = batch_data.loc[:, active_genes_batch]
        
        gene_mask = select_genes(batch_data, n=1500, threshold=0)
        
        first_group = first_group.loc[:, gene_mask]
        batch_data = batch_data.loc[:, gene_mask]
        
        first_group_raw = first_group.values
        first_group_raw = np.log1p(first_group_raw)
        first_group_raw -= np.mean(first_group_raw, axis=0)
        first_group_raw /= np.std(first_group_raw, axis=0) + 1e-9
        
        batch_data_raw = batch_data.values
        batch_data_raw = np.log1p(batch_data_raw)
        batch_data_raw -= np.mean(batch_data_raw, axis=0)
        batch_data_raw /= np.std(batch_data_raw, axis=0) + 1e-9
    
        affinities = affinity.PerplexityBasedNN(
            first_group_raw,
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=3,
        )
        
        embedding = TSNEEmbedding(
            first_group_tsne,
            affinities,
            negative_gradient_method="fft",
            n_jobs=8
        )
        
        new_embedding = embedding.prepare_partial(batch_data_raw, initialization="median", k=10)
        
        new_embedding.optimize(250, learning_rate=0.1, momentum=0.8, inplace=True)
        
        colors = get_colors_for(perturbations_first)
        
        cell_order = list(colors.keys())
        num_cell_types = len(np.unique(perturbations_first))
        
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        
        plot(first_group_tsne, perturbations_first['perturbation_group'], tissues_first['tissue_super'], ax=ax[0], title="Reference embedding", colors=colors, s=69, label_order=cell_order,
                  legend_kwargs=dict(loc="upper center", bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure, labelspacing=1, ncol=num_cell_types // 2 + 1))

        colors_bw = {1: "#666666"}
        plot(first_group_tsne, np.ones_like(perturbations_first['perturbation_group']), tissues_first['tissue_super'], ax=ax[1], colors=colors_bw, alpha=0.05, s=30, draw_legend=False)
        plot(new_embedding, perturbations_batch['perturbation_group'], tissues_batch['tissue_super'], ax=ax[1], colors=colors, draw_legend=False, s=69, label_order=cell_order, alpha=0.5)

        ijk +=1
        
        if ijk > 10 :
            break

"""
