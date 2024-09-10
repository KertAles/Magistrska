# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:21:25 2024

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

import openTSNE

import global_values as gv

def plot(
    x,
    y,
    y2,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.55), "s": kwargs.get("s", 69)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
        classes2 = np.unique(y2)
    else:
        classes = np.unique(y)
        classes2 = np.unique(y2)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
        
    else :
        colors = {k: v for k, v in zip(classes, colors)}
        
    point_colors = np.array(list(map(colors.get, y)))
    
    markers_list = ['o','s','^','P','*','v', 'D']
    markers = {m: v for m, v in zip(classes2, markers_list)}

    for class2 in classes2 :
        indices = np.where(y2 == class2)[0].astype(int)
        ax.scatter(x[indices, 0], x[indices, 1], c=point_colors[indices],
                   marker=markers[class2], rasterized=True, edgecolors='k',
                   **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker='s',
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_handles.extend([
            matplotlib.lines.Line2D(
                [],
                [],
                marker=markers[yi2],
                color="w",
                markerfacecolor='k',
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi2,
                markeredgecolor="k",
            )
            for yi2 in classes2
        ])
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


def plot_1(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.35), "s": kwargs.get("s", 24)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
        
    else :
        colors = {k: v for k, v in zip(classes, colors)}
        
    point_colors = np.array(list(map(colors.get, y)))
    
    
    ax.scatter(x[:, 0], x[:, 1], c=point_colors,
               rasterized=True,
               **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=24, alpha=0.5
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker='s',
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)



def TSNE_plot(data, labels1, target_name1, labels2, target_name2) :
    tsne = manifold.TSNE()
    data_transformed = tsne.fit_transform(data)
    
    target_df1 = pd.concat([pd.DataFrame(data_transformed), labels1], axis=1)
    target_df2 = pd.concat([pd.DataFrame(data_transformed), labels2], axis=1)
    
    sns.pairplot(target_df1, hue=target_name1, height=16)
    sns.pairplot(target_df2, hue=target_name2, height=16)

def UMAP_plot(data, labels1, target_name1, labels2, target_name2) :
    mapper = umap.UMAP().fit(data)
    umap.plot.points(mapper, labels=labels1[target_name1], width=600, height=600)
    umap.plot.points(mapper, labels=labels2[target_name2], width=600, height=600)


def open_TSNE_plot(data, labels1, target_name1, labels2, target_name2) :
    
    embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=8,
    random_state=3,
    ).fit(data)
    
    colors = ['grey', 'red', 'lime', 'teal',
              'blue', 'indigo', 'fuchsia', 'chocolate', 'yellow',
              'darkgreen', 'cyan', 'royalblue', 'tomato', 'skyblue',
              'thistle', 'crimson']
    
    plot(embedding_pca_cosine, labels1[target_name1], labels2[target_name2], colors=colors)
    #plot(embedding_pca_cosine, labels2[target_name2], labels1[target_name1])


if __name__ == '__main__':
    tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)
    #tpm_table.set_index('SRR_accession', inplace=True)
    #tpm_table = pd.read_table('./data/athaliana_metadata.tsv', index_col=0)
    #tpm_table = pd.read_table(gv.METADATA_PATH, index_col=0)
    
    #exp_count = tpm_table['experiment_library_selection'].value_counts()
    #instr_count = tpm_table['experiment_instrument_model'].value_counts()
    
    #tpm_table = tpm_table[tpm_table['tissue_super'].isin(['mature_leaf'])]
    #tpm_table = tpm_table[tpm_table['perturbation_group'].isin(['control', 'chemical stress'])]
    tpm_table = tpm_table[tpm_table['tissue_super'] != 'senescence']
    tpm_table.dropna(inplace=True)
    
    batch_count = tpm_table['sra_study'].value_counts()
    chosen_batches = []

    for batch in batch_count.index :
        batch_data = tpm_table[tpm_table['sra_study'] == batch]
        
        if 'control' not in batch_data['perturbation_group'].unique() or 'chemical stress' not in batch_data['perturbation_group'].unique():
            continue
        
        chosen_batches.append(batch)
    
    #tpm_table = tpm_table[tpm_table['sra_study'].isin(chosen_batches)]
    #tpm_table = tpm_table[tpm_table['perturbation_group'].isin(['control', 'chemical stress'])]
    
    #tpm_table.to_csv('joined_tpm.tsv', sep="\t") 
  
    chosen_rank = 2
    
    rank_file = open(gv.CKN_GENE_RANKS, 'r')
    genes_in_lcc = []

    for line in rank_file:
        gene_rank = int(line.split('\t')[1])
        if gene_rank <= chosen_rank :
            genes_in_lcc.append(line.split('\t')[0])
            
    batch_count = tpm_table['sra_study'].value_counts()
    
    #tpm_table = tpm_table.drop(tpm_table[tpm_table['perturbation_group'] != 'environmental stress'].index)
    
    perturbations = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
    #scnd_perturbations = tpm_table.loc[:, tpm_table.columns == "secondary_perturbation"]
    tissues = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
    sra_studies = tpm_table.loc[:, tpm_table.columns == "sra_study"]
    
    tpm_table.drop("tissue_super", axis=1, inplace=True)
    tpm_table.drop("perturbation_group", axis=1, inplace=True)
    #tpm_table.drop("secondary_perturbation", axis=1, inplace=True)
    tpm_table.drop("sra_study", axis=1, inplace=True)
    
    #tpm_table = tpm_table.drop(columns=[col for col in tpm_table if col.split('.')[0] not in genes_in_lcc])
    
    data_raw = tpm_table.values
    
    data_log = np.log1p(data_raw)
    
    
    
    
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
    
    plot_1(embedding_pca_cosine, perturbations['perturbation_group'], title='t-SNE plot of perturbation groups')
    plot_1(embedding_pca_cosine, tissues['tissue_super'], title='t-SNE plot of tissue types')
    plot_1(embedding_pca_cosine, sra_studies['sra_study'], draw_legend=False, title='t-SNE plot of batches')
    
    #open_TSNE_plot(data_log, perturbations, 'perturbation_group', tissues, 'tissue_super')
    #open_TSNE_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')
    #TSNE_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')
    #UMAP_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')