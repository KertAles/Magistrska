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
import umap
import umap.plot

import openTSNE

from global_values import JOINED_DATA_PATH_GROUPED


def plot(
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
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

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
                marker="s",
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
    
    plot(embedding_pca_cosine, labels1[target_name1])
    plot(embedding_pca_cosine, labels2[target_name2])

tpm_table = pd.read_table(JOINED_DATA_PATH_GROUPED)
#tpm_table.to_csv('joined_tpm.tsv', sep="\t") 

tpm_table.drop("idx", axis=1, inplace=True)

perturbations = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
tissues = tpm_table.loc[:, tpm_table.columns == "tissue_super"]

tpm_table.drop("tissue_super", axis=1, inplace=True)
tpm_table.drop("perturbation_group", axis=1, inplace=True)

data_raw = tpm_table

data_raw = data_raw.values

data_log = np.log10(data_raw + 1e-6)


open_TSNE_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')
TSNE_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')
UMAP_plot(data_log, tissues, 'tissue_super', perturbations, 'perturbation_group')