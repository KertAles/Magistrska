# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:21:25 2024

@author: alesk
"""

import numpy as np
import pandas as pd
from sklearn import manifold
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

    ax.figure.savefig(f'{title}.png')

def TSNE_plot(data, labels1, target_name1, labels2, target_name2) :
    tsne = manifold.TSNE()
    data_transformed = tsne.fit_transform(data)
    
    target_df1 = pd.concat([pd.DataFrame(data_transformed), labels1], axis=1)
    target_df2 = pd.concat([pd.DataFrame(data_transformed), labels2], axis=1)
    
    sns.pairplot(target_df1, hue=target_name1, height=16)
    sns.pairplot(target_df2, hue=target_name2, height=16)


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


def plot_tsne(data, labels1, target_name1, labels2, target_name2, labels3, target_name3):
    embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=8,
    random_state=3,
    ).fit(data)
    
    plot_1(embedding_pca_cosine, labels1[target_name1], title='t-SNE plot of perturbation groups')
    plot_1(embedding_pca_cosine, labels2[target_name2], title='t-SNE plot of tissue types')
    plot_1(embedding_pca_cosine, labels3[target_name3], draw_legend=False, title='t-SNE plot of batches')


def save_tsne(tpm_table, display_name) :
    
    perturbations = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
    tissues = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
    sra_studies = tpm_table.loc[:, tpm_table.columns == "sra_study"]
    
    tpm_table.drop("tissue_super", axis=1, inplace=True)
    tpm_table.drop("perturbation_group", axis=1, inplace=True)
    tpm_table.drop("sra_study", axis=1, inplace=True)

    data_raw = tpm_table.values
    #data_log = np.log1p(data_raw)
    data_log=data_raw
    
    embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=8,
    random_state=3,
    ).fit(data_log)
    
    plot_1(embedding_pca_cosine, perturbations['perturbation_group'], title=f'{display_name} - t-SNE plot of perturbation groups')
    plot_1(embedding_pca_cosine, tissues['tissue_super'], title=f'{display_name} - t-SNE plot of tissue types')
    plot_1(embedding_pca_cosine, sra_studies['sra_study'], draw_legend=False, title=f'{display_name} - t-SNE plot of batches')

if __name__ == '__main__':
    measurements = [
            {'tpm_path' : gv.GROUPED_DATA, 'T' : False, 'meta_path' : None, 'display_name' : 'Unprocessed data'},
            {'tpm_path' : gv.PROPORTIONAL_DATA_CONTROLS, 'T' : False, 'meta_path' : None, 'display_name' : 'Proportional gene expression'},
            {'tpm_path' : './data/vae_cov_transformed_filtered.tsv', 'T' : False, 'meta_path' : None, 'display_name' : 'Data subset'},
            {'tpm_path' : './data/athaliana_annotated.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'Unprocessed data 2'},
            {'tpm_path' : './data/combat.tsv', 'T' : True, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'ComBat'},
            {'tpm_path' : './data/combat_seq.tsv', 'T' : True, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'ComBat-seq'},
            {'tpm_path' : './data/vae_transformed.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'scVI VAE'},
            {'tpm_path' : './data/vae_cov_transformed.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'scVI VAE w covariates'},
            {'tpm_path' : './data/vae_smol_transformed.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'scVI VAE smol'},
            {'tpm_path' : './data/vae_cov_transformed_smol.tsv', 'T' : False, 'meta_path' : './data/metadata_T.tsv', 'display_name' : 'scVI VAE smol w/ covariates'},
            
        ]

    for measurement in measurements :
        tpm_table = pd.read_table(measurement['tpm_path'], index_col=0)

        
        if measurement['T'] :
            tpm_table = tpm_table.T
             
        
        if measurement['meta_path'] == None :
            perturbation_count = tpm_table['perturbation_group'].value_counts()
            perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
            
            tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
            tissue_count = tpm_table['tissue_super'].value_counts()
        else :
            metadata_table = pd.read_table(measurement['meta_path'], index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
            tpm_table = tpm_table.join(metadata_table, how='inner')
            
            #tpm_table.dropna(inplace=True)
            
            perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
            perturbation_count = perturbations['perturbation_group'].value_counts()
            
            tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
            tissue_count = tissues['tissue_super'].value_counts()
            
        #if 'sra_study' in tpm_table.columns :
        #    tpm_table.drop("sra_study", axis=1, inplace=True)
          
        #if 'tissue_super' in tpm_table.columns :
        #    tpm_table.drop("tissue_super", axis=1, inplace=True)
            
        #if 'perturbation_group' in tpm_table.columns :
        #    tpm_table.drop("perturbation_group", axis=1, inplace=True)
          
        if 'secondary_perturbation' in tpm_table.columns :
            tpm_table.drop("secondary_perturbation", axis=1, inplace=True)
    
        #tpm_table = tpm_table.drop(columns=[col for col in tpm_table if col.split('.')[0] not in genes_list])
        
        print('Drawing plots for ' + measurement['tpm_path'])
        
        save_tsne(tpm_table, measurement['display_name'])
    
        