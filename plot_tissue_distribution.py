# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:31:57 2024

@author: alesk
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

import global_values as gv


if __name__ == '__main__':
    tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)
    #tpm_table.set_index('SRR_accession', inplace=True)
    
    tpm_table = tpm_table[tpm_table['tissue_super'] != 'senescence']
    
    tissues_col = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
    tissues_col = tissues_col.values
    encoder = preprocessing.LabelEncoder()
    encoder.fit(np.stack(tissues_col).reshape(-1, 1))
    
    distribution_dict = {}
    
    for index, row in tpm_table.iterrows():
        perturbation = row['perturbation_group']
        tissue = row['tissue_super']
        
        if not perturbation in distribution_dict :
            distribution_dict[perturbation] = np.zeros(len(encoder.classes_))
        
        tissue_code = encoder.transform(np.array([tissue]))
        
        distribution_dict[perturbation][tissue_code] += 1
        
    
    distribution_dict_copy = distribution_dict.copy()
    
    for perturbation, tissue_count in distribution_dict.items():
        distribution_dict[perturbation] /= np.sum(tissue_count)
    
    tissue_dict = {}
    tissues = encoder.classes_
    i = 0
    for tissue in tissues :
        tissue_dict[tissue] = np.zeros(len(distribution_dict.keys()))
        j = 0
        for perturbation, tissue_count in distribution_dict.items() :
            tissue_dict[tissue][j] = tissue_count[i]
            j += 1
        i += 1
        
        
    fig, ax = plt.subplots(figsize=(12,6))
    width = 0.5
    
    bottom = np.zeros(len(distribution_dict.keys())) 
    
    for tissue, pert_count in tissue_dict.items():
        p = ax.bar(distribution_dict.keys(), pert_count, width, bottom=bottom, label=tissue)
        bottom += pert_count
    
    
    ax.set_title("Proportion of tissue types per perturbation group")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    plt.show()
    
    
    tissue_dict = {}
    tissues = encoder.classes_
    i = 0
    for tissue in tissues :
        tissue_dict[tissue] = np.zeros(len(distribution_dict_copy.keys()))
        j = 0
        for perturbation, tissue_count in distribution_dict_copy.items() :
            tissue_dict[tissue][j] = tissue_count[i]
            j += 1
        i += 1
        
        
    for tissue, pert_count in tissue_dict.items():
        tissue_dict[tissue] /= np.sum(pert_count)
    
    distribution_dict = {}
    perts = distribution_dict_copy.keys()
    i = 0
    for pert in perts :
        distribution_dict[pert] = np.zeros(len(tissues))
        j = 0
        for tissue, pert_count in tissue_dict.items() :
            distribution_dict[pert][j] = pert_count[i]
            j += 1
        i += 1
    
    
    fig, ax = plt.subplots(figsize=(20,6))
    width = 0.5
    tissues = encoder.classes_
    bottom = np.zeros(len(tissues))
    
    for perturbation, tissue_count in distribution_dict.items():
        p = ax.bar(tissues, tissue_count, width, bottom=bottom, label=perturbation)
        bottom += tissue_count
        
    
    ax.set_title("Proportion of perturbation groups per tissue type")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    plt.show()