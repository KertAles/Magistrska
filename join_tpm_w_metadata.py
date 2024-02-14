# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:18:53 2023

@author: alesk
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing

from global_values import TPM_PATH, METADATA_PATH, JOINED_DATA_PATH, JOINED_DATA_PATH_GROUPED, JOINED_DATA_PATH_CKN, GENES_IN_CKN_PATH

allowed_tissues = ['young_seedling', 'seed',
                   'mature_leaf', 'mature_root', 'mature_flower',
                   'senescence']

if __name__ == '__main__':
    
    tpm_table = pd.read_table(TPM_PATH)
    metadata_table = pd.read_table(METADATA_PATH)
    separate_CKN_genes = True
    
    
    if separate_CKN_genes :
        separated_genes = []
        
        genes_file = open(GENES_IN_CKN_PATH, 'r')
        
        for gene in genes_file :
            separated_genes.append(gene[:-1])
            
        genes_file.close()
    
    dict_list = []
    
    gene_id_col = None
        
    for column in tpm_table :
        if column != 'gene_id' :
            experiment_id = column[:-4]
            experiment_metadata = metadata_table.loc[metadata_table['SRR_accession'] == experiment_id]
            
            """
            if experiment_metadata['tissue_super'].values in allowed_tissues:
                experiment_data = tpm_table[column].values
                
                curr_dict = {}
                curr_dict['tissue_super'] = experiment_metadata['tissue_super'].values[0]
                for gene_id, value in zip(gene_id_col, experiment_data) :
                    curr_dict[gene_id] = value
                dict_list.append(curr_dict)
            
            
            if experiment_metadata['perturbation_group'].values[0] != 'unknown' :
                experiment_data = tpm_table[column].values
                    
                curr_dict = {}
                perturbation_group = experiment_metadata['perturbation_group'].values[0]
                
                if perturbation_group == 'unstressed' :
                    perturbation_group = 'control'
                    
                curr_dict['perturbation_group'] = perturbation_group
                for gene_id, value in zip(gene_id_col, experiment_data) :
                    curr_dict[gene_id] = value
                dict_list.append(curr_dict)
            """
            if experiment_metadata['perturbation_group'].values[0] != 'unknown':
                experiment_data = tpm_table[column].values
                    
                curr_dict = {}
                perturbation_group = experiment_metadata['perturbation_group'].values[0]
                tissue = experiment_metadata['tissue_super'].values[0]
                
                if 'senescence' in tissue :
                    tissue = 'senescence'
                
                if 'seed' in tissue and 'seedling' not in tissue :
                    tissue = 'seed'
                
                if perturbation_group == 'unstressed' :
                    perturbation_group = 'control'
                    
                if tissue in allowed_tissues:
                    curr_dict['perturbation_group'] = perturbation_group
                    curr_dict['tissue_super'] = tissue
                    for gene_id, value in zip(gene_id_col, experiment_data) :
                        gene_name = gene_id.split('.')[0]
                        if not separate_CKN_genes :
                            curr_dict[gene_id] = value
                        elif separate_CKN_genes and gene_name in separated_genes :
                            curr_dict[gene_id] = value
                    dict_list.append(curr_dict)
        else :
            gene_id_col = tpm_table[column].values
        
    joined_tpm = pd.DataFrame(dict_list)
    
    if separate_CKN_genes :
        joined_tpm.to_csv(JOINED_DATA_PATH_CKN, sep="\t")
    else :
        joined_tpm.to_csv(JOINED_DATA_PATH_GROUPED, sep="\t")
        
        
        
        
    tpm_table = pd.read_table(TPM_PATH)
    metadata_table = pd.read_table(METADATA_PATH)
    separate_CKN_genes = False
    
    
    if separate_CKN_genes :
        separated_genes = []
        
        genes_file = open(GENES_IN_CKN_PATH, 'r')
        
        for gene in genes_file :
            separated_genes.append(gene[:-1])
            
        genes_file.close()
    
    dict_list = []
    
    gene_id_col = None
        
    for column in tpm_table :
        if column != 'gene_id' :
            experiment_id = column[:-4]
            experiment_metadata = metadata_table.loc[metadata_table['SRR_accession'] == experiment_id]
            
            """
            if experiment_metadata['tissue_super'].values in allowed_tissues:
                experiment_data = tpm_table[column].values
                
                curr_dict = {}
                curr_dict['tissue_super'] = experiment_metadata['tissue_super'].values[0]
                for gene_id, value in zip(gene_id_col, experiment_data) :
                    curr_dict[gene_id] = value
                dict_list.append(curr_dict)
            
            
            if experiment_metadata['perturbation_group'].values[0] != 'unknown' :
                experiment_data = tpm_table[column].values
                    
                curr_dict = {}
                perturbation_group = experiment_metadata['perturbation_group'].values[0]
                
                if perturbation_group == 'unstressed' :
                    perturbation_group = 'control'
                    
                curr_dict['perturbation_group'] = perturbation_group
                for gene_id, value in zip(gene_id_col, experiment_data) :
                    curr_dict[gene_id] = value
                dict_list.append(curr_dict)
            """
            if experiment_metadata['perturbation_group'].values[0] != 'unknown':
                experiment_data = tpm_table[column].values
                    
                curr_dict = {}
                perturbation_group = experiment_metadata['perturbation_group'].values[0]
                tissue = experiment_metadata['tissue_super'].values[0]
                
                if 'senescence' in tissue :
                    tissue = 'senescence'
                
                if 'seed' in tissue and 'seedling' not in tissue :
                    tissue = 'seed'
                
                if perturbation_group == 'unstressed' :
                    perturbation_group = 'control'
                    
                if tissue in allowed_tissues:
                    curr_dict['perturbation_group'] = perturbation_group
                    curr_dict['tissue_super'] = tissue
                    for gene_id, value in zip(gene_id_col, experiment_data) :
                        gene_name = gene_id.split('.')[0]
                        if not separate_CKN_genes :
                            curr_dict[gene_id] = value
                        elif separate_CKN_genes and gene_name in separated_genes :
                            curr_dict[gene_id] = value
                    dict_list.append(curr_dict)
        else :
            gene_id_col = tpm_table[column].values
        
    joined_tpm = pd.DataFrame(dict_list)
    
    if separate_CKN_genes :
        joined_tpm.to_csv(JOINED_DATA_PATH_CKN, sep="\t")
    else :
        joined_tpm.to_csv(JOINED_DATA_PATH_GROUPED, sep="\t")