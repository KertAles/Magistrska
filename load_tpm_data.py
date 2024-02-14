# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:09:08 2023

@author: alesk
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing

from global_values import JOINED_DATA_PATH


class NonPriorData(Dataset) :
    
    def __init__(self, tpm_path, transformation='none', target='tissue', epsilon=1e-6):
        tpm_table = pd.read_table(tpm_path)
        #tpm_table.to_csv('joined_tpm.tsv', sep="\t") 
        self.transformation = transformation
        self.target = target
        
        tpm_table.drop("idx", axis=1, inplace=True)
        
        if self.target == 'perturbation' :
            tpm_table.drop("tissue_super", axis=1, inplace=True)
            data_raw = tpm_table.loc[:, tpm_table.columns != "perturbation_group"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
        elif self.target == 'tissue' :
            tpm_table.drop("perturbation_group", axis=1, inplace=True)
            data_raw = tpm_table.loc[:, tpm_table.columns != "tissue_super"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
        elif self.target == 'both' :
            tissue_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
            tpm_table.drop("tissue_super", axis=1, inplace=True)
            perturbation_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
            data_raw = tpm_table.loc[:, tpm_table.columns != "perturbation_group"]

        else :
            print('INVALID TARGET')
        
        data_raw = data_raw.values
        if self.target != 'both' :
            gt_raw = gt_raw.values
        else :
            tissue_raw = tissue_raw.values
            perturbation_raw = perturbation_raw.values
            
        if self.transformation == 'log2' :
            data_raw = np.log2(data_raw + epsilon)
        elif self.transformation == 'log10' :
            data_raw = np.log10(data_raw + epsilon)
        elif self.transformation != 'none' :
            print('INVALID TRANSFORMATION')
            
            
        self.data = np.stack(data_raw)
        
        if self.target != 'both' :
            self.onehot = preprocessing.OneHotEncoder()
            self.gt = self.onehot.fit_transform(np.stack(gt_raw).reshape(-1, 1)).todense()
            print(self.onehot.categories_)
        else :
            self.onehot_tissue = preprocessing.OneHotEncoder()
            self.onehot_perturbation = preprocessing.OneHotEncoder()
            
            self.gt_tissues = self.onehot_tissue.fit_transform(np.stack(tissue_raw).reshape(-1, 1)).todense()
            self.gt_perturbations = self.onehot_perturbation.fit_transform(np.stack(perturbation_raw).reshape(-1, 1)).todense()
            
            print(self.onehot_tissue.categories_)
            print(self.onehot_perturbation.categories_)
        print(np.max(self.data))
        print(np.min(self.data))
        
        if self.transformation == 'none' :
            max_val = np.max(self.data)
            self.data /= max_val
        elif self.transformation == 'log10' :
            self.data /= 6
        elif self.transformation == 'log2' :
            self.data /= 20
    def __len__(self):
        return self.data.shape[0] 
        
    
    def __getitem__(self, idx):
        
        if self.target != 'both' :
            tpm, gt = (self.data[idx, :], self.gt[idx])
            ret_data = {'tpm' : tpm, 'classification' : gt}
        else :
            tpm, gt_t, gt_p = (self.data[idx, :], self.gt_tissues[idx], self.gt_perturbations[idx])
            ret_data = {'tpm' : tpm, 'tissue' : gt_t, 'perturbation' : gt_p}
        return ret_data
    
if __name__ == '__main__':
    dataset = NonPriorData(JOINED_DATA_PATH, transformation='log10', target='both')
    dataset.__getitem__(0)