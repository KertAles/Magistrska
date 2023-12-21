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
    
    def __init__(self, tpm_path, transformation='none', target='tissues', epsilon=1e-6):
        tpm_table = pd.read_table(tpm_path)
        #tpm_table.to_csv('joined_tpm.tsv', sep="\t") 
        self.transformation = transformation
        self.target = target
        
        tpm_table.drop("idx", axis=1, inplace=True)
        
        if self.target == 'perturbation' :
            tpm_table.drop("tissue_super", axis=1, inplace=True)
            data_raw = tpm_table.loc[:, tpm_table.columns != "perturbation_group"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
        elif self.target == 'tissues' :
            tpm_table.drop("perturbation_group", axis=1, inplace=True)
            data_raw = tpm_table.loc[:, tpm_table.columns != "tissue_super"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
            
        data_raw = data_raw.values
        gt_raw = gt_raw.values
        
        if self.transformation == 'log2' :
            data_raw = np.log2(data_raw + epsilon)
        elif self.transformation == 'log10' :
            data_raw = np.log10(data_raw + epsilon)
        
        self.onehot = preprocessing.OneHotEncoder()
        
        self.data = np.stack(data_raw)
        self.gt = self.onehot.fit_transform(np.stack(gt_raw).reshape(-1, 1)).todense()
        
        print(self.onehot.categories_)
        test = self.gt
        
        max_val = np.max(self.data)
        self.data /= max_val
        
    def __len__(self):
        return self.gt.shape[0] 
        
    
    def __getitem__(self, idx):
        
        tpm, gt = (self.data[idx, :], self.gt[idx])

        return {
            'tpm': tpm.copy(),
            'classification': gt.copy()
        }
    
if __name__ == '__main__':
    dataset = NonPriorData(JOINED_DATA_PATH)
    dataset.__getitem__(0)