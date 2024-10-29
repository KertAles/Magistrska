# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:09:08 2023

@author: alesk
"""

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing

import global_values as gv


class NonPriorData(Dataset) :
    
    def __init__(self, tpm_path, metadata_path=None, transformation='none', target='tissue', epsilon=1e-6, batches=[]):
        tpm_table = pd.read_table(tpm_path, index_col=0)
        #tpm_table.set_index('SRR_accession', inplace=True)
        #tpm_table.to_csv('joined_tpm.tsv', sep="\t") 
        self.transformation = transformation
        self.target = target
        
        #tpm_table = tpm_table[tpm_table['perturbation_group'].notin(['control', 'chemical stress'])]
        
        if metadata_path is not None :
            metadata_table = pd.read_table(metadata_path, index_col=0)
            metadata_table = metadata_table.rename(columns={'SRAStudy' : 'sra_study'})
            metadata_table = metadata_table[['perturbation_group', 'tissue_super', 'sra_study']]
            tpm_table = metadata_table.join(tpm_table, how='inner')
        #perturbations = tpm_table['perturbation_group']
        #tpm_table['perturbation_group'] = tpm_table['perturbation_group'].apply(
        #    lambda x: 'stressed' if 'control' not in x else x)
        
        if self.target != 'both' :
            self.onehot = preprocessing.OneHotEncoder()
            
            if self.target == 'perturbation' :
                self.onehot.fit(np.stack(tpm_table.loc[:, tpm_table.columns == "perturbation_group"].values).reshape(-1, 1))
            elif self.target == 'tissue' :
                self.onehot.fit(np.stack(tpm_table.loc[:, tpm_table.columns == "tissue_super"].values).reshape(-1, 1))
            elif self.target == 'secondary_perturbation' :
                self.onehot.fit(np.stack(tpm_table.loc[:, tpm_table.columns == "secondary_perturbation"].values).reshape(-1, 1))
            
            print(self.onehot.categories_)
        else :
            self.onehot_tissue = preprocessing.OneHotEncoder()
            self.onehot_perturbation = preprocessing.OneHotEncoder()
            
            self.onehot_tissue.fit(np.stack(tpm_table.loc[:, tpm_table.columns == "tissue_super"].values).reshape(-1, 1))
            self.onehot_perturbation.fit(np.stack(tpm_table.loc[:, tpm_table.columns == "perturbation_group"].values).reshape(-1, 1))
            
            print(self.onehot_tissue.categories_)
            print(self.onehot_perturbation.categories_)

        if len(batches) > 0 :
            tpm_table = tpm_table[tpm_table['sra_study'].isin(batches)]
        
        if self.target == 'perturbation' :
            data_raw = tpm_table.loc[:, tpm_table.columns != "perturbation_group"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
        elif self.target == 'tissue' :
            data_raw = tpm_table.loc[:, tpm_table.columns != "tissue_super"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
        elif self.target == 'both' :
            tissue_raw = tpm_table.loc[:, tpm_table.columns == "tissue_super"]
            perturbation_raw = tpm_table.loc[:, tpm_table.columns == "perturbation_group"]
            data_raw = tpm_table.loc[:, tpm_table.columns != "perturbation_group"]
        elif self.target == 'secondary_perturbation' :
            tpm_table = tpm_table.drop(tpm_table[tpm_table['perturbation_group'] != 'environmental stress'].index)
            data_raw = tpm_table.loc[:, tpm_table.columns != "secondary_perturbation"]
            gt_raw = tpm_table.loc[:, tpm_table.columns == "secondary_perturbation"]
        else :
            print('INVALID TARGET')
            
        if 'sra_study' in data_raw.columns :
            data_raw.drop("sra_study", axis=1, inplace=True)
            
        if 'tissue_super' in data_raw.columns :
            data_raw.drop("tissue_super", axis=1, inplace=True)
            
        if 'perturbation_group' in data_raw.columns :
            data_raw.drop("perturbation_group", axis=1, inplace=True)
          
        if 'secondary_perturbation' in data_raw.columns :
            data_raw.drop("secondary_perturbation", axis=1, inplace=True)
        
        #tpm_cols = pd.read_table('data/columns.tsv')
        #data_raw = data_raw[data_raw.columns.intersection(tpm_cols.columns)]
        
        self.num_of_genes = len(data_raw.columns)
        self.columns = data_raw.columns
        
        data_raw = data_raw.values * 492263.0
        
        max_val = np.max(data_raw)
        min_val = np.min(data_raw)
        max_val -= min_val
        
        if self.target != 'both' :
            gt_raw = gt_raw.values
        else :
            tissue_raw = tissue_raw.values
            perturbation_raw = perturbation_raw.values
            
        data_raw -= min_val
        if self.transformation == 'log2' :
            #mini = -4.0
            #mini = np.min(data_raw)
            data_raw = np.log1p(data_raw)
            max_val = np.log1p(max_val)
        elif self.transformation == 'log10' :
            data_raw = np.log10(data_raw + 1)
            max_val = np.log10(max_val + 1)
        elif self.transformation != 'none' :
            print('INVALID TRANSFORMATION')
            
            
        self.data = np.stack(data_raw)
        
        if self.target != 'both' :
            self.gt = self.onehot.transform(np.stack(gt_raw).reshape(-1, 1)).todense()
        else :
            self.gt_tissues = self.onehot_tissue.transform(np.stack(tissue_raw).reshape(-1, 1)).todense()
            self.gt_perturbations = self.onehot_perturbation.transform(np.stack(perturbation_raw).reshape(-1, 1)).todense()
        print(np.max(self.data))
        print(np.min(self.data))
        
        print(max_val)
        #max_val = np.max(self.data)
        #max_val = 2.9
        if self.transformation == 'none' :
            self.data /= 10.5
        elif self.transformation == 'log10' :
            self.data /= 10.5
        elif self.transformation == 'log2' :
            self.data /= 10.5
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
    dataset = NonPriorData(gv.GROUPED_DATA, transformation='log10', target='both')
    item = dataset.__getitem__(0)