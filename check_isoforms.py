# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:46:36 2024

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

import global_values as gv

tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)
#tpm_table.set_index('SRR_accession', inplace=True)
#tpm_table.to_csv('joined_tpm.tsv', sep="\t") 

#tpm_table.drop("idx", axis=1, inplace=True)

tpm_table.drop("tissue_super", axis=1, inplace=True)
tpm_table.drop("perturbation_group", axis=1, inplace=True)
f = open("data/isoform_count.txt", "a")

checked_genes = []
for name_isoform, values in tpm_table.items():
   name_gene = name_isoform.split('.')[0]
   if name_gene not in checked_genes :
       checked_genes.append(name_gene)
       isoforms = tpm_table.filter(regex=name_gene)
       
       idx_dic = {}
       for col in isoforms.columns:
           idx_dic[col] = tpm_table.columns.get_loc(col)
           
       #print(idx_dic)
       i = -1
       j = 0
       for isoform in idx_dic :
           j += 1
           if i == -1 :
               i = idx_dic[isoform]
           else :
               diff = idx_dic[isoform] - i
               
               if diff != 1 :
                   print('Difference between two genes is too large!')
                   print(diff)
                   print('')
                   
               i = idx_dic[isoform]
       f.write(name_gene + '\t' + str(j) + '\n')
       

f.close()