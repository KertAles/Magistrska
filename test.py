# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:30:27 2024

@author: alesk
"""

import pandas as pd
import global_values as gv


meta = pd.read_table(gv.METADATA_PATH, index_col=1)
tpm2 = pd.read_table(gv.TPM_PATH, index_col=0)


grouped = tpm2.join(meta)


#meta2 = pd.read_table('C:/Faks/Magistrska/data/athaliana_metadata.tsv')
#tpm2 = tpm2[tpm2['tissue_super'] != 'senescence']


#batch_count = tpm2['sra_study'].value_counts()

#pert_count = tpm2['perturbation_group'].value_counts()

#tissue_count = tpm2['tissue_super'].value_counts()