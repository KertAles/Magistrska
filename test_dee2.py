# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:34:01 2024

@author: alesk
"""

import pandas as pd

metadata_table = pd.read_table('./data/athaliana_metadata.tsv')

exp_strat = metadata_table['experiment_library_strategy'].value_counts()

exp_selec = metadata_table['experiment_library_selection'].value_counts()

exp_model = metadata_table['experiment_instrument_model'].value_counts()

print(exp_model)