# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:04:25 2024

@author: alesk
"""

import pandas as pd
import numpy as np
import global_values as gv

from inmoose.pycombat.pycombat_seq import pycombat_seq

data = pd.read_table(gv.GROUPED_DATA, index_col=0)

batches = data['sra_study'].values

data = data.drop(columns=['perturbation_group', 'tissue_super', 'sra_study'])

data = data.transpose()


corrected = pycombat_seq(data, batches)
pi