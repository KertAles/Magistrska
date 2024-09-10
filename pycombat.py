# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:07:19 2024

@author: alesk
"""

import pandas as pd
import numpy as np
from combat.pycombat import pycombat
import global_values as gv

data = pd.read_table(gv.GROUPED_DATA, index_col=0)

studies = data['sra_study']
data = data.drop(columns=['perturbation_group', 'tissue_super', 'sra_study'])

data = data.dropna()

data = data.apply(np.log1p)
data = data.transpose()
#data = data.values
studies = studies.values

corrected = pycombat(data, studies)