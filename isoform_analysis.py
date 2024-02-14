# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:37:45 2024

@author: alesk
"""

import numpy as np
import matplotlib.pyplot as plt
from global_values import ISOFORM_COUNT_PATH

isoform_file = open(ISOFORM_COUNT_PATH, "r")
isoform_count = {}
isoform_list = []
for line in isoform_file:
    name_gene = line.split('\t')[0]
    num_isoforms = int(line.split('\t')[1])
    isoform_count[name_gene] = num_isoforms
    isoform_list.append(num_isoforms)
    

maximum = max(isoform_count.values())
keys = filter(lambda x:isoform_count[x] == maximum, isoform_count.keys())

print("Maximal number of isoforms: ")
print(maximum)
for key in keys :
    print(key)
    
isoform_list_np = np.array(isoform_list)

print("Mean value and standard deviation: ")
print(np.mean(isoform_list_np))
print(np.std(isoform_list_np))

plt.figure(figsize=(10,6))
plt.title('Distribution of number of isoforms')
n, bins, patches = plt.hist(isoform_list, bins=26)
plt.show()

isoform_list_cut = list(filter(lambda a: a not in [1, 2, 3, 4, 5], isoform_list))

plt.figure(figsize=(10,6))
plt.title('Distribution of number of isoforms without numbers <6')
n, bins, patches = plt.hist(isoform_list_cut, bins=21)
plt.show()