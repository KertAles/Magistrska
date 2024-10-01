# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:46:01 2024

@author: alesk
"""

import pandas as pd
import numpy as np

#test = pd.read_table('C:/Faks/Magistrska/data/athaliana_trimmed_2.tsv')

#test.set_index('gene_id', inplace=True)

#test = test.astype(float)

#test.T.to_csv('C:/Faks/Magistrska/data/athaliana_trimmed_T.tsv', sep="\t")

qc_data = pd.read_table('./data/athaliana_qc.tsv')
qc_data = qc_data[qc_data['category'] == 'QcPassRate']
qc_data['value'] = qc_data['value'].apply(lambda x: float(x[:-1]))

qc_dict = {}

for index, row in qc_data.iterrows() : 
    qc_dict[row['srr']] = row['value']

"""
ckn_file = open('./data/CKN_gene_ranks.txt', 'r')
genes_list = ['SRR_accession']
gene_positions = {}
i = 0
for line in ckn_file :
    gene_id = line.split('\t')[0]
    rank = int(line.split('\t')[1])
    
    if gene_id[:2] == 'AT' and '|' not in gene_id:
        gene_positions[gene_id] = i
        genes_list.append(gene_id)
        i += 1
        
vect_len = len(gene_positions)
"""

file_in = open('./data/athaliana_se.tsv', 'r')
file_out = open('./data/athaliana_trimmed_3.tsv', 'w')

file_out.write('\t'.join(genes_list) + '\t' + 'qc_rate' + '\n')
curr_srr = ''
write_out = ''
qc_cutoff = 99.0
gene_expressions = []
for line in file_in :
    print(line)
    line_split = line.split('\t')
    srr_acc = line_split[0]
    gene = line_split[1]
    
    if srr_acc != curr_srr :
        if write_out != '':
            write_out = write_out + '\t'.join(map(str, gene_expressions))
            file_out.write(write_out + '\t' + str(qc_dict[curr_srr]) + '\n')
            
        curr_srr = srr_acc
        gene_expressions = np.zeros(vect_len, dtype=int)
        
        if qc_dict[srr_acc] >= qc_cutoff:
            write_out = srr_acc + '\t'
            
    if gene in gene_positions:
        value = line_split[2]
        
        gene_expressions[gene_positions[gene]] = int(value)
        
file_out.close()
    
