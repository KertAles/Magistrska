# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:58:39 2024

@author: alesk
"""
import pandas as pd
from global_values import NODES_CKN_ANNOT_PATH, ISOFORM_COUNT_PATH, GENES_IN_CKN_PATH, GENES_NOT_IN_CKN_PATH


isoform_file = open(ISOFORM_COUNT_PATH, "r")

gene_list = []
for line in isoform_file:
    name_gene = line.split('\t')[0]
    gene_list.append(name_gene)

nodes_table = pd.read_table(NODES_CKN_ANNOT_PATH, lineterminator='\n')

separated_genes = []
genes_not_present = []

for gene in gene_list :
    if gene in set(nodes_table['node_ID']) :
        separated_genes.append(gene)
    else :
        genes_not_present.append(gene)
        
        
genes_in_ckn = open(GENES_IN_CKN_PATH, "w")
for gene in separated_genes :
    genes_in_ckn.write(gene + '\n')
        
genes_in_ckn.close()


genes_notin_ckn = open(GENES_NOT_IN_CKN_PATH, "w")
for gene in genes_not_present :
    genes_notin_ckn.write(gene + '\n')
        
genes_notin_ckn.close()