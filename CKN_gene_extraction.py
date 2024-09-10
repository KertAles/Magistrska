# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:58:39 2024

@author: alesk
"""
import pandas as pd
import global_values as gv


isoform_file = open(gv.ISOFORM_COUNT_PATH, "r")

gene_list = []
for line in isoform_file:
    name_gene = line.split('\t')[0]
    gene_list.append(name_gene)

nodes_table = pd.read_table(gv.NODES_CKN_ANNOT_PATH, lineterminator='\n')

separated_genes = []
genes_not_present = []

for gene in gene_list :
    if gene in set(nodes_table['node_ID']) :
        separated_genes.append(gene)
    else :
        genes_not_present.append(gene)
        
        
genes_in_ckn = open(gv.GENES_IN_CKN_PATH, "w")
for gene in separated_genes :
    genes_in_ckn.write(gene + '\n')
        
genes_in_ckn.close()


genes_notin_ckn = open(gv.GENES_NOT_IN_CKN_PATH, "w")
for gene in genes_not_present :
    genes_notin_ckn.write(gene + '\n')
        
genes_notin_ckn.close()