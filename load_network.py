# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:06:32 2023

@author: alesk
"""

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

import global_values as gv


#tpm_table = pd.read_table(gv.TPM_PATH)

print('blah')


nodes_table = pd.read_table(gv.NODES_CKN_ANNOT_PATH, lineterminator='\n')
conn_table = pd.read_table(gv.INTERACTIONS_CKN_ANNOT_PATH)
tpm_table = pd.read_table('./data/averaged_data.tsv', index_col=0)

gene_max_rank = {}
gene_tissues = {}

for node in nodes_table['node_ID'] :
    if node in tpm_table.columns :
        gene_max_rank[node] = []
        gene_tissues[node] = nodes_table[nodes_table['node_ID'] == node]['tissue'].values[0]


tissues = ['seed', 'root', 'leaf', 'stem', 'flower']

for tissue in tissues :

    G = nx.DiGraph()
    
    for ind in nodes_table.index:
        if (tissue in nodes_table['tissue'][ind] or nodes_table['tissue'][ind] == 'not assigned') and nodes_table['node_ID'][ind] in tpm_table.columns:
            G.add_node(nodes_table['node_ID'][ind])
        
    for ind in conn_table.index:
        test = conn_table['source'][ind]
        if conn_table['source'][ind] in tpm_table.columns and conn_table['target'][ind] in tpm_table.columns :
            src = gene_tissues[conn_table['source'][ind]]
            dst = gene_tissues[conn_table['target'][ind]]
            
            if ((tissue in src or src == 'not assigned')
                and (tissue in dst or dst == 'not assigned')) :
                G.add_edge(conn_table['source'][ind], conn_table['target'][ind])
                if conn_table['isDirected'][ind] == 0 :
                    G.add_edge(conn_table['target'][ind], conn_table['source'][ind])
            
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    
    for node in largest_cc :
        gene_max_rank[node].append(tissue)
        
    ccs = nx.strongly_connected_components(G)
    print(f'Nodes in graph for {tissue}: {len(G)}')
    #print(f'Number of components : {len(ccs)}')
    print('Components by size: ')
    for cc in ccs:
        if len(cc) > 5 :
            print(len(cc))
        
    print('done')
    
for node in gene_max_rank :
    gene_max_rank[node] = ','.join(gene_max_rank[node])


ckn_genes_max_rank = open(gv.CKN_GENE_TISSUE, "w")
for gene in gene_max_rank :
    ckn_genes_max_rank.write(gene + '\t' + str(gene_max_rank[gene]) + '\n')
        
ckn_genes_max_rank.close()


"""
nodes_table = pd.read_table(gv.NODES_CKN_ANNOT_PATH, lineterminator='\n')
conn_table = pd.read_table(gv.INTERACTIONS_CKN_ANNOT_PATH)
tpm_table = pd.read_table('./data/averaged_data.tsv', index_col=0)

gene_max_rank = {}

for node in nodes_table['node_ID'] :
    if node in tpm_table.columns :
        gene_max_rank[node] = 5


for i in range(4,-1,-1) :

    G = nx.DiGraph()
    
    for ind in nodes_table.index:
        if nodes_table['node_ID'][ind] in tpm_table.columns :
            G.add_node(nodes_table['node_ID'][ind])
        
    for ind in conn_table.index:
        if conn_table['rank'][ind] <= i and conn_table['source'][ind] in tpm_table.columns and conn_table['target'][ind] in tpm_table.columns :
            G.add_edge(conn_table['source'][ind], conn_table['target'][ind])
            if conn_table['isDirected'][ind] == 0 :
                G.add_edge(conn_table['target'][ind], conn_table['source'][ind])
        
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    
    for node in largest_cc :
        gene_max_rank[node] = i


ckn_genes_max_rank = open(gv.CKN_GENE_RANKS, "w")
for gene in gene_max_rank :
    ckn_genes_max_rank.write(gene + '\t' + str(gene_max_rank[gene]) + '\n')
        
ckn_genes_max_rank.close()
"""
"""
import seaborn as sns



G = nx.Graph()

 
node_uniques = list(nodes_table['node_type'].value_counts().index)
cols = sns.color_palette(palette='Set2', n_colors=len(node_uniques))
node_cols = []

nodes_table.set_index('node_ID', inplace=True)

for node in nodes_table.index:
     G.add_node(node)
     
     
     
for ind in conn_table.index:
     if conn_table['rank'][ind] <= 2 :
         G.add_edge(conn_table['source'][ind], conn_table['target'][ind])
     
largest_cc = list(max(nx.connected_components(G), key=len))
S = G.subgraph(largest_cc).copy()

for node in S.nodes :
    node_cols.append(cols[node_uniques.index(nodes_table['node_type'][node])])

fig = plt.figure(figsize=[24, 24]) # create the canvas for plotting

nx.draw(S, pos=nx.spring_layout(S), node_size=120, node_color=node_cols, alpha=0.75, nodelist=largest_cc)

"""
"""

fig = plt.figure(figsize=[48, 48]) # create the canvas for plotting
ax1 = plt.subplot(1,1,1) 


nx.draw_spring(G, node_size=30, node_color=node_cols, label=uniques, ax=ax1)



"""