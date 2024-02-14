# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:06:32 2023

@author: alesk
"""

import networkx as nx
import pandas as pd
import randomcolor
from matplotlib import pyplot as plt

NODES_ANNOT_PATH = 'data/PSS_nodes.tsv'
INTERACTIONS_ANNOT_PATH = 'data/PSS_conn.tsv'

TPM_PATH = 'data/TPM_table.tsv'


tpm_table = pd.read_table(TPM_PATH)

print('blah')

nodes_table = pd.read_table(NODES_ANNOT_PATH, lineterminator='\n')
conn_table = pd.read_table(INTERACTIONS_ANNOT_PATH)

uniques = ["Complex", "ForeignCoding", "PlantCoding", "ForeignAbiotic", "Metabolite", "Condition", "PlantAbstract", "ForeignEntity", "Process", "PlantNonCoding", "ForeignNonCoding"]
colors = ["#f5bf42", "#42f2f5", "#57f542", "#5d42f5", "#f542e3", "#f54296", "#b0f542", "#42f5c2", "#4248f5", "#4bf542", "#ecf542"]
uniques_col = {}
for typ, col in zip(uniques, colors) :
    uniques_col[typ] = col
G = nx.Graph()
node_cols = []
labels = {}

for ind in nodes_table.index:
    node_type = nodes_table['node_type'][ind]
    #if node_type not in uniques :
    #    uniques.append(node_type)
    #    uniques_col[node_type] = randomcolor.RandomColor().generate()[0]
    G.add_node(nodes_table['name'][ind])
    node_cols.append(uniques_col[node_type])
    
    labels[nodes_table['name'][ind]] = nodes_table['name'][ind]
    
for ind in conn_table.index:
    G.add_edge(conn_table['source'][ind], conn_table['target'][ind])
    
fig = plt.figure(figsize=[48, 48]) # create the canvas for plotting
ax1 = plt.subplot(1,1,1) 


nx.draw_spring(G, node_size=30, node_color=node_cols, label=uniques, ax=ax1)



