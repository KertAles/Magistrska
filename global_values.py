# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:28:56 2023

@author: alesk
"""

# Folder paths
MODELS_PATH = 'models'
CHECKPOINTS_PATH = 'models/checkpoints'
RESULTS_PATH ='results'

# Original datasets
NODES_PSS_ANNOT_PATH = 'data/PSS_nodes.tsv'
INTERACTIONS_PSS_ANNOT_PATH = 'data/PSS_conn.tsv'
NODES_CKN_ANNOT_PATH = 'data/CKN_nodes.tsv'
INTERACTIONS_CKN_ANNOT_PATH = 'data/CKN_conn.tsv'
TPM_PATH = 'data/TPM_table.tsv'
METADATA_PATH = 'data/short_metadata_table_16_11_22-added_age_group_supergroups.txt'

METADATA_PROC_PATH = 'data/metadata_proc.tsv'


# Gene "metadata"
ISOFORM_COUNT_PATH = 'data/isoform_count.txt'
GENES_IN_CKN_PATH = 'data/genes_CKN.txt'
GENES_NOT_IN_CKN_PATH = 'data/genes_notin_CKN.txt'
CKN_GENE_RANKS = 'data/CKN_gene_ranks.txt'
CKN_GENE_TISSUE = 'data/CKN_gene_tissues.txt'


### Processed datasets

# TPM joined with metadata - base
JOINED_DATA = 'data/joined_tpm.tsv'

# TPM with grouped tissues
GROUPED_DATA = 'data/grouped_tpm.tsv'

# TPM that uses secondary perturbation descriptions
EXTENDED_DATA = 'data/extended_tpm.tsv'
EXTENDED_GROUPED_DATA = 'data/extended_grouped_tpm.tsv'

# TPM with proportional gene expression w.r.t. control samples
PROPORTIONAL_DATA_CONTROLS = 'data/proportional_tpm_w_controls.tsv'
PROPORTIONAL_DATA_NO_CONTROLS = 'data/proportional_tpm_wo_controls.tsv'

VAE_GROUPED_DATA = 'data/vae_joined.tsv'



