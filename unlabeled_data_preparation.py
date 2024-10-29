# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:40:28 2024

@author: alesk
"""
import pandas as pd
import numpy as np

import global_values as gv

def process_unlabeled_gene_expression() :
    qc_data = pd.read_table('./data/athaliana_qc.tsv')
    qc_data = qc_data[qc_data['category'] == 'QcPassRate']
    qc_data['value'] = qc_data['value'].apply(lambda x: float(x[:-1]))
    
    qc_dict = {}
    
    for index, row in qc_data.iterrows() : 
        qc_dict[row['srr']] = row['value']
    
    ckn_file = open('./data/CKN_gene_ranks.txt', 'r')
    genes_list = ['SRR_accession']
    gene_positions = {}
    i = 0
    for line in ckn_file :
        gene_id = line.split('\t')[0]
        #rank = int(line.split('\t')[1])
        
        if gene_id[:2] == 'AT' and '|' not in gene_id:
            gene_positions[gene_id] = i
            genes_list.append(gene_id)
            i += 1
            
    vect_len = len(gene_positions)
    
    
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
        

def process_metadata(limit_models=False, limit_small_batches=False):
    
    allowed_tissues = ['young_seedling', 'seed',
                       'mature_leaf', 'mature_root', 'mature_flower']
    
    allowed_strats = ['RNA-Seq']
    allowed_selecs = ['cDNA', 'RANDOM', 'PolyA', 'Oligo-dT']
    
    #tpm_table = pd.read_table('./data/athaliana_trimmed_2.tsv')
    metadata_table = pd.read_table('./data/athaliana_metadata.tsv')
    metadata_table_2 = pd.read_table(gv.METADATA_PATH)
    
    #tpm_table.drop_duplicates('SRR_accession', inplace=True)
    metadata_table.drop_duplicates('SRR_accession', inplace=True)
    
    metadata_table.set_index('SRR_accession', inplace=True)
    metadata_table_2.set_index('SRR_accession', inplace=True)
    
    
    metadata_table = metadata_table[['study_accession', 'experiment_library_strategy', 'experiment_library_selection', 'experiment_instrument_model']]
    metadata_table = metadata_table.rename(columns={"study_accession": "sra_study"})
    
    
    metadata_table_2 = metadata_table_2[metadata_table_2['perturbation_group'] != 'unknown']
    
    metadata_table_2['tissue_super'] = metadata_table_2['tissue_super'].apply(lambda x: 'senescence' if 'senescence' in x else x)
    metadata_table_2['tissue_super'] = metadata_table_2['tissue_super'].apply(lambda x: 'seed' if 'seed' in x and 'seedling' not in x else x)
        
    metadata_table_2 = metadata_table_2[metadata_table_2['tissue_super'].isin(allowed_tissues)]
    metadata_table_2['perturbation_group'] = metadata_table_2['perturbation_group'].apply(lambda x: 'control' if x == 'unstressed' else x)
    
    
    metadata_table_2 = metadata_table_2[['perturbation_group', 'tissue_super']]
    #tpm_table.set_index('SRR_accession', inplace=True)
    #tpm_table = metadata_table.join(tpm_table)
    metadata_table = metadata_table.join(metadata_table_2)
    
    metadata_table = metadata_table[metadata_table['experiment_library_strategy'].isin(allowed_strats)]
    metadata_table = metadata_table[metadata_table['experiment_library_selection'].isin(allowed_selecs)]
    
    if limit_models :
        model_count = metadata_table['experiment_instrument_model'].value_counts()
        chosen_models = []
        
        
        for count, model in zip(model_count, model_count.index)  :
            if count > 100 :
                chosen_models.append(model)
        
        metadata_table = metadata_table[metadata_table['experiment_instrument_model'].isin(chosen_models)]
        
    if limit_small_batches :
        batch_count = metadata_table['sra_study'].value_counts()
        chosen_batches = []
        
        
        for count, batch in zip(batch_count, batch_count.index)  :
            if count > 10 :
                chosen_batches.append(batch)
        
        metadata_table = metadata_table[metadata_table['sra_study'].isin(chosen_batches)]
    
    metadata_table.to_csv('./data/metadata_T.tsv', sep="\t")



if __name__ == '__main__':
    process_unlabeled_gene_expression()
    process_metadata()