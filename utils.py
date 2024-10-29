# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:52:28 2024

@author: alesk
"""
import pandas as pd
from decimal import Decimal
import global_values as gv

def get_model_identifier(loss, transformation, target, otherClass, split_isoforms, special_id='', model_type='baseline') :
    return f'{model_type}_{"splitIsoforms_" if split_isoforms else "_"}{special_id}_{loss}_{transformation}_{target}{"_otherVector" if otherClass else "_noOtherVector"}'


def generate_isoform_file(table_path=gv.GROUPED_DATA) :
    tpm_table = pd.read_table(table_path, index_col=0)

    tpm_table.drop("tissue_super", axis=1, inplace=True)
    tpm_table.drop("perturbation_group", axis=1, inplace=True)
    f = open("data/isoform_count.txt", "a")

    checked_genes = []
    for name_isoform, values in tpm_table.items():
       name_gene = name_isoform.split('.')[0]
       if name_gene not in checked_genes :
           checked_genes.append(name_gene)
           isoforms = tpm_table.filter(regex=name_gene)
           
           idx_dic = {}
           for col in isoforms.columns:
               idx_dic[col] = tpm_table.columns.get_loc(col)
               
           #print(idx_dic)
           i = -1
           j = 0
           for isoform in idx_dic :
               j += 1
               if i == -1 :
                   i = idx_dic[isoform]
               else :
                   diff = idx_dic[isoform] - i
                   
                   if diff != 1 :
                       print('Difference between two genes is too large!')
                       print(diff)
                       print('')
                       
                   i = idx_dic[isoform]
           f.write(name_gene + '\t' + str(j) + '\n')
           
    f.close()
    
def average_isoforms_in_data(table_path='./data/vae_cov_transformed_filtered.tsv', out_path='./data/vae_cov_filtered_avg.tsv', has_meta=True) :
    tpm_table = pd.read_table(table_path, sep='\t', index_col=0)
    
    if has_meta :
        metadata_table = tpm_table[['perturbation_group', 'tissue_super', 'sra_study']]
        tpm_table = tpm_table.drop(['perturbation_group', 'tissue_super', 'sra_study'], axis=1)
    else :
        metadata_table = pd.read_table(gv.GROUPED_DATA, sep='\t', index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]

    keep_columns = []
    rename_cols = {}

    for col in tpm_table.columns :
        gene = col.split('.')[0]
        
        if gene not in keep_columns :
            tpm_table[col] = tpm_table[[column for column in tpm_table.columns if gene in column]].mean(axis=1)
            
            keep_columns.append(gene)
            rename_cols[col] = gene
     
    tpm_table = tpm_table.rename(columns=rename_cols)
    tpm_table = tpm_table[keep_columns]


    tpm_table = metadata_table.join(tpm_table, how='inner')


    tpm_table.to_csv(out_path, sep="\t")
    
    
def dee2_metadata_counts() :
    metadata_table = pd.read_table('./data/athaliana_metadata.tsv')
    
    exp_strat = metadata_table['experiment_library_strategy'].value_counts()
    exp_selec = metadata_table['experiment_library_selection'].value_counts()
    exp_model = metadata_table['experiment_instrument_model'].value_counts()
    
    return exp_strat, exp_selec, exp_model


def enrichr_table_to_latex(enrichr_base, enrichr_ckn, p_value=0.125) :
    enrichr_base = enrichr_base[enrichr_base['P-value'] < p_value]
    enrichr_ckn = enrichr_ckn[enrichr_ckn['P-value'] < p_value]

    enrichr_ckn = enrichr_ckn.sort_values(by='P-value')
    enrichr_base = enrichr_base.sort_values(by='P-value')

    base_table = ''
    for index, row in enrichr_base.iterrows():
        pvalue = '%.2E' % Decimal(row["P-value"])
        base_table += f'{row["Term"]} & {row["Overlap"]} & {pvalue} & {int(row["Combined Score"])} \\\\ \\hline \n'
        
        
    ckn_table = ''
    for index, row in enrichr_ckn.iterrows():
        pvalue = '%.2E' % Decimal(row["P-value"])
        ckn_table += f'{row["Term"]} & {row["Overlap"]} & {pvalue} & {int(row["Combined Score"])} \\\\ \\hline \n'
        
    return base_table, ckn_table