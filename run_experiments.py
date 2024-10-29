# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:46:38 2024

@author: alesk
"""

import numpy as np
import pandas as pd
import json

from predict import predict_model, get_model_identifier
from training import NonPriorTraining
import global_values as gv

models = ['baseline', 'ckn']
losses = ['focal']
transformations = ['log2']
targets = ['tissue', 'perturbation', 'both']
other_classes = [False, True]
data = ['./data/vae_cov_transformed_filtered.tsv']
metadata = [None]
split_isoforms = [False, True]


tissues = ['mature_flower', 'mature_leaf', 'mature_root', 'seed', 'young_seedling']
perturbations = ['chemical stress', 'control', 'environmental stress', 'mechanical stress', 'mutant']

with open('results/pred_results.txt', 'w') as convert_file: 
    print('file cleared.')
    
if __name__ == '__main__' :
    
    
    tpm_table = pd.read_table('./data/vae_cov_transformed_filtered.tsv', index_col=0)

    batches = tpm_table['sra_study'].unique()

    num_folds = 6
    num_batches = len(batches)
    k_fold_size = int(num_batches / num_folds)

    np.random.seed(42)
    k_folds = []

    for i in range(num_folds-1) :
        fold = np.random.choice(batches, k_fold_size, replace=False)
        
        k_folds.append(fold)
        
        batches = np.setdiff1d(batches, fold)

    k_folds.append(batches)

    fold_file = open('folds_file.txt', 'w')

    for fold in k_folds:
        fold_file.write('\t'.join(fold) + '\n')

    fold_file.close()
    
    
    
    i = 1
    results = {}
    for model in models :
        for data_path, metadata_path in zip(data, metadata) :
            i += 1
            for loss in losses :
                for transformation in transformations :
                    for target in targets :
                        for split_isoform in split_isoforms :
                            for other_class in other_classes :
                                
                                if model == 'ckn' and not split_isoform:
                                    data_path = './data/vae_cov_filtered_avg.tsv'
                                
                                data_sign = data_path.split('/')[-1].split('.')[0]
                                
                                if target == 'both' and other_class :
                                    continue
                                
                                model_id = get_model_identifier(loss, transformation, target, other_class, split_isoform, special_id=data_sign, model_type=model)
                                
                                results_tissue = {'macro avg' : 0, 'weighted avg' : 0}
                                for tissue in tissues :
                                    results_tissue[tissue] = 0
                                    
                                results_perturbation = {'macro avg' : 0, 'weighted avg' : 0}
                                for perturbation in perturbations :
                                    results_perturbation[perturbation] = 0
                                    
                                    
                                for i in range(num_folds):
                                    
                                    test_batches = k_folds[i]
                                    if i == 0 :
                                        val_batches = k_folds[-1]
                                    else :
                                        val_batches = k_folds[i-1]
                                        
                                    train_batches = np.concatenate(k_folds)
                                    
                                    train_batches = np.setdiff1d(train_batches, np.concatenate([test_batches, val_batches]))
                                    
                                    
                                    print(f'Training {model_id} on {data_path}')
                                    
                                    agent = NonPriorTraining(model_type=model,
                                                             loss=loss,
                                                             transformation=transformation,
                                                             target=target,
                                                             include_other_class=other_class,
                                                             split_isoforms=split_isoform,
                                                             special_id=data_sign)
                                    
                                    agent.train(data_path,
                                                metadata_path = metadata_path,
                                                split_batches = True,
                                                train_batches = train_batches,
                                                val_batches = val_batches,
                                                test_batches = test_batches,
                                                epochs=10)
                                    
                                    agent.save_model(fold=i+1)
                                    
                                    result = predict_model(data_path, 
                                                  metadata_path,
                                                  model_type=model,
                                                  split_batches=True,
                                                  test_batches = test_batches,
                                                  return_scores=True,
                                                  loss=loss,
                                                  transformation=transformation,
                                                  target=target,
                                                  include_other_class=other_class,
                                                  split_isoforms=split_isoform,
                                                  special_id=data_sign,
                                                  fold=i+1)
                                    
                                    if target == 'both' :
                                        tissue_results = result[0]
                                        perturbation_results = result[1]
                                    
                                    if target == 'perturbation' :
                                        tissue_results = []
                                        perturbation_results = result
                                        
                                    if target == 'tissue' :
                                        tissue_results = result
                                        perturbation_results = []
                                    
                                    
                                    for tissue_result in tissue_results :
                                        if tissue_result == 'macro avg' or tissue_result == 'weighted avg' :
                                            results_tissue[tissue_result] += tissue_results[tissue_result]['f1-score']
                                        elif tissue_result != 'accuracy':
                                            tissue_num = int(tissue_result)
                                            results_tissue[tissues[tissue_num]] += tissue_results[tissue_result]['f1-score']
                                            
                                    for perturbation_result in perturbation_results :
                                        if perturbation_result == 'macro avg' or perturbation_result == 'weighted avg':
                                            results_perturbation[perturbation_result] += perturbation_results[perturbation_result]['f1-score']
                                        elif perturbation_result != 'accuracy' :
                                            perturbation_num = int(perturbation_result)
                                            results_perturbation[perturbations[perturbation_num]] += perturbation_results[perturbation_result]['f1-score']
                                
                                for tissue in results_tissue:
                                    results_tissue[tissue] /= num_folds
                                
                                for perturbation in results_perturbation:
                                    results_perturbation[perturbation] /= num_folds
                                    
                                
                                results[model_id] = [results_tissue, results_perturbation]
                                
                            
                                
                            
with open('results/pred_results.txt', 'w') as convert_file: 
    convert_file.write(json.dumps(results))
    convert_file.close()   
        