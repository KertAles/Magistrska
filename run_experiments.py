# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:46:38 2024

@author: alesk
"""

from predict import predict_model, get_model_identifier
from training import NonPriorTraining
import global_values as gv
import numpy as np
import pandas as pd
import json
tpm_table = pd.read_table(gv.GROUPED_DATA, index_col=0)

batches = tpm_table['sra_study'].unique()

num_folds = 6
num_batches = len(batches)
k_fold_size = int(num_batches / num_folds)

np.random.seed(42)
k_folds = []

for i in range(num_folds) :
    fold = np.random.choice(batches, k_fold_size, replace=False)
    
    k_folds.append(fold)
    
    batches = np.setdiff1d(batches, fold)

k_folds.append(batches)

fold_file = open('folds_file.txt', 'w')

for fold in k_folds:
    fold_file.write('\t'.join(fold) + '\n')

fold_file.close()

losses = ['focal']
transformations = ['log2']
targets = ['perturbation', 'both', 'tissue']
other_classes = [False, True]
#data = ['./data/harmony_joined.tsv']
#data = [ './data/limma_data.tsv', './data/averaged_data.tsv', './data/vae_transformed5.tsv']
#metadata = [None, None, './data/metadata_proc.tsv']
data = ['./data/vae_cov_transformed.tsv']
metadata = ['./data/metadata_T.tsv']
split_isoforms = [False, True]
#losses = ['bceLogits']
#transformations = ['log10']
#targets = ['both']

#split_isoforms = [False]
#data_path = JOINED_DATA_PATH_GROUPED

tissues = ['mature_flower', 'mature_leaf', 'mature_root', 'seed', 'young_seedling']
perturbations = ['chemical stress', 'control', 'environmental stress', 'mechanical stress', 'mutant']

if __name__ == '__main__' :
    i = 1
    results = {}
    for data_path, metadata_path in zip(data, metadata) :
        data_sign = data_path.split('/')[-1].split('.')[0]
        i += 1
        for loss in losses :
            for transformation in transformations :
                for target in targets :
                    for split_isoform in split_isoforms :
                        for other_class in other_classes :
                            
                            if target == 'both' and other_class :
                                continue
                            
                            model_id = get_model_identifier(loss, transformation, target, other_class, split_isoform, special_id=data_sign, model_type='baseline')
                            
                            results_tissue = {'avg' : 0}
                            for tissue in tissues :
                                results_tissue[tissue] = 0
                                
                            results_perturbation = {'avg' : 0}
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
                                
                                agent = NonPriorTraining(model_type='baseline',
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
                                              model_type='baseline',
                                              split_batches=True,
                                              test_batches = test_batches,
                                              return_scores=True,
                                              loss=loss,
                                              transformation=transformation,
                                              target=target,
                                              include_other_class=other_class,
                                              split_isoforms=split_isoform,
                                              special_id=data_sign)
            
                                for tissue_result in result[0] :
                                    if tissue_result == 'macro avg':
                                        results_tissue['avg'] += result[0][tissue_result]['f1-score']
                                    elif tissue_result != 'accuracy' and tissue_result != 'weighted avg' :
                                        tissue_num = int(tissue_result)
                                        results_tissue[tissues[tissue_num]] += result[0][tissue_result]['f1-score']
                                        
                                for perturbation_result in result[1] :
                                    if perturbation_result == 'macro avg':
                                        results_perturbation['avg'] += result[1][perturbation_result]['f1-score']
                                    elif perturbation_result != 'accuracy' and perturbation_result != 'weighted avg' :
                                        perturbation_num = int(perturbation_result)
                                        results_perturbation[perturbations[perturbation_num]] += result[1][perturbation_result]['f1-score']
                            
                            for tissue in results_tissue:
                                results_tissue[tissue] /= num_folds
                            
                            for perturbation in results_perturbation:
                                results_perturbation[perturbation] /= num_folds
                                
                            results[model_id] = [results_tissue, results_perturbation]
                            
                            print(results)
                            
                            
    with open('results/pred_results.txt', 'w') as convert_file: 
        convert_file.write(json.dumps(results))