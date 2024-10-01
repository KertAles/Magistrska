# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:40:02 2023

@author: alesk
"""


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from training import FCN, FCN_CKN
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

import global_values as gv
from load_tpm_data import NonPriorData
import numpy as np

from tabulate import tabulate

def get_model_identifier(loss, transformation, target, otherClass, split_isoforms, special_id='', model_type='baseline') :
    return f'{model_type}_{"splitIsoforms_" if split_isoforms else "_"}{special_id}_{loss}_{transformation}_{target}{"_otherVector" if otherClass else "_noOtherVector"}'

def predict_model(data_path,
                  metadata_path=None,
                  split_batches=True,
                  test_batches = [],
                  return_scores = False,
                  model_type='baseline',
                  loss='crossentropy',
                  transformation='log2',
                  target='both',
                  include_other_class=False,
                  split_isoforms=False,
                  special_id='') :
    
    isoforms_file = open(gv.ISOFORM_COUNT_PATH, 'r')
    isoform_count = []
    
    if split_isoforms :
        for line in isoforms_file:
            num_isoforms = int(line.split('\t')[1])
            isoform_count.append(num_isoforms)

    model_id = get_model_identifier(loss, transformation, target, include_other_class, split_isoforms, special_id, model_type=model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if split_batches :
        if len(test_batches) == 0 :
            test_file = open('test_batches.txt', 'r')
            test_batches = []
            
            for batch in test_file:
                test_batches.append(batch[:-1])
            
        dataset = NonPriorData(tpm_path=data_path,
                               metadata_path=metadata_path,
                               transformation=transformation,
                               target='both',
                               batches=test_batches)  
    else :
        if include_other_class :
            dataset = NonPriorData(tpm_path=data_path,
                                   metadata_path=metadata_path,
                                   transformation=transformation,
                                   target='both')
        else :
            dataset = NonPriorData(tpm_path=data_path,
                                   metadata_path=metadata_path,
                                   transformation=transformation,
                                   target=target)
        
        val_percent = 0.2
        
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        
        n_test = len(val_set)
        val_set, test_set  = random_split(val_set, [n_test//2, n_test - n_test//2], generator=torch.Generator().manual_seed(0))
        
    input_size = dataset.num_of_genes
    
    if include_other_class or target == 'both' :
        perturbation_output_size = len(dataset.onehot_perturbation.categories_[0])
        tissue_output_size = len(dataset.onehot_tissue.categories_[0])
        
        num_of_tissues = 0
        
        model = FCN_CKN(output_size=perturbation_output_size + tissue_output_size, columns=dataset.columns.tolist())
        num_of_tissues = tissue_output_size
    else :
        first_output_size = len(dataset.onehot.categories_[0])
        second_output_size = 0
        num_of_tissues = 0
        
        input_size = first_output_size
        num_of_tissues = first_output_size

    
    if model_type == 'baseline' :
        if include_other_class :
            if target == 'tissue' :
                model = FCN(input_size=input_size,
                            output_size=tissue_output_size,
                               include_other_class=include_other_class,
                               other_vector_size=perturbation_output_size,
                               isoforms_count=isoform_count)
            elif target == 'perturbation' :
                model = FCN(input_size=input_size,
                            output_size=perturbation_output_size,
                               include_other_class=include_other_class,
                               other_vector_size=tissue_output_size,
                               isoforms_count=isoform_count)
        else :
            if target != 'both' :
                model = FCN(input_size=input_size,
                            output_size=first_output_size,
                               include_other_class=include_other_class,
                               other_vector_size=second_output_size,
                               isoforms_count=isoform_count)
            elif target == 'both' :
                model = FCN(input_size=input_size,
                            output_size=perturbation_output_size + tissue_output_size,
                               include_other_class=False,
                               isoforms_count=isoform_count)
                num_of_tissues = tissue_output_size
    elif model_type == 'ckn' :
        if include_other_class :
            if target == 'tissue' :
                model = FCN_CKN(output_size=tissue_output_size, columns=dataset.columns.tolist(), dropout=0.2)
               
            elif target == 'perturbation' :
                model = FCN_CKN(input_size=input_size,
                               output_size=perturbation_output_size,
                               include_other_class=include_other_class,
                               other_vector_size=tissue_output_size,
                               isoforms_count=isoform_count)
        else :
            if target != 'both' :
                model = FCN_CKN(output_size=first_output_size, columns=dataset.columns.tolist(), dropout=0.2)
               
            elif target == 'both' :
                model = FCN_CKN(output_size=perturbation_output_size + tissue_output_size, columns=dataset.columns.tolist(), dropout=0.2)
                num_of_tissues = tissue_output_size

    model.load_state_dict(torch.load(f'{gv.MODELS_PATH}/model_{model_id}.pt'))
    model.to(device=device)
    model.eval()
    
    
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    
    num_test_batches = len(test_loader)
    
    y_pred = []
    y_true = []
    
    y_pred_tissue = []
    y_true_tissue = []
    
    y_pred_perturbation = []
    y_true_perturbation = []

    # iterate over the validation set
    for batch in tqdm(test_loader, total=num_test_batches, desc='Test round', unit='batch', leave=False):
        tpm_data = batch['tpm']
        if target != 'both' :
            if include_other_class:
                class_true = batch[target][:, 0, :]
                if target == 'perturbation' :
                    class_true_tissue = batch['tissue'][:, 0, :]
                    class_true_tissue = class_true_tissue.to(device=device, dtype=torch.float32)
                else :
                    class_true_perturbation = batch['perturbation'][:, 0, :]
                    class_true_perturbation = class_true_perturbation.to(device=device, dtype=torch.float32)
                tpm_data = tpm_data.to(device=device, dtype=torch.float32)
                class_true = class_true.to(device=device, dtype=torch.float32)
            else :
                class_true = batch['classification'][:, 0, :]
                tpm_data = tpm_data.to(device=device, dtype=torch.float32)
                class_true = class_true.to(device=device, dtype=torch.float32)
        else :
            tpm_data, class_true_tissue, class_true_perturbation = batch['tpm'], batch['tissue'], batch['perturbation']

            tpm_data = tpm_data.to(device=device, dtype=torch.float32)
            class_true_tissue = class_true_tissue.to(device=device, dtype=torch.float32)
            class_true_perturbation = class_true_perturbation.to(device=device, dtype=torch.float32)
            
            
        with torch.no_grad():
            if include_other_class :
                if target == 'tissue' :
                    class_pred = model(tpm_data, class_true_perturbation)
                elif target == 'perturbation' :
                    class_pred = model(tpm_data, class_true_tissue)
            else:
                class_pred = model(tpm_data)
            if target != 'both' :
                output = (torch.max(torch.exp(class_pred), 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction
                
                labels = (torch.max(torch.exp(class_true), 1)[1]).data.cpu().numpy()
                y_true.extend(labels) # Save Truth
            else :
                output_tissue = (torch.max(torch.exp(class_pred[:, :num_of_tissues]), 1)[1]).data.cpu().numpy()
                y_pred_tissue.extend(output_tissue) # Save Prediction
                
                labels_tissue = (torch.max(torch.exp(class_true_tissue[:, 0, :]), 1)[1]).data.cpu().numpy()
                y_true_tissue.extend(labels_tissue) # Save Truth
                
                output_perturbation = (torch.max(torch.exp(class_pred[:, num_of_tissues:]), 1)[1]).data.cpu().numpy()
                y_pred_perturbation.extend(output_perturbation) # Save Prediction
                
                labels_perturbation = (torch.max(torch.exp(class_true_perturbation[:, 0, :]), 1)[1]).data.cpu().numpy()
                y_true_perturbation.extend(labels_perturbation) # Save Truth
                
    
    if target != 'both' :
        cf_matrix = confusion_matrix(y_true, y_pred)
        class_report = metrics.classification_report(y_true, y_pred)
        
        if return_scores :
            class_report_dict = metrics.classification_report(y_true, y_pred, output_dict=True)
        
        classification_report = open(f'{gv.RESULTS_PATH}/report_{model_id}.txt', 'w')
        classification_report.write(class_report)
        classification_report.write('\n')
        classification_report.write(tabulate(cf_matrix))
        classification_report.write('\n')
        if include_other_class :
            if target == 'perturbation' :
                classification_report.write('\t'.join(category for category in dataset.onehot_perturbation.categories_[0]))
                classification_report.write('\n')
                classification_report.close()
                print(dataset.onehot_perturbation.categories_)
            else :
                classification_report.write('\t'.join(category for category in dataset.onehot_tissue.categories_[0]))
                classification_report.write('\n')
                classification_report.close()
                print(dataset.onehot_tissue.categories_)
        else :
            classification_report.write('\t'.join(category for category in dataset.onehot.categories_[0]))
            classification_report.write('\n')
            classification_report.close()
            print(dataset.onehot.categories_)

        
    else :
        cf_matrix_tissue = confusion_matrix(y_true_tissue, y_pred_tissue)
        cf_matrix_perturbation = confusion_matrix(y_true_perturbation, y_pred_perturbation)
        
        class_report_tissue = metrics.classification_report(y_true_tissue, y_pred_tissue)
        class_report_perturbation = metrics.classification_report(y_true_perturbation, y_pred_perturbation)
        
        if return_scores :
            class_report_tissue_dict = metrics.classification_report(y_true_tissue, y_pred_tissue, output_dict=True)
            class_report_perturbation_dict = metrics.classification_report(y_true_perturbation, y_pred_perturbation, output_dict=True)
            
            class_report_dict = [class_report_tissue_dict, class_report_perturbation_dict]
            
        classification_report = open(f'{gv.RESULTS_PATH}/report_{model_id}.txt', 'w')
        classification_report.write(class_report_tissue)
        classification_report.write('\n')
        classification_report.write(tabulate(cf_matrix_tissue))
        classification_report.write('\n')
        classification_report.write('\t'.join(category for category in dataset.onehot_tissue.categories_[0]))
        classification_report.write('\n')
        classification_report.write(class_report_perturbation)
        classification_report.write('\n')
        classification_report.write(tabulate(cf_matrix_perturbation))
        classification_report.write('\n')
        classification_report.write('\t'.join(category for category in dataset.onehot_perturbation.categories_[0]))
        classification_report.write('\n')
        classification_report.close()
        
        print(dataset.onehot_tissue.categories_)
        print(dataset.onehot_perturbation.categories_)
    
    #print(dataset.onehot.transform(['mature_flower', 'mature_leaf', 'mature_root', 'mature_seed',
    #       'mature_seedling', 'seed_seed', 'senescence_senescence_green',
    #       'senescence_senescence_reproductive', 'young_seed',
    #       'young_seedling']))
    if return_scores :
        return class_report_dict
    else :
        return None

if __name__ == '__main__':
    loss='focal'
    transformation = 'log2'
    target = 'secondary_perturbation'
    include_other_class=False
    split_isoforms=False
    
    predict_model(gv.EXTENDED_GROUPED_DATA, loss=loss, transformation=transformation, target=target, include_other_class=include_other_class)
