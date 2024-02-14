# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:40:02 2023

@author: alesk
"""


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from training import FCN
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from global_values import MODELS_PATH, JOINED_DATA_PATH, JOINED_DATA_PATH_GROUPED, RESULTS_PATH, ISOFORM_COUNT_PATH
from load_tpm_data import NonPriorData
import numpy as np

from tabulate import tabulate

def get_model_identifier(loss, transformation, target, otherClass, split_isoforms) :
    return f'{"splitIsoforms" if split_isoforms else "nonprior"}_{loss}_{transformation}_{target}{"_otherVector" if otherClass else "_noOtherVector"}'

def predict_model(data_path, loss='crossentropy', transformation='log2', target='both', include_other_class=False, split_isoforms=False) :
    
    isoforms_file = open(ISOFORM_COUNT_PATH, 'r')
    isoforms_count = []
    
    if split_isoforms :
        for line in isoforms_file:
            num_isoforms = int(line.split('\t')[1])
            isoforms_count.append(num_isoforms)
    
    if target == 'tissue' :
        model = FCN(output_size=6, include_other_class=include_other_class, other_vector_size=5, isoforms_count=isoforms_count)
    elif target == 'perturbation' :
        model = FCN(output_size=5, include_other_class=include_other_class, other_vector_size=6, isoforms_count=isoforms_count)
    elif target == 'both' :
        model = FCN(output_size=11, include_other_class=False, isoforms_count=isoforms_count)

    model_id = get_model_identifier(loss, transformation, target, include_other_class, split_isoforms)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f'{MODELS_PATH}/model_{model_id}.pt'))
    model.to(device=device)
    model.eval()
    
    if include_other_class :
        dataset = NonPriorData(tpm_path=data_path,
                               transformation=transformation,
                               target='both')
    else :
        dataset = NonPriorData(tpm_path=data_path,
                               transformation=transformation,
                               target=target)
    
    val_percent = 0.2
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    n_test = len(val_set)
    val_set, test_set  = random_split(val_set, [n_test//2, n_test - n_test//2], generator=torch.Generator().manual_seed(0))
    
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    
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
                output_tissue = (torch.max(torch.exp(class_pred[:, :6]), 1)[1]).data.cpu().numpy()
                y_pred_tissue.extend(output_tissue) # Save Prediction
                
                labels_tissue = (torch.max(torch.exp(class_true_tissue[:, 0, :]), 1)[1]).data.cpu().numpy()
                y_true_tissue.extend(labels_tissue) # Save Truth
                
                output_perturbation = (torch.max(torch.exp(class_pred[:, 6:]), 1)[1]).data.cpu().numpy()
                y_pred_perturbation.extend(output_perturbation) # Save Prediction
                
                labels_perturbation = (torch.max(torch.exp(class_true_perturbation[:, 0, :]), 1)[1]).data.cpu().numpy()
                y_true_perturbation.extend(labels_perturbation) # Save Truth
                
    
    if target != 'both' :
        cf_matrix = confusion_matrix(y_true, y_pred)
        class_report = metrics.classification_report(y_true, y_pred)
        
        classification_report = open(f'{RESULTS_PATH}/report_{model_id}.txt', 'w')
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
        
        classification_report = open(f'{RESULTS_PATH}/report_{model_id}.txt', 'w')
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
    

if __name__ == '__main__':
    loss='crossentropy'
    transformation = 'log2'
    target = 'perturbation'
    include_other_class=True
    
    predict_model(JOINED_DATA_PATH_GROUPED, loss=loss, transformation=transformation, target=target, include_other_class=include_other_class)
