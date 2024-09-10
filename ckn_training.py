# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:10:31 2024

@author: alesk
"""

# Based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
import random
import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

from load_tpm_data import NonPriorData
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import global_values as gv
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class FCN_CKN(nn.Module):
    def __init__(self, output_size=6, tf_groups_file='data/tf_groups.txt', binding_groups_file='data/binding_groups.txt', columns=[]):
        super().__init__()
        
        tf_file = open(tf_groups_file, 'r')
        
        groups = []
        for line in tf_file :
            genes = line[:-1].split('\t')
            groups.append([columns.index(gene) for gene in genes if gene in columns])
        
        self.tf_groups = groups
        tf_file.close()
        
        bindings_file = open(binding_groups_file, 'r')
        
        groups = []
        for line in bindings_file :
            genes = line[:-1].split('\t')
            groups.append([columns.index(gene) for gene in genes if gene in columns])
        
        self.binding_groups = groups
        bindings_file.close()
        
        
        num_of_tf_groups = 0
        self.tf_fcs = nn.ModuleList()
        for tf_group in self.tf_groups :
            num_of_tf_groups += 1
            self.tf_fcs.append(nn.Linear(len(tf_group), 1))
        
        
        num_of_binding_groups = 0
        self.binding_fcs = nn.ModuleList()
        for binding_group in self.binding_groups :
            num_of_binding_groups += 1
            self.binding_fcs.append(nn.Linear(len(binding_group), 1))
        
        input_size = num_of_tf_groups + num_of_binding_groups
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.do1 = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.do2 = nn.Dropout(p=0.15)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.do3 = nn.Dropout(p=0.15)
        self.fc4 = nn.Linear(128, output_size)
        
        self.leaky = nn.LeakyReLU(0.05)
        
        
    def forward(self, x, vector_other=None, tissues_size=0):
        
        group_stack = []
        for tf_fc, tf_group in zip(self.tf_fcs, self.tf_groups) :
            group_stack.append(tf_fc(x[:, tf_group]))
        
        for binding_fc, binding_group in zip(self.binding_fcs, self.binding_groups) :
            group_stack.append(binding_fc(x[:, binding_group]))
        
        x = torch.cat(group_stack, 1) 
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky(x)
        x = self.do3(x)
        x = self.fc4(x)
        if tissues_size > 0 :
            x = torch.cat([nn.Softmax(dim=1)(x[:, :tissues_size]),
                           nn.Softmax(dim=1)(x[:, tissues_size:])], 1)
            
        return x
    
    
class NonPriorTraining:
            
    def __init__(self, loss='crossentropy', transformation='none', target='tissue', include_other_class=False, gamma=2.0, split_isoforms=False, special_id=''):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        self.special_id = special_id
        self.split_isoforms = split_isoforms
        self.target = target
        self.transformation = transformation
        self.include_other_class = include_other_class
        self.loss = loss
        self.gamma = gamma
        
        if self.target == 'both' :
            self.include_other_class = False
        
        self.isoform_count = []
        
        if self.split_isoforms :
            isoform_file = open(gv.ISOFORM_COUNT_PATH, 'r')
            
            for line in isoform_file:
                num_isoforms = int(line.split('\t')[1])
                self.isoform_count.append(num_isoforms)

        if self.loss == 'crossentropy' or self.loss=='focal' :
            self.criterion = nn.CrossEntropyLoss()
        else :
            self.criterion = nn.BCEWithLogitsLoss()
        
    def save_model(self) :
        torch.save(self.fcn.state_dict(), f'{gv.MODELS_PATH}/model_{self.get_model_identifier()}.pt')
    
    def get_model_identifier(self) :
        return f'{"splitIsoforms" if self.split_isoforms else "nonprior"}{self.special_id}_{self.loss}_{self.transformation}_{self.target}{"_otherVector" if self.include_other_class else "_noOtherVector"}'
    
    
    def preprocess_state(self, state):
        return [torch.tensor(np.array([state]), dtype=torch.float32)]
        
    def evaluate(self, dataloader):
        self.fcn.eval()
        num_val_batches = len(dataloader)
        ce_loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            if self.include_other_class :
                
                if self.target == 'tissue' :
                    tpm_data, class_true = batch['tpm'], batch['tissue'][:, 0, :]
                    vector_other = batch['perturbation'][:, 0, :]
                elif self.target == 'perturbation' :
                    tpm_data, class_true = batch['tpm'], batch['perturbation'][:, 0, :]
                    vector_other = batch['tissue'][:, 0, :]
                    
                tpm_data = tpm_data.to(device=self.device, dtype=torch.float32)
                class_true = class_true.to(device=self.device, dtype=torch.float32)
                vector_other = vector_other.to(device=self.device, dtype=torch.float32)
                
            elif self.target != 'both' :
                tpm_data, class_true = batch['tpm'], batch['classification'][:, 0, :]
                
                tpm_data = tpm_data.to(device=self.device, dtype=torch.float32)
                class_true = class_true.to(device=self.device, dtype=torch.float32)
                
            else :
                tpm_data, class_true_tissue, class_true_perturbation = batch['tpm'], batch['tissue'][:, 0, :],  batch['perturbation'][:, 0, :]
                
                tpm_data = tpm_data.to(device=self.device, dtype=torch.float32)
                class_true_tissue = class_true_tissue.to(device=self.device, dtype=torch.float32)
                class_true_perturbation = class_true_perturbation.to(device=self.device, dtype=torch.float32)
                

            with torch.no_grad():
                if self.include_other_class :
                    class_pred = self.fcn(tpm_data, vector_other)
                else :
                    class_pred = self.fcn(tpm_data)
                if self.target != 'both' :
                    loss = self.criterion(class_pred, class_true)
                    
                    if self.loss == 'focal' :
                        pt = torch.exp(-loss)
                        loss = ((1-pt)**self.gamma * loss).mean() 
                    
                    ce_loss += loss.item()
                else : 
                    class_pred_tissue = class_pred[:, :self.num_of_tissues]
                    class_pred_perturbation = class_pred[:, self.num_of_tissues:]
                    
                    loss_tissue = self.criterion(class_pred_tissue, class_true_tissue)
                    loss_perturbation = self.criterion(class_pred_perturbation, class_true_perturbation)
                    
                    if self.loss == 'focal' :
                        pt = torch.exp(-loss_tissue)
                        loss_tissue = ((1-pt)**self.gamma * loss_tissue).mean() 
                        
                        pt = torch.exp(-loss_perturbation)
                        loss_perturbation = ((1-pt)**self.gamma * loss_perturbation).mean() 
                        
                        
                    loss = loss_tissue + loss_perturbation
                    
                    ce_loss += loss.item()

        self.fcn.train()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return ce_loss
        return ce_loss / num_val_batches

    def train(self, data_path, 
                  metadata_path = None,
                  epochs: int = 100,
                  batch_size: int = 16,
                  learning_rate: float = 0.0005,
                  val_percent: float = 0.2,
                  save_checkpoint: bool = False) :
        # 1. Create dataset
        #dataset = get_set(dir_train)

        """
        if self.include_other_class :
            dataset = NonPriorData(tpm_path=data_path,
                                   metadata_path=metadata_path,
                                   transformation=self.transformation,
                                   target='both')
        else :
            dataset = NonPriorData(tpm_path=data_path,
                                   metadata_path=metadata_path,
                                   transformation=self.transformation,
                                   target=self.target)

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        
        n_test = len(val_set)
        val_set, test_set  = random_split(val_set, [n_test//2, n_test - n_test//2], generator=torch.Generator().manual_seed(0))
        
        # 3. Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        
        """
        
        train_file = open('train_batches.txt', 'r')
        val_file = open('val_batches.txt', 'r')
        test_file = open('test_batches.txt', 'r')
        
        train_batches = []
        val_batches = []
        test_batches = []
        
        for batch in train_file:
            train_batches.append(batch[:-1])
            
        for batch in val_file :
            val_batches.append(batch[:-1])
            
        for batch in test_file :
            test_batches.append(batch[:-1])
            
        train_set = NonPriorData(tpm_path=data_path,
                                 metadata_path=metadata_path,
                                 transformation=self.transformation,
                                  target=self.target,
                                  batches=train_batches)
        val_set = NonPriorData(tpm_path=data_path,
                               metadata_path=metadata_path,
                                 transformation=self.transformation,
                                  target=self.target,
                                  batches=val_batches)
        
        test_set = NonPriorData(tpm_path=data_path,
                               metadata_path=metadata_path,
                                 transformation=self.transformation,
                                  target=self.target,
                                  batches=test_batches)
        
        dataset = train_set
        
        
        n_train = len(train_set)
        n_val = len(val_set)
        
        
        
        loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
        
        if self.include_other_class or self.target == 'both' :
            perturbation_output_size = len(dataset.onehot_perturbation.categories_[0])
            tissue_output_size = len(dataset.onehot_tissue.categories_[0])
        else :
            first_output_size = len(dataset.onehot.categories_[0])
            second_output_size = 0
        
        input_size = dataset.num_of_genes
        self.num_of_tissues = 0

        if self.include_other_class :
            if self.target == 'tissue' :
                self.fcn = FCN_CKN(output_size=tissue_output_size, columns=dataset.columns.tolist())
               
            elif self.target == 'perturbation' :
                self.fcn = FCN_CKN(input_size=input_size,
                               output_size=perturbation_output_size,
                               include_other_class=self.include_other_class,
                               other_vector_size=tissue_output_size,
                               isoforms_count=self.isoform_count)
        else :
            if self.target != 'both' :
                self.fcn = FCN_CKN(output_size=first_output_size, columns=dataset.columns.tolist())
               
            elif self.target == 'both' :
                self.fcn = FCN_CKN(output_size=perturbation_output_size + tissue_output_size, columns=dataset.columns.tolist())
                self.num_of_tissues = tissue_output_size
                
        self.fcn.to(device=self.device)
        
        
        # (Initialize logging)
        #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        #experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        #                              val_percent=val_percent, save_checkpoint=save_checkpoint))
        
        """
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
        ''')
        """
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = torch.optim.Adam(self.fcn.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,9], gamma=0.3)
        
        global_step = 0
        self.f_loss = open(f'{gv.RESULTS_PATH}/loss_{self.get_model_identifier()}.txt', 'w')
        # 5. Begin training
        for epoch in range(1, epochs+1):
            self.fcn.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                k = 0
                for batch in train_loader:
                    tpm = batch['tpm']
                    if self.target != 'both' :
                        if self.include_other_class:
                            class_true = batch[self.target][:, 0, :]
                            if self.target == 'perturbation' :
                                class_true_tissue = batch['tissue'][:, 0, :]
                                class_true_tissue = class_true_tissue.to(device=self.device, dtype=torch.float32)
                            else :
                                class_true_perturbation = batch['perturbation'][:, 0, :]
                                class_true_perturbation = class_true_perturbation.to(device=self.device, dtype=torch.float32)
                            tpm_data = tpm.to(device=self.device, dtype=torch.float32)
                            class_true = class_true.to(device=self.device, dtype=torch.float32)
                        else :
                            class_true = batch['classification'][:, 0, :]
                            tpm_data = tpm.to(device=self.device, dtype=torch.float32)
                            class_true = class_true.to(device=self.device, dtype=torch.float32)
                    else :
                        class_true_tissue = batch['tissue'][:, 0, :]
                        class_true_perturbation = batch['perturbation'][:, 0, :]

                        tpm_data = tpm.to(device=self.device, dtype=torch.float32)
                        class_true_tissue = class_true_tissue.to(device=self.device, dtype=torch.float32)
                        class_true_perturbation = class_true_perturbation.to(device=self.device, dtype=torch.float32)
                        
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.utils.clip_grad_norm_(self.fcn.parameters(), 1.0)
                    
                    #with torch.cuda.amp.autocast(enabled=True):
                    if self.include_other_class :
                        if self.target == 'tissue' :
                            class_preds = self.fcn(tpm_data, class_true_perturbation)
                        elif self.target == 'perturbation' :
                            class_preds = self.fcn(tpm_data, class_true_tissue)
                    else:
                        if self.target == 'both' :
                            class_preds = self.fcn(tpm_data, tissues_size=tissue_output_size)
                        else :
                            class_preds = self.fcn(tpm_data)
                        
                        
                    if self.target != 'both' :
                        loss = self.criterion(class_preds, class_true)
                        if self.loss == 'focal' :
                            pt = torch.exp(-loss)
                            loss = ((1-pt)**self.gamma * loss).mean() 
                    else : 
                        class_pred_tissue = class_preds[:, :self.num_of_tissues]
                        class_pred_perturbation = class_preds[:, self.num_of_tissues:]
                        
                        loss_tissue = self.criterion(class_pred_tissue, class_true_tissue)
                        loss_perturbation = self.criterion(class_pred_perturbation, class_true_perturbation)
                        
                        if self.loss == 'focal' :
                            pt = torch.exp(-loss_tissue)
                            loss_tissue = ((1-pt)**self.gamma * loss_tissue).mean() 
                            
                            pt = torch.exp(-loss_perturbation)
                            loss_perturbation = ((1-pt)**self.gamma * loss_perturbation).mean() 
                            
                        loss = loss_tissue + loss_perturbation
                    
                            
                    #loss /= float(sep_preds.size(2))
                    
                    loss.backward()
                    optimizer.step() 
                    
                    #grad_scaler.scale(loss).backward()
                    
                    #grad_scaler.step(optimizer)
                    #grad_scaler.update()

                    global_step += 1

                    epoch_loss += loss.item()
                    #experiment.log({
                    #    'train loss': loss.item(),
                    #    'step': global_step,
                    #    'epoch': epoch
                    #})
                    k += 1
                    #if k % 100 == 0 :
                    pbar.update(tpm.shape[0])
                    pbar.set_postfix(**{'loss (batch)': epoch_loss / k})   
                        
                    
            # Evaluation round
        
            scheduler.step()
            val_score = self.evaluate(val_loader)
            print('Validation score: ' + str(val_score))
            test_score = self.evaluate(test_loader)
            print('Test score: ' + str(test_score))
            self.f_loss.write(str(val_score))
            self.f_loss.write('\n')
            
            self.predict_model(test_set, dataset, self.loss, self.transformation, self.target, self.include_other_class, self.split_isoforms)


            if save_checkpoint:
                Path(gv.CHECKPOINTS_PATH).mkdir(parents=True, exist_ok=True)
                torch.save(self.fcn.state_dict(), f'{gv.CHECKPOINTS_PATH}/checkpoint_{self.get_model_identifier()}_{epoch}.pth')
                #logging.info(f'Checkpoint {epoch} saved!')
        self.f_loss.close()
        
    def predict_model(self, dataset, dataset_oh, loss='crossentropy', transformation='log2', target='both', include_other_class=False, split_isoforms=False, special_id='') :
            
            isoforms_file = open(gv.ISOFORM_COUNT_PATH, 'r')
            isoform_count = []
            
            if split_isoforms :
                for line in isoforms_file:
                    num_isoforms = int(line.split('\t')[1])
                    isoform_count.append(num_isoforms)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            """
            
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
            """
            
              
            if include_other_class or target == 'both' :
                perturbation_output_size = len(dataset_oh.onehot_perturbation.categories_[0])
                tissue_output_size = len(dataset_oh.onehot_tissue.categories_[0])
                
                num_of_tissues = 0
                num_of_tissues = tissue_output_size
            else :
                first_output_size = len(dataset_oh.onehot.categories_[0])
                second_output_size = 0
                num_of_tissues = 0
                
                num_of_tissues = first_output_size
            #dataset = test_set
        
            self.fcn.eval()
            
            
            loader_args = dict(batch_size=2, num_workers=1, pin_memory=True)
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
                            class_pred = self.fcn(tpm_data, class_true_perturbation)
                        elif target == 'perturbation' :
                            class_pred = self.fcn(tpm_data, class_true_tissue)
                    else:
                        if target == 'both' :
                            class_pred = self.fcn(tpm_data, tissues_size=num_of_tissues)
                        else :
                            class_pred = self.fcn(tpm_data)
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
                print(class_report)
                print(tabulate(cf_matrix))
                if include_other_class :
                    if target == 'perturbation' :
                        print(dataset_oh.onehot_perturbation.categories_)
                    else :
                        print(dataset_oh.onehot_tissue.categories_)
                else :
                    print(dataset_oh.onehot.categories_)

                
            else :
                cf_matrix_tissue = confusion_matrix(y_true_tissue, y_pred_tissue)
                cf_matrix_perturbation = confusion_matrix(y_true_perturbation, y_pred_perturbation)
                
                class_report_tissue = metrics.classification_report(y_true_tissue, y_pred_tissue)
                class_report_perturbation = metrics.classification_report(y_true_perturbation, y_pred_perturbation)
                
                print(class_report_tissue)
                print(tabulate(cf_matrix_tissue))
                print(dataset_oh.onehot_tissue.categories_)
                
                print(class_report_perturbation)
                print(tabulate(cf_matrix_perturbation))
                print(dataset_oh.onehot_perturbation.categories_)
            
            #print(dataset.onehot.transform(['mature_flower', 'mature_leaf', 'mature_root', 'mature_seed',
            #       'mature_seedling', 'seed_seed', 'senescence_senescence_green',
            #       'senescence_senescence_reproductive', 'young_seed',
            #       'young_seedling']))
            
from torchsummary import summary        
        
if __name__ == '__main__':
    agent = NonPriorTraining(loss='focal', transformation='log2', target='perturbation', include_other_class=False, split_isoforms=False)
    agent.train('data/limma_data.tsv')
    agent.save_model()
    
    #fcn = FCN(output_size=5)
    #summary(fcn, input_size=(26234,), batch_size=2)