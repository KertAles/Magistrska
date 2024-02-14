# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:07:44 2023

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

from global_values import MODELS_PATH, CHECKPOINTS_PATH, JOINED_DATA_PATH_GROUPED, RESULTS_PATH, ISOFORM_COUNT_PATH



class FCN(nn.Module):
    def __init__(self, output_size=6, include_other_class=False, other_vector_size=0, isoforms_count=[]):
        super().__init__()
        
        input_size = 48359
        
        self.isoforms_count = isoforms_count
        if len(self.isoforms_count) != 0 :
            num_of_isoforms = 0
            self.isoforms_fcs = nn.ModuleList()
            for isoform in isoforms_count :
                num_of_isoforms += 1
                self.isoforms_fcs.append(nn.Linear(isoform, 1))
            input_size = num_of_isoforms
            
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        if include_other_class :
            self.fc4 = nn.Linear(128 + other_vector_size, output_size)
        else :
            self.fc4 = nn.Linear(128, output_size)
        self.include_other_class = include_other_class
        
        self.leaky = nn.LeakyReLU(0.05)
        
        
    def forward(self, x, vector_other=None):
        
        if len(self.isoforms_count) != 0 :
            grouped_isoforms = []
            i = 0
            for fc, isoform_count in zip(self.isoforms_fcs, self.isoforms_count):
                grouped_isoforms.append(fc(x[:, i:i+isoform_count]))
                i += isoform_count
            x = torch.cat(grouped_isoforms, 1)
            
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky(x)
        #x = nn.Dropout(p=0.3)(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky(x)
        if self.include_other_class :
            x = torch.cat((x, vector_other), 1)
        x = self.fc4(x)
        #x = nn.Softmax(dim=1)(x)
        
        return x
    
    
class NonPriorTraining:
            
    def __init__(self, loss='crossentropy', transformation='none', target='tissue', include_other_class=False, gamma=2.0, split_isoforms=False):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        
        self.split_isoforms = split_isoforms
        self.target = target
        self.transformation = transformation
        self.include_other_class = include_other_class
        self.loss = loss
        self.gamma = gamma
        
        self.isoform_count = []
        
        if self.split_isoforms :
            isoform_file = open(ISOFORM_COUNT_PATH, 'r')
            
            for line in isoform_file:
                num_isoforms = int(line.split('\t')[1])
                self.isoform_count.append(num_isoforms)
        
        if self.target == 'tissue' :
            self.fcn = FCN(output_size=6, include_other_class=include_other_class, other_vector_size=5, isoforms_count=self.isoform_count)
        elif self.target == 'perturbation' :
            self.fcn = FCN(output_size=5, include_other_class=include_other_class, other_vector_size=6, isoforms_count=self.isoform_count)
        elif self.target == 'both' :
            self.fcn = FCN(output_size=11, include_other_class=False, isoforms_count=self.isoform_count)
        self.fcn.to(device=self.device)
        
        
        if self.loss == 'crossentropy' or self.loss=='focal' :
            self.criterion = nn.CrossEntropyLoss()
        else :
            self.criterion = nn.BCEWithLogitsLoss()
        
    def save_model(self) :
        torch.save(self.fcn.state_dict(), f'{MODELS_PATH}/model_{self.get_model_identifier()}.pt')
    
    def get_model_identifier(self) :
        return f'{"splitIsoforms" if self.split_isoforms else "nonprior"}_{self.loss}_{self.transformation}_{self.target}{"_otherVector" if self.include_other_class else "_noOtherVector"}'
    
    
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
                    class_pred_tissue = class_pred[:, :6]
                    class_pred_perturbation = class_pred[:, 6:]
                    
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
                  epochs: int = 10,
                  batch_size: int = 32,
                  learning_rate: float = 0.00025,
                  val_percent: float = 0.2,
                  save_checkpoint: bool = False) :
        # 1. Create dataset
        #dataset = get_set(dir_train)
        if self.include_other_class :
            dataset = NonPriorData(tpm_path=data_path,
                                   transformation=self.transformation,
                                   target='both')
        else :
            dataset = NonPriorData(tpm_path=data_path,
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
        self.f_loss = open(f'{RESULTS_PATH}/loss_{self.get_model_identifier()}.txt', 'w')
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
                        class_preds = self.fcn(tpm_data)
                    if self.target != 'both' :
                        loss = self.criterion(class_preds, class_true)
                        if self.loss == 'focal' :
                            pt = torch.exp(-loss)
                            loss = ((1-pt)**self.gamma * loss).mean() 
                    else : 
                        class_pred_tissue = class_preds[:, :6]
                        class_pred_perturbation = class_preds[:, 6:]
                        
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
            self.f_loss.write(str(val_score))
            self.f_loss.write('\n')


            if save_checkpoint:
                Path(CHECKPOINTS_PATH).mkdir(parents=True, exist_ok=True)
                torch.save(self.fcn.state_dict(), f'{CHECKPOINTS_PATH}/checkpoint_{self.get_model_identifier()}_{epoch}.pth')
                #logging.info(f'Checkpoint {epoch} saved!')
        self.f_loss.close()
if __name__ == '__main__':
    agent = NonPriorTraining(loss='focal', transformation='log2', target='perturbation', include_other_class=True, split_isoforms=True)
    agent.train(JOINED_DATA_PATH_GROUPED)
    agent.save_model()

