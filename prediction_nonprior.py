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

from global_values import MODELS_PATH, CHECKPOINTS_PATH, JOINED_DATA_PATH

f_loss = open('loss.txt', 'w')

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48359, 8000)
        self.fc2 = nn.Linear(8000, 2000)
        self.fc3 = nn.Linear(2000, 10)
        self.leaky = nn.LeakyReLU(0.05)
        
        
    def forward(self, x): 

        x = self.fc1(x)
        x = self.leaky(x)
        #x = nn.Dropout(p=0.3)(x)
        x = self.fc2(x)
        x = self.leaky(x)
        x = self.fc3(x)
        #x = nn.Softmax(dim=1)(x)
        
        return x
    
    
class NonPriorTraining:
            
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        self.fcn = FCN()
        self.fcn.to(device=self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    def save_model(self) :
        torch.save(self.fcn.state_dict(), '{}/model_nonprior.pt'.format(MODELS_PATH))
    
    
    def preprocess_state(self, state):
        return [torch.tensor(np.array([state]), dtype=torch.float32)]
        
    def evaluate(self, dataloader):
        self.fcn.eval()
        num_val_batches = len(dataloader)
        ce_loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            tpm_data, class_true = batch['tpm'], batch['classification'][:, 0, :]
            # move images and labels to correct device and type
            tpm_data = tpm_data.to(device=self.device, dtype=torch.float32)
            class_true = class_true.to(device=self.device, dtype=torch.float32)

            with torch.no_grad():
                class_pred = self.fcn(tpm_data)
                loss = self.criterion(class_pred, class_true)
                ce_loss += loss.item()
               

        self.fcn.train()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return ce_loss
        return ce_loss / num_val_batches

    def train(self, epochs: int = 10,
                  batch_size: int = 8,
                  learning_rate: float = 0.001,
                  val_percent: float = 0.1,
                  save_checkpoint: bool = True) :
        # 1. Create dataset
        #dataset = get_set(dir_train)
        dataset = NonPriorData(JOINED_DATA_PATH)

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        #grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        global_step = 0

        # 5. Begin training
        for epoch in range(1, epochs+1):
            self.fcn.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                k = 0
                for batch in train_loader:
                    tpm = batch['tpm']
                    true_sep = batch['classification'][:, 0, :]


                    tpm_data = tpm.to(device=self.device, dtype=torch.float32)
                    true_sep = true_sep.to(device=self.device, dtype=torch.float32)
                    
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.utils.clip_grad_norm_(self.fcn.parameters(), 1.0)
                    
                    #with torch.cuda.amp.autocast(enabled=True):
                    sep_preds = self.fcn(tpm_data)
                    loss = self.criterion(sep_preds, true_sep)
                            
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
                    
                    #if k % 100 == 0 :
                    pbar.update(tpm.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item()})   
                        
                    k += 1
            # Evaluation round
        
                            
            val_score = self.evaluate(val_loader)
            print('Validation score: ' + str(val_score))
            f_loss.write(str(val_score))
            f_loss.write('\n')


            if save_checkpoint:
                Path(CHECKPOINTS_PATH).mkdir(parents=True, exist_ok=True)
                torch.save(self.fcn.state_dict(), str(CHECKPOINTS_PATH + 'checkpoint_nonprior_epoch{}.pth'.format(epoch)))
                #logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    agent = NonPriorTraining()
    agent.train()
    agent.save_model()

