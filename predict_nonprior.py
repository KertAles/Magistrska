# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:40:02 2023

@author: alesk
"""


from sklearn.metrics import confusion_matrix
from prediction_nonprior import FCN
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from global_values import MODELS_PATH, JOINED_DATA_PATH
from load_tpm_data import NonPriorData
import numpy as np


if __name__ == '__main__':
    model = FCN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODELS_PATH + "/model_nonprior.pt"))
    model.to(device=device)
    model.eval()
    
    dataset = NonPriorData(JOINED_DATA_PATH)
    val_percent = 0.1
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    num_val_batches = len(val_loader)
    ce_loss = 0
    
    
    y_pred = []
    y_true = []

    # iterate over the validation set
    for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        tpm_data, class_true = batch['tpm'], batch['classification'][:, 0, :]
        # move images and labels to correct device and type
        tpm_data = tpm_data.to(device=device, dtype=torch.float32)
        class_true = class_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            class_pred = model(tpm_data)
            
            output = (torch.max(torch.exp(class_pred), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = (torch.max(torch.exp(class_true), 1)[1]).data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    print('blah')
    
    print(dataset.onehot.categories_)
    
    #print(dataset.onehot.transform(['mature_flower', 'mature_leaf', 'mature_root', 'mature_seed',
    #       'mature_seedling', 'seed_seed', 'senescence_senescence_green',
    #       'senescence_senescence_reproductive', 'young_seed',
    #       'young_seedling']))