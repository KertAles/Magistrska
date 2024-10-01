
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:26:11 2024

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
from torch import nn

import captum

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)


        # AlexNet model 
        modules = list(self.model.modules())[1:]

        # travese the modules，register forward hook & backward hook
        # for the ReLU
        for module in modules:
            if isinstance(module, nn.LeakyReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        # register backward hook for the first conv layer
        first_layer = modules[0] 
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class):
        input_image.requires_grad_()
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        
        grad_target_map = torch.zeros(model_output.shape,
                                      dtype=torch.float)
        if target_class is not None:
            target_class = target_class.argmax()
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1
        
        model_output.backward(grad_target_map)
        
        result = self.image_reconstruction.data[0]
        return result#.numpy()
    
    
    
def get_model_identifier(loss, transformation, target, otherClass, split_isoforms) :
    return f'{"splitIsoforms" if split_isoforms else "nonprior"}_{loss}_{transformation}_{target}{"_otherVector" if otherClass else "_noOtherVector"}'

if __name__ == '__main__':
    
    loss='focal'
    transformation = 'log2'
    target = 'both'
    include_other_class=False
    split_isoforms=False
    special_id = 'blah'
    split_batches = True
    data_path='./data/grouped_tpm_avg.tsv'
    metadata_path=None
    model_type = 'ckn'
    
    isoforms_file = open(gv.ISOFORM_COUNT_PATH, 'r')
    isoform_count = []
    
    if split_isoforms :
        for line in isoforms_file:
            num_isoforms = int(line.split('\t')[1])
            isoform_count.append(num_isoforms)

    #model_id = get_model_identifier(loss, transformation, target, include_other_class, split_isoforms, special_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fold_file = open('folds_file.txt', 'r')
    
    test_batches = fold_file.readline()[:-1].split('\t')
    fold_file.close()
    
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
        test_set = dataset
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
        
        #model = FCN_CKN(output_size=perturbation_output_size + tissue_output_size, columns=dataset.columns.tolist())
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

    #model.load_state_dict(torch.load(f'{gv.MODELS_PATH}/model_{model_id}.pt'))
    model.load_state_dict(torch.load('models/model_ckn__grouped_tpm_avg_focal_log2_both_noOtherVector.pt'))
    model.to(device=device)
    model.eval()
    
    #for param in model.parameters():
    #    print(param)
        
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    
    num_test_batches = len(test_loader)
    
    guided_bp = Guided_backprop(model)
    lrp = captum.attr.GuidedBackprop(model)
    
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
            class_true_tissue = class_true_tissue.to(device=device, dtype=torch.float32)[:, 0, :]
            class_true_perturbation = class_true_perturbation.to(device=device, dtype=torch.float32)[:, 0, :]
            class_true = torch.cat([class_true_tissue,
                           class_true_perturbation], 1)
        
        result = guided_bp.visualize(tpm_data, class_true)
        
        
        result = lrp.attribute(inputs=tpm_data)
        
        print(result)
        print(result.permute(1,2,0).numpy())
        print('blah')
        break
    
    
