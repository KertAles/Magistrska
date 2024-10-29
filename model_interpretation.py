
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:26:11 2024

@author: alesk
"""


from sklearn.metrics import roc_curve, auc
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training import FCN, FCN_CKN
import global_values as gv
from load_tpm_data import NonPriorData
from utils import get_model_identifier, enrichr_table_to_latex

from copy import deepcopy

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        #def first_layer_hook_fn(module, grad_in, grad_out):
        #    self.image_reconstruction = grad_in[0] 

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
        
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, torch.nn.LeakyReLU):
                return (torch.clamp(grad_in[0], min=0.),)

        for i, module in enumerate(self.model.modules()):
            if isinstance(module, torch.nn.LeakyReLU):
                #print(test_model.named_modules())
                module.register_backward_hook(relu_hook_function)
        """
        # AlexNet model 
        modules = list(self.model.modules())[1:]

        # travese the modules，register forward hook & backward hook
        # for the ReLU
        for module in modules:
            if isinstance(module, nn.LeakyReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        # register backward hook for the first conv layer
        #first_layer = modules[0] 
        #first_layer.register_backward_hook(first_layer_hook_fn)
        """
    def visualize(self, input_image, target_class, other_class=None):
        input_image.requires_grad_()
        model_output = self.model(input_image, other_class)
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
        
        result = input_image.grad
        #result = self.image_reconstruction.data[0]
        return result, pred_class#.numpy()
    
    
if __name__ == '__main__':
    
    loss='focal'
    transformation = 'log2'
    target = 'perturbation'
    include_other_class=False
    split_isoforms=True
    split_batches = True
    
    metadata_path='./data/metadata_T.tsv'
    #model_type = 'ckn'

    
    isoforms_file = open(gv.ISOFORM_COUNT_PATH, 'r')
    isoform_count = []
    
    if split_isoforms :
        for line in isoforms_file:
            num_isoforms = int(line.split('\t')[1])
            isoform_count.append(num_isoforms)

    #model_id = get_model_identifier(loss, transformation, target, include_other_class, split_isoforms, special_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tpm_table = pd.read_table('./data/vae_cov_tpm_avg.tsv', index_col=0)
    metadata_table = pd.read_table(metadata_path, index_col=0)
    metadata_table = metadata_table.rename(columns={'SRAStudy' : 'sra_study'})
    metadata_table = metadata_table[['perturbation_group', 'tissue_super', 'sra_study']]
    tpm_table = metadata_table.join(tpm_table, how='inner')
    
    batches = tpm_table['sra_study'].unique()
    tissues = tpm_table['tissue_super']

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
    
    tissues = ['mature_flower', 'mature_leaf', 'mature_root', 'seed', 'young_seedling']
    perturbations = ['chemical stress', 'control', 'environmental stress', 'mechanical stress', 'mutant']
    #perturbations = ['control', 'environmental stress']
    results_tissue = []
    results_perturbation = []
    
    for fold in range(num_folds) :
        tissue_res = {'ckn': {}, 'baseline' :{}}
        for tissue in tissues :
            tissue_res['ckn'][tissue] = {'count' : 0, 'sum' : []}
            tissue_res['baseline'][tissue] = {'count' : 0, 'sum': []}
        results_tissue.append(tissue_res)
        
        perturbation_res = {'ckn': {}, 'baseline' :{}}
        for perturbation in perturbations :
            perturbation_res['ckn'][perturbation] = {'count' : 0, 'sum' : []}
            perturbation_res['baseline'][perturbation] = {'count' : 0, 'sum' : []}
        results_perturbation.append(perturbation_res)
            
    
    for model_type in ['baseline', 'ckn'] :
        for fold in range(num_folds):
            
            #if model_type == 'ckn' :
            #    data_path='./data/vae_cov_tpm_avg.tsv'
            #else :
            data_path = './data/vae_cov_transformed.tsv'
            metadata_path = './data/metadata_T.tsv'
            special_id = data_path.split('/')[-1].split('.')[0]
            
            test_batches = k_folds[fold]
        
            if split_batches :
                if len(test_batches) == 0 :
                    test_file = open('test_batches.txt', 'r')
                    test_batches = []
                    
                    for batch in test_file:
                        test_batches.append(batch[:-1])
                  
                if include_other_class :
                    dataset = NonPriorData(tpm_path=data_path,
                                           metadata_path=metadata_path,
                                           transformation=transformation,
                                           target='both',
                                           batches=test_batches)
                else :
                    dataset = NonPriorData(tpm_path=data_path,
                                           metadata_path=metadata_path,
                                           transformation=transformation,
                                           target=target,
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
                        model = FCN_CKN(num_genes=input_size,
                                           output_size=tissue_output_size,
                                           device=device,
                                           split_isoforms=split_isoforms,
                                           include_other_class=include_other_class,
                                           other_vector_size=perturbation_output_size,
                                           columns=dataset.columns.tolist(),
                                           dropout=0.2)
                       
                    elif target == 'perturbation' :
                        model = FCN_CKN(num_genes=input_size,
                                           output_size=perturbation_output_size,
                                           device=device,
                                           split_isoforms=split_isoforms,
                                           include_other_class=include_other_class,
                                           other_vector_size=tissue_output_size,
                                           columns=dataset.columns.tolist(),
                                           dropout=0.2)
                else :
                    if target != 'both' :
                        model = FCN_CKN(num_genes=input_size,
                                           output_size=first_output_size,
                                           device=device,
                                           split_isoforms=split_isoforms,
                                           include_other_class=include_other_class,
                                           other_vector_size=second_output_size,
                                           columns=dataset.columns.tolist(),
                                           dropout=0.2)
                       
                    elif target == 'both' :
                        model = FCN_CKN(num_genes=input_size,
                                           output_size=perturbation_output_size + tissue_output_size,
                                           device=device,
                                           split_isoforms=split_isoforms,
                                           include_other_class=include_other_class,
                                           other_vector_size=perturbation_output_size,
                                           columns=dataset.columns.tolist(),
                                           dropout=0.2)
                        num_of_tissues = tissue_output_size
        
            #model.load_state_dict(torch.load(f'{gv.MODELS_PATH}/model_{model_id}.pt'))
            model_id = get_model_identifier(loss, transformation, target, include_other_class, split_isoforms, special_id, model_type=model_type)
            
            model.load_state_dict(torch.load(f'{gv.MODELS_PATH}/model_{model_id}_{str(fold+1)}.pt', map_location=torch.device('cpu')))
            model.to(device=device)
            model.eval()
            
            #for param in model.parameters():
            #    print(param)
                
            loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
            test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
            
            num_test_batches = len(test_loader)
            
            guided_bp = Guided_backprop(model)
            #lrp = captum.attr.GuidedBackprop(model)
            
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
                
                result, prediction = guided_bp.visualize(tpm_data, class_true)
                
                if class_true.argmax().item() == prediction :
                    
                    true_idx = (torch.max(torch.exp(class_true), 1)[1]).data.cpu().numpy()[0]
                    
                    if target == 'tissue' :
                        results_tissue[fold][model_type][tissues[true_idx]]['count'] += 1
                        if 'sum' in results_tissue[fold][model_type][tissues[true_idx]] :   
                            results_tissue[fold][model_type][tissues[true_idx]]['sum'].append(result.data.numpy()[0, :])
                    else :
                        results_perturbation[fold][model_type][perturbations[true_idx]]['count'] += 1
                        if 'sum' in results_perturbation[fold][model_type][perturbations[true_idx]] :   
                            results_perturbation[fold][model_type][perturbations[true_idx]]['sum'].append(result.data.numpy()[0, :])

    """
    if target == 'tissue' :
        for fold in range(num_folds) :
            for tissue in results_tissue[fold]['ckn'] :
                results_tissue[fold]['ckn'][tissue]['avg'] = results_tissue[fold]['ckn'][tissue]['sum'] / results_tissue[fold]['ckn'][tissue]['count']
            
            for tissue in results_tissue[fold]['baseline'] :
                results_tissue[fold]['baseline'][tissue]['avg'] = results_tissue[fold]['baseline'][tissue]['count'] / results_tissue[fold]['baseline'][tissue]['count']
                
        results_tissue_copy = deepcopy(results_tissue)
    else :
        for fold in range(num_folds) :
            for perturbation in results_perturbation[fold]['ckn'] :
                results_perturbation[fold]['ckn'][perturbation]['avg'] = results_perturbation[fold]['ckn'][perturbation]['sum'] / results_perturbation[fold]['ckn'][perturbation]['count']
                #results_perturbation['ckn'][perturbation]['sum'] /= results_perturbation['ckn'][perturbation]['count']
            
            for perturbation in results_perturbation[fold]['baseline'] :
                results_perturbation[fold]['baseline'][perturbation]['avg'] = results_perturbation[fold]['baseline'][perturbation]['sum'] / results_perturbation[fold]['baseline'][perturbation]['count']    
                
        results_perturbation_copy = deepcopy(results_perturbation)
    """  
    
    results_tissue_copy = deepcopy(results_tissue)
    results_perturbation_copy = deepcopy(results_perturbation)
    
    tissues = ['mature_leaf', 'mature_root', 'mature_flower', 'seed']
    
    perturbation_genes = {
        'chemical stress' : ['RESPONSE_TO_CADMIUM_ION',
        'RESPONSE_TO_CARBOHYDRATE_STIMULUS',
        'RESPONSE_TO_DISACCHARIDE_STIMULUS',
        'RESPONSE_TO_ETHYLENE_STIMULUS',
        'RESPONSE_TO_HYDROGEN_PEROXIDE',
        'RESPONSE_TO_NITRATE',
        'RESPONSE_TO_NUTRIENT_LEVELS',
        'RESPONSE_TO_OSMOTIC_STRESS',
        'RESPONSE_TO_FRUCTOSE_STIMULUS',
        'RESPONSE_TO_METAL_ION',
        'RESPONSE_TO_MONOSACCHARIDE_STIMULUS',
        'RESPONSE_TO_SALT_STRESS',
        'RESPONSE_TO_CHEMICAL_STIMULUS'],
        'environmental stress' : [
            'RESPONSE_TO_ABSENCE_OF_LIGHT',
            'RESPONSE_TO_BACTERIUM',
            'RESPONSE_TO_BLUE_LIGHT',
            'RESPONSE_TO_FAR_RED_LIGHT',
            'RESPONSE_TO_COLD',
            'RESPONSE_TO_FREEZING',
            'RESPONSE_TO_FUNGUS',
            'RESPONSE_TO_HEAT',
            'RESPONSE_TO_HIGH_LIGHT_INTENSITY',
            'RESPONSE_TO_LIGHT_STIMULUS',
            'RESPONSE_TO_LIGHT_INTENSITY',
            'RESPONSE_TO_MOLECULE_OF_BACTERIAL_ORIGIN',
            'RESPONSE_TO_MOLECULE_OF_FUNGAL_ORIGIN',
            'RESPONSE_TO_HYPOXIA',
            'RESPONSE_TO_OXIDATIVE_STRESS',
            'RESPONSE_TO_OXYGEN_LEVELS',
            'RESPONSE_TO_OZONE',
            'RESPONSE_TO_RED_LIGHT',
            'RESPONSE_TO_RED_OR_FAR_RED_LIGHT',
            'RESPONSE_TO_TEMPERATURE_STIMULUS',
            'RESPONSE_TO_UV',
            'RESPONSE_TO_WATER_DEPRIVATION',
            'RESPONSE_TO_WATER_STIMULUS'],
        'mechanical stress' : [
            'RESPONSE_TO_WOUNDING'
            ],
        'mutant' : [
            'RESPONSE_TO_DNA_DAMAGE_STIMULUS'
            ]
        }
    
    def get_gene_relevance_scores(curr_class, results, tpm_table) :
        ckn_nodes = pd.read_table(gv.NODES_CKN_ANNOT_PATH)
        ckn_nodes = ckn_nodes[ckn_nodes['node_ID'].isin(list(tpm_table.columns))]
        
        
        ckn_tissue = curr_class.split('_')[-1]
        
        all_genes = list(tpm_table.columns)[3:]
        nodes = list(ckn_nodes['node_ID'])
        y_test = list(ckn_nodes['tissue'].str.contains(ckn_tissue))
        
        
        for line in isoforms_file:
            num_isoforms = int(line.split('\t')[1])
            isoform_count.append(num_isoforms)
        
        
        ckn_relevance_scores = []
        baseline_relevance_scores = []
        for fold in range(num_folds) :
            new_baseline_relevance = np.zeros(len(all_genes))
            j = 0
            avg_relevance = np.mean(results[fold]['baseline'][curr_class]['sum'], axis=0)
            if avg_relevance.size > 1 :
                for isoform_cnt, i in zip(isoform_count, range(len(new_baseline_relevance))) :
                    
                    new_baseline_relevance[i] = np.max(avg_relevance[j:j+isoform_cnt])
                    j += isoform_cnt
            
            results[fold]['baseline'][curr_class]['avg'] = deepcopy(new_baseline_relevance)
            baseline_relevance_scores.append(results[fold]['baseline'][curr_class]['avg'])
            
            new_baseline_relevance = np.zeros(len(all_genes))
            j = 0
            avg_relevance = np.mean(results[fold]['ckn'][curr_class]['sum'], axis=0)
            if avg_relevance.size > 1 :
                for isoform_cnt, i in zip(isoform_count, range(len(new_baseline_relevance))) :
                    
                    new_baseline_relevance[i] = np.max(avg_relevance[j:j+isoform_cnt])
                    j += isoform_cnt
                
            results[fold]['ckn'][curr_class]['avg'] = deepcopy(new_baseline_relevance)
            ckn_relevance_scores.append(deepcopy(results[fold]['ckn'][curr_class]['avg']))
       
        y_pred_baseline_return = []
        y_pred_ckn_return = []
        for fold in range(num_folds):
            #tissue_res_ckn = np.mean(ckn_relevance_scores, axis=0)
            #tissue_res_base = np.mean(baseline_relevance_scores, axis=0)
            tissue_res_ckn = ckn_relevance_scores[fold]
            tissue_res_base = baseline_relevance_scores[fold]
            
            y_pred_proba_ckn = np.zeros(ckn_nodes.shape[0])
            y_pred_proba_base = np.zeros(ckn_nodes.shape[0])
            
            for node, i in zip(nodes, range(len(nodes))):
                
                gene_idx = all_genes.index(node)
                
                y_pred_proba_ckn[i] = tissue_res_ckn[gene_idx]
                y_pred_proba_base[i] = tissue_res_base[gene_idx]
                
            y_pred_baseline_return.append(y_pred_proba_base)
            y_pred_ckn_return.append(y_pred_proba_ckn)
        
        return y_test, y_pred_ckn_return, y_pred_baseline_return
    
    
    if target == 'tissue' :
        results_tissue = deepcopy(results_tissue_copy)
        for curr_tissue in tissues :
    
            y_test, y_pred_proba_ckn, y_pred_proba_base = get_gene_relevance_scores(curr_tissue, results_tissue, tpm_table)
            
            plt.figure(figsize=(8, 6), dpi=80)  
            tprs_ckn = []
            tprs_base = []
            
            relevance_ckn = []
            relevance_base = []
            
            mean_fpr = np.arange(0, 10, 0.002)
            for fold in range(num_folds) :
                
                gene_sets = {}
        
                with open('./data/arabidopsis.gmt') as gmt:
                    for line in gmt :
                        split = line.split('\t')
                        gene_sets[split[0]] = split[-1].split(',')
                
                
                # Calculate ROC curve
                if sum(y_pred_proba_ckn[fold]) != 0 :
                    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba_ckn[fold])
                    roc_auc1 = auc(fpr1, tpr1)
                    
                    interp_tpr = np.interp(mean_fpr, fpr1, tpr1)
                    interp_tpr[0] = 0.0
                    
                    tprs_ckn.append(interp_tpr)
                    relevance_ckn.append(y_pred_proba_ckn[fold])
                    
                if sum(y_pred_proba_base[fold]) != 0 :
                    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba_base[fold]) 
                    roc_auc2 = auc(fpr2, tpr2)
                    
                    interp_tpr = np.interp(mean_fpr, fpr2, tpr2)
                    interp_tpr[0] = 0.0
                    
                    tprs_base.append(interp_tpr)
                    relevance_base.append(y_pred_proba_base[fold])
                # Plot the ROC curve
            relevance_ckn = np.mean(relevance_ckn, axis=0)
            relevance_base = np.mean(relevance_base, axis=0)
            
            fpr1, tpr1, thresholds1 = roc_curve(y_test, relevance_ckn, drop_intermediate=False)
            
            tpr1_tmp = np.interp(mean_fpr, fpr1, tpr1)
            tpr1_tmp[0] = 0.0
            fpr1_tmp = np.interp(mean_fpr, tpr1, fpr1)
            fpr1_tmp[0] = 0.0
            
            fpr1 = fpr1_tmp
            tpr1 = tpr1_tmp
            
            roc_auc1 = auc(fpr1, tpr1)
            
            std_tpr1 = np.std(tprs_ckn, axis=0)
            tprs1_upper = np.minimum(tpr1 + std_tpr1, 1)
            tprs1_lower = np.maximum(tpr1 - std_tpr1, 0)
              
            
            fpr2, tpr2, thresholds2 = roc_curve(y_test, relevance_base, drop_intermediate=False) 
            
            tpr2_tmp = np.interp(mean_fpr, fpr2, tpr2)
            tpr2_tmp[0] = 0.0
            fpr2_tmp = np.interp(mean_fpr, tpr2, fpr2)
            fpr2_tmp[0] = 0.0
            
            fpr2 = fpr2_tmp
            tpr2 = tpr2_tmp
            
            roc_auc2 = auc(fpr2, tpr2)
            
            std_tpr2 = np.std(tprs_base, axis=0)
            tprs2_upper = np.minimum(tpr2 + std_tpr2, 1)
            tprs2_lower = np.maximum(tpr2 - std_tpr2, 0)
            
            plt.fill_between(
                fpr1,
                tprs1_lower,
                tprs1_upper,
                color="blue",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            
            plt.fill_between(
                fpr2,
                tprs2_lower,
                tprs2_upper,
                color="orange",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            
            plt.plot(fpr1, tpr1, label='CKN model  (area = %0.3f)' % roc_auc1)
            plt.plot(fpr2, tpr2, label='Baseline model (area = %0.3f)' % roc_auc2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {curr_tissue} genes')
            plt.legend()
            plt.show()
            
    else :
        
        
        #perturbations = ['chemical stress', 'environmental stress']
        #perturbations = ['environmental stress']

        results_perturbation = deepcopy(results_perturbation_copy)
        
        for curr_perturbation in perturbations:
            #curr_perturbation = 'mutant'
            y_test, y_pred_proba_ckn, y_pred_proba_base = get_gene_relevance_scores(curr_perturbation, results_perturbation, tpm_table)
            
            
            ckn_nodes = pd.read_table(gv.NODES_CKN_ANNOT_PATH)
            ckn_nodes = ckn_nodes[ckn_nodes['node_ID'].isin(list(tpm_table.columns))]
            
            plt.figure(figsize=(8, 6), dpi=80)  
            tprs_ckn = []
            tprs_base = []
            
            relevance_ckn = []
            relevance_base = []
            
            mean_fpr = np.arange(0, 10, 0.002)
            for fold in range(num_folds) :
                
                gene_sets = {}
        
                with open('./data/arabidopsis.gmt') as gmt:
                    for line in gmt :
                        split = line.split('\t')
                        gene_sets[split[0]] = split[-1].split(',')
                
                
                pert_gene_set = []
                
                for gene_set in perturbation_genes[curr_perturbation] :
                    pert_gene_set.extend(gene_sets[gene_set])
                
                pert_gene_set = list(set(pert_gene_set))
                
                print(curr_perturbation)
                print(len(pert_gene_set))
                
                y_test = list(ckn_nodes['node_ID'].isin(pert_gene_set))
                
                # Calculate ROC curve
                if sum(y_pred_proba_ckn[fold]) != 0 :
                    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba_ckn[fold])
                    roc_auc1 = auc(fpr1, tpr1)
                    
                    interp_tpr = np.interp(mean_fpr, fpr1, tpr1)
                    interp_tpr[0] = 0.0
                    
                    tprs_ckn.append(interp_tpr)
                    relevance_ckn.append(y_pred_proba_ckn[fold])
                    
                if sum(y_pred_proba_base[fold]) != 0 :
                    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba_base[fold]) 
                    roc_auc2 = auc(fpr2, tpr2)
                    
                    interp_tpr = np.interp(mean_fpr, fpr2, tpr2)
                    interp_tpr[0] = 0.0
                    
                    tprs_base.append(interp_tpr)
                    relevance_base.append(y_pred_proba_base[fold])
                # Plot the ROC curve
            relevance_ckn = np.mean(relevance_ckn, axis=0)
            relevance_base = np.mean(relevance_base, axis=0)
            
            fpr1, tpr1, thresholds1 = roc_curve(y_test, relevance_ckn, drop_intermediate=False)
            
            tpr1_tmp = np.interp(mean_fpr, fpr1, tpr1)
            tpr1_tmp[0] = 0.0
            fpr1_tmp = np.interp(mean_fpr, tpr1, fpr1)
            fpr1_tmp[0] = 0.0
            
            fpr1 = fpr1_tmp
            tpr1 = tpr1_tmp
            
            roc_auc1 = auc(fpr1, tpr1)
            
            std_tpr1 = np.std(tprs_ckn, axis=0)
            tprs1_upper = np.minimum(tpr1 + std_tpr1, 1)
            tprs1_lower = np.maximum(tpr1 - std_tpr1, 0)
              
            
            fpr2, tpr2, thresholds2 = roc_curve(y_test, relevance_base, drop_intermediate=False) 
            
            tpr2_tmp = np.interp(mean_fpr, fpr2, tpr2)
            tpr2_tmp[0] = 0.0
            fpr2_tmp = np.interp(mean_fpr, tpr2, fpr2)
            fpr2_tmp[0] = 0.0
            
            fpr2 = fpr2_tmp
            tpr2 = tpr2_tmp
            
            roc_auc2 = auc(fpr2, tpr2)
            
            std_tpr2 = np.std(tprs_base, axis=0)
            tprs2_upper = np.minimum(tpr2 + std_tpr2, 1)
            tprs2_lower = np.maximum(tpr2 - std_tpr2, 0)
            
            plt.fill_between(
                fpr1,
                tprs1_lower,
                tprs1_upper,
                color="blue",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            
            plt.fill_between(
                fpr2,
                tprs2_lower,
                tprs2_upper,
                color="orange",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            
            plt.plot(fpr1, tpr1, label='CKN model  (area = %0.3f)' % roc_auc1)
            plt.plot(fpr2, tpr2, label='Baseline model (area = %0.3f)' % roc_auc2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {curr_perturbation} genes')
            plt.legend()
            plt.show()
            
    
    """
    meta = pd.read_table(gv.METADATA_PATH)
    
    meta_env = meta[meta['perturbation_group'] == 'environmental stress']
    meta_chem = meta[meta['perturbation_group'] == 'chemical stress']
    meta_mech = meta[meta['perturbation_group'] == 'mechanical stress']
    meta_mut = meta[meta['perturbation_group'] == 'mutant']
    
    
    sub_env0 = meta_env['perturbation_group_1_1'].value_counts()
    sub_env1 = meta_env['perturbation_group_2_1'].value_counts()
    sub_env2 = meta_env['perturbation_group_2_2'].value_counts()
    sub_env3 = meta_env['perturbation_group_2_3'].value_counts()
    
    sub_chem0 = meta_chem['perturbation_group_1_1'].value_counts()
    sub_chem1 = meta_chem['perturbation_group_2_1'].value_counts()
    sub_chem2 = meta_chem['perturbation_group_2_2'].value_counts()
    sub_chem3 = meta_chem['perturbation_group_2_3'].value_counts()
    
    
    sub_mech0 = meta_mech['perturbation_group_1_1'].value_counts()
    sub_mech1 = meta_mech['perturbation_group_2_1'].value_counts()
    sub_mech2 = meta_mech['perturbation_group_2_2'].value_counts()
    sub_mech3 = meta_mech['perturbation_group_2_3'].value_counts()
    
    sub_mut0 = meta_mut['perturbation_group_1_1'].value_counts()
    sub_mut1 = meta_mut['perturbation_group_2_1'].value_counts()
    sub_mut2 = meta_mut['perturbation_group_2_2'].value_counts()
    sub_mut3 = meta_mut['perturbation_group_2_3'].value_counts()
    """
    
    import gseapy as gp

    i = 0
    gene_sets = {}

    with open('./data/arabidopsis.gmt') as gmt:
        for line in gmt :
            split = line.split('\t')
            gene_sets[split[0]] = split[-1].split(',')
            
    
    gene_sets = {}

    with open('./data/ath_gomapmen.gmt') as gmt:
        for line in gmt :
            g_split = line.split('\t')
            gene_set_name = ' '.join(g_split[0].split(' ')[1:])
            #if not 'protein' in gene_set_name and not 'RNA' in gene_set_name :
            gene_sets[gene_set_name] = g_split[1:]
            
            
    results_perturbation = deepcopy(results_perturbation_copy)
    #results_perturbation = deepcopy(results_tissue_copy)
    #for curr_perturbation in perturbations :
    for curr_perturbation in perturbations:   
        
        
        y_test, y_pred_proba_ckn, y_pred_proba_base = get_gene_relevance_scores(curr_perturbation, results_perturbation, tpm_table)
        
        relevance_ckn = []
        relevance_base = []
        for fold in range(num_folds):
            relevance_ckn.append(y_pred_proba_ckn[fold])
            relevance_base.append(y_pred_proba_base[fold])
        
        y_pred_proba_ckn = np.mean(relevance_ckn, axis=0)
        y_pred_proba_base = np.mean(relevance_base, axis=0)
        
        ckn_nodes = pd.read_table(gv.NODES_CKN_ANNOT_PATH)
        ckn_nodes = ckn_nodes[ckn_nodes['node_ID'].isin(list(tpm_table.columns))]
        ckn_nodes_list = list(ckn_nodes['node_ID'])
        
      
        ckn_relevant_genes = np.flip(np.argsort(y_pred_proba_ckn)[-1000:])
        gene_list_ckn = [ckn_nodes_list[i] for i in ckn_relevant_genes]
        
        base_relevant_genes = np.flip(np.argsort(y_pred_proba_base)[-1000:])
        gene_list_base = [ckn_nodes_list[i] for i in base_relevant_genes]
        
        enr_base = gp.enrichr(gene_list=gene_list_base, # or "./tests/data/gene_list.txt",
                     gene_sets=gene_sets, # don't forget to set organism to the one you desired! e.g. Yeast
                     background=None, # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
                     outdir=None,
                     verbose=True)
        
        enr_ckn = gp.enrichr(gene_list=gene_list_ckn, # or "./tests/data/gene_list.txt",
                     gene_sets=gene_sets, # don't forget to set organism to the one you desired! e.g. Yeast
                     background=None, # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
                     outdir=None,
                     verbose=True)
        
        
        base_res = enr_base.results
        ckn_res = enr_ckn.results
        
        gp.dotplot(enr_base.res2d, figsize=(3,5), title=f"Baseline model, {curr_perturbation}", cmap = plt.cm.autumn_r)
        plt.show()
    
        gp.dotplot(enr_ckn.res2d, figsize=(3,5), title=f"CKN-based model, {curr_perturbation}", cmap = plt.cm.autumn_r)
        plt.show()
        
        
        baseline_latex_table, ckn_latex_table = enrichr_table_to_latex(base_res, ckn_res, p_value=0.125)
        
        
        
        ckn_relevant_genes = np.flip(np.argsort(y_pred_proba_ckn))
        ckn_sorted_relevance = np.flip(np.sort(y_pred_proba_ckn))
        gene_list_ckn = [ckn_nodes_list[i] for i in ckn_relevant_genes]
        
        base_relevant_genes = np.flip(np.argsort(y_pred_proba_base))
        base_sorted_relevance = np.flip(np.sort(y_pred_proba_base))
        gene_list_base = [ckn_nodes_list[i] for i in base_relevant_genes]
        
        tpm_avg = pd.read_table('./data/vae_cov_tpm_avg.tsv', index_col=0)
        metadata_table = pd.read_table('./data/metadata_T.tsv', index_col=0)
        
        tpm_avg = tpm_avg.join(metadata_table, how='inner')
        tpm_avg = tpm_avg[tpm_avg['perturbation_group'].isin([curr_perturbation, 'control'])]
        classes = list(tpm_avg['perturbation_group'])
        tpm_avg = tpm_avg.drop(['sra_study', 'perturbation_group', 'tissue_super', 'experiment_library_strategy', 'experiment_library_selection', 'experiment_instrument_model'], axis=1)
        
        tpm_avg = tpm_avg[np.intersect1d(tpm_avg.columns, gene_list_base)]
        tpm_avg = tpm_avg.reindex(gene_list_base, axis=1)
        
        tpm_avg *= 492263.0
        
        #cols = tpm_avg.columns
        #remove_cols = list(set(cols).difference(set(gene_list_base)))
        #tpm_avg = tpm_avg.drop(remove_cols, axis=1)
    
        #classes = ['chemical stress'] * 304
        
        test = pd.DataFrame(data=gene_list_base)
        test.insert(1, 1, base_sorted_relevance, True)
        pre_res = gp.prerank(rnk=test,
                     gene_sets=gene_sets,
                     verbose=True,
                    )
        terms = pre_res.res2d.Term
        axs = pre_res.plot(terms=terms[:5],
                   ofname=f"Baseline model, {curr_perturbation}",
                   #legend_kws={'loc': (1.2, 0)}, # set the legend loc
                   show_ranking=True, # whether to show the second yaxis
                   figsize=(5,8)
                  )
        
        
        #base_gsea = gp.gsea(tpm_avg.T, gene_sets, classes)
        #terms = base_gsea.res2d.Term
        #axs = base_gsea.plot(terms[:10], ofname=f"Baseline model, {curr_perturbation}", show_ranking=False, legend_kws={'loc': (1.05, 0)}, )
        
        
        
        tpm_avg = pd.read_table('./data/vae_cov_tpm_avg.tsv', index_col=0)
        metadata_table = pd.read_table('./data/metadata_T.tsv', index_col=0)
        
        tpm_avg = tpm_avg.join(metadata_table, how='inner')
        tpm_avg = tpm_avg[tpm_avg['perturbation_group'].isin([curr_perturbation, 'control'])]
        classes = list(tpm_avg['perturbation_group'])
        tpm_avg = tpm_avg.drop(['sra_study', 'perturbation_group', 'tissue_super', 'experiment_library_strategy', 'experiment_library_selection', 'experiment_instrument_model'], axis=1)
        
        tpm_avg = tpm_avg[np.intersect1d(tpm_avg.columns, gene_list_ckn)]
        tpm_avg = tpm_avg.reindex(gene_list_ckn, axis=1)
        
        tpm_avg *= 492263.0
        
        #cols = tpm_avg.columns
        #remove_cols = list(set(cols).difference(set(gene_list_ckn)))
        #tpm_avg = tpm_avg.drop(remove_cols, axis=1)
        test = pd.DataFrame(data=gene_list_ckn)
        test.insert(1, 1, ckn_sorted_relevance, True)
        pre_res = gp.prerank(rnk=test,
                     gene_sets=gene_sets,
                     verbose=True,
                    )
        terms = pre_res.res2d.Term
        axs = pre_res.plot(terms=terms[:5],
                   ofname=f"CKN-based model, {curr_perturbation}",
                   #legend_kws={'loc': (1.2, 0)}, # set the legend loc
                   show_ranking=True, # whether to show the second yaxis
                   figsize=(5,8)
                  )
        
        base_res.to_csv(f'plots/enrichr_baseline_{curr_perturbation}_unfiltered.csv')
        ckn_res.to_csv(f'plots/enrichr_ckn_{curr_perturbation}_unfiltered.csv')
        
        