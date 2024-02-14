# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:46:38 2024

@author: alesk
"""

from predict_nonprior import predict_model
from training_nonprior import NonPriorTraining
from global_values import MODELS_PATH, CHECKPOINTS_PATH, JOINED_DATA_PATH_GROUPED, RESULTS_PATH

losses = ['crossentropy', 'focal', 'bceLogits']
transformations = ['none', 'log2', 'log10']
targets = ['perturbation', 'both']
other_classes = [False, True]
#split_isoforms = [False, True]
#losses = ['bceLogits']
#transformations = ['log10']
#targets = ['both']

split_isoforms = [True]
data_path = JOINED_DATA_PATH_GROUPED


if __name__ == '__main__' :
    for loss in losses :
        for transformation in transformations :
            for target in targets :
                for split_isoform in split_isoforms :
                    for other_class in other_classes :
                        
                        if target == 'both' and other_class :
                            continue
                        
                        print(f'Training {"splitIsoforms" if split_isoform else "nonprior"}_{loss}_{transformation}_{target}_{"otherClass" if other_class else "noOtherClass"} on {data_path}')
                        
                        agent = NonPriorTraining(loss=loss, transformation=transformation, target=target, include_other_class=other_class)
                        agent.train(data_path)
                        agent.save_model()
                            
                        predict_model(data_path, loss=loss, transformation=transformation, target=target, include_other_class=other_class)
    
        