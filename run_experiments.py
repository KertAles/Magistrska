# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:46:38 2024

@author: alesk
"""

from predict import predict_model, get_model_identifier
from training import NonPriorTraining
import global_values as gv

losses = ['focal']
transformations = ['log2']
targets = ['both']
other_classes = [False]
#data = ['./data/harmony_joined.tsv']
data = [ './data/limma_data.tsv', './data/averaged_data.tsv', './data/vae_transformed5.tsv']
metadata = [None, None, './data/metadata_proc.tsv']
#data = [gv.PROPORTIONAL_DATA_CONTROLS, gv.GROUPED_DATA]
#split_isoforms = [False, True]
#losses = ['bceLogits']
#transformations = ['log10']
#targets = ['both']

split_isoforms = [False]
#data_path = JOINED_DATA_PATH_GROUPED


if __name__ == '__main__' :
    i = 1
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
                            
                            model_id = get_model_identifier(loss, transformation, target, other_class, split_isoform, special_id=data_sign, model_type='ckn')
                            
                            print(f'Training {model_id} on {data_path}')
                            
                            agent = NonPriorTraining(model_type='ckn', loss=loss, transformation=transformation, target=target, include_other_class=other_class, split_isoforms=split_isoform, special_id=data_sign)
                            agent.train(data_path, metadata_path, epochs=10)
                            agent.save_model()
                                
                            predict_model(data_path, metadata_path, model_type='ckn', loss=loss, transformation=transformation, target=target, include_other_class=other_class, split_isoforms=split_isoform, special_id=data_sign)
        
            