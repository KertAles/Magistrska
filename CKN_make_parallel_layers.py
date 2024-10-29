# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:59:04 2024

@author: alesk
"""

import global_values as gv
import pandas as pd
import numpy as np

def generate_CKN_layers(rank=2) :
    connection_data = pd.read_table(gv.INTERACTIONS_CKN_ANNOT_PATH)
    
    connection_data = connection_data[connection_data['rank'] <= rank]
    
    tf_data = connection_data[connection_data['isTFregulation'] == 1]
    
    binding_data = connection_data[connection_data['type'] == 'binding']
    
    
    tf_targets = tf_data['target'].unique()
    
    tf_file = open('data/tf_groups.txt', 'w')
    
    for target in tf_targets :
        line = target
        
        source_tfs_curr = tf_data[tf_data['target'] == target]
        source_tfs_curr = source_tfs_curr['source'].unique()
        
        line = line + '\t' + '\t'.join(source_tfs_curr) + '\n'
    
        tf_file.write(line)
    
    tf_file.close()
    
    bindings = np.concatenate([binding_data['source'].unique(), binding_data['target'].unique()])
    bindings_file = open('data/binding_groups.txt', 'w')
    
    for binding in bindings :
        line = binding
        
        source_bindings = binding_data[binding_data['source'] == binding]
        target_bindings = binding_data[binding_data['target'] == binding]
        
        neighbours = np.concatenate([source_bindings['target'].unique(),
                                    target_bindings['source'].unique()])
        
        neighbours = list(set(neighbours))
        
        line = line + '\t' + '\t'.join(neighbours) + '\n'
        
        bindings_file.write(line)
        
    bindings_file.close()

if __name__ == '__main__':
    generate_CKN_layers(rank=2)