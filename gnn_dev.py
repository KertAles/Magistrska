# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:40:16 2024

@author: alesk
"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tabulate import tabulate

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv, HeteroGraphConv, GATConv
from dgl.data import DGLDataset
from torch.utils.data import random_split

import global_values as gv

from sklearn import preprocessing


class CKNDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="CKN_dataset")
    
    def process(self):
        self.graphs = []
        self.labels = []
        
        nodes_data = pd.read_csv(gv.NODES_CKN_ANNOT_PATH, sep='\t')
        edges_data = pd.read_csv(gv.INTERACTIONS_CKN_ANNOT_PATH, sep='\t')

        chosen_rank = 3
        #tpm_table = pd.read_table('./data/vae_transformed5.tsv', index_col=0)
        tpm_table = pd.read_table('./data/averaged_data.tsv', index_col=0)
        
        tpm_table.dropna(inplace=True)
        
        #metadata_table = pd.read_table('./data/metadata_proc.tsv', index_col=0)[['perturbation_group', 'tissue_super', 'sra_study']]
        #tpm_table = metadata_table.join(tpm_table, how='inner')
         
        #tpm_table = tpm_table[tpm_table["tissue_super"].isin(["young_seedling", "mature_leaf", "mature_root"])]
        #tpm_table = tpm_table[tpm_table["perturbation_group"].isin(["environmental stress", "chemical stress", "control"])]
        
        perturbations = pd.DataFrame(tpm_table['perturbation_group']).copy()
        perturbations['perturbation_group'] = perturbations['perturbation_group'].apply(
                            lambda x: 'stressed' if 'control' not in x else x)
        perturbation_count = perturbations['perturbation_group'].value_counts()
        
        self.perturbation_encoder = preprocessing.LabelEncoder()
        self.perturbation_encoder.fit_transform(perturbations)
        
        tissues = pd.DataFrame(tpm_table['tissue_super']).copy()
        tissue_count = tissues['tissue_super'].value_counts()
            
        if 'sra_study' in tpm_table.columns :
            tpm_table.drop("sra_study", axis=1, inplace=True)
        if 'tissue_super' in tpm_table.columns :
            tpm_table.drop("tissue_super", axis=1, inplace=True)
        if 'perturbation_group' in tpm_table.columns :
            tpm_table.drop("perturbation_group", axis=1, inplace=True)
        if 'secondary_perturbation' in tpm_table.columns :
            tpm_table.drop("secondary_perturbation", axis=1, inplace=True)
        """
        rank_file = open(gv.CKN_GENE_RANKS, 'r')
        genes_list = []

        for line in rank_file:
            gene_rank = int(line.split('\t')[1])
            if gene_rank <= chosen_rank :
                genes_list.append(line.split('\t')[0])
                          
        tpm_table = tpm_table.drop(columns=[col for col in tpm_table if col not in genes_list])
    
        edges_data = edges_data[edges_data['rank'] <= chosen_rank]
        edges_data = edges_data[edges_data['source'].isin(tpm_table.columns)]
        edges_data = edges_data[edges_data['target'].isin(tpm_table.columns)]
        
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(pd.concat([edges_data['source'], edges_data['target']]))
        
        edges_data['source'] = self.label_encoder.transform(edges_data['source'])
        edges_data['target'] = self.label_encoder.transform(edges_data['target'])
        
        undirected_edges = edges_data[edges_data['isDirected'] == 0]
        
        self.dim_nfeats = len(tpm_table.columns)
        
        for column in tpm_table.columns: 
            tpm_table[column] = tpm_table[column]  / tpm_table[column].abs().max() 
                
        for srr_accession, row in tpm_table.iterrows() :
            
            edges_src = torch.from_numpy(np.concatenate([edges_data["source"].to_numpy(), undirected_edges["target"].to_numpy()]))
            edges_dst = torch.from_numpy(np.concatenate([edges_data["target"].to_numpy(), undirected_edges["source"].to_numpy()]))
            
            g = dgl.graph(
                (edges_src, edges_dst)
            )
            
            #print(g.num_nodes())
            #g = dgl.add_self_loop(g)
            
            nodes = self.label_encoder.inverse_transform(g.nodes().numpy())
            
            node_features = torch.from_numpy(row[nodes].to_numpy())
            node_features = node_features.unsqueeze(1)
            g.ndata["attr"] = node_features
            
            
            self.graphs.append(g)
            self.labels.append(perturbations.loc[srr_accession].values[0])
        """
        
        tissues = ['seed', 'root', 'leaf', 'flower']
        edges_data_copy = edges_data.copy()
        
        for tissue in tissues :
            tissue_file = open(gv.CKN_GENE_TISSUE, 'r')
            genes_list = []

            for line in tissue_file:
                gene_tissue = line.split('\t')[1]
                if tissue in gene_tissue :
                    genes_list.append(line.split('\t')[0])
            
            tpm_table_tissue = tpm_table.drop(columns=[col for col in tpm_table if col not in genes_list])
        
            #edges_data = edges_data[edges_data['rank'] <= chosen_rank]
            edges_data = edges_data_copy[edges_data_copy['source'].isin(tpm_table_tissue.columns)]
            edges_data = edges_data[edges_data['target'].isin(tpm_table_tissue.columns)]
            
            
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(pd.concat([edges_data['source'], edges_data['target']]))
            
            edges_data['source'] = self.label_encoder.transform(edges_data['source'])
            edges_data['target'] = self.label_encoder.transform(edges_data['target'])
            
            undirected_edges = edges_data[edges_data['isDirected'] == 0]
            
            self.dim_nfeats = len(tpm_table.columns)
            
            for column in tpm_table.columns: 
                tpm_table[column] = tpm_table[column]  / tpm_table[column].abs().max() 
            
            edges_src = torch.from_numpy(np.concatenate([edges_data["source"].to_numpy(), undirected_edges["target"].to_numpy()]))
            edges_dst = torch.from_numpy(np.concatenate([edges_data["target"].to_numpy(), undirected_edges["source"].to_numpy()]))
            
            
            for srr_accession, row in tpm_table.iterrows() :
                
                
                g = dgl.graph(
                    (edges_src, edges_dst)
                )
                
                #print(g.num_nodes())
                g = dgl.add_self_loop(g)
                
                nodes = self.label_encoder.inverse_transform(g.nodes().numpy())
                
                node_features = torch.from_numpy(np.log1p(row[nodes].to_numpy()))
                g.ndata["attr"] = node_features.unsqueeze(1)
                
                
                self.graphs.append(g)
                self.labels.append(perturbations.loc[srr_accession].values[0])
        
        
        self.class_encoder = preprocessing.LabelEncoder()
        
        self.num_classes = len(self.perturbation_encoder.classes_)
        self.dim_nfeats = 1
        
        
        self.labels = self.class_encoder.fit_transform(self.labels)
        self.labels = torch.LongTensor(self.labels)
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class GCN(nn.Module):
    """
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")
    
    """
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats*2)
        self.conv2 = GraphConv(h_feats*2, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        #self.conv1 = GATConv(in_feats, h_feats*2, num_heads=8, feat_drop=0.2, attn_drop=0.2)
        #self.conv2 = GATConv(h_feats*2, h_feats, num_heads=8, feat_drop=0.2, attn_drop=0.2)
        #self.conv3 = GATConv(h_feats, num_classes, num_heads=8, feat_drop=0.2, attn_drop=0.2)
        
        #self.fcn = torch.nn.Linear(21964, num_classes)
        self.sm = torch.nn.Softmax(dim=1)
    def forward(self, g, in_feat, shape):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        #h = F.relu(h)
        #h = h.flatten()
        #h = h.reshape((shape[0], 21964))
        #h = self.fcn(h)
        #return h
        
        g.ndata["h"] = h
        #return dgl.mean_nodes(g, "h")
        return self.sm(dgl.mean_nodes(g, "h"))


batch_size=3

dataset = CKNDataset()

n_train = int(len(dataset) * 0.8)
n_test = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))

train_dataloader = GraphDataLoader(train_set, shuffle=True, batch_size=batch_size)
test_dataloader = GraphDataLoader(test_set, shuffle=False, drop_last=True, batch_size=batch_size)

"""
dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)


n_train = int(len(dataset) * 0.3)
n_test = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))


train_dataloader = GraphDataLoader(train_set, shuffle=True, batch_size=16)
test_dataloader = GraphDataLoader(test_set, shuffle=False, drop_last=True, batch_size=16)

""" 

class_counts = [0.0, 0.0]

for batched_graph, labels in train_dataloader :
    for label in labels :
        class_counts[label] += 1
        
class_sum = class_counts[0] + class_counts[1]
class_counts[0] = class_counts[0] / (class_sum)
class_counts[1] = class_counts[1] / (class_sum)
   
# Create the model with given dimensions
model = GCN(dataset.dim_nfeats, 32, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

gamma = 2.0
epochs = 10

y_train = []

for epoch in range(epochs):
    epoch_loss = 0
    with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        k = 0
        weight = torch.Tensor(class_counts)
        for batched_graph, labels in train_dataloader:
            test = batched_graph.ndata["attr"].float()
            pred = model(batched_graph, batched_graph.ndata["attr"].float(), labels.shape)
            
            loss = F.cross_entropy(pred, labels, weight=weight)
            pt = torch.exp(-loss)
            loss = ((1-pt)**gamma * loss).mean() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test = labels.data.cpu().numpy()
            y_train.extend(test)
            epoch_loss += loss.item()
            #experiment.log({
            #    'train loss': loss.item(),
            #    'step': global_step,
            #    'epoch': epoch
            #})
            k += 1
            #if k % 100 == 0 :
            pbar.update(labels.shape[0])
            pbar.set_postfix(**{'loss (batch)': epoch_loss / k})   
            
    y_pred = []
    y_true = []

    class_0 = np.array([])
    class_1 = np.array([])

    for batched_graph, class_true in test_dataloader:
        class_pred = model(batched_graph, batched_graph.ndata["attr"].float(), class_true.shape)
        
        out_vect = class_pred.data.cpu().numpy()
        class_0 = np.append(class_0, out_vect[:, 0])
        class_1 = np.append(class_1, out_vect[:, 1])
        output = torch.max(torch.exp(class_pred), 1)[1].data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        #class_true = (torch.max(torch.exp(class_true), 1)[1]).data.cpu().numpy()
        class_true = class_true.data.cpu().numpy()
        y_true.extend(class_true) # Save Truth
       
    roc_auc = metrics.roc_auc_score(y_true, class_1)
    
    print(f'Current AUC: {roc_auc}')
    

y_pred = []
y_true = []

class_0 = np.array([])
class_1 = np.array([])

for batched_graph, class_true in tqdm(test_dataloader, total=len(test_dataloader), desc='Test round', unit='batch', leave=False):
    class_pred = model(batched_graph, batched_graph.ndata["attr"].float(), class_true.shape)
    
    out_vect = class_pred.data.cpu().numpy()
    class_0 = np.append(class_0, out_vect[:, 0])
    class_1 = np.append(class_1, out_vect[:, 1])
    output = torch.max(torch.exp(class_pred), 1)[1].data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    
    #class_true = (torch.max(torch.exp(class_true), 1)[1]).data.cpu().numpy()
    class_true = class_true.data.cpu().numpy()
    y_true.extend(class_true) # Save Truth
   
roc_auc = metrics.roc_auc_score(y_true, class_1)

    
cf_matrix = confusion_matrix(y_true, y_pred)
class_report = metrics.classification_report(y_true, y_pred)

classification_report = open(f'{gv.RESULTS_PATH}/report_GNN.txt', 'w')
classification_report.write(class_report)
classification_report.write('\n')
classification_report.write(tabulate(cf_matrix))
classification_report.write('\n')

#classification_report.write('\t'.join(category for category in dataset.class_encoder.classes_))
classification_report.write('\n')
classification_report.close()
print(dataset.class_encoder.classes_)




"""
thresholds = map(lambda x: x/100000.0, range(60000, 100000, 1))

accs

for th in thresholds :
    th_preds = 
    
    
"""  
    

