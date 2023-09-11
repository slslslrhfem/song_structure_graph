import torch_geometric
import torch
import numpy as np
import dgl
from pathlib import Path
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hyperparameter import Hyperparameter as hp
from pytorch_lightning.loggers import WandbLogger
from utils import pypianoroll_to_CONLON, get_conlon_tensor, np_tensor_decoder, np_tensor_encoder
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset
from tqdm import tqdm
from AE import AE, Encoder_AE
import copy
from functools import partial

import random

class GraphDataset(dgl.data.DGLDataset):
    def __init__(self):
        if hp.inst_num==17:
            ROOT = Path("processed_pattern_with_order_17/conformed_lpd_17/lpd_17_cleansed")
        else:
            ROOT = Path("processed_pattern_with_order_5/conformed_lpd_5/lpd_5_cleansed")
        self.files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.bin'))

        # Data object is from PYG, It can be replaced with DGL.graph object in the same way. 
        # However, when N is large, this will case OOM.
    def __getitem__(self,index):
        graph_list, _ = dgl.load_graphs(str(self.files[index]))
        graph = graph_list[0]
        
        label = graph.ndata['genre'][0]
        
         # genre label
        return graph, [label]
    
    def __len__(self):
        return len(self.files)



def train_graph2vec():
    dataset = GraphDataset()
    """
    label_num = [0 for i in range(17)]
    for data in tqdm(dataset):
        label_num[data[1][0].item()]+=data[0].number_of_nodes()
    class_weight = label_num/np.sum(label_num)
    print(class_weight) 
    #use for compute genre class weight
    """ 

    train_size = int(len(dataset)*0.9)
    validation_size = len(dataset)-train_size
    train_dataset = Subset(dataset,range(train_size))
    validation_dataset = Subset(dataset,range(train_size, train_size + validation_size))# use non-random split for test generation quality

    train_loader = GraphDataLoader(train_dataset, batch_size=hp.GEM_batch_size, shuffle=True,num_workers=5)#, collate_fn=_collate_fn)
    valid_loader = GraphDataLoader(validation_dataset, batch_size=4, shuffle=False,num_workers=5)#, collate_fn=_collate_fn)
    classifier_model = graph_emb_classifier(hp.GEM_input_dim, 512, hp.AE_embedding_dim, 19)
   
    wandb_logger = WandbLogger()
    if hp.inst_num==17:
        checkpoint_callback = ModelCheckpoint(dirpath = 'checkpoints/graph_checkpoints_17/', save_top_k = 10, filename='graph-{epoch:02d}-{val_mask_loss:.4f}-{val_loss:.4f}', monitor="val_loss")
    else:
        checkpoint_callback = ModelCheckpoint(dirpath = 'checkpoints/graph_checkpoints_5/', save_top_k = 10, filename='graph-{epoch:02d}-{val_mask_loss:.4f}-{val_loss:.4f}', monitor="val_loss")
    trainer = pl.Trainer(logger = wandb_logger, accelerator='cpu', devices=1 if torch.cuda.is_available() else None,
                          max_epochs=2000,detect_anomaly=True,callbacks=[checkpoint_callback]) 
    trainer.fit(classifier_model, train_loader, valid_loader)
    return 0

class graph_emb_classifier(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, embedding_dimension,n_classes):
        super(graph_emb_classifier, self).__init__()
        self.inst_emb = nn.Embedding(hp.inst_num, embedding_dimension//2)
        self.key_emb = nn.Embedding(24, embedding_dimension//2)
        self.position_emb = nn.Embedding(128, 256)
        #self.efeat_linear = nn.Linear(4,hp.GEM_input_dim)# need for GINEConv
        self.image_bn = nn.BatchNorm1d(hp.AE_embedding_dim)
        self.conlon_bn = nn.BatchNorm1d(hp.AE_embedding_dim)
        #self.resnet = resnet18()
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, 4, num_bases=4)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, 4,num_bases=4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.embedding = nn.Linear(hidden_dim, embedding_dimension)
        #self.embedding = dglnn.RelGraphConv(hidden_dim, embedding_dimension, 4, num_bases=4)#activation=partial(F.softmax, dim=1))
        #self.embedding_bn = nn.BatchNorm1d(embedding_dimension)
        self.classify = nn.Linear(embedding_dimension, n_classes)
        self.transform_edge = dgl.DropEdge(p=0.3)
        self.transform_node = dgl.DropNode(p=0.3)
        #self.batch_norm = nn.BatchNorm2d(n_classes)
        self.class_weight=[0.10694549, 0.03150136, 0.19173273, 0.11884681, 0.01055109, 0.09231478,
 0.01340586, 0.01327084, 0.05172588, 0.01827875, 0.0451845,  0.00540816,
 0.0047017,  0.01478502, 0.00703567, 0.01824981, 0.14966582, 0.03171113, 0.07468462]
        
    def forward(self, g, h, efeat):
        # Apply graph convolution and activation.zx
        h = F.relu(self.bn1(self.conv1(g, h, efeat)))
        h = F.relu(self.bn2(self.conv2(g, h, efeat)))
        h = self.embedding(h)#(g, h, efeat)
        embedding = h
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h') 
            classification = self.classify(hg)
            return embedding, classification
    
    def training_step(self, batch, batch_idx):
        graph, labels = batch
        graph = self.transform_edge(graph)
        mask_idx = np.random.choice(len(graph.ndata['CONLON']), int(0.3*len(graph.ndata['CONLON'])))
        target = copy.deepcopy(graph.ndata['CONLON'][mask_idx])
        for idx in mask_idx:
            graph.ndata['CONLON'][idx] = torch.zeros_like(graph.ndata['CONLON'][idx]) # Note that this All zero tensor is used for CONLON latent tensor, not encoder(empty CONLON).

        masked_graph = graph
        
        CONLON = masked_graph.ndata['CONLON'].to(torch.float32) 
        img_feature = CONLON#torch.flatten(CONLON,1)#self.resnet(CONLON)##s(CONLON)#CONLON
        inst = self.inst_emb(masked_graph.ndata['inst'])
        key = self.key_emb(masked_graph.ndata['key'])
        fix_class = torch.cat([inst,key],dim=1).to(torch.float32)

        #inst_emb = self.instrument_embedding(inst)
        #key_emb = self.key_label_embedding(key)
        class_feature = self.image_bn(fix_class) # this is node feature, but always given for training & inference
        img_feature = self.conlon_bn(img_feature) # this is node feature, but partially masked for training, and generated from scratch autoregressively in inference.

        feature = class_feature + img_feature # this is node feature, adding fixed feature and partially masked feature
        position_emb = self.position_emb(masked_graph.ndata['pattern_order'])   #this is position of node feature.
        feature = torch.cat([feature, position_emb], dim=1).to(torch.float32) #use concat for position embedding. 
        efeat = masked_graph.edata['edge_feature']
        #efeat = self.efeat_linear(efeat.to(torch.float32))
        embedding, logits = self(masked_graph, feature, efeat)
        one_shot_labels = F.one_hot(labels[0], num_classes=19).to(torch.float32)
        classification_loss = F.cross_entropy(logits, one_shot_labels,weight=torch.tensor(self.class_weight))#)
        
        masking_prediction_loss = F.mse_loss(embedding[mask_idx], target) # 
        loss = classification_loss + masking_prediction_loss
        pred = logits.argmax(1)
        acc = (pred==labels[0]).float().mean()
        metrics = {'acc': acc, 'cls' : classification_loss,'mask' : masking_prediction_loss} 
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        del target
        return loss

    def validation_step(self, batch, batch_idx):
        graph, labels = batch
        mask_idx = np.random.choice(len(graph.ndata['CONLON']), int(0.3*len(graph.ndata['CONLON'])))
        target = copy.deepcopy(graph.ndata['CONLON'][mask_idx])
        for idx in mask_idx:
            graph.ndata['CONLON'][idx] = torch.zeros_like(graph.ndata['CONLON'][idx]) # this All zero tensor is CONLON latent tensor, not encoder(zero CONLON).

        masked_graph = graph
        CONLON = masked_graph.ndata['CONLON'].to(torch.float32)  # (batch * flatten image)
        img_feature = CONLON#torch.flatten(CONLON,1)#self.resnet(CONLON)#self.resnet(CONLON)
        inst = self.inst_emb(masked_graph.ndata['inst'])
        key = self.key_emb(masked_graph.ndata['key'])
        fix_class = torch.cat([inst,key],dim=1).to(torch.float32)
        #print(inst.shape, img_feature.shape)
        class_feature = self.image_bn(fix_class)
        img_feature = self.conlon_bn(img_feature) #normalize


        feature = class_feature + img_feature#torch.cat([inst,key,img_feature], dim=1).to(torch.float32)
        position_emb = self.position_emb(masked_graph.ndata['pattern_order'])   #this is position of node feature.
        feature = torch.cat([feature, position_emb], dim=1).to(torch.float32)
        efeat = masked_graph.edata['edge_feature']
        #efeat = self.efeat_linear(efeat.to(torch.float32)) #need if use GINEConv
        embedding, logits = self(masked_graph, feature, efeat)
        one_shot_labels = F.one_hot(labels[0], num_classes=19).to(torch.float32)
        classification_loss = F.cross_entropy(logits, one_shot_labels,weight=torch.tensor(self.class_weight))#)
        masking_prediction_loss = F.mse_loss(embedding[mask_idx], target) #
        loss = classification_loss + masking_prediction_loss
        pred = logits.argmax(1)
        acc = (pred==labels[0]).float().mean()
        self.log('val_cls_loss',classification_loss)
        self.log('val_mask_loss',masking_prediction_loss)
        self.log('val_acc',acc)
        self.log('val_loss', loss)
        del target

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr = hp.GEM_learning_rate, momentum = 0.9)
        return opt
    
def node_masking(graph,p): # To get pianoroll before masking with deep copy, we don't used this augmentation method in training
    masked=[]
    for node_idx in range(len(graph.ndata['CONLON'])):
        if random.random()<p:
            graph.ndata['CONLON'][node_idx] = torch.zeros_like(graph.ndata['CONLON'][node_idx]) # this All zero tensor is CONLON latent tensor, not encoder(zero CONLON).
            masked.append(node_idx)
    return graph, masked


