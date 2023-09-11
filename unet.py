import torch_geometric
import dgl
from pathlib import Path
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from AE import AE
from hyperparameter import Hyperparameter as hp
from tqdm import tqdm
import argparse
import logging
import math
import os
import random
from pathlib import Path
from graph import graph_emb_classifier
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version

from tqdm.auto import tqdm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from graph import node_masking
import math
from typing import List

import numpy as np
import torch
import copy
import pickle
import torch.nn as nn
import wandb



class Graph_Dataset(dgl.data.DGLDataset):
    def __init__(self):
        if hp.inst_num==17:
            ROOT = Path("processed_pattern_with_order_17/conformed_lpd_17/lpd_17_cleansed")
        else:
            ROOT = Path("processed_pattern_with_order_5/conformed_lpd_5/lpd_5_cleansed")
        self.files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.bin'))

    def __getitem__(self,index):   

        graph_list, _ = dgl.load_graphs(str(self.files[index]))
        graph = graph_list[0]

        return graph
    
    def __len__(self):
        return len(self.files)


def preprocess_unet():
    dataset = Graph_Dataset()
    graph_encoder = graph_emb_classifier(hp.GEM_input_dim, 512, hp.AE_embedding_dim, 19)
    if hp.inst_num==17:
        folder_path = 'graph_checkpoints_17'
    else:
        folder_path = 'graph_checkpoints_5'
    best_checkpoint, lowest_val_mask_loss = find_lowest_val_mask_loss_checkpoint(folder_path)
    print(best_checkpoint)
    checkpoint = torch.load(os.path.join(folder_path, best_checkpoint))
    graph_encoder.load_state_dict(checkpoint["state_dict"])
    dataloader = GraphDataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=10,
    )
    file_num=0
    for data in tqdm(dataloader):
        masked_idx=[]
        conlons=[]# get data before masking
        for node_idx in range(len(data.ndata['CONLON'])):
            if random.random()<0.3:
                conlons.append(data.ndata['CONLON'][node_idx])
                masked_idx.append(node_idx) 

        conlon_latent = copy.deepcopy(conlons)

        for node_idx in masked_idx:
            data.ndata['CONLON'][node_idx] = torch.zeros_like(data.ndata['CONLON'][node_idx]) # this All zero tensor is CONLON latent tensor, not encoder(zero CONLON).

        masked_graph = data


        CONLON = masked_graph.ndata['CONLON'].to(torch.float32) 
        img_feature = CONLON#torch.flatten(CONLON,1)#self.resnet(CONLON)##s(CONLON)#CONLON
        inst = graph_encoder.inst_emb(masked_graph.ndata['inst'])
        key = graph_encoder.key_emb(masked_graph.ndata['key'])
        fix_class = torch.cat([inst,key],dim=1).to(torch.float32)
        class_feature = graph_encoder.image_bn(fix_class)
        img_feature = graph_encoder.image_bn(img_feature)
        feature = class_feature + img_feature
        position_emb = graph_encoder.position_emb(masked_graph.ndata['pattern_order'])
        feature = torch.cat([feature, position_emb], dim=1).to(torch.float32) 
        efeat = masked_graph.edata['edge_feature']
        node_representation, _ = graph_encoder(masked_graph, feature, efeat) # get node_representation from masked_graph
        
        if hp.inst_num==17:
            os.makedirs('processed_pattern_unet_17/', exist_ok=True)
            for i, idx in enumerate(masked_idx):
                pair = [node_representation[idx].detach().numpy(), conlon_latent[i].detach().numpy()]#representation of masked pattern and its CONLON latent pair
                with open('processed_pattern_unet_17/'+str(file_num)+".pkl","wb") as f:
                    pickle.dump(pair, f)
                file_num+=1
        else:
            os.makedirs('processed_pattern_unet_5/', exist_ok=True)
            for i, idx in enumerate(masked_idx):
                pair = [node_representation[idx].detach().numpy(), conlon_latent[i].detach().numpy()]#representation of masked pattern and its CONLON latent pair
                with open('processed_pattern_unet_5/'+str(file_num)+".pkl","wb") as f:
                    pickle.dump(pair, f)
                file_num+=1

class unet_Dataset(dgl.data.DGLDataset):
    def __init__(self):
        if hp.inst_num==17:
            ROOT = Path("processed_pattern_unet_17")
        else:
            ROOT = Path("processed_pattern_unet_5")
        self.files = sorted(ROOT.glob('*.pkl'))

    def __getitem__(self,index):   
        with open(self.files[index], "rb") as f:
            graph_representation, CONLON_latent = pickle.load(f)
        return graph_representation, CONLON_latent
    
    def __len__(self):
        return len(self.files)



class Unet(pl.LightningModule):
    def __init__(self):
        super(Unet, self).__init__()
        self.unet = UNET_1D(1,128,7,3)
        self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, graph_latent):

        return self.unet(graph_latent)

    
    def training_step(self, batch, batch_idx):
        graph_latent, CONLON_latent = batch
        graph_latent = graph_latent.unsqueeze(dim=1)
        CONLON_latent = CONLON_latent.unsqueeze(dim=1)

        CONLON_pred = self.forward(graph_latent)
        loss = self.loss_fn(CONLON_pred, CONLON_latent)


        
        return loss

    def validation_step(self, batch, batch_idx):
        graph_latent, CONLON_latent = batch
        graph_latent = graph_latent.unsqueeze(dim=1)
        CONLON_latent = CONLON_latent.unsqueeze(dim=1)

        CONLON_pred = self.forward(graph_latent)
        loss = self.loss_fn(CONLON_pred, CONLON_latent)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = 0.001,weight_decay=1e-5)
        return opt
    

def train_unet():

    unet=Unet()
    dataset = unet_Dataset()
    train_length = int(len(dataset)*0.9)
    valid_length = len(dataset)-train_length
    train_set, valid_set = torch.utils.data.random_split(dataset,[train_length, valid_length])
    train_loader = DataLoader(dataset=train_set,
                          batch_size = 256,
                          shuffle=True, drop_last=True,num_workers=12)
    valid_loader = DataLoader(dataset=valid_set,
                          batch_size = 256,
                        drop_last=True,num_workers=12)
    wandb_logger = WandbLogger()
    if hp.inst_num==17:
        checkpoint_callback = ModelCheckpoint(dirpath = 'checkpoints/Unet_checkpoints_17/', save_top_k = 10, filename='unet_{epoch:02d}_{val_loss:.5f}', monitor="val_loss")
    else:
        checkpoint_callback = ModelCheckpoint(dirpath = 'checkpoints/Unet_checkpoints_5/', save_top_k = 10, filename='unet_{epoch:02d}_{val_loss:.5f}', monitor="val_loss")
    trainer = pl.Trainer(logger = wandb_logger, accelerator='auto', devices=1 if torch.cuda.is_available() else None,
                          max_epochs=20,detect_anomaly=True,callbacks=[checkpoint_callback])
    trainer.fit(unet, train_loader, valid_loader)





class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=4)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=16)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=64)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,4, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,4, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,4, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1,padding = 3)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        #out = nn.functional.softmax(out,dim=2)
        
        return out