
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pypianoroll
import pretty_midi
from pretty_midi import PrettyMIDI as pm
import pickle
from matplotlib import pyplot as plt
import os
from itertools import combinations
from pattern_classes import pattern
import torch
from torch.utils.data import Dataset
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from hyperparameter import Hyperparameter as hp
import torchvision
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(123)
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from utils import pypianoroll_to_CONLON, get_conlon_tensor, np_tensor_decoder, np_tensor_encoder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import torch.nn.functional as F
#wandb.init(project="pattern_representation_vqvae")


"""
Lots of source code from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=Pl56hjlBI8ym.
"""


class AE_dataset(Dataset):
    def __init__(self):
        self.files = sorted(list(Path("pattern_conlon_image").glob('*/*.npy')))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        np_dict = np.load(self.files[index],allow_pickle=True)
        CONLON_image = np_tensor_decoder(np_dict) # channel, time, pitch
        #CONLON_image = np.sum(CONLON_image,axis=0) # used for represent all instruments in one pianoroll.
        #CONLON_image = np.expand_dims(CONLON_image,axis=0)       
        return torch.tensor(CONLON_image,dtype = torch.float)



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder_AE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder_AE, self).__init__()

        self._conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,dilation=12,padding=0)

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self.linear = nn.Sequential(
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(num_hiddens*32*96,256)
        )

    def forward(self, inputs):
        inputs= self._conv_0(inputs)
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._residual_stack(x)

        return self.linear(x)

class Decoder_AE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder_AE, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(256,128*32*96),
            nn.GELU()
        )

        self._conv_1 = nn.Conv2d(in_channels=128,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = x.reshape(x.shape[0],-1, 96,32)
        x = self._conv_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class AE(pl.LightningModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,  embedding_dim, data_variance):
        super(AE, self).__init__()
        
        self._encoder = Encoder_AE(2, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self._decoder = Decoder_AE(128,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

        self.data_variance = data_variance

    def forward(self, x):
        z = self._encoder(x)
        x_recon = self._decoder(z)
        return z, x_recon 
    
    def training_step(self, batch, batch_idx):
        images = batch
        data_recon = self.forward(images)[1]
        recon_error = F.mse_loss(data_recon, images) / self.data_variance
        self.log("recon_loss", recon_error)
        return recon_error
    
    def validation_step(self, batch, batch_idx):
        images = batch
        data_recon = self.forward(images)[1]
        recon_error = F.mse_loss(data_recon, images) / self.data_variance
        self.log("val_loss",recon_error)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor":"val_loss"}



def train_AE():
    dataset = AE_dataset()
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset)-train_len

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size = hp.vq_batch_size,
                          shuffle=True, drop_last=True,num_workers=12) #collate_fn=numpy_collate)

    valid_loader = DataLoader(dataset=test_dataset,
                         batch_size=hp.batch_size,
                         shuffle=False, drop_last=True,num_workers=12)#, collate_fn=numpy_collate)
    
    train_loader_forvar = DataLoader(train_dataset, batch_size=10000)
    train_dataset_array = next(iter(train_loader_forvar))[0].numpy()

    data_variance = 6.355786#np.var(train_dataset_array) 
    model = AE(hp.vq_num_hiddens, hp.vq_num_residual_layers, hp.vq_num_residual_hiddens,
              hp.vq_embedding_dim, data_variance)
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                         dirpath = 'ae_checkpoints/', save_top_k=10, filename='ae-{epoch:02d}-{val_loss:.2f}')
    trainer = pl.Trainer(logger=wandb_logger, accelerator="auto", devices=1 if torch.cuda.is_available() else None,
                         max_epochs=100, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)