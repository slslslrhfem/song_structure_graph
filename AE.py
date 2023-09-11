
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



class AE_dataset(Dataset):
    def __init__(self):
        self.files = sorted(list(Path("pattern_conlon_image_5").glob('*/*.npy')))
        if hp.inst_num==17:
            self.transform = transforms.Compose([transforms.Normalize((0.0705, 0.0111), (2.6271, 0.9481))])
        else:
            self.transform = transforms.Compose([transforms.Normalize((0.1332, 0.0161), (3.2939, 0.6091))])
        # note that mean = tensor([0.0705, 0.0111]), std = tensor([2.6271, 0.9481]) for our data


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        np_dict = np.load(self.files[index],allow_pickle=True)
        CONLON_image = np_tensor_decoder(np_dict) # channel, time, pitch
        CONLON_image = torch.tensor(CONLON_image,dtype = torch.float)# self.transform(torch.tensor(CONLON_image,dtype = torch.float))
        #CONLON_image = np.sum(CONLON_image,axis=0) # used for represent all instruments in one pianoroll.
        #CONLON_image = np.expand_dims(CONLON_image,axis=0)       
        return self.transform(CONLON_image)



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
        self._bn0 = nn.BatchNorm2d(in_channels)
        self._bn1 = nn.BatchNorm2d(num_hiddens // 2)
        self._bn2 = nn.BatchNorm2d(num_hiddens)
        self.linear = nn.Sequential(
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(num_hiddens*32*48,hp.AE_embedding_dim),
        )

    def forward(self, inputs):
        inputs= self._conv_0(inputs)
        inputs= self._bn0(inputs)
        inputs = F.relu(inputs)
        
        x = self._conv_1(inputs)
        x = self._bn1(x)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = self._bn2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._residual_stack(x)
        
        
        return self.linear(x)
        

class Decoder_AE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder_AE, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(hp.AE_embedding_dim,128*32*48),
            nn.GELU()
        )

        self._bn1 = nn.BatchNorm2d(num_hiddens)
        self._bn_trans1 = nn.BatchNorm2d(num_hiddens // 2)

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
        x = x.reshape(x.shape[0], -1, 48, 32)
        x = self._conv_1(x)
        x = self._bn1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = self._bn_trans1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class AE(pl.LightningModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,  embedding_dim):
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

        self.reg_weight= 10
        self.kernel_type = 'imq'

    def forward(self, x):
        z = self._encoder(x)
        x_recon = self._decoder(z)
        return z, x_recon 
    
    def training_step(self, batch, batch_idx):
        images = batch
        z, data_recon = self.forward(images)
        recon_error = F.mse_loss(data_recon, images)
        batch_size = images.size(0)   # You can try Wasserstein AE for single track
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr
        mmd_loss = self.compute_mmd(z, reg_weight)
        
        loss = recon_error  + mmd_loss * 0.1

        self.log("recon_loss", recon_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("mmd_loss", mmd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch
        z, data_recon = self.forward(images)
        recon_error = F.mse_loss(data_recon, images)
        
        batch_size = images.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr
        mmd_loss = self.compute_mmd(z, reg_weight) 
        

        loss = recon_error  + mmd_loss * 0.1
        self.log("val_recon_loss", recon_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mmd_loss", mmd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor":"val_loss"}

    def compute_kernel(self, x1, x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self, x1, x2, eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1, x2, eps = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z, reg_weight):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = reg_weight * prior_z__kernel.mean() + \
              reg_weight * z__kernel.mean() - \
              2 * reg_weight * priorz_z__kernel.mean()
        return mmd
    
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        x=x.float()
        y=y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

def train_AE():
    dataset = AE_dataset()
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset)-train_len
    """
    mean = 0.
    std = 0.
    loader = DataLoader(dataset)
    for images in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(mean, std)
    """

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size = hp.AE_batch_size,
                          shuffle=True, drop_last=True,num_workers=12) #collate_fn=numpy_collate)

    valid_loader = DataLoader(dataset=test_dataset,
                         batch_size=hp.batch_size,
                         shuffle=False, drop_last=True,num_workers=12)#, collate_fn=numpy_collate)
    
    #train_loader_forvar = DataLoader(train_dataset, batch_size=10000)
    #train_dataset_array = next(iter(train_loader_forvar))[0].numpy()
    model = AE(hp.AE_num_hiddens, hp.AE_num_residual_layers, hp.AE_num_residual_hiddens,
              hp.AE_embedding_dim)
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(monitor='val_recon_loss',
                                         dirpath = 'checkpoints/ae_checkpoints_'+str(hp.inst_num)+'/', save_top_k=10, filename='ae-{epoch:02d}-{val_recon_loss:.4f}')
    trainer = pl.Trainer(logger=wandb_logger, accelerator="auto", devices=1 if torch.cuda.is_available() else None,
                         max_epochs=100, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)