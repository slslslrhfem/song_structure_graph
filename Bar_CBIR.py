# Code for Bar contents based information Retrieval

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
import wandb
from AE import AE_dataset
#wandb.init(project="pattern_representation")
torch.autograd.set_detect_anomaly(True)

def image_process_CBIR_CONLON(root):
    ROOT = Path(root)
    files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.npz'))
    pianoroll = pypianoroll.load(files[0])
    pretty = pypianoroll.to_pretty_midi(pianoroll)
    inst_name=[]
    for insts in pretty.instruments:
        inst_name.append(insts.name)
    for j,file in tqdm(enumerate(files)): 
        conlon_full=[]
        pianoroll = pypianoroll.load(file)
        CONLON = pypianoroll_to_CONLON(pianoroll)     
        iter=0
        for insts in CONLON:
            for channel in insts:
                conlon_full.append(np.array(channel))
        conlon_full = np.moveaxis(np.array(conlon_full),0,-1) 
        get_conlon_tensor(np.array(conlon_full),file)
    print("done!")


class CBIR_dataset(Dataset):
    def __init__(self):
        self.files = sorted(list(Path("bar_conlon_image").glob('*/*.npy')))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        np_dict = np.load(self.files[index],allow_pickle=True)
        CONLON_image = np_tensor_decoder(np_dict)
        CONLON_image = np.transpose(CONLON_image,(2,0,1)) #[channel, time, pitch]

        #CONLON_image = np.sum(CONLON_image,axis=0) # used for represent all instruments in one channel.(sum or concat all channel) used for image based similarity
        #CONLON_image = np.expand_dims(CONLON_image,axis=0)
        
        #CONLON_image -= np.min(CONLON_image)
        #CONLON_image /= np.max(CONLON_image)# normalize to [0,1]. note that it doesn't make each image's mean and std same.
        #this is general method in image process, but we can't denormalize this precisely, so we can use this only for CBIR. 

        return torch.tensor(CONLON_image,dtype = torch.float)

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

class Encoder(nn.Module):

    def __init__(self,hp,p,t):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image.
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = hp.dim_h
        self.net = nn.Sequential(
            nn.Conv2d(hp.n_channel, c_hid, kernel_size=3, padding=1, stride=2), # 96x128 => 48x64
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 48x64 => 24x32
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 24x32 => 12x16
            nn.GELU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*int(p/8)*int(t/8)*c_hid, hp.n_z)
        )

    def forward(self, x):
        return self.net(x)
class Decoder(nn.Module):

    def __init__(self,hp,p,t):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = hp.dim_h
        self.p = p
        self.t= t
        self.linear = nn.Sequential(
            nn.Linear(hp.n_z, 2*int(p/8)*int(t/8)*c_hid),
            nn.GELU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, hp.n_channel, kernel_size=3, output_padding=1, padding=1, stride=2), 
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, int(self.p/8), int(self.t/8))
        x = self.net(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()

        self.n_channel = hp.n_channel
        self.dim_h = hp.dim_h
        self.n_z = hp.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

class wasserstein_VAE(pl.LightningModule):
    def __init__(self,p,t): # p for pitch, t for time
        super(wasserstein_VAE,self).__init__()
        self.encoder = Encoder(hp,p,t)
        self.decoder = Decoder(hp,p,t)
        self.discriminator = Discriminator(hp)
    
    def forward(self, x):
        return self.encoder(x)
    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch
        # train discriminator
        if optimizer_idx == 0:
            
            z_fake = torch.randn(images.size()[0], hp.n_z) * hp.sigma
            z_fake = z_fake.to(self.device)
            d_fake = self.discriminator(z_fake)
            z_real = self.encoder(images)
            d_real = self.discriminator(z_real)
            eps = 1e-10
            gan_loss = hp.gan_LAMBDA * (-torch.log(d_fake + eps).mean()-torch.log(1 - d_real + eps).mean())
            self.log("gan_loss",gan_loss, prog_bar=True)
            return gan_loss

        # train encoder and decoder
        if optimizer_idx == 1:
            batch_size = images.size()[0]
            z_real = self.encoder(images)
            x_recon = self.decoder(z_real)
            d_real = self.discriminator(self.encoder(Variable(images.data)))
            recon_loss = nn.MSELoss()(x_recon, images)
            d_loss = -hp.LAMBDA * (torch.log(d_real)).mean()
            self.log("recon_loss",recon_loss,prog_bar=True)
            self.log("d_loss",d_loss, prog_bar=True)
            return recon_loss + d_loss



    def configure_optimizers(self):
        opt_d = optim.Adam(self.discriminator.parameters(), lr = 0.5 * hp.lr)
        opt_autoencoder = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr = hp.lr)
        return [opt_d, opt_autoencoder], []
    
    def training_epoch_end(self,outs):
        batch_size = hp.batch_size
        test_loader = self.trainer._data_connector._val_dataloader_source.dataloader()
        test_iter = iter(test_loader)
        test_data = next(test_iter)
        z_real = self.encoder(Variable(test_data).cuda())
        reconst = self.decoder(z_real).cpu().view(batch_size, hp.n_channel, 96, 128)
        reconst = torch.transpose(torch.sum(reconst, axis=1),1,2).view(batch_size,1,128,96)
        test_data = torch.transpose(torch.sum(test_data.view(batch_size, hp.n_channel, 96, 128),axis=1),1,2).view(batch_size,1,128,96)
        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')


        save_image(test_data, './data/reconst_images/wae_gan_input.png')
        save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (self.current_epoch + 1))
        torch.save(self.encoder.state_dict(),"models/encoder"+str(self.current_epoch)+"dim_h"+str(hp.dim_h)+"latent"+str(hp.n_z)+"chan"+str(hp.n_channel)+".h5")
        torch.save(self.decoder.state_dict(),"models/decoder"+str(self.current_epoch)+"dim_h"+str(hp.dim_h)+"latent"+str(hp.n_z)+"chan"+str(hp.n_channel)+".h5")
        torch.save(self.discriminator.state_dict(),"models/discriminator"+str(self.current_epoch)+"dim_h"+str(hp.dim_h)+"latent"+str(hp.n_z)+"chan"+str(hp.n_channel)+".h5")


def train_CBIR():
    dataset = CBIR_dataset()
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset)-train_len
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size = hp.batch_size,
                          shuffle=True, drop_last=True,num_workers=12) 

    test_loader = DataLoader(dataset=test_dataset,
                         batch_size=hp.batch_size,
                         shuffle=False, drop_last=True)
    print(train_dataset[0].shape)
    model = wasserstein_VAE(train_dataset[0].shape[1],train_dataset[0].shape[2])
    wandb_logger = WandbLogger()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = 'checkpoints/', filename ='{epoch:02d}-check')
    #note that saveing models and checkpoints is different.
    trainer = pl.Trainer(logger=wandb_logger, callbacks=checkpoint_callback, accelerator="auto", devices=1 if torch.cuda.is_available() else None, max_epochs=100,detect_anomaly=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)#, ckpt_path="checkpoints/~.ckpt")

def train_AE():
    dataset= VQVAE_dataset()
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset)-train_len
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size = hp.batch_size,
                          shuffle=True, drop_last=True,num_workers=12) 

    test_loader = DataLoader(dataset=test_dataset,
                         batch_size=hp.batch_size,
                         shuffle=False, drop_last=True,num_workers=12)
    print(train_dataset[0].shape)
    model = wasserstein_VAE(train_dataset[0].shape[1],train_dataset[0].shape[2])
    wandb_logger = WandbLogger()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = 'ae_checkpoints/', filename ='{epoch:02d}-check')
    #note that saveing models and checkpoints is different.
    trainer = pl.Trainer(logger=wandb_logger, callbacks=checkpoint_callback, accelerator="auto", devices=1 if torch.cuda.is_available() else None, max_epochs=100,detect_anomaly=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)#, ckpt_path="checkpoints/~.ckpt")