import sys
from collect_npz import preprocess_npz
from Bar_CBIR import image_process_CBIR_CONLON, train_CBIR
from utils import *
from Bar_CBIR import CBIR_dataset
import torch
import numpy as np
from AE import  train_AE
from graph import train_graph2vec
#from diffusion import train_diffusion
from generate import generate_music, test_training_set
from torch.utils.data import random_split
from unet import preprocess_unet, train_unet
from hyperparameter import Hyperparameter as hp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():

    if sys.argv[1] == 'dataset_conformity': # to remove files which has tempo change, no metadata for genre and key.
        if hp.inst_num == 17:
            dataset_conformity_check("lpd_17/lpd_17_cleansed")
        else:
            dataset_conformity_check("lpd_5/lpd_5_cleansed")

    if sys.argv[1] == 'process_CBIR': #data processing for training wasserstein Autoencoder for CBIR
        if hp.inst_num == 17:
            image_process_CBIR_CONLON("conformed_lpd_17/lpd_17_cleansed")
        else:
            image_process_CBIR_CONLON("conformed_lpd_5/lpd_5_cleansed")

    if sys.argv[1] == 'train_CBIR': # training wasserstein Autoencoder
        train_CBIR()

    if sys.argv[1] == 'process_pattern': # data processing for training AE for single track, with pattern CONLON
        if hp.inst_num == 17:
            preprocess_npz("conformed_lpd_17/lpd_17_cleansed",False)
        else:
            preprocess_npz("conformed_lpd_5/lpd_5_cleansed",False)

    if sys.argv[1] == 'train_AE': # training AE for pattern CONLON
        train_AE()
    
    if sys.argv[1] == 'preprocess_npz': # do song structure analysis and conduct song structure graph with single track AE
        if hp.inst_num == 17:
            preprocess_npz("conformed_lpd_17/lpd_17_cleansed",True)
        else:
            preprocess_npz("conformed_lpd_5/lpd_5_cleansed",True)

    
    
    if sys.argv[1] == 'train_graph2vec':
        train_graph2vec()

    if sys.argv[1] == 'preprocess_unet': # get latent from graph and autoencoder. 
        preprocess_unet()

    if sys.argv[1] == 'train_unet':
        train_unet()
        pass

    if sys.argv[1] == 'generate_music': # it gives metrics too.
        generate_music()
        pass

    if sys.argv[1] == 'test_trainingset':
        test_training_set("conformed_lpd_17/lpd_17_cleansed")


    """
    some test codes here

    """

    """
    CONLON = CBIR_dataset()[0]


    for data in CBIR_dataset():
        CONLON = torch.cat((CONLON,data))

    CONLON = np.moveaxis(CONLON.numpy(),-1,0)
    print(CONLON.shape)
    new_CON=[]
    for i in range(17):
        CON=[]
        for j in range(2):
            CON.append(CONLON[i*2+j])
        new_CON.append(CON)


    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    pattern = CONLON_to_patterns(np.array(new_CON),inst_list)
    patterns_to_midi(pattern,inst_list)
    """
    """
     test codes
    """


if __name__=="__main__":
    main()