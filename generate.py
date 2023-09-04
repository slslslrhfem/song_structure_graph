
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pypianoroll
import pretty_midi
from pretty_midi import PrettyMIDI as pm
import pickle
from matplotlib import pyplot as plt
from pattern_classes import pattern
from Bar_CBIR import Encoder, Decoder, Discriminator
from utils import pypianoroll_to_CONLON, np_tensor_encoder, np_tensor_decoder, get_meta
from collect_npz import *
from graph import GraphDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Subset
from AE import AE, Decoder_AE
torch.manual_seed(123)
import copy
from graph import graph_emb_classifier
from unet import Unet

def instrument_name_to_program(inst):
    program_list = [0,0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    return program_list[inst_list.index(inst)]

def patterns_to_midi(patterns, inst_name,file_name):
    tempo=120
    result_track = pretty_midi.PrettyMIDI()
    instruments=[]
    UPs = []
    KSs = []
    NDs = []
    VAs = []
    DAs = []
    for inst in inst_name:
        inst_program = instrument_name_to_program(inst)
        if inst == 'Drums':
            instruments.append(pretty_midi.Instrument(program=inst_program,is_drum=True,name=inst))
        else:
            instruments.append(pretty_midi.Instrument(program=inst_program,name=inst))
    for pattern in patterns:
        try:
            key_map = [1,0,1,0,1,1,0,1,0,1,0,1]
            key = pattern.key
            if key<12:
                key_map = np.roll(key_map,key)
            else:
                key_map = np.roll(key_map,key+3) #나중에 방향 보셈 아마 맞음
            used_pitches=[]
            KS = 0
            VA = 0
            DA = 0
            program_idx = inst_name.index(pattern.inst)
            
            if pattern.CONLON.shape ==(5,):
                conlon=np_tensor_decoder(pattern.CONLON)
            else:
                conlon = pattern.CONLON.detach().numpy()

            velocity_channel = conlon[0]
            duration_channel = conlon[1]
            indices = np.where(velocity_channel>1)
            for i in range(len(indices[0])):
                velocity = int(velocity_channel[indices[0][i]][indices[1][i]])
                if velocity>127:
                    velocity=127
                VA += velocity
                duration = (duration_channel[indices[0][i]][indices[1][i]]) * 60 / tempo / 24
                if duration<0.125:
                    duration=0.125 # for prevent some issue in 0 duration notes..
                DA += duration
                start = (indices[0][i] * 60 / tempo / 24) + pattern.pattern_order * 96 * 60 / tempo / 6
                end = float(start+duration) # why start+duration returns torch tensor..
                pitch = int(indices[1][i])
                if pitch not in used_pitches:
                    used_pitches.append(pitch)
                if key_map[pitch%12] ==1:
                    KS+=1 
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
                instruments[program_idx].notes.append(note)
            
            ND = len(indices[0])
            UP = len(used_pitches)
            KS = KS / ND
            VA = VA / ND
            DA = DA / ND
            UPs.append(UP)
            KSs.append(KS)
            NDs.append(ND)
            VAs.append(VA)
            DAs.append(DA)
        except: # for ND = 0.
            pass
    for inst in instruments:
        result_track.instruments.append(inst)
    result_track.write(file_name)
    if inst=='Drums':
        Metrics= None
    else:
        Metrics = [np.average(NDs), np.average(UPs), np.average(KSs), np.average(VAs), np.average(DAs)]
    return result_track, Metrics



def test_training_set(root, graph=True):
    ROOT = Path(root)
    files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.npz'))

    final_metric=np.array([0,0,0,0,0.0])
    encoder, decoder, discriminator = Encoder(hp,128,96), Decoder(hp,128,96), Discriminator(hp) # use only encoder, this encoder is for generate song structure graph.
    encoder.load_state_dict(torch.load("models/encoder20dim_h64latent32chan34.h5"))
    encoder.eval()
    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock']
    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)
    for file in tqdm(files[:1000]): 
        #print(file)
        filename = "processed_pattern_with_order/"+str(file)[:-4]+".bin"
        pianoroll = pypianoroll.load(file)
        #pypianoroll.write("test.midi",pianoroll)
        CONLON = pypianoroll_to_CONLON(pianoroll)
        start_timings, SMM_CBIR = CONLON_to_starttiming(CONLON,encoder)
        #print(start_timings, "this is start_timing for first track, filename is ", filename)
        key_number, key_timing, genre = get_meta(file,genre_dict)
        patterns = CONLON_to_patterns(CONLON,inst_list, key_number, key_timing, genre, start_timings = start_timings)
        song, metrics = patterns_to_midi(patterns, inst_list, "wanttobe3.midi")
        #print(metrics)
        if metrics is not None:
            final_metric += np.array(metrics)
        gc.collect()
    final_metric = np.array(final_metric)
    print(final_metric/1000, "ND, UP, KS, VA, DA")
    return 0

def generate_music():

    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)

    encoder, decoder, discriminator = Encoder(hp,128,96), Decoder(hp,128,96), Discriminator(hp) # use only encoder, this encoder is for generate song structure graph.
    encoder.load_state_dict(torch.load("models/encoder20dim_h64latent32chan34.h5"))
    encoder.eval()

    unet = Unet()
    checkpoint = torch.load("Unet_checkpoints/unet_epoch=40_val_loss=0.72.ckpt")
    unet.load_state_dict(checkpoint["state_dict"])

    dataset = GraphDataset()
    train_size = int(len(dataset)*0.99)
    validation_size = len(dataset)-train_size
    train_dataset = Subset(dataset,range(train_size))
    validation_dataset = Subset(dataset,range(train_size, train_size + validation_size))# use non-random split for test generation quality
    valid_loader = GraphDataLoader(validation_dataset, batch_size=1, shuffle=False,num_workers=15)#, collate_fn=_collate_fn)
    AutoEncoder = AE(hp.vq_num_hiddens, hp.vq_num_residual_layers, hp.vq_num_residual_hiddens,
              hp.vq_embedding_dim, 6.355786)
    checkpoint = torch.load("ae_checkpoints/ae-epoch=65-val_loss=0.10.ckpt")
    AutoEncoder.load_state_dict(checkpoint["state_dict"])
    decoder_AE = AutoEncoder._decoder
    graph_encoder = graph_emb_classifier(hp.GEM_input_dim, 512, 256, 17)
    checkpoint = torch.load("graph_checkpoints/no_bn_2layer_graph_posenc-epoch=40-acc=0.33-val_mask_loss=1.03-val_loss=1.36.ckpt")
    graph_encoder.load_state_dict(checkpoint["state_dict"])

    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock']

    
# Generation Test
    print("start generation test..")
    song_num=0
    final_metric=np.array([0,0,0,0,0.0])
    for graph in tqdm(valid_loader):
        masked=[]
        graph = graph[0] # unbatch, loader's batch_size is 1
        for i in range(len(graph.ndata['CONLON'])):
            if graph.ndata['pattern_order'][i]!=0 and graph.ndata['pattern_order'][i]!=1:#if not first/second pattern
                graph.ndata['CONLON'][i]=torch.zeros_like(graph.ndata['CONLON'][i])
                masked.append(i)
        graph_generated = graph_generation(graph, masked, graph_encoder,unet)
        patterns = graph_to_patterns(graph_generated,decoder_AE, inst_list,genre_list)
        song, metrics = patterns_to_midi(patterns, inst_list, "generation/"+str(song_num)+".midi")
        if metrics is not None:
            final_metric += np.array(metrics)
        song_num+=1
    final_metric = np.array(final_metric)
    print(final_metric/len(validation_dataset), "ND, UP, KS, VA, DA")
    
# Inpainting Test
    print("start inpainting test..")
    final_metric=np.array([0,0,0,0,0.0])
    song_num=0
    for graph in tqdm(valid_loader):
        graph = graph[0]
        mask_idx = np.random.choice(len(graph.ndata['CONLON']), int(0.3*len(graph.ndata['CONLON'])))
        target = copy.deepcopy(graph.ndata['CONLON'][mask_idx])
        for idx in mask_idx:
            graph.ndata['CONLON'][idx]=torch.zeros_like(graph.ndata['CONLON'][idx])
        graph_generated = graph_generation(graph, mask_idx, graph_encoder,unet)
        patterns = graph_to_patterns(graph_generated,decoder_AE, inst_list,genre_list)
        song, metrics = patterns_to_midi(patterns, inst_list, "inpainting/"+str(song_num)+".midi")
        if metrics is not None:
            final_metric += np.array(metrics)
        song_num+=1
    final_metric = np.array(final_metric)
    print(final_metric/len(validation_dataset), "ND, UP, KS, VA, DA")



def graph_to_patterns(graph, decoder, inst_list,genre_list):
    patterns=[]
    insts = graph.ndata["inst"]
    pattern_orders = graph.ndata["pattern_order"]
    keys = graph.ndata["key"]
    genres = graph.ndata["genre"]
    CONLONs = graph.ndata["CONLON"]
    if CONLONs.shape[1:] != (128,384):
        if torch.sum(CONLONs)==0:
            CONLONs = torch.zeros(2,128,384)
        else:
            CONLONs = decoder(CONLONs)

    for inst, pattern_order, key, genre, CONLON in zip(insts, pattern_orders, keys, genres, CONLONs):
        patterns.append(pattern(inst_list[inst.item()], int(pattern_order.item()), int(pattern_order.item()*4), key.item(), genre_list[genre.item()], CONLON))
    
    return patterns

    
def graph_generation(masked_graph, masked, graph_encoder,unet):
    for masked_idx in tqdm(masked):
        CONLON = masked_graph.ndata['CONLON'].to(torch.float32) 
        img_feature = CONLON#torch.flatten(CONLON,1)#self.resnet(CONLON)##s(CONLON)#CONLON
        inst = graph_encoder.inst_emb(masked_graph.ndata['inst'])
        key = graph_encoder.key_emb(masked_graph.ndata['key'])
        fix_class = torch.cat([inst,key],dim=1).to(torch.float32)

        #inst_emb = self.instrument_embedding(inst)
        #key_emb = self.key_label_embedding(key)
        class_feature = graph_encoder.image_bn(fix_class) # this is node feature, but always given for training & inference
        img_feature = graph_encoder.image_bn(img_feature) # this is node feature, but partially masked for training, and generated from scratch autoregressively in inference.

        feature = class_feature + img_feature # this is node feature, adding fixed feature and partially masked feature
        position_emb = graph_encoder.position_emb(masked_graph.ndata['pattern_order'])   #this is position of node feature.
        feature = torch.cat([feature, position_emb], dim=1).to(torch.float32) #use concat for position embedding. 
        efeat = masked_graph.edata['edge_feature']
        #efeat = self.efeat_linear(efeat.to(torch.float32))
        masked_graph.ndata['CONLON'][masked_idx]=unet(torch.unsqueeze(torch.unsqueeze(graph_encoder(masked_graph, feature, efeat)[0][masked_idx],dim=0),dim=0)) # fill masked part one by one to control generate process autoregressively.
        #graph_encoder(masked_graph, feature, efeat)[0][masked_idx]# -> no unet
    return masked_graph