
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
from Bar_CBIR import Encoder, Decoder, Discriminator, wasserstein_VAE
from utils import pypianoroll_to_CONLON, np_tensor_encoder, np_tensor_decoder, get_meta, midi_pattern_to_CONLON
from collect_npz import *
from graph import GraphDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Subset
from AE import AE, Decoder_AE
torch.manual_seed(123)
import copy
from graph import graph_emb_classifier
from unet import Unet
from collections import Counter
import math
from utils import find_lowest_val_mask_loss_checkpoint, CBIR_checkpoint_path, get_best_unet_checkpoint_path, find_best_checkpoint

def instrument_name_to_program(inst):
    program_list = [0,0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    return program_list[inst_list.index(inst)]

def patterns_to_midi(patterns, inst_name,file_name, masked):
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
    for i, pattern in enumerate(patterns):
        try:
            local_track = pretty_midi.PrettyMIDI()
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
            
            local_program = instrument_name_to_program(inst_name[program_idx])
            if pattern.inst == 'Drums':
                local_inst = pretty_midi.Instrument(program=local_program,is_drum=True, name=inst)
            else:
                local_inst = pretty_midi.Instrument(program=local_program, name=inst)
            
            if pattern.CONLON.shape ==(5,):
                conlon=np_tensor_decoder(pattern.CONLON)
            else:
                conlon = pattern.CONLON.detach().numpy()

            velocity_channel = conlon[0]
            duration_channel = conlon[1]
            indices = np.where(velocity_channel>2) 
            for i in range(len(indices[0])):

                velocity = int(velocity_channel[indices[0][i]][indices[1][i]])
                VA += velocity
                if velocity>127:
                    velocity=127
                if velocity<100:
                    velocity=100 # Use higher value for more clear hearing. For Compute Metric(VA), use velocity value before control

                duration = (duration_channel[indices[0][i]][indices[1][i]]) * 120 / tempo / 24
                DA += duration
                if duration<0.125:
                    duration=0.125 # for prevent some issue in 0 duration notes..
                start = (indices[0][i] * 120 / tempo / 24) + pattern.pattern_order * 48 * 120 / tempo / 6
                end = float(start+duration) 
                local_start = (indices[0][i] * 120 / tempo / 24)
                local_end = float(local_start+duration)
                pitch = int(indices[1][i])
                if pitch not in used_pitches:
                    used_pitches.append(pitch)
                if key_map[pitch%12] ==1:
                    KS+=1 
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
                local_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=local_start, end=local_end)
                instruments[program_idx].notes.append(note)
                local_inst.notes.append(local_note)
            local_track.instruments.append(local_inst)
            folder_name = str(file_name).replace('.midi','')
            local_file_path = str(file_name).replace('.midi','')+"/"+str(pattern.inst)+"_"+str(pattern.pattern_order)+".midi"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            local_track.write(local_file_path)
            ND = len(indices[0])
            UP = len(used_pitches)
            KS = KS / ND
            VA = VA / ND
            DA = DA / ND
            if i in masked: #Metric used for only model-generated pattern with masking
                UPs.append(UP)
                KSs.append(KS)
                NDs.append(ND)
                VAs.append(VA)
                DAs.append(DA)
        except Exception as e: # for ND = 0.
            pass
    for inst in instruments:
        result_track.instruments.append(inst)
    result_track.write(file_name)
    if inst=='Drums':
        Metrics= None
    else:
        Metrics = [np.average(NDs), np.average(UPs), np.average(KSs), np.average(VAs), np.average(DAs)]
        #print(Metrics)
    return result_track, Metrics



def test_training_set(root, graph=True):
    #we use some models for only get 
    ROOT = Path(root)
    files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.npz'))
    if hp.inst_num==17:
        CBIR_checkpoint_dir = "checkpoints/CBIR_checkpoints_17"
    else:
        CBIR_checkpoint_dir = "checkpoints/CBIR_checkpoints_5"


    final_metric=np.array([0,0,0,0,0.0])
    wae = wasserstein_VAE(128,48)
    CBIR_checkpoint = CBIR_checkpoint_path(CBIR_checkpoint_dir)
    checkpoint = torch.load(CBIR_checkpoint)
    wae.load_state_dict(checkpoint["state_dict"])
    encoder = wae.encoder
    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock']
    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)
    all_num=0
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
        song, metrics = patterns_to_midi(patterns, inst_list, "wanttobe3.midi",range(len(patterns)))
        
        if metrics is not None and not math.isnan(metrics[2]):
            final_metric += np.array(metrics)
            all_num += 1
        gc.collect() 
    final_metric = np.array(final_metric)
    print(final_metric/all_num, "ND, UP, KS, VA, DA")
    return 0

def generate_music():

    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)

    if hp.inst_num==17:
        graph_checkpoint_dir = 'checkpoints/graph_checkpoints_17'
        ae_checkpoint_dir = "checkpoints/ae_checkpoints_17/"
        unet_checkpoint_dir = "checkpoints/Unet_checkpoints_17"
        CBIR_checkpoint_dir = "checkpoints/CBIR_checkpoints_17"
    else:
        graph_checkpoint_dir = 'checkpoints/graph_checkpoints_5'
        ae_checkpoint_dir = "checkpoints/ae_checkpoints_5/"
        unet_checkpoint_dir = "checkpoints/Unet_checkpoints_5"
        CBIR_checkpoint_dir = "checkpoints/CBIR_checkpoints_5"

    wae = wasserstein_VAE(128,48)
    CBIR_checkpoint = CBIR_checkpoint_path(CBIR_checkpoint_dir)
    checkpoint = torch.load(CBIR_checkpoint)
    wae.load_state_dict(checkpoint["state_dict"])
    encoder = wae.encoder

    unet = Unet()
    unet_checkpoint_path = get_best_unet_checkpoint_path(unet_checkpoint_dir)
    checkpoint = torch.load(unet_checkpoint_path)
    unet.load_state_dict(checkpoint["state_dict"])

    dataset = GraphDataset()
    train_size = int(len(dataset)*0.99)
    validation_size = len(dataset)-train_size
    train_dataset = Subset(dataset,range(train_size))
    validation_dataset = Subset(dataset,range(train_size, train_size + validation_size))# use non-random split for test generation quality
    valid_loader = GraphDataLoader(validation_dataset, batch_size=1, shuffle=False)#, collate_fn=_collate_fn)
    

    
    best_checkpoint_path = find_best_checkpoint(ae_checkpoint_dir)
    AutoEncoder = AE(hp.AE_num_hiddens, hp.AE_num_residual_layers, hp.AE_num_residual_hiddens,
              hp.AE_embedding_dim)
    checkpoint = torch.load(best_checkpoint_path)
    AutoEncoder.load_state_dict(checkpoint["state_dict"])
    decoder_AE = AutoEncoder._decoder
    
    graph_encoder = graph_emb_classifier(hp.GEM_input_dim, 512, hp.AE_embedding_dim, 19)

    graph_best_checkpoint, lowest_val_mask_loss = find_lowest_val_mask_loss_checkpoint(graph_checkpoint_dir)
    checkpoint = torch.load(graph_best_checkpoint)
    graph_encoder.load_state_dict(checkpoint["state_dict"])

    print(CBIR_checkpoint, unet_checkpoint_path, best_checkpoint_path, graph_best_checkpoint)

    if hp.inst_num==17:
        inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    else:
        inst_list = ['Drums', 'Piano', 'Guitar', 'Bass','Strings']
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock','latin', 'catchy']
    
    
    print("start conditioned generation test..")
    condition_CONLON = midi_pattern_to_CONLON("condition.mid").float()
    condition_CONLON = AutoEncoder._encoder(condition_CONLON.unsqueeze(dim=0))
    final_metric=np.array([0,0,0,0,0.0])
    song_num=0
    for graph in tqdm(valid_loader):
        graph = graph[0]
        valid_nodes = torch.nonzero(graph.ndata['inst'], as_tuple=True)[0]
        edge_feature_count = {}
        for node in valid_nodes:
            src, _ = graph.out_edges(node)
            count = torch.sum(graph.edata['edge_feature'][src] == 3).item()
            edge_feature_count[node.item()] = count
        max_count = max(edge_feature_count.values())
        most_connected_nodes = [node for node, count in edge_feature_count.items() if count == max_count]
        inst_values = [graph.ndata['inst'][node].item() for node in most_connected_nodes]
        counter = Counter(inst_values)

        # Calculate which instrument plays the most repeated melody
        most_common_inst, freq = counter.most_common(1)[0]
        masked = []
        for idx in range(len(graph.ndata['CONLON'])):

            graph.ndata['key'][idx] = 0 # pretty_midi.KeySignature module, key_number(int) : Key number according to [0, 11] Major, [12, 23] minor. For example, 0 is C Major, 12 is C minor.

            if graph.ndata['inst'][idx] == most_common_inst:
                graph.ndata['CONLON'][idx]=condition_CONLON
            else:
                graph.ndata['CONLON'][idx]=torch.zeros_like(graph.ndata['CONLON'][idx])
                masked.append(idx)
        graph_generated = graph_generation(graph, masked, graph_encoder,unet)
        patterns = graph_to_patterns(graph_generated,decoder_AE, inst_list,genre_list)
        song, metrics = patterns_to_midi(patterns, inst_list, "condition/"+str(song_num)+".midi", masked)
        if metrics is not None:
            final_metric += np.array(metrics)
        song_num+=1
        del valid_nodes, edge_feature_count, most_connected_nodes, inst_values
        del graph_generated, patterns, song
        torch.cuda.empty_cache()
        gc.collect()
    final_metric = np.array(final_metric)
    print(final_metric/len(validation_dataset), "ND, UP, KS, VA, DA")
    
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
        song, metrics = patterns_to_midi(patterns, inst_list, "generation/"+str(song_num)+".midi", masked)
        if metrics is not None:
            final_metric += np.array(metrics)
        song_num+=1
        #print(metrics, "ND, UP, KS, VA, DA")
        del graph_generated, patterns, song
        torch.cuda.empty_cache()
        gc.collect()
    final_metric = np.array(final_metric)
    print(final_metric/len(validation_dataset), "ND, UP, KS, VA, DA")
    
# Inpainting Test
    print("start inpainting test..")
    final_metric=np.array([0,0,0,0,0.0])
    song_num=0
    for graph in tqdm(valid_loader):
        graph = graph[0]
        mask_idx = np.random.choice(len(graph.ndata['CONLON']), int(0.3*len(graph.ndata['CONLON'])))
        for idx in mask_idx:
            graph.ndata['CONLON'][idx]=torch.zeros_like(graph.ndata['CONLON'][idx])
        graph_generated = graph_generation(graph, mask_idx, graph_encoder,unet)
        patterns = graph_to_patterns(graph_generated,decoder_AE, inst_list,genre_list)
        song, metrics = patterns_to_midi(patterns, inst_list, "inpainting/"+str(song_num)+".midi",masked)
        if metrics is not None:
            final_metric += np.array(metrics)
        song_num+=1
        del graph_generated, patterns, song
        torch.cuda.empty_cache()
        gc.collect()
    final_metric = np.array(final_metric)
    print(final_metric/len(validation_dataset), "ND, UP, KS, VA, DA")
# Generation with melody condition test
    


def graph_to_patterns(graph, decoder, inst_list,genre_list):
    patterns=[]
    insts = graph.ndata["inst"]
    pattern_orders = graph.ndata["pattern_order"]
    keys = graph.ndata["key"]
    genres = graph.ndata["genre"]
    CONLONs = graph.ndata["CONLON"]

    if hp.inst_num==17:
        inv_normalize = transforms.Normalize(
        mean=[-0.0705/2.6271, -0.0111/0.9481],
        std=[1/2.6271, 1/0.9481])
    else:
        inv_normalize = transforms.Normalize(
        mean=[-0.1332/3.2939, -0.0161/0.6091],
        std=[1/3.2939, 1/0.6091])


    if CONLONs.shape[1:] != (128,192):
        if torch.sum(CONLONs)==0:
            CONLONs = torch.zeros(2,128,192)
        else:
            CONLONs = inv_normalize(decoder(CONLONs))


    for inst, pattern_order, key, genre, CONLON in zip(insts, pattern_orders, keys, genres, CONLONs):
        patterns.append(pattern(inst_list[inst.item()], int(pattern_order.item()), int(pattern_order.item()*4), key.item(), genre_list[genre.item()], CONLON))
    
    return patterns

def drop_all_edges(graph):
    edge_ids = graph.edges(form='uv')
    graph.remove_edges(edge_ids[0])
    return graph

def graph_generation(masked_graph, masked, graph_encoder,unet):
    #masked_graph = drop_all_edges(masked_graph) some tests..
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