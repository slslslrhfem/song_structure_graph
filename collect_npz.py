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
from hyperparameter import Hyperparameter as hp
import torch
from numba import njit
from Bar_CBIR import Encoder, Decoder, Discriminator, wasserstein_VAE
from utils import pypianoroll_to_CONLON, np_tensor_encoder, np_tensor_decoder, get_meta
from scipy.signal import argrelextrema
import dgl
import cv2
import gc
from AE import AE
import re
from functools import partial
import parmap
import multiprocessing as mp
from utils import CBIR_checkpoint_path, find_best_checkpoint
 
def preprocess_npz(root, graph=True):
    error_count=0
    ROOT = Path(root)
    device = torch.device('cuda')
    files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.npz'))
    wae = wasserstein_VAE(128,48)
    if hp.inst_num==17:
        CBIR_checkpoint_dir = "CBIR_checkpoints_17/"
        ae_checkpoint_dir = "ae_checkpoints_17/"
    else:
        CBIR_checkpoint_dir = "CBIR_checkpoints_5/"
        ae_checkpoint_dir = "ae_checkpoints_5/"
    CBIR_checkpoint = CBIR_checkpoint_path(CBIR_checkpoint_dir)
    checkpoint = torch.load("CBIR_checkpoints_17/epoch=07-recon_loss=1.048136-check.ckpt")
    wae.load_state_dict(checkpoint["state_dict"])
    encoder = wae.encoder
    encoder.eval()

    # 체크포인트 디렉토리 설정
    best_checkpoint_path = find_best_checkpoint(ae_checkpoint_dir)
    print("best,", best_checkpoint_path)

    if hp.inst_num==17:
        inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    else:
        inst_list = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock','latin', 'catchy']
    graph_list=[]
    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)
    
    if graph:
        for file in tqdm(files):
            npz_process(file, encoder, genre_dict, genre_list, inst_list, graph, best_checkpoint_path)
    else:
        parmap.map(partial(npz_process, encoder=encoder, genre_dict=genre_dict, genre_list=genre_list, inst_list=inst_list, graph=graph, device=device), files, pm_pbar=True, pm_processes=int(mp.cpu_count()/2))
    #multiprocess with GPU is not good idea, and we use GPU when constructing graph.
    return 0

def npz_process(file, encoder, genre_dict, genre_list, inst_list, graph, best_checkpoint_path):
    if hp.inst_num==17:
        filename = "processed_pattern_with_order_17/"+str(file)[:-4]+".bin"
    else:
        filename = "processed_pattern_with_order_5/"+str(file)[:-4]+".bin"
    pianoroll = pypianoroll.load(file)
    #pypianoroll.write("test.midi",pianoroll)
    CONLON = pypianoroll_to_CONLON(pianoroll)
    start_timings, SMM_CBIR = CONLON_to_starttiming(CONLON,encoder)
    #print(start_timings, "this is start_timing for first track, filename is ", filename)
    key_number, key_timing, genre = get_meta(file,genre_dict)
    patterns = CONLON_to_patterns(CONLON,inst_list, key_number, key_timing, genre, start_timings = start_timings)
    
    if graph:
        try: # dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got k and k-1 instead. for some case. there is some case that returns node feature's shape not correctly. 
            pattern_graph = patterns_to_pattern_graph(patterns,CONLON, SMM_CBIR,inst_list,genre_list, best_checkpoint_path)
            save_graph(pattern_graph, filename, True)
        except Exception as e:
            print(e)
    else: 
        #save_pattern_midi(patterns,filename) # For training REMI DAAE
        save_pattern_image(patterns,filename) # For training CONLON AE
    del filename, pianoroll, CONLON, start_timings, SMM_CBIR, patterns
    gc.collect()

def save_pattern_image(patterns,filename):
    for i,np_dict in enumerate(patterns):
        if len(np_dict.CONLON[0])>0: # empty tensor
            #if np_dict.inst != 'Drums':
            os.makedirs(os.path.dirname("pattern_conlon_image_5/"+str(os.path.basename(filename))[:-4]+'/'+str(i).zfill(4)), exist_ok=True)
            np.save("pattern_conlon_image_5/"+str(os.path.basename(filename))[:-4]+'/'+str(i).zfill(4), np_dict.CONLON)


def save_graph(pattern_graph, filename, save_pattern_image=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dgl.save_graphs(filename,pattern_graph)
    """
    if save_pattern_image:
        for i,np_tensors in enumerate(pattern_graph.ndata['CONLON']):
            np_dict = np_tensor_encoder(np_tensors)

            if len(np_dict[0])>0: # empty tensor can occur nan loss
                os.makedirs(os.path.dirname("pattern_conlon_image/"+str(os.path.basename(filename))[:-4]+'/'+str(i).zfill(4)), exist_ok=True)
                np.save("pattern_conlon_image/"+str(os.path.basename(filename))[:-4]+'/'+str(i).zfill(4), np_dict)
    """



def patterns_to_pattern_graph(patterns, CONLON,CBIR,inst_list,genre_list, best_checkpoint_path):
    u,v, features = get_edges(patterns,CBIR) # (u,v,f), which is tensor of from node u, to node v, edge feature f.
    CONLONs = []
    insts = []
    genres = []
    keys = []
    start_timings = []
    pattern_orders = []

    for pattern in patterns:
        #if pattern.inst != 'Drums': 큰 성능 향상 없음
        CONLONs.append(torch.tensor(np_tensor_decoder(pattern.CONLON)))
        insts.append(inst_list.index(pattern.inst))
        keys.append(pattern.key)
        genres.append(genre_list.index(pattern.genre))
        start_timings.append(pattern.start_timing)
        pattern_orders.append(pattern.pattern_order)
    max_order = max(pattern_orders)
    max_orders = [max_order for i in range(len(patterns))]
    AutoEncoder = AE(hp.AE_num_hiddens, hp.AE_num_residual_layers, hp.AE_num_residual_hiddens,
              hp.AE_embedding_dim)
    
    checkpoint = torch.load(best_checkpoint_path)
    AutoEncoder.load_state_dict(checkpoint["state_dict"])
    CONLONs = torch.stack(CONLONs,dim=0).to(torch.float)
    CONLONs = AutoEncoder(CONLONs)[0]

    g=dgl.graph((u,v))
    g.edata['edge_feature'] = features#torch.stack(features,dim=0) #features for relgraph, stack for GINEconv
    g.ndata['CONLON'] = CONLONs
    g.ndata['inst'] = torch.tensor(insts)
    g.ndata['key'] = torch.tensor(keys)
    g.ndata['genre'] = torch.tensor(genres)
    g.ndata['start_timing'] = torch.tensor(start_timings) # need for last generation. it means actual time of pattern. 
    g.ndata['pattern_order'] = torch.tensor(pattern_orders) # need for last generation & positional embedding.
    g.ndata['max_order'] = torch.tensor(max_orders) # just for easy implementation of positional encoding.
    #for generation process, pattern order * pattern_length = start_timing.
    #g.ndata['trackroll'] -> can be added with heuristic manner, which is important in complimantary music genertaion.

    return g

def get_edges(patterns, CBIR):
    from_node=[]
    to_node=[]
    feature=[]
    #edges=[]#array of [from_node, to_node, feature]. feature is one hot vector.

    #edge1, edge2, edge3, edge4 = 0,0,0,0
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)): # for all case, i<j
            #for relGraph
            if patterns[i].pattern_order == patterns[j].pattern_order: # Same Time Edge
                from_node.append(i)
                to_node.append(j)
                feature.append(0)
                from_node.append(j)
                to_node.append(i)
                feature.append(0)
                #edge1+=1
            if patterns[i].pattern_order+1 == patterns[j].pattern_order and patterns[i].inst == patterns[j].inst: # Same inst flow Edge
                from_node.append(i)
                to_node.append(j)
                feature.append(1)# Same inst flow Edge is directed edge.
                
                #edge2+=1
            if check_same_song_structure(patterns[i],patterns[j],CBIR):
                from_node.append(i)
                to_node.append(j)
                feature.append(2)
                from_node.append(j)
                to_node.append(i)
                feature.append(2)
                #edge3+=1
            if check_same_contour(patterns[i],patterns[j]):
                from_node.append(i)
                to_node.append(j)
                feature.append(3)
                from_node.append(j)
                to_node.append(i)
                feature.append(3)
                #edge4+=1
            
    return from_node, to_node, torch.tensor(feature)# feature

def check_same_song_structure(pattern1, pattern2, CBIR):
    if pattern1.pattern_order == pattern2.pattern_order:
        return 0
    else:
        sim=0
        for i in range(hp.pattern_length):
            if pattern1.start_timing+i >= len(CBIR) or pattern2.start_timing+i >= len(CBIR):# can occur in outro, which has shorter length than normal pattern length.
                pass
            else:
                sim += CBIR[pattern1.start_timing+i][pattern2.start_timing+i]
        if sim > 0.8*hp.pattern_length and pattern1.inst==pattern2.inst: # sim=1*hp.pattern_length(-epsilon) for exactly same pattern
            return 1
        return 0

def check_same_contour(pattern1, pattern2):
    pattern1_conlon = np_tensor_decoder(pattern1.CONLON)[1]#use only duration channel for contour matching
    pattern2_conlon = np_tensor_decoder(pattern2.CONLON)[1]#use only duration channel
    contour_similarity = cv2.matchShapes(pattern1_conlon, pattern2_conlon,cv2.CONTOURS_MATCH_I3,0)
    if contour_similarity<0.1:
        return 1
    return 0

def CONLON_to_starttiming(CONLON,encoder):

    get_similarity_njit = njit(get_similarity)
    get_novelty_curve_njit = njit(get_novelty_curve)
    pattern_length = hp.pattern_length
    
    CONLON_tensor_list = torch.tensor(get_tensor_list(CONLON),dtype=torch.float32)
    
    CONLON_latent = encoder(CONLON_tensor_list)
    
    similarity_CBIR = get_similarity_njit(CONLON_latent.detach().numpy())
    #similarity_CBIR = 1 - (similarity_CBIR / (np.max(similarity_CBIR) + 1e-5)) # mse similarity
    similarity_CBIR = similarity_CBIR / (np.max(similarity_CBIR) + 1e-5) # cosine similarity
    #plt.imshow(similarity_CBIR, interpolation=None, )
    #plt.savefig("CBIR_cosine_fig")
    novelty_curve = get_novelty_curve_njit(similarity_CBIR, pattern_length, method='radially')
    plt.plot(novelty_curve)
    plt.savefig("novelty_curve")
    localmaxima = argrelextrema(np.array(novelty_curve), np.greater)[0]
    starting_points=[0]
    
    novelty_avg = np.average(novelty_curve)
    for maxima in localmaxima:
        if novelty_curve[maxima]>novelty_avg*0.5: 
            starting_points.append(maxima)  # add some pattern starting point with novelty approach
    close_points = np.where(np.diff(starting_points) < pattern_length)[0]
    starting_points = np.delete(starting_points, close_points) # remove some pattern starting points with close points.

    add_points=[]
    starting_points = np.append(starting_points,len(novelty_curve))

    for i,maxima in enumerate(starting_points[:-1]): # add some pattern starting points with repetitive approach. 
        #if verse has length 16 in time 4~20, Then starting points has 4 and 20 but not 8, 12, 16(with 4 pattern length.). 
        # we can add them if point 8, 12, 16 is repetition(or similar) points of 4. 
        while maxima + 2*pattern_length <=starting_points[i+1]:
            if similarity_CBIR[maxima][maxima + pattern_length]>0.5:
                add_points.append(maxima+pattern_length)
            maxima = maxima + pattern_length
    starting_points = np.sort(np.concatenate((starting_points, np.array(add_points))))
    return starting_points, similarity_CBIR

def get_novelty_curve(smm, pattern_length, method='none'):
    pattern_length = int(pattern_length)
    total_time_length = smm.shape[0]
    novelty_curve=[]
    dist = np.arange(-pattern_length, pattern_length+1,1)
    full_size_kernel = np.zeros((pattern_length*2+1, pattern_length*2+1))
    for i in range(pattern_length*2+1):
        for j in range(pattern_length*2+1):
            std = np.std(dist)
            p = i-pattern_length
            q = j-pattern_length
            i_sign = np.sign(p)
            j_sign = np.sign(q)
            if method == 'radially':
                value_function = np.exp( -1/(2*pattern_length)*(np.square(p)+np.square(q)) )
            else:
                value_function = 1
            full_size_kernel[i][j]=i_sign*j_sign*value_function
    for i in range(total_time_length):
        if i<pattern_length:
            #std = np.std(dist[pattern_length-i:])
            kernel = full_size_kernel[pattern_length-i:,pattern_length-i:]  * np.exp(-np.square(std))
            view = smm[:i+pattern_length+1,:i+pattern_length+1]
            novelty_curve.append(np.sum(kernel*view))

        elif i+pattern_length>=total_time_length:
            #std = np.std(dist[:pattern_length+total_time_length-i])
            kernel = full_size_kernel[:pattern_length+total_time_length-i,:pattern_length+total_time_length-i]  * np.exp(-np.square(std))
            view = smm[i-pattern_length:,i-pattern_length:]
            novelty_curve.append(np.sum(kernel*view))

        else:
            std = np.std(dist)
            kernel = full_size_kernel * np.exp(-np.square(std))
            view = smm[i-pattern_length:i+pattern_length+1, i-pattern_length:i+pattern_length+1]
            novelty_curve.append(np.sum(kernel*view))

    return novelty_curve


    
def get_tensor_list(CONLON):
    conlon_full=[]
    for insts in CONLON:
        for channel in insts:
            conlon_full.append(channel)
    #print(np.array(conlon_full).shape)
    conlon_full = np.moveaxis(np.array(conlon_full),0,-1)
    #print(conlon_full.shape)

    num_note_per_bar = 48
    num_bar = conlon_full.shape[0]//num_note_per_bar
    tensor_list = np.split(conlon_full[:num_bar*num_note_per_bar], num_bar)

    return np.transpose(tensor_list,(0,3,1,2)) # batch, channel, time, pitch

@njit
def dot(x, y):
    s = 0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s

def get_similarity(CONLON_latent):
    latent_length = CONLON_latent.shape[1]
    num_bar = len(CONLON_latent)
    
    corr_list = []
    for i in range(num_bar):
        #for j in range(i+1, num_bar):
        adding_list=[]
        for j in range(num_bar):
            former = CONLON_latent[i]
            latter = CONLON_latent[j]
            #adding_list.append((np.square(former-latter)).mean()) #mse similarity. can be changed with cosine similarity.
            adding_list.append(dot(former, latter)/(2*np.linalg.norm(former)*np.linalg.norm(latter))+1/2) # cosine similarity + 1, to make [-1,1] -> [0,1]
        corr_list.append(adding_list)
    
    
    return corr_list


def get_json():
    json_data = json.loads("midi_info.json")
    print(json_data)

def CONLON_to_patterns(whole_CONLON,inst_name, key_number, key_timing, genre ,start_timings):
    #whole_CONLON -> (# of insts, 2, time, pitch)
    patterns=[]
    pattern_length =  48 * 4 #resolution of bar *  # of bar. Default CONLON resolution is 96.
    for inst, CONLON in enumerate(whole_CONLON):
        #if inst_name[inst] != 'Drums': # 큰 성능 향상은 없다.
        for i,start_timing in enumerate(start_timings):
            for key_idx, timings in enumerate(key_timing): # if key changes at 0, 48, 60 with C, B, C, key should be B with start timing 48 ~ 59. pattern with start timing 57~59 can include both keys, but it's rare case for this set.
                if start_timing>=timings:
                    key = key_number[key_idx]
            try:
                key
            except NameError:
                key =0  # if key_timing not starting with zero(which is not expected value in MIDI..)

            start_index = int(start_timing*96)
            end_index = int(start_timing*96+pattern_length)
            available_pattern = CONLON[0][start_index:end_index,:]
            if np.sum(available_pattern) != 0:
                if available_pattern.shape[0] == pattern_length:
                    encoded_conlon = np_tensor_encoder(CONLON[:,start_index:end_index,:])
                else: # if outro has shorter length than pattern_length.
                    before_pad = CONLON[:,start_index:end_index,:]
                    padded_CONLON = np.zeros((2,pattern_length,128))
                    padded_CONLON[:before_pad.shape[0],:before_pad.shape[1],:before_pad.shape[2]]=before_pad
                    encoded_conlon = np_tensor_encoder(padded_CONLON)
                patterns.append(pattern(inst_name[inst],i, int(start_timing), key, genre, encoded_conlon))    
    return patterns