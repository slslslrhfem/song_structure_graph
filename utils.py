
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
from hyperparameter import Hyperparameter as hp
import re

def np_tensor_decoder(np_dict):
    CONLON = np.zeros(np_dict[4])
    for i,j,k,val in zip(np_dict[0],np_dict[1],np_dict[2],np_dict[3]):# time, pitch, channel, value
        if k%2 == 0: # velocity channel
            CONLON[i][j][k]=val#/hp.normalize_factor  #dividing with factor, vel_channel and dur_channel has same mean.
        else: # duration channel
            CONLON[i][j][k]=val 
    return CONLON 

def get_conlon_tensor(pianoroll,file): 
    
    num_note_per_bar = 96
    num_bar = pianoroll.shape[0]//num_note_per_bar
    tensor_list = np.split(pianoroll[:num_bar*num_note_per_bar], num_bar)
    for i,np_tensors in enumerate(tensor_list):
        np_dict = np_tensor_encoder(np_tensors)

        if len(np_dict[0])>0: # empty tensor can occur nan loss
            os.makedirs(os.path.dirname("bar_conlon_image/"+str(os.path.basename(file))[:-4]+'/'+str(i).zfill(4)), exist_ok=True)
            np.save("bar_conlon_image/"+str(os.path.basename(file))[:-4]+'/'+str(i).zfill(4), np_dict)
    tensor_list = np.array(tensor_list)
    return torch.tensor(tensor_list)

def np_tensor_encoder(CONLON):
    idx = list(np.where(CONLON>0))
    values=[]
    for i,j,k in zip(idx[0], idx[1], idx[2]):
        values.append(CONLON[i][j][k])
    idx.append(np.array(values))
    idx.append(CONLON.shape)
    return np.array(idx,dtype=object)
    #module for saving image as tokens, since it has too large size

def get_meta(file,genre_dict):
    #print(file)
    splited = str(file).split('\\')
    msd_name = splited[5]
    lmd_name = splited[6][:-4]
    pm = pretty_midi.PrettyMIDI('datasets/lmd_full/'+lmd_name[0]+'/'+lmd_name+'.mid')

    #Genre labeling
    if msd_name in genre_dict.keys():
        genre = genre_dict[msd_name]
    else: # There is no genre tag for this song in amg, lastfm, tagtraum label, which is given in LPD dataset.
        genre = None
    #Key Signature labeling
    key_numbers = [k.key_number for k in pm.key_signature_changes]
    key_timings = [pm.time_to_tick(k.time)//480 for k in pm.key_signature_changes] #//480 to make measure tick -> bar(in integer)

    return key_numbers, key_timings, genre
    

def dataset_conformity_check(root):
    #Especially, remove songs which has tempo change
    ROOT = Path(root)
    files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.npz'))
    pianoroll = pypianoroll.load(files[0])
    pretty = pypianoroll.to_pretty_midi(pianoroll)
    inst_name=[]
    genre_list = ['country', 'piano', 'rock', 'pop', 'folk', 'electronic', 'rap', 'chill', 'dance', 'jazz', 'rnb', 'reggae', 'house', 'techno', 'trance', 'metal', 'pop_rock']#num of each genre : {'country': 943, 'piano': 406, 'rock': 1915, 'pop': 1242, 'folk': 100,  'electronic': 919,
    #'rap': 196,  'chill': 140, 'favorites': 353, 'dance': 508, 'jazz': 248, 'rnb': 443, 'reggae': 81, 'house': 64, 'techno': 168, 'trance': 79, 'metal': 264, 'pop_rock': 1635}
    #doesn't include some minor genre, ex) 'hardcore' : only 18 song and some 'weird' genre names. ex) 'amazing' or 'favorite'
    file_num=0
    for insts in pretty.instruments:
        inst_name.append(insts.name)
    with open('datasets/genre.pickle', 'rb') as f:
        genre_dict = pickle.load(f)
    for j,file in tqdm(enumerate(files)): 
        key, key_timing, genre = get_meta(file,genre_dict)
        pianoroll = pypianoroll.load(file)
        try:
            if len(key)==0 or genre not in genre_list:
                #print(genre, key)
                #print("metadata is not given, or not given in genre_list.")
                raise
            CONLON = pypianoroll_to_CONLON(pianoroll)    
            """
            iter=0
            for insts in CONLON:
                for channel in insts:
                    if iter==0:
                        conlon_full = np.array(channel)
                    else:
                        conlon_full = np.concatenate((conlon_full, np.array(channel)),axis=1)
                    iter=1
            if j==0:
                tensor_list = get_conlon_tensor(conlon_full)
            else:
                pass
                #tensor_list = torch.cat((tensor_list,get_conlon_tensor(conlon_full)),0)
            """
            filename = "conformed_lpd_17/"+str(file)[7:]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pypianoroll.save(filename,pianoroll)
            file_num+=1
        except Exception as e:
            #print("indexerror can be occur if bpm changes in small portion (ex 120->121), or some issue in resolution change.")
            #print(e)
            pass

    print("conformity check complete! available pianoroll number : ", file_num)


def instrument_name_to_program(inst):
    program_list = [0,0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
    inst_list = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    return program_list[inst_list.index(inst)]

def patterns_to_midi(patterns, inst_name):
    tempo=120
    result_track = pretty_midi.PrettyMIDI()
    instruments=[]
    for inst in inst_name:
        inst_program = instrument_name_to_program(inst)
        if inst == 'Drums':
            instruments.append(pretty_midi.Instrument(program=inst_program,is_drum=True,name=inst))
        else:
            instruments.append(pretty_midi.Instrument(program=inst_program,name=inst))
    for pattern in patterns[:124]:
        program_idx = inst_name.index(pattern.inst)
        conlon = pattern.CONLON
        velocity_channel = conlon[0]
        duration_channel = conlon[1]
        indices = np.where(velocity_channel>20)
        for i in range(len(indices[0])):
            velocity = int(velocity_channel[indices[0][i]][indices[1][i]])
            duration = (duration_channel[indices[0][i]][indices[1][i]]) * 60 / tempo / 24
            start = pattern.start_timing * 60 / tempo / 24 + (indices[0][i] * 60 / tempo / 24) + pattern.pattern_order * 192 * 60 / tempo / 6
            end = start+duration
            pitch = int(indices[1][i])
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            instruments[program_idx].notes.append(note)
    for inst in instruments:
        result_track.instruments.append(inst)
    result_track.write('pretty_test2.mid')
    return result_track
def pypianoroll_to_CONLON(pianoroll):
    #Note that output is (34, N, 128) 34 is for 2 channel and 17 instrument.
    if np.max(pianoroll.tempo)/np.min(pianoroll.tempo)>1.0001:
        #print("Song has multiple tempo")
        raise

    tempo = pianoroll.tempo[0]
    resol = int(pianoroll.resolution) # 24 -> 12, same as 96 notes per bar -> 48 notes per bar
    pretty = pypianoroll.to_pretty_midi(pianoroll)
    n_beats = pretty.get_beats(0)
    
    
    whole_CONLON=[]
    """
    for i in range(17):
        if np.array(pianoroll.tracks[i].pianoroll).shape[0] != 0:
            CONLON_shape = np.array(pianoroll.tracks[i].pianoroll).shape # (N, 128) )
    """
    CONLON_shape = (len(n_beats)*resol,128)
    for i, inst in enumerate(pretty.instruments):
        CONLON=[]
        duration_channel = np.zeros(CONLON_shape)
        velocity_channel = np.zeros(CONLON_shape)
        for note in inst.notes:
            note_start = note.start / 60 * tempo * resol
            note_end = note.end / 60 * tempo * resol
            if round(note_start)>=CONLON_shape[0]:
                velocity_channel[CONLON_shape[0]-1][note.pitch]=note.velocity 
                duration_channel[CONLON_shape[0]-1][note.pitch]=round(note_end-note_start)
            else:
                #print(note.start, tempo, resol, note_start, round(note_start), velocity_channel.shape)
                velocity_channel[round(note_start)][note.pitch]=note.velocity 
                duration_channel[round(note_start)][note.pitch]=round(note_end-note_start)
            
        CONLON.append(velocity_channel)
        CONLON.append(duration_channel)
        whole_CONLON.append(CONLON)
    return np.array(whole_CONLON)
