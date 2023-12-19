#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']="0"
import shutil
import numpy as np
import yaml
import torch
import psutil
from torch import nn
from random import shuffle
from dataloader import multiple_smile_to_hot, multiple_smile_to_hot_v1, multiple_smile_to_hot_v2, grammar_one_hot_to_smile, get_nearest_pairs,timehms, list2dictionary
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt

import deepsmiles
converterDeepSMILES = deepsmiles.Converter(rings=True, branches=True) 
from selfies import decoder as decoderSELFIES
import selfies as sf

from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from datasets import SmilesDataset, SelfiesDataset, SmilesCollate
from functions import read_smiles

import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-nltanh', dest='nltanh', help='Change relu activation function for a tanh in the first layer of encoder', action='store_true',default=False)
parser.add_argument('-scheduler', dest='scheduler',help='Add an scheduler with patience of 2 for encoder and decoder', action='store_true',default=False)
parser.add_argument('-reduction0', dest='reduction0',help='Select OHE', action='store_true',default=False)
parser.add_argument('-reduction1', dest='reduction1',help='Select between reduction1 type OHE', action='store_true',default=False)
parser.add_argument('-reduction2', dest='reduction2',help='Select between reduction2 type OHE', action='store_true',default=False)
parser.add_argument('-directory_name', nargs='?', const='default_directory', help='Name of the directory to create or move files to', default=None)
parser.add_argument('-encoding', nargs='?', const='encoding', help='smiles, deepsmiles or selfies', default=None)
parser.add_argument('-smiles_file', nargs='?', const='smiles_dataset', help='smiles dataset to train', default=None)
parser.add_argument('-dataset_name', nargs='?', const='Dataset_name', help='Dataset_name', default=None)
parser.add_argument('-size', nargs='?', const='Dataset size', help='Dataset size', default=None)
parser.add_argument('-index', nargs='?', const='Index of dataset', help='Index of dataset', default=None)
parser.add_argument('-equivalentModel', dest='equivalentModel',help='Increase neurons on first layer to 350', action='store_true',default=False)

args = parser.parse_args()


def _make_dir(directory):
    os.makedirs(directory)

def log_file_name(type_of_encoding, dataset_name, size, index):
    csv_log_name = './VDR_Type_' + str(type_of_encoding)

    if args.reduction0:
        csv_log_name = csv_log_name + '_v0' # Regular One Hot Encoding (OHE)    
    if args.reduction1:
        csv_log_name = csv_log_name + '_v1' # rOHEv1
    if args.reduction2:
        csv_log_name = csv_log_name + '_v2'  # rOHEv2
    if args.nltanh:
        csv_log_name = csv_log_name + '_tanh' # Change to tanh first act. function from first layer of encoder
    if args.scheduler:
        csv_log_name = csv_log_name + '_scheduler' # Add a scheduler to training
    if args.dataset_name:
        csv_log_name = csv_log_name +'_' + str(args.dataset_name)
    if args.size:
        csv_log_name = csv_log_name +'_' + str(args.size)
    if args.index:
        csv_log_name = csv_log_name +'_' +str(args.index)

    csv_log_name = csv_log_name +'.csv'
    return csv_log_name


def save_models(encoder, decoder, epoch,type_of_encoding, dataset_name, size, index):
    out_dir = './Type_' + str(type_of_encoding) 

    if args.reduction0:
        out_dir = out_dir + '_v0'  # Regular One Hot Encoding (OHE)
    if args.reduction1:
        out_dir = out_dir + '_v1'  # rOHEv1
    if args.reduction2:
        out_dir = out_dir + '_v2'  # rOHEv2
    if args.nltanh:
        out_dir = out_dir + '_tanh'  # Change to tanh first act. function from first layer of encoder
    if args.scheduler:
        out_dir = out_dir + '_scheduler' # Add a scheduler to training
    if args.dataset_name:
        out_dir = out_dir +'_' + str(args.dataset_name)
    if args.size:
        out_dir = out_dir +'_' + str(args.size)
    if args.index:
        out_dir = out_dir +'_' + str(args.index)


    out_dirmove = out_dir + '_saved_models/'
    out_dir = out_dir + '_saved_models/{}'.format(epoch)

    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))
    return out_dirmove



class VAE_encode(nn.Module):
    def __init__(self, layer_1d, layer_2d, layer_3d, latent_dimension,activation):
        """
        Fully Connected layers for the RNN.
        """
        super(VAE_encode, self).__init__()
        # Reduce dimension upto second last layer of Encoder
        self.encode_4d = nn.Sequential(
            nn.Linear(len_max_molec1Hot, layer_1d),
            activation,
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU(),
        )
        
        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)
        
        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)
    
    def reparameterize(self, mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        #print('reparameterize(self, mu, log_var)')
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Go down to dim-4
        h1 = self.encode_4d(x)
        # Go down to dim-2 & produce the mean & variance
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class VAE_decode(nn.Module):
    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,pairs):
        """
        Through Decoder
        """
        super(VAE_decode, self).__init__()
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num
        self.pairs = pairs

        # Simple Decoder
        self.decode_RNN  = nn.GRU(
                input_size  = latent_dimension,
                hidden_size = gru_neurons_num,
                num_layers  = gru_stack_size,
                batch_first = False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, self.pairs[1]*len_alphabet),
        )


    def init_hidden(self, batch_size = 1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size, self.gru_neurons_num)


    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)        # fully connected layer

        return decoded, hidden


def IsCorrectSMILES(smiles):
    try:
        resMol=MolFromSmiles(smiles, sanitize=True)
    except Exception:
        resMol=None

    if resMol==None:
        return 0
    else:
        return 1



def sample_latent_space(latent_dimension):
    model_encode.eval() # 
    model_decode.eval() # 

    fancy_latent_point=torch.normal(torch.zeros(latent_dimension),torch.ones(latent_dimension))
    hidden = model_decode.init_hidden()
    gathered_atoms = []
    for ii in range(len_max_molec):                 
        fancy_latent_point = fancy_latent_point.reshape(1, 1, latent_dimension)
        fancy_latent_point=fancy_latent_point.to(device)
        decoded_one_hot, hidden = model_decode(fancy_latent_point, hidden)
        decoded_one_hot = decoded_one_hot.flatten()
        decoded_one_hot = decoded_one_hot.detach()
        soft = nn.Softmax(0)
        decoded_one_hot = soft(decoded_one_hot)
        _,MaxIdx=decoded_one_hot.max(0)
        gathered_atoms.append(MaxIdx.data.cpu().numpy().tolist())

    model_encode.train()
    model_decode.train()
    return gathered_atoms


def list_to_molecule_string(mollist,current_alphabet):
    molecule=''
    current_alphabet = {v: k for k, v in current_alphabet.items()}
    # print('Mollist',mollist,current_alphabet)
    for ii in mollist:
        molecule+=current_alphabet[ii]    
        molecule=molecule.replace(' ','')
    return(molecule)


def latent_space_quality(latent_dimension, dictionary, sample_num=100, stereochem=False):
    total_samples=0
    total_correct=0
    all_correct_molecules=[];
    print('latent_space_quality: sample_num: ',sample_num)
    while total_samples<=sample_num:
        Molecule=''
        while len(Molecule)==0:
            is_decoding_error=0
            if type_of_encoding==0: # SMILES
                Molecule=list_to_molecule_string(sample_latent_space(latent_dimension),dictionary)

            if type_of_encoding==1: # DeepSMILES
                Molecule=list_to_molecule_string(sample_latent_space(latent_dimension),dictionary)
                try:
                    Molecule=converterDeepSMILES.decode(Molecule)
                except Exception:
                    is_decoding_error=1
                    Molecule='err'

            if type_of_encoding==2: # SELFIES
                Molecule=list_to_molecule_string(sample_latent_space(latent_dimension),dictionary)
                Molecule = decoderSELFIES(Molecule)
                if Molecule == None:
                    Molecule = ''

        total_samples+=1
        if is_decoding_error==0:
            isItCorrect=IsCorrectSMILES(Molecule)
        else:
            isItCorrect=0

        if isItCorrect==1:
            total_correct+=1
            SameMol=0
            for jj in range(len(all_correct_molecules)):
                if Molecule==all_correct_molecules[jj]:
                    SameMol=1
                    break

            if SameMol==0:
                all_correct_molecules.append(Molecule)
    return total_correct, len(all_correct_molecules)

def quality_in_validation_set(data_valid,KLD_alpha):
    x = [i for i in range(len(data_valid))]  # random shuffle input
    shuffle(x)
    data_valid = data_valid[x]

    quality_list=[]
    for batch_iteration in range(min(25,num_batches_valid)):  # batch iterator

        current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size
        inp_smile_hot = data_valid[current_smiles_start : current_smiles_stop]

        inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
        latent_points, mus, log_vars = model_encode(inp_smile_encode)
        latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])

        hidden = model_decode.init_hidden(batch_size = batch_size)
        decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2] * pairs[1]).to(device)
        for seq_index in range(inp_smile_hot.shape[1]):
            decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)
            decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]

        decoded_one_hot = decoded_one_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2] * pairs[1])

        # Recovering target index from input data
        if args.reduction0: 
            _, label_atoms  = inp_smile_hot.max(2)
        if args.reduction1:
            label_atoms = pairs[1]*inp_smile_hot.max(2)[1] + pairs[1]*inp_smile_hot.max(2)[0] - 1
        if args.reduction2:
            label_atoms = pairs[1]*inp_smile_hot.max(2)[1] + torch.log2(inp_smile_hot.max(2)[0]*(2**(pairs[1] - 1)))

        label_atoms = label_atoms.reshape(batch_size * inp_smile_hot.shape[1]).type(torch.LongTensor)
        label_atoms = label_atoms.to(device)

        # assess reconstruction quality
        _, decoded_max_indices = decoded_one_hot.max(1)
        _, input_max_indices   = inp_smile_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2]).max(1)

        differences = 1. - torch.abs(decoded_max_indices - label_atoms)
        differences = torch.clamp(differences, min = 0., max = 1.).double()
        quality     = 100. * torch.mean(differences)
        quality     = quality.detach().cpu().numpy()
        quality_list.append(quality)

    return(np.mean(quality_list))

def train_model(data_train, data_valid, num_epochs, latent_dimension, tensorBoard_graphing, checkpoint, lr_enc, lr_dec, KLD_alpha, sample_num, dictionary, dataset_name, size,index,device):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ',num_epochs)

    if tensorBoard_graphing:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model_encode.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(model_decode.parameters(), lr=lr_dec)

    #  Scheduler
    if args.scheduler:
        scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_encoder, mode='max', factor=0.975, patience=2, verbose=True)
        scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decoder, mode='max', factor=0.975, patience=2, verbose=True)

    data_train = data_train.clone().detach()
    data_train=data_train.to(device)

    # Patience = 20 epochs
    # early_stopping = EarlyStopping(patience=20*num_batches_train)  
    

    quality_valid_list=[0,0,0,0];
    for epoch in range(num_epochs):
        x = [i for i in range(len(data_train))]  # random shuffle input
        shuffle(x)

        #B = [data[iai] for ii in x]         # Shuffled inputs (TODO: unnecesary variable)
        data_train  = data_train[x]
        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            loss, recon_loss, kld = 0., 0., 0.

            current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size
            inp_smile_hot = data_train[current_smiles_start : current_smiles_stop]
            inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
            latent_points, mus, log_vars = model_encode(inp_smile_encode)
            latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])

            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

            hidden = model_decode.init_hidden(batch_size = batch_size)
            decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2] * pairs[1]).to(device)
            for seq_index in range(inp_smile_hot.shape[1]):
                decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)
                decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]

            decoded_one_hot = decoded_one_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2] * pairs[1])
            # Recovering target index from input data
            if args.reduction0: 
                _, label_atoms  = inp_smile_hot.max(2)
            if args.reduction1: 
                label_atoms = pairs[1]*inp_smile_hot.max(2)[1] + pairs[1]*inp_smile_hot.max(2)[0] - 1   
            if args.reduction2:
                label_atoms = pairs[1]*inp_smile_hot.max(2)[1] + torch.log2(inp_smile_hot.max(2)[0]*(2**(pairs[1] - 1)))

            label_atoms  = label_atoms.reshape(batch_size * inp_smile_hot.shape[1]).long()

            criterion   = torch.nn.CrossEntropyLoss()
            recon_loss += criterion(decoded_one_hot, label_atoms)


            loss += recon_loss + KLD_alpha * kld
            if tensorBoard_graphing:
                writer.add_scalar('Batch Loss', loss, epoch*(num_batches_train) + batch_iteration)

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model_decode.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 == 0:
                end = time.time()

                # assess reconstruction quality
                _, decoded_max_indices = decoded_one_hot.max(1)
                _, input_max_indices   = inp_smile_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2]).max(1)

                differences = 1. - torch.abs(decoded_max_indices - label_atoms)
                differences = torch.clamp(differences, min = 0., max = 1.).double()
                quality     = 100. * torch.mean(differences)
                quality     = quality.detach().cpu().numpy()

                qualityValid=quality_in_validation_set(data_valid,KLD_alpha)

                # Early stopping criteria
                # early_stopping(qualityValid)
                new_line = 'Epoch: %d,  Batch: %d / %d,\t(loss_training: %.4f\t| quality: %.4f | quality_valid: %.4f)\tELAPSED TIME: %.5f' % (epoch, batch_iteration, num_batches_train, loss.item(), quality, qualityValid, end - start)
                print(new_line)
                start = time.time()
        

        ################################################################
        #
        ################################################################
        model_encode.eval()
        model_decode.eval()
        with torch.no_grad():
            inp_smile_hot_valid = data_valid  # Asumiendo que los datos de validación ya están en forma de tensor
            inp_smile_encode_valid = inp_smile_hot_valid.reshape(inp_smile_hot_valid.shape[0], inp_smile_hot_valid.shape[1] * inp_smile_hot_valid.shape[2])
            latent_points_valid, mus_valid, log_vars_valid = model_encode(inp_smile_encode_valid)
            latent_points_valid = latent_points_valid.reshape(1, inp_smile_hot_valid.shape[0], latent_points_valid.shape[1])
    
            kld_valid = -0.5 * torch.mean(1. + log_vars_valid - mus_valid.pow(2) - log_vars_valid.exp())
    
            hidden_valid = model_decode.init_hidden(batch_size=inp_smile_hot_valid.shape[0])
            decoded_one_hot_valid = torch.zeros(inp_smile_hot_valid.shape[0], inp_smile_hot_valid.shape[1], inp_smile_hot_valid.shape[2] * pairs[1]).to(device)

            for seq_index in range(inp_smile_hot_valid.shape[1]):
                decoded_one_hot_line_valid, hidden_valid = model_decode(latent_points_valid, hidden_valid)
                decoded_one_hot_valid[:, seq_index, :] = decoded_one_hot_line_valid[0]
    
            decoded_one_hot_valid = decoded_one_hot_valid.reshape(inp_smile_hot_valid.shape[0] * inp_smile_hot_valid.shape[1], inp_smile_hot_valid.shape[2] * pairs[1])

            if args.reduction0:
                _, label_atoms_valid = inp_smile_hot_valid.max(2)
            if args.reduction1: 
                label_atoms_valid = pairs[1] * inp_smile_hot_valid.max(2)[1] + pairs[1] * inp_smile_hot_valid.max(2)[0] -1
            if args.reduction2:
                label_atoms_valid = pairs[1] * inp_smile_hot_valid.max(2)[1] + torch.log2(inp_smile_hot_valid.max(2)[0]*(2**(pairs[1] - 1)))
            
            label_atoms_valid = label_atoms_valid.reshape(inp_smile_hot_valid.shape[0] * inp_smile_hot_valid.shape[1] ).type(torch.LongTensor)
            label_atoms_valid = label_atoms_valid.to(device)

            criterion   = torch.nn.CrossEntropyLoss()     
            recon_loss_valid = criterion(decoded_one_hot_valid, label_atoms_valid)
    
            loss_valid = recon_loss_valid + KLD_alpha * kld_valid
            print('loss_valid',loss_valid.item())
        
        model_encode.train()
        model_decode.train()
        ###########################################################################
            



        qualityValid = quality_in_validation_set(data_valid,KLD_alpha)
        quality_valid_list.append(qualityValid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 30.:
            corr, unique = latent_space_quality(latent_dimension,dictionary,sample_num = sample_num,stereochem=False)
        else:
            corr, unique = -1., -1.

        new_line = 'Validity: %.5f %% | Diversity: %.5f %% | Reconstruction: %.5f %%' % (corr * 100. / sample_num, unique * 100. / sample_num, qualityValid)
        file_line = '%.5f,%.5f,%.5f ' % (corr * 100. / sample_num, unique * 100. / sample_num, qualityValid)

        print(new_line)
        with open(csv_log_name, 'a') as content:
            if epoch ==0:
                content.write('Validity,Diversity,Reconstruction' + '\n')
                content.write(file_line + '\n')
            else:
                content.write(file_line + '\n')

        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        # if quality_valid_list[-1] < 70 and epoch > 10:
        #     break

        if quality_increase > 20:  # increase less than one percent in 3 episodes
            print('Early stopping criteria')
            break

        # if early_stopping.stop:
        #     print('Early stopping criteria')
        #     break

        if quality_increase == 1:
            if checkpoint:
                out_dir = save_models(model_encode, model_encode, epoch,type_of_encoding, dataset_name,size,index)
        
        if args.scheduler:
            scheduler_encoder.step(qualityValid)
            scheduler_decoder.step(qualityValid)
    return out_dir

class EarlyStopping():
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset.

    Note that patience here is measured in steps, rather than in epochs,
    because the size of an epoch will not be consistent if the size of the
    dataset changes.

    Inspired by:
    https://github.com/Bjarten/early-stopping-pytorch
    https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/finetune_lm.py
    """

    def __init__(self, patience=100):
        """
        Args:
            patience: (int) if the validation loss fails to improve for this
              number of consecutive batches, training will be stopped
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.stop = False
        print("Instantiated early stopping with patience=" + str(self.patience))

    def __call__(self, val_loss):
        # Do nothing if early stopping is disabled
        if self.patience > 0:
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss >= self.best_loss:
                # Loss is not decreasing
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
                    # print("Stopping early with best loss " + str(self.best_loss))
            else:
                # Loss is decreasing
                self.best_loss = val_loss
                # Reset counter
                self.counter = 0



if __name__ == '__main__':
    try:
        startt = time.time()
        print("Initial time",startt)
        dataset_name = args.dataset_name
        size = args.size
        index = args.index
        fileSettings = 'settings.yml'
        
        if args.reduction1:
            reduction_type = 'reduced1'
        if args.reduction2:
            reduction_type = 'reduced2'
        if args.reduction0:
            reduction_type = 'regular'

        # type_of_encoding and hyperparametersFile
        if args.encoding == 'smiles':
            print('SMILES Case')
            type_of_encoding = 0
            file_name = args.smiles_file
            dataset = SmilesDataset(smiles_file=args.smiles_file, reduction_type=reduction_type)

        elif args.encoding == 'deepsmiles':
            print('DeepSMILES Case')
            type_of_encoding = 1
            file_name = args.smiles_file
            dataset = SmilesDataset(smiles_file=args.smiles_file, reduction_type=reduction_type)

        elif args.encoding == 'selfies':
            print('SELFIES Case')
            type_of_encoding = 2
            file_name = args.smiles_file
            dataset = SelfiesDataset(selfies_file=args.smiles_file, reduction_type=reduction_type)

        dictionary = dataset.vocabulary.dictionary
        del dictionary['<PAD>']
        del dictionary['SOS']
        del dictionary['EOS']
        encoding_alphabet = list(dictionary) + [' ']

        if args.reduction0:
            pairs, pads = [len(encoding_alphabet),1], 0
        if args.reduction1 or args.reduction2:
            pairs, pads = get_nearest_pairs(len(encoding_alphabet))
        if pads > 0:
            for i in range(pads):
                encoding_alphabet = encoding_alphabet + ['<'+str(i)+'>']
            dictionary = list2dictionary(encoding_alphabet)
            pairs, pads = get_nearest_pairs(len(encoding_alphabet))
        else:
            dictionary = list2dictionary(encoding_alphabet)

        print('dictionary       ',dictionary)
        print('SettingsFile     ',fileSettings)
        print('Type of encoding ',type_of_encoding)
        print('Dataset Name     ',file_name)
        
        
        if os.path.exists(fileSettings):
            user_settings = yaml.load(open(fileSettings, "r"), Loader=yaml.Loader)
            settings = user_settings
        else:
            print("Expected a file settings.yml but didn't find it.")
            print("Create a file with the default settings.")
            print("Please check the file before restarting the code.")
            print()
            exit()

        cuda_device = settings['data']['cuda_device']
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        print('os environ',
              os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.is_available())

        # create log name
        csv_log_name = log_file_name(type_of_encoding,dataset_name,size,index)
        print('Log_metrics_file:', csv_log_name)
        
        content = open(csv_log_name, 'w')
        content.close()


        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(0)
            # Keep track of memory usage before the model is trained
            start_memory = torch.cuda.max_memory_allocated(device=device)
        else:
            device = torch.device("cpu")
            # Keep track of memory usage before the model is trained
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss

        data_parameters = settings['data']
        batch_size = data_parameters['batch_size']

        
        try:
            smiles_list = read_smiles(file_name)
        except:
            df = pd.read_csv(file_name)
            smiles_list = np.asanyarray(df.smiles)

        if type_of_encoding ==2:
            largest_smile_len = max([len(list(sf.split_selfies(smile))) for smile in smiles_list])
        else:
            largest_smile_len = max([len(smile) for smile in smiles_list])
        largest_smile_len = int(10*torch.ceil(torch.tensor([largest_smile_len/10])).item())
        # print('Largest smiles', largest_smile_len, int(1.25*largest_smile_len))
        print('Acquiring data...')
        
        # print('Indexes to OHE', indexes)
        if args.reduction0:
            data = multiple_smile_to_hot(smiles_list, largest_smile_len, encoding_alphabet, type_of_encoding,dictionary)
        if args.reduction1:
            data = multiple_smile_to_hot_v1(smiles_list, largest_smile_len, encoding_alphabet, type_of_encoding,dictionary)
        if args.reduction2:
            data = multiple_smile_to_hot_v2(smiles_list, largest_smile_len, encoding_alphabet, type_of_encoding,dictionary)
        print('Data Acquired.')

        len_max_molec = data.shape[1]
        len_alphabet = data.shape[2]
        len_max_molec1Hot = len_max_molec * len_alphabet
        print('Len Max molecule', len_max_molec)
        print('Len alphabet',len_alphabet)
        print('Len max molecule ohe', len_max_molec1Hot)

        if args.nltanh:
            activation = nn.Tanh()
        else:
            activation = nn.ReLU()

        encoder_parameter = settings['encoder']
        encoder_parameter.update({'activation': activation})


        if args.equivalentModel:
            # l1 = settings['encoder']['layer_1d']
            # l2 = settings['encoder']['layer_2d']
            # layerEquivalent = (len_max_molec * len(encoding_alphabet) + 1 + l2)/(len_max_molec * len_alphabet + 1 + l2)
            # layerEquivalent = int(100 * layerEquivalent)
            # print('Equivalent first layer ',layerEquivalent)
            # 'layer_1d: 545'
            encoder_parameter.update({'layer_1d': 350 })


        decoder_parameter = settings['decoder']
        decoder_parameter.update({'pairs': pairs})
        
        training_parameters = settings['training']
        training_parameters.update({'device': device})

        model_encode = VAE_encode(**encoder_parameter)
        model_decode = VAE_decode(**decoder_parameter)

        total_params_encoder = sum(param.numel() for param in model_encode.parameters() if param.requires_grad)
        total_params_decoder = sum(param.numel() for param in model_decode.parameters() if param.requires_grad)
        
        print('Parameters encoder',total_params_encoder)
        for param_tensor in model_encode.state_dict():
            print(param_tensor, "\t", model_encode.state_dict()[param_tensor].size())
        
        print('\nParameters decoder',total_params_decoder)
        for param_tensor in model_decode.state_dict():
            print(param_tensor, "\t", model_decode.state_dict()[param_tensor].size())   

        model_encode.train()
        model_decode.train()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('*'*17)
        print('Device:',device )
        print('*'*17)

        data = torch.tensor(data, dtype=torch.float).to(device)

        train_valid_test_size=[0.5, 0.5, 0.0]
        x = [i for i in range(len(data))]  # random shuffle input
        shuffle(x)
        data = data[x]
        idx_traintest=int(len(data)*train_valid_test_size[0])
        idx_trainvalid=idx_traintest+int(len(data)*train_valid_test_size[1])
        data_train=data[0:idx_traintest]
        data_valid=data[idx_traintest:idx_trainvalid]
        data_test=data[idx_trainvalid:]

        num_batches_train = int(len(data_train) / batch_size)
        num_batches_valid = int(len(data_valid) / batch_size)

        model_encode = VAE_encode(**encoder_parameter).to(device)
        model_decode = VAE_decode(**decoder_parameter).to(device)
        print("start training")
        out_dir = train_model(data_train=data_train, data_valid=data_valid, **training_parameters,dictionary=dictionary, dataset_name=args.dataset_name,size=args.size,index=args.index)
        print('Out directory:',out_dir)

        

        if device == torch.device("cuda"):
            # Keep track of memory usage after the model is trained
            end_memory = torch.cuda.max_memory_allocated(device=device)
        else:
            process = psutil.Process(os.getpid())
            end_memory = process.memory_info().rss
    

        endd = time.time()
        print("Initial time",startt)
        print("Final time",endd)
        print("Global elapsed time:", endd - startt, 'seconds\nTIME H-M-S:',timehms(endd - startt))
        print("Memory used during training:", end_memory - start_memory, "bytes")

        if args.directory_name:
            directory_name = args.directory_name
    
            # Check if the directory already exists
            if os.path.isdir(directory_name):
                print(f'The directory {directory_name} already exists.')
                # Move the calculated files to the existing directory
                shutil.move(csv_log_name, directory_name)
                shutil.move(out_dir, directory_name)
            else:
                print(
                    f'The directory {directory_name} does not exist. Creating the directory...')
                # Create the directory
                os.mkdir(directory_name)
                # Move the calculated files to the newly created directory
                shutil.move(csv_log_name, directory_name)
                shutil.move(out_dir, directory_name)


    # except Exception as e:
    except AttributeError:
        _, error_message,_ = sys.exc_info()
        print(error_message)

