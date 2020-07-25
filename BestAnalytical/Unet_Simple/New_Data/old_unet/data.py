import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import sys, os
import matplotlib.pyplot as plt

###################################################################
# This function implements the transformation we do to the data
def transform(x):
    return np.log10(x+1.0) #x**0.25
###################################################################

###################################################################
# This function reads all data, normalizes it, and adds Gaussian Noise
def read_all_data(maps, normalize=True, noise=False):

    # Load estimated HI (input) maps
    HI_esti = np.load('Esti_transform_large.npy')

    # Load HI and DM (True) maps 
    DM_True = np.load('DM_transform_large.npy')
    HI_True = np.load('HI_transform_large.npy')
    
    # Take the log
    DM_True = transform(DM_True)
    HI_True = transform(HI_True)
  
    # normalize data
    if normalize:
        # Mean and std values
        DM_True_mean, DM_True_std = np.mean(DM_True), np.std(DM_True)
        HI_True_mean, HI_True_std = np.mean(HI_True), np.std(HI_True)
        HI_esti_mean, HI_esti_std = np.mean(HI_esti), np.std(HI_esti)
    
        # Normalize Matrices
        DM_True = (DM_True - DM_True_mean)/DM_True_std
        HI_True = (HI_True - HI_True_mean)/HI_True_std
        HI_esti = (HI_esti - HI_esti_mean)/HI_esti_std

    # Convert to torch tensor
    DM_True = torch.tensor(DM_True, dtype=torch.float)
    HI_True = torch.tensor(HI_True, dtype=torch.float)
    HI_esti = torch.tensor(HI_esti, dtype=torch.float)

    return DM_True, HI_True, HI_esti
###################################################################

###################################################################
# Define Dataset 
class make_Dataset(Dataset):
    
    def __init__(self, name, seed, maps, DM0, HI0, HI1):
         
        # shuffle the maps (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        np.random.seed(seed)
        indexes = np.arange(maps)
        np.random.shuffle(indexes)

        if   name=='train':  size, offset = int(maps*0.8), int(maps*0.0)
        elif name=='valid':  size, offset = int(maps*0.1), int(maps*0.8)
        elif name=='test':   size, offset = int(maps*0.1), int(maps*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, 2, 32, 32), dtype=torch.float)
        self.output = torch.zeros((size, 1, 32, 32), dtype=torch.float)
        
        # do a loop over all elements in the dataset
        for i in range(size):
            
            # find the index of the map
            j = indexes[i+offset]

            # load maps
            self.input [i,0,:,:] = DM0[j]
            self.input [i,1,:,:] = HI1[j]
            self.output[i,0,:,:] = HI0[j]
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
###################################################################

###################################################################
# This function creates the different datasets
def create_datasets(name, seed, maps, DM0, HI0, HI1, batch_size):

    data_set     = make_Dataset(name, seed, maps, DM0, HI0, HI1)
    data_loader  = DataLoader(dataset=data_set, batch_size=batch_size, 
                               shuffle=True)
    return data_loader
###################################################################

###################################################################
# This function creates the test set for making images
def create_testset_images(seed, maps, DM0, HI0, HI1):

    test_Dataset  = make_Dataset('test', seed, maps, DM0, HI0, HI1)
    test_loader   = DataLoader(dataset=test_Dataset,  batch_size=1, 
                               shuffle=False)

    return test_loader
###################################################################
