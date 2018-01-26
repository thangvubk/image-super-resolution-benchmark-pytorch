from __future__ import division
import torch
import os
import h5py
import glob
import math
import numpy as np
from torch.utils.data import Dataset


class Trainset(Dataset):
    def __init__(self, path):
        f = h5py.File(path)
        self.inputs = np.array(f.get('data'))
        self.labels = np.array(f.get('label'))
        #self.inputs = self.inputs[0:10000]
        #self.labels = self.labels[0:10000]
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        inp = self.inputs[idx]
        lbl = self.labels[idx]
        inp = inp - 0.5
        lbl = lbl - 0.5
        return torch.from_numpy(inp).float(), torch.from_numpy(lbl).float() 
        #return inp, lbl

