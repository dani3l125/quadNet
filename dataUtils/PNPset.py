import os

from torch.utils.data import Dataset
import os.path as path
import csv
import numpy as np
import random
import torch
import pandas as pd
from multiprocessing import Process
import glob
import yaml

with open(r'PNPData/config.yml', 'r') as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

class PNPset(Dataset):
    def __init__(self):
        self.s_paths = glob.glob(cfg["DATASET"]["NAME"]+'/*/*/system.npy')\
                       + glob.glob(cfg["DATASET"]["NAME"]+'/*/system.npy')\
                       + glob.glob(cfg["DATASET"]["NAME"]+'system.npy')
        self.l_paths = glob.glob(cfg["DATASET"]["NAME"] + '/*/*/labels.npy')\
                       + glob.glob(cfg["DATASET"]["NAME"] + '/*/labels.npy')\
                       + glob.glob(cfg["DATASET"]["NAME"] + 'labels.npy')
        self.samples = [torch.tensor(np.load(path)) for path in self.s_paths]
        self.labels = [torch.tensor(np.load(path)) for path in self.l_paths]
        self.dim = self.samples[0].shape[0]
        self.coefficients = self.samples[0].shape[1]

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = PNPset()
    for item in dataset:
        print(item)
