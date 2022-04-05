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

with open(r'config.yml', 'r') as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

class PNPset(Dataset):
    def __init__(self):
        self.paths = glob.glob(cfg["DATASET"]["NAME"]+'/*/*/system.npy')
        self.samples = [torch.tensor(np.load(path)) for path in self.paths]
        self.dim = self.samples[0].shape[0]
        self.coefficients = self.samples[0].shape[1]

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def getData(self, size=100000):
        return


if __name__ == '__main__':
    dataset = PNPset()
    for item in dataset:
        print(item)
