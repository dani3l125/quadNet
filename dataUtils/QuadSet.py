import os

from torch.utils.data import Dataset
import os.path as path
import csv
import numpy as np
import random
import torch
import pandas as pd

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['roots']
par_path = os.path.join("Data", "quadratic.csv")


class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples, self.roots_n = self.genData(size)

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx], self.roots_n[idx]]

    def genData(self, size=100000):
        if True or not path.exists(par_path) or not path.exists(roots_path):
            if path.exists(par_path):
                os.remove(par_path)
            if path.exists(roots_path):
                os.remove(roots_path)
            with open(par_path, 'w', encoding='UTF8') as p, open(roots_path, 'w', encoding='UTF8') as r:
                writer_r = csv.writer(r)
                writer_r.writerow(header_r)
                writer_p = csv.writer(p)
                writer_p.writerow(header_p)
                for i in range(int(size)):
                    a = random.uniform(r_min, r_max)
                    b = random.uniform(r_min, r_max)
                    c = random.uniform(r_min, r_max)
                    par_l = [a, b, c]
                    root_l = np.roots(par_l)
                    writer_r.writerow(
                        [root_l[0] if np.isreal(root_l[0]) else None, root_l[1] if np.isreal(root_l[1]) else None])
                    writer_p.writerow(par_l)
        inputs = pd.read_csv(par_path).dropna().reset_index(drop=True)
        outputs = pd.read_csv(roots_path).dropna().reset_index(drop=True)
        return inputs, outputs
