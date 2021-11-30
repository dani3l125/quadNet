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
headers = ['coefficients', 'root1', 'root2']
data_path = os.path.join("Data", "quadratic.csv")


class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples, self.roots = self.getData(size)

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx], self.roots[idx]]

    def genData(self, size=100000):
        if True or not path.exists(data_path):
            if path.exists(data_path):
                os.remove(data_path)
            with open(data_path, 'w', encoding='UTF8') as data:
                writer = csv.writer(data)
                writer.writerow(headers)
                for i in range(int(size)):
                    a = random.uniform(r_min, r_max)
                    b = random.uniform(r_min, r_max)
                    c = random.uniform(r_min, r_max)
                    par_l = [a, b, c]
                    root_l = np.roots(par_l)
                    writer.writerow(
                        [par_l, root_l[0] if np.isreal(root_l[0]) else None,
                         root_l[1] if np.isreal(root_l[1]) else None])

    def getData(self, size=100000):
        data = pd.read_csv(path).dropna().reset_index(drop=True).head(size)
        par = data[:, headers[0]]
        root1 = data[:, headers[1]]
        root2 = data[:, headers[2]]
        return par, np.array(root1, root2).T
