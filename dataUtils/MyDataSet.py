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
headers = ['a', 'b', 'c', 'root1', 'root2']
data_path = os.path.join("Data", "quadratic.csv")


class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples, self.roots = self.getData(size)

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx], self.roots[idx]]

    def getData(self, size=100000):
        data = pd.read_csv(data_path).dropna().reset_index(drop=True).head(size)
        par = np.array([data.loc[:, headers[0]], data.loc[:, headers[1]], data.loc[:, headers[2]]]).T
        roots = np.array([data.loc[:, headers[3]], data.loc[:, headers[4]]]).T
        return par, roots


def genData(size=100000):
    if path.exists(data_path):
        os.remove(data_path)
    with open(data_path, 'w', encoding='UTF8') as data:
        writer = csv.writer(data)
        writer.writerow(headers)
        i = 0
        while i in range(int(size)):
            a = random.uniform(r_min, r_max)
            b = random.uniform(r_min, r_max)
            c = random.uniform(r_min, r_max)
            par_l = [a, b, c]
            root_l = np.roots(par_l)
            if np.isreal(root_l[0]) and np.isreal(root_l[1]):
                root_l = np.sort(root_l)
                writer.writerow(
                    [a, b, c, root_l[0], root_l[1]])
                i += 1
