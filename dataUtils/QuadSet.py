import os

from torch.utils.data import Dataset
import os.path as path
import csv
import numpy as np
import random
import torch

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['class']
par_path = 'C:/Users/user/Documents/GitHub/quadNet/data/parameters.csv'
roots_path = 'C:/Users/user/Documents/GitHub/quadNet/data//roots.csv'


class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples, self.roots_n = self.genData(size)

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx], self.roots_n[idx]]

    def genData(self, size=100000):
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
                # roots = []
                # for r in root_l:
                #     if np.isreal(r):
                #         roots.append(float(1))
                #     else:
                #         roots.append(float(0))
                # for r in root_l:
                #     roots.append(np.real(r))
                # writer_r.writerow(roots)
                # sum of probabilities is 1 ->can use softmax.
                if np.isreal(root_l[0]) and np.isreal(root_l[1]):
                    writer_r.writerow([0])
                elif np.isreal(root_l[0]) or np.isreal(root_l[1]):
                    writer_r.writerow([1])
                else:
                    writer_r.writerow([2])
                writer_p.writerow(par_l)
        inputs = torch.from_numpy(np.loadtxt(par_path, dtype=np.float32, delimiter=",", skiprows=1))
        outputs = torch.from_numpy(np.loadtxt(roots_path, dtype=np.float32, delimiter=",", skiprows=1))
        return inputs.unsqueeze(dim=1), outputs.type(torch.LongTensor)
