from torch.utils.data import Dataset
import os.path as path
import csv
import numpy as np
import random

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['exist1', 'root1', 'exist2', 'root2']
par_path = 'Data/parameters.csv'
roots_path = 'Data/roots.csv'

class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples, self.annotations = self.genData(size)

    def __len__(self):
        return self.samples.size(dim=0)

    def __getitem__(self, idx):
        return {'parameters': self.samples[idx], 'roots': self.annotations[idx]}

    def genData(self, size=100000):
        if not path.exists(par_path):  # generate dataset if needed
            with open(par_path, 'w', encoding='UTF8') as p, open(roots_path, 'w', encoding='UTF8') as r:
                writer_r = csv.writer(r)
                writer_r.writerow(header_r)
                writer_p = csv.writer(p)
                writer_p.writerow(header_p)
                for i in range(size):
                    a = random.uniform(r_min, r_max)
                    b = random.uniform(r_min, r_max)
                    c = random.uniform(r_min, r_max)
                    par_l = [a, b, c]
                    root_l = np.roots(par_l)
                    roots = []
                    for r in root_l:
                        if np.isreal(r):
                            roots.append(float(1))
                        else:
                            roots.append(float(0))
                        roots.append(np.real(r))
                    writer_r.writerow(roots)
                    writer_p.writerow(par_l)
            inputs = torch.from_numpy(np.loadtxt(par_path, dtype=np.float32, delimiter=",", skiprows=1))
            outputs = torch.from_numpy(np.loadtxt(roots_path, dtype=np.float32, delimiter=",", skiprows=1))
            inputs.unsqueeze(0)
            inputs.unsqueeze(1)
            outputs.unsqueeze(0)
            outputs.unsqueeze(1)
            return inputs, outputs
