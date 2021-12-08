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

r_min = -10000.0
r_max = 10000.0
headers = ['a1', 'a2', 'a3', 'a4', 'root1', 'root2', 'root3']
data_path = os.path.join("CubeData", "NewCubic.csv")
n_threads = 1


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
        return par.astype(np.float64), roots.astype(np.float64)


def genData(size=100000):
    threads = []
    if path.exists(data_path):
        os.remove(data_path)
    # with open(os.path.join("Data2", "headers.csv"), 'w', encoding='UTF8') as data:
    #     writer = csv.writer(data)
    #     writer.writerow(headers)
    for t in range(n_threads):
        threads.append(Process(target=subData, args=(int(size / n_threads), t)))
        threads[t].start()
    for t in range(n_threads):
        threads[t].join()
    all_filenames = glob.glob("CubeData" + "/*.csv")
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(data_path, index=False)


def subData(size, index):
    i = 0
    if path.exists(os.path.join("CubeData", "cubic{}.csv".format(index))):
        os.remove(os.path.join("CubeData", "cubic{}.csv".format(index)))
    with open(os.path.join("CubeData", "cubic{}.csv".format(index)), 'w') as data:
        writer = csv.writer(data)
        writer.writerow(headers)
        while i in range(int(size)):
            a = random.uniform(r_min, r_max)
            b = random.uniform(r_min, r_max)
            c = random.uniform(r_min, r_max)
            d = random.uniform(r_min, r_max)
            par_l = [a, b, c, d]
            root_l = np.roots(par_l)
            if np.isreal(root_l[0]) and np.isreal(root_l[1]) and np.isreal(root_l[2]):
                root_l = np.sort(root_l)
                writer.writerow(
                    np.array([a, b, c, d, root_l[0], root_l[1], root_l[2]]))
                i += 1
