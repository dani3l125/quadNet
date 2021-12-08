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

with open("config.yml", "r") as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

headers = ['a{}'.format(i) for i in range(cfg['DATASET']['PROBLEM_DEGREE'] + 1)]
data_path = os.path.join("CubeData", cfg['DATASET']['NAME'])


class Quadset(Dataset):
    def __init__(self, size=100000):
        self.samples = self.getData(size)

    def __len__(self):
        len(self.samples)
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def getData(self, size=100000):
        data = pd.read_csv(data_path).dropna().reset_index(drop=True).head(size)
        par = np.array(
            [data.loc[:, headers[i]] for i in range(cfg['DATASET']['PROBLEM_DEGREE'] + 1)]).T
        return par.astype(np.float64)


def genData(size=100000):
    threads = []
    if path.exists(data_path):
        os.remove(data_path)
    # with open(os.path.join("Data2", "headers.csv"), 'w', encoding='UTF8') as data:
    #     writer = csv.writer(data)
    #     writer.writerow(headers)
    for t in range(cfg['DATASET']['THREADS']):
        threads.append(Process(target=subData, args=(int(size / cfg['DATASET']['THREADS']), t)))
        threads[t].start()
    for t in range(cfg['DATASET']['THREADS']):
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
            writer.writerow(
                np.random.uniform(low=-cfg['DATASET']['RANGE'], high=cfg['DATASET']['RANGE'],
                                  size=(cfg['DATASET']['PROBLEM_DEGREE'] + 1,)))
            i += 1
