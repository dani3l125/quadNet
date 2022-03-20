import numpy as np
from multiprocessing import Process
import glob
import yaml
import csv
import os.path as path
import os
from scipy.stats import special_ortho_group as random_rotation
import pandas as pd

with open(r'config.yml', 'r') as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)


def gen_data(data_path: str = cfg["DATASET"]["NAME"]):
    threads = []
    if path.exists(data_path):
        os.remove(data_path)

    os.makedirs(data_path, exist_ok=True)

    for t in range(cfg['DATASET']['THREADS']):
        points = np.random.random_sample((cfg['DATASET']['POINTS'], cfg['DATASET']['DIMENSION']))
        threads.append(Process(target=generate_rot_tr, args=(points, t)))
        threads[t].start()
    # points = np.random.random_sample((cfg['DATASET']['POINTS'], cfg['DATASET']['DIMENSION']))
    # generate_rot_tr(points, 0)


def generate_rot_tr(points: np.ndarray, index: int):
    file_path = path.join(cfg["DATASET"]["NAME"], f'group{index}')
    if path.exists(file_path):
        os.remove(file_path)
    os.mkdir(file_path)

    np.save(path.join(cfg['DATASET']['NAME'], f'group{index}', 'points.npy'), points)

    for i in range(cfg['DATASET']['SIZE']):
        rotation = random_rotation.rvs(cfg['DATASET']['DIMENSION'])
        translation = np.random.random_sample(cfg['DATASET']['DIMENSION'])
        file_path = path.join(cfg["DATASET"]["NAME"], f'group{index}', f'transformation{i}')
        if path.exists(file_path):
            os.remove(file_path)
        os.mkdir(file_path)
        np.save(path.join(file_path, 'points.npy'),
                rotation @ points.T + translation[:, np.newaxis])
        np.save(path.join(file_path, 'rotation.npy'), rotation)
        np.save(path.join(file_path, 'translation.npy'), translation)


if __name__ == '__main__':
    gen_data()
