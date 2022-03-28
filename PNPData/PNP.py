import numpy as np
from multiprocessing import Process
import glob
import yaml
import csv
import os.path as path
import os
from scipy.stats import special_ortho_group as random_rotation
import pandas as pd
import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


with open(r'config.yml', 'r') as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)


def gen_data(data_path: str = cfg["DATASET"]["NAME"]):
    threads = []
    if path.exists(data_path):
        os.remove(data_path)

    os.makedirs(data_path, exist_ok=True)

    for t in range(cfg['DATASET']['THREADS']):
        threads.append(Process(target=generate_points, args=(t,)))
        threads[t].start()
    for t in range(cfg['DATASET']['THREADS']):
        threads[t].join()
    # points = np.random.random_sample((cfg['DATASET']['POINTS'], cfg['DATASET']['DIMENSION']))
    # generate_rot_tr(points, 0)


def generate_points(t: int):
    for index in range(cfg["DATASET"]["EPOCHS"]):
        points = np.random.random_sample((cfg['DATASET']['POINTS'], cfg['DATASET']['DIMENSION']))
        generate_rot_tr(points, t, index)


def generate_rot_tr(points: np.ndarray, thread: int, index: int):
    file_path = path.join(cfg["DATASET"]["NAME"], f'group_{thread}_{index}')
    if path.exists(file_path):
        os.remove(file_path)
    os.mkdir(file_path)

    np.save(path.join(cfg['DATASET']['NAME'], f'group_{thread}_{index}', 'points.npy'), points)

    for i in range(cfg['DATASET']['SIZE']):
        rotation = random_rotation.rvs(cfg['DATASET']['DIMENSION'])
        translation = np.random.random_sample(cfg['DATASET']['DIMENSION'])
        file_path = path.join(cfg["DATASET"]["NAME"], f'group_{thread}_{index}', f'transformation{i}')
        if path.exists(file_path):
            os.remove(file_path)
        os.mkdir(file_path)
        np.save(path.join(file_path, 'points.npy'),
                rotation @ points.T + translation[:, np.newaxis])
        np.save(path.join(file_path, 'rotation.npy'), rotation)
        np.save(path.join(file_path, 'translation.npy'), translation)
        np.save(path.join(file_path, 'system.npy'),
                get_pol_system(q_points=(rotation @ points.T + translation[:, np.newaxis]).T, p_points=points))


class Indexer:
    def __init__(self, dim):
        self.tree = [np.zeros((dim - i), dtype=int) for i in range(dim)]
        self.count = 0
        for son in self.tree:
            for i in range(son.size):
                son[i] = self.count
                self.count += 1

    def __getitem__(self, index):
        i, j = index
        if i <= j:
            return self.tree[i][j - i]
        return self.tree[j][i - j]

    def __len__(self):
        return self.count


def eta_indexer(index):
    return (index - 1) * index // 2


def get_pol_system(q_points: np.ndarray, p_points: np.ndarray):
    # Initialize constants
    n, d = p_points.shape
    dim = d * d + d + d + ((d * (d - 1)) // 2)
    size = ncr(dim + 2, 2)
    system = np.zeros((dim, size))

    # Initialize indexes structure

    system_indexer = Indexer(dim + 1)
    constraints_indexer = Indexer(d)

    # constraints
    for i in range(d):
        for j in range(i + 1):
            for k in range(d):
                system[constraints_indexer[i, j], system_indexer[i * d + k, j * d + k]] = 1
                if i == j:
                    system[constraints_indexer[i, j], system_indexer[dim, dim]] = -1

    # derivative with respect to translation
    for k in range(d):
        system[k + len(constraints_indexer), system_indexer[dim, dim]] = -2 * np.sum(q_points[:, k])  # x_0 ** 2
        system[k + len(constraints_indexer), system_indexer[d * d + k, dim]] = 2  # t_k * x_0
        for m in range(d):
            system[k + len(constraints_indexer), system_indexer[k * d + m, dim]] = -2 * np.sum(
                p_points[:, m])  # r_km * x_0

    # derivative with respect to rotation
    for m in range(d):
        for k in range(d):
            system[m * d + k + d + len(constraints_indexer), system_indexer[dim, dim]] = -2 * np.sum(
                q_points[:, m] * p_points[:, k])  # x_0 ** 2
            system[m * d + k + d + len(constraints_indexer), system_indexer[d * d + m, dim]] = -2 * np.sum(
                p_points[:, k])  # t_m * x_0
            system[m * d + k + d + len(constraints_indexer), system_indexer[
                d * d + d + m, m * d + k]] = 2  # lambda_m * r_mk
            for gamma in range(m):
                system[m * d + k + d + len(constraints_indexer), system_indexer[
                    d * d + d + d + eta_indexer(m) + gamma, gamma * d + k]] = 1  # eta^m_gamma * r_gammak
            for beta in range(m + 1, d):
                system[m * d + k + d + len(constraints_indexer), system_indexer[
                    beta * d + k, eta_indexer(beta) + m]] = 1  # r_betak * eta^beta_m

    return system


if __name__ == '__main__':
    gen_data()
