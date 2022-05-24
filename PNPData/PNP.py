import numpy as np
from multiprocessing import Process
import glob
import yaml
import csv
import os.path as path
import os
from scipy.stats import special_ortho_group as random_rotation
import pandas as pd
import argparse
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
from dataUtils.PNPset import *
from torchvision.models import resnet50
from scipy.special import comb

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gen', type=int, default=0,
                    help='Generate data if needed (random point clouds)')
parser.add_argument('--one', type=int, default=0,
                    help='an integer for the accumulator')
parser.add_argument('--bs', type=int, default=64,
                    help='batch size')
parser.add_argument('--unsup', type=int, default=0,
                    help='weather to use labels or distance from 0')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500


def ncr(n, r):
    return comb(n, r, exact=True)


with open(r'PNPData/config.yml', 'r') as cfg:
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


def generate_rot_tr(p_points: np.ndarray, thread: int, index: int):
    file_path = path.join(cfg["DATASET"]["NAME"], f'group_{thread}_{index}')
    if path.exists(file_path):
        os.remove(file_path)
    os.mkdir(file_path)

    np.save(path.join(cfg['DATASET']['NAME'], f'group_{thread}_{index}', 'points.npy'), p_points)

    for i in range(cfg['DATASET']['SIZE']):
        rotation = random_rotation.rvs(cfg['DATASET']['DIMENSION'])
        translation = np.random.random_sample(cfg['DATASET']['DIMENSION'])
        q_points = (rotation @ p_points.T + translation[:, np.newaxis]).T
        lagrange = get_lagrange_coefficients(q_points=q_points, p_points=p_points, translation=translation,
                                             rotation=rotation)

        labels = np.concatenate([rotation.ravel(), translation, lagrange])
        file_path = path.join(cfg["DATASET"]["NAME"], f'group_{thread}_{index}', f'transformation{i}')
        if path.exists(file_path):
            os.remove(file_path)
        os.mkdir(file_path)
        np.save(path.join(file_path, 'points.npy'),
                rotation @ p_points.T + translation[:, np.newaxis])
        np.save(path.join(file_path, 'rotation.npy'), rotation)
        np.save(path.join(file_path, 'translation.npy'), translation)
        np.save(path.join(file_path, 'system.npy'),
                get_pol_system(q_points=q_points, p_points=p_points))
        np.save(path.join(file_path, 'labels.npy'), labels)


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


def get_pol_system(*, q_points: np.ndarray, p_points: np.ndarray):
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
        system[k + len(constraints_indexer), system_indexer[d * d + k, dim]] = 2 * n  # t_k * x_0
        for m in range(d):
            system[k + len(constraints_indexer), system_indexer[k * d + m, dim]] = 2 * np.sum(
                p_points[:, m])  # r_km * x_0

    # derivative with respect to rotation
    for m in range(d):
        for k in range(d):
            system[m * d + k + d + len(constraints_indexer), system_indexer[dim, dim]] = -2 * np.sum(
                q_points[:, m] * p_points[:, k])  # x_0 ** 2
            system[m * d + k + d + len(constraints_indexer), system_indexer[d * d + m, dim]] = 2 * np.sum(
                p_points[:, k])  # t_m * x_0
            system[m * d + k + d + len(constraints_indexer), system_indexer[
                d * d + d + m, m * d + k]] = 2  # lambda_m * r_mk
            for gamma in range(m):
                system[m * d + k + d + len(constraints_indexer), system_indexer[
                    d * d + d + d + eta_indexer(m) + gamma, gamma * d + k]] = 1  # eta^m_gamma * r_gammak
            for beta in range(m + 1, d):
                system[m * d + k + d + len(constraints_indexer), system_indexer[
                    d * d + d + d + eta_indexer(beta) + m, beta * d + k]] = 1  # r_betak * eta^beta_m

    return system / np.linalg.norm(system, axis=1)


def get_lagrange_coefficients(*, q_points: np.ndarray, p_points: np.ndarray, translation: np.ndarray,
                              rotation: np.ndarray):
    # Initialize constants
    n, d = p_points.shape
    size = d + ((d * (d - 1)) // 2)
    system = np.zeros((d ** 2, size))
    free_coefficients = np.zeros((d ** 2,))

    # derivative with respect to rotation

    for m in range(d):
        for k in range(d):
            free_coefficients[m * d + k] = -2 * translation[m] * np.sum(p_points[:, k]) + 2 * np.sum(
                q_points[:, m] * p_points[:, k])  # free coefficient
            system[m * d + k, m] = 2 * rotation[m, k]  # lambda_m
            for i in range(m):
                system[m * d + k, d + eta_indexer(m) + i] = rotation[i, k]  # eta^m_i
            for i in range(m + 1, d):
                system[m * d + k, d + eta_indexer(i) + m] = rotation[i, k]  # eta^i_m

    return (np.linalg.pinv(system) @ free_coefficients[:, np.newaxis]).squeeze()


def resize_vector(vector):
    vector = torch.cat((vector, torch.ones((vector.shape[0], 1), device=device)), dim=-1)
    vector_indexer = Indexer(vector.shape[1])
    resized = torch.zeros((vector.shape[0], len(vector_indexer)), device=device)
    for i in range(vector.shape[1]):
        for j in range(i, vector.shape[1]):
            resized[:, vector_indexer[i, j]] = vector[:, i] * vector[:, j]
    return resized.type(torch.float64)


if __name__ == '__main__':
    if args.gen:
        gen_data()

    dataset = PNPset()
    if args.one:
        train_ds = dataset
        val_ds = dataset
    else:
        train_ds, val_ds = random_split(dataset, (int(0.8 * len(dataset)), int(0.2 * len(dataset))))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 18)
    model = model.double().to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    for e in range(EPOCHS):
        print(f'===Epoch {e + 1}===')
        running_loss = 0.0

        # Training epoch
        for i, systems in enumerate(train_dl):
            optimizer.zero_grad()
            flatten_in_sys = systems.reshape((systems.shape[0], -1)).type(torch.float64).to(device)
            out = model(
                nn.functional.pad(systems.unsqueeze(1).repeat(1, 3, 1, 1), (17, 17, 103, 103)).type(torch.float64).to(
                    device))
            resized = resize_vector(out)
            values = torch.matmul(systems.type(torch.float64).to(device), resized.T.type(torch.float64).to(device))
            loss = criterion(values, torch.zeros_like(values).to(device))
            loss.backward()
            optimizer.step()
            print(f'Epoch:{e + 1}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')

        loss_mean = 0

        with torch.no_grad():
            for i, (systems, labels) in enumerate(val_dl):
                flatten_in_sys = systems.reshape((systems.shape[0], -1)).type(torch.float64)
                out = model(nn.functional.pad(systems.unsqueeze(1).repeat(1, 3, 1, 1), (17, 17, 103, 103)).type(
                    torch.float64).to(device))
                if args.unsup:
                    loss = criterion(values, torch.zeros_like(values).to(device))
                    resized = resize_vector(out).to(device)
                    values = torch.matmul(systems.type(torch.float64).to(device),
                                          resized.T.type(torch.float64).to(device))
                else:
                    loss = criterion(out, labels)
                loss_mean = ((loss_mean * i) + loss) / (i + 1)
                print(f'Epoch:{e + 1}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
                scheduler.step(loss)

        scheduler.step(loss_mean)

        torch.save(model.state_dict(), f'solver{e}.pth')
