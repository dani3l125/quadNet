import os.path
import sys
import numpy as np
from scipy.stats import special_ortho_group as random_rotation
import yaml
import torch

sys.path.append(os.path.join('..', 'PNPData'))

from PNPData.PNP import get_lagrange_coefficients, get_pol_system

with open(os.path.join('..', 'PNPData', 'config.yml'), 'r') as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)


def generate_batch(batch_size=1):
    systems_list = []
    labels_list = []
    for i in range(batch_size):
        p_points = np.random.random_sample((cfg['DATASET']['POINTS'], cfg['DATASET']['DIMENSION']))
        rotation = random_rotation.rvs(cfg['DATASET']['DIMENSION'])
        translation = np.random.random_sample(cfg['DATASET']['DIMENSION'])
        q_points = (rotation @ p_points.T + translation[:, np.newaxis]).T
        lagrange = get_lagrange_coefficients(q_points=q_points, p_points=p_points, translation=translation,
                                             rotation=rotation)
        labels = np.concatenate([rotation.ravel(), translation, lagrange])
        system = get_pol_system(q_points=q_points, p_points=p_points)
        systems_list.append(system)
        labels_list.append(labels)
    return torch.from_numpy(np.stack(systems_list, axis=0)), torch.from_numpy(np.stack(labels_list, axis=0))


if __name__ == '__main__':
    print(generate_batch())
