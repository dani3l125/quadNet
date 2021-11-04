import random
import csv
import os.path as path
import numpy as np
import torch
import torchvision.models as models

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['root1', 'root2']
par_path = 'Data/parameters.csv'
roots_path = 'Data/roots.csv'

def main():
    if not path.exists(par_path):  # generate dataset if needed
        # TODO: add one root, no roots. deal with loss function
        with open(par_path, 'w', encoding='UTF8') as p, open(roots_path, 'w', encoding='UTF8') as r:
            writer_r = csv.writer(r)
            writer_r.writerow(header_r)
            writer_p = csv.writer(p)
            writer_p.writerow(header_p)
            for i in range(1000000):
                r1 = random.uniform(r_min, r_max)
                r2 = random.uniform(r_min, r_max)
                if r1 > r2:  # assuming ordered roots
                    temp = r1
                    r1 = r2
                    r2 = temp
                a = 1
                b = r1 + r2
                c = r1 * r2
                if random.getrandbits(1):  # possible to remove 'if'
                    a = random.uniform(r_min, r_max)
                    b *= a
                    c *= a
                writer_r.writerow([r1, r2])
                writer_p.writerow([a, b, c])

    #TODO: visualize with pandas

    # Create network
    net = models.resnet18()  #TODO find best model

    # Prepare the data
    # with open(par_path, 'w', encoding='UTF8') as p, open('Data/roots.csv', 'w', encoding='UTF8') as r:
    inputs_file = torch.from_numpy(np.loadtxt(par_path, dtype=np.float32, delimiter=",", skiprows=1))
    outputs_file = torch.from_numpy(np.loadtxt(roots_path, dtype=np.float32, delimiter=",", skiprows=1))
    inputs_file.unsqueeze(0)
    inputs_file.unsqueeze(1)
    print(inputs_file)
    print(outputs_file)

if __name__ == "__main__":
    main()

