import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
from PiNet import PiNet
import matplotlib.pyplot as plt


def generate_batch(degree=2, batch_size=64):
    poly_coeff = np.zeros((batch_size, degree + 1))
    roots_matrix = np.zeros((batch_size, degree))
    for b in range(batch_size):
        roots = np.random.random_sample(degree)
        roots_matrix[b] = roots
        poly_coeff[b] = np.poly(roots)
        # for k in range(degree + 1):
        #     poly_coeff[b, k] = sum([np.prod(roots[list(x)]) for x in combinations(list(range(degree)), k)]) * (
        #         1 if k % 2 == 0 else -1)
    return poly_coeff, roots_matrix


def train_pinet(model, batch_size=64):
    device = next(model.parameters()).device
    n_batches = 100
    n_epochs = 100
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    losses = np.zeros((2, n_epochs))

    for e in range(n_epochs):
        print(f'===Epoch {e + 1}===')
        epoch_loss = 0
        for i in range(n_batches):
            batch, roots = generate_batch(model.in_size - 1, batch_size)
            batch = torch.tensor(batch, device=device)
            roots = torch.tensor(roots, device=device)
            optimizer.zero_grad()
            # losses = model.training_step(batch)
            out = model(batch)
            loss = criterion(out, roots)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{e + 1:5d}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
            epoch_loss += loss
        epoch_loss /= n_batches
        losses[0, e] = e + 1
        losses[1, e] = epoch_loss

        if e % 50 == 0:
            torch.save(model.state_dict(), f'solver{e}.pth')
        plt.figure()
        plt.title('training loss')
        plt.plot(losses[0, :e], losses[1, :e])
        plt.savefig(f'train_conv.png')


if __name__ == '__main__':
    file_name = os.path.join(r'experiments', r'output.txt')
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass
    with open(file_name, 'w') as output_file:
        model = PiNet(12, 3, 2, output_file)
        model.to('cuda')
        train_pinet(model)
