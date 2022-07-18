import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from UnsupModel import UnsupModel
from dataset import generate_batch
import matplotlib.pyplot as plt


def train_unsup(model, system, labels):
    device = next(model.parameters()).device
    n_batches = 100
    n_epochs = 100
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = np.zeros((3, n_epochs))

    for e in range(n_epochs):
        print(f'===Epoch {e + 1}===')
        epoch_loss = 0
        epoch_distance = 0
        for i in range(n_batches):
            optimizer.zero_grad()
            # losses = model.training_step(batch)
            out = model(system)
            loss = criterion(torch.tensor(0.), (out ** 2).max())
            loss.backward()
            optimizer.step()
            print(f'Epoch:{e + 1:5d}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
            epoch_loss += loss
            epoch_distance += torch.norm((labels - out))
        epoch_loss /= n_batches
        epoch_distance /= n_batches
        losses[0, e] = e + 1
        losses[1, e] = epoch_loss
        losses[2, e] = epoch_distance

        if e % 50 == 0:
            torch.save(model.state_dict(), f'solver{e}.pth')
        plt.figure()
        plt.title('training loss')
        plt.plot(losses[0, :e], losses[1, :e])
        plt.savefig(f'train_loss.png')
        plt.figure()
        plt.title('training distance from real solution')
        plt.plot(losses[0, :e], losses[2, :e])
        plt.savefig(f'train_dist.png')


if __name__ == '__main__':
    system, labels = generate_batch()
    system = system.squeeze()
    labels = labels.squeeze()
    model = UnsupModel(18, 2)
    train_unsup(model, system, labels)
