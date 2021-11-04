import random
import csv
import os.path as path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import QuadSet as Dataset
import matplotlib.pyplot as plt
import DeviceDataLoader as DDL

r_min = -10000.0
r_max = 10000.0
data_size = 100000
val_size = 20000
header_p = ['a', 'b', 'c']
header_r = ['exist1', 'root1', 'exist2', 'root2']
par_path = 'Data/parameters.csv'
roots_path = 'Data/roots.csv'
# Hyperparams, loss, optimizer:
num_epochs = 10
lr = 0.001
opt_func = torch.optim.Adam
loss_func = torch.nn.PairwiseDistance()


def training_step(net, batch):
    parameters, roots = batch
    print("here")
    out = net(parameters)[0]  # Generate predictions
    loss = loss_func(torch.tensor(out), roots)  # Calculate loss
    return loss


def validation_step(net, batch):
    parameters, roots = batch
    out = net(parameters)  # Generate predictions
    loss = loss_func(out, roots)  # Calculate loss
    acc = accuracy(out, roots)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(net, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def evaluate(model, val_loader):
    with torch.no_grad():
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# TODO: add lr scheduler
def train(epochs, lr, model, train_loader, val_loader, opt_f=torch.optim.SGD):
    history = []
    optimizer = opt_f(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def epoch_end(net, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def main():
    # Initialize dataset
    dataset = Dataset.Quadset()

    train_ds, val_ds = torch.utils.data.random_split(dataset, (data_size - val_size, val_size))

    # Initialize data loaders
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=256,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False
                                           )

    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=256,
                                         shuffle=True,
                                         num_workers=4,
                                         pin_memory=False
                                         )

    device = DDL.get_default_device()

    train_dl = DDL.DeviceDataLoader(train_dl, device)
    val_dl = DDL.DeviceDataLoader(val_dl, device)

    # Initialize model
    model = torch.nn.LSTM(input_size=3,
                          hidden_size=4,
                          num_layers=10
                          )

    DDL.to_device(model, device)

    history = train(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)

    torch.save(model.state_dict(), 'lstm_quadratic.pth')


if __name__ == "__main__":
    main()
