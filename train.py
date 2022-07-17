import torch
import numpy as np
from torch.utils.data import DataLoader
from dataUtils.MyDataSet import *
import matplotlib.pyplot as plt
from dataUtils import PNPset
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import random_split
import argparse
from torch.optim import Adam, lr_scheduler
from PNPData.PNP import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--func', type=str, default='relu',
                    help='Activation function: none/relu/hardshrink/tanhshrink')
parser.add_argument('--name', type=int, default=0,
                    help='of graphs')
parser.add_argument('--gen', type=int, default=0,
                    help='Generate data if needed (random point clouds)')
parser.add_argument('--one', type=int, default=0,
                    help='an integer for the accumulator')
parser.add_argument('--bs', type=int, default=32,
                    help='batch size')
parser.add_argument('--unsup', type=int, default=0,
                    help='weather to use labels or distance from 0')
parser.add_argument('--inf_data', type=int, default=0,
                    help='train with infinite dataset')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500

activ_func = nn.Tanhshrink()
# if args.func == 'none':
#     activ_func = nn.Identity()
# elif args.func == 'hardshrink':
#     activ_func = nn.Hardshrink()
# elif args.func == 'tanhshrink':
#     activ_func = nn.Tanhshrink()


r_min = -10000.0
r_max = 10000.0
data_size = 100  # 3 * 10e+6
val_size = int(4 * 10e+6)  # 6 * 10e+5
# Hyperparams, optimizer:
num_epochs = 750
lr = 0.05
batch_size = 1
degree = 15
save_epoch = 20
opt_func = torch.optim.Adam
schedule_func = torch.optim.lr_scheduler.StepLR


def evaluate(model, val_loader, train_loader):
    with torch.no_grad():
        model.eval()
        outputs_val = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs_val)


# TODO: add lr scheduler
def fit(epochs, lr, model, train_loader, val_loader, opt_f=torch.optim.SGD,
        schedule_func=torch.optim.lr_scheduler.StepLR):
    history = []
    optimizer = opt_f(model.parameters(), lr)
    scheduler = schedule_func(optimizer, 75)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader, train_loader)
        result['val_mse'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if epoch % save_epoch == 0:
            # save the model
            torch.save(model.state_dict(), 'QuadNet_ep' + str(epoch) + '.pth')
    return history


def plot_accuracies(history):
    entropy = [x['val_entropy_loss'] for x in history]
    plt.plot(entropy, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Existing roots loss vs. No. of epochs')

    mse = [x['val_mse'] for x in history]
    plt.plot(entropy, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Roots estimation MSE vs. No. of epochs')


def train():
    dataset = PNPset.PNPset()
    if args.one:
        train_ds = dataset
        val_ds = dataset
    else:
        train_ds, val_ds = random_split(dataset, (int(0.8 * len(dataset)), int(0.2 * len(dataset))))
    train_dl = DataLoader(train_ds, batch_size=args.bs, pin_memory=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.bs, pin_memory=True, num_workers=4)

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 18)
    model = model.double().to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    losses = np.zeros((3, EPOCHS))

    for e in range(EPOCHS):
        print(f'===Epoch {e + 1}===')
        loss_mean = 0

        # Training epoch
        for i, (systems, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            out = model(
                nn.functional.pad(systems.unsqueeze(1).repeat(1, 3, 1, 1), (17, 17, 103, 103)).type(torch.float64).to(
                    device))
            if args.unsup:
                loss = criterion(values, torch.zeros_like(values).to(device))
                resized = resize_vector(out).to(device)
                values = torch.matmul(systems.type(torch.float64).to(device),
                                      resized.T.type(torch.float64).to(device))
            else:
                loss = criterion(out, labels.to(device))
            loss.backward()
            optimizer.step()
            print(f'Epoch:{e + 1}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
            loss_mean = ((loss_mean * i) + loss) / (i + 1)

        losses[0, e] = e + 1
        losses[1, e] = loss_mean
        loss_mean = 0

        with torch.no_grad():
            for i, (systems, labels) in enumerate(val_dl):
                out = model(nn.functional.pad(systems.unsqueeze(1).repeat(1, 3, 1, 1), (17, 17, 103, 103)).type(
                    torch.float64).to(device))
                if args.unsup:
                    loss = criterion(values, torch.zeros_like(values).to(device))
                    resized = resize_vector(out).to(device)
                    values = torch.matmul(systems.type(torch.float64).to(device),
                                          resized.T.type(torch.float64).to(device))
                else:
                    loss = criterion(out, labels.to(device))
                loss_mean = ((loss_mean * i) + loss) / (i + 1)
                print(f'Epoch:{e + 1}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
            scheduler.step(loss_mean)

        scheduler.step(loss_mean)
        losses[2, e] = loss_mean

        torch.save(model.state_dict(), f'solver{e}.pth')

        plt.figure()
        plt.title('training loss')
        plt.plot(losses[0, :e], losses[1, :e])
        plt.savefig(f'train{args.name}.png')
        plt.figure()
        plt.title('validation loss')
        plt.plot(losses[0, :e], losses[2, :e])
        plt.savefig(f'val{args.name}.png')


def generate_batch(batch_size=64):
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


def train_inf_data():
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.relu = activ_func
    model.maxpool = nn.Identity()
    model = model.double().to(device)

    n_batches = 1000

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    losses = np.zeros((2, EPOCHS))

    for e in range(EPOCHS):
        print(f'===Epoch {e + 1}===')
        epoch_loss = 0
        # Training epoch
        for i in range(n_batches):
            systems, labels = generate_batch(batch_size=args.bs)
            optimizer.zero_grad()
            out = model(
                nn.functional.pad(systems.unsqueeze(1).repeat(1, 3, 1, 1), (17, 17, 103, 103)).type(torch.float64).to(
                    device))
            if args.unsup:
                loss = criterion(values, torch.zeros_like(values).to(device))
                resized = resize_vector(out).to(device)
                values = torch.matmul(systems.type(torch.float64).to(device),
                                      resized.T.type(torch.float64).to(device))
            else:
                loss = criterion(out, labels.to(device))
            loss.backward()
            optimizer.step()
            print(f'Epoch:{e + 1}, Batch:{i + 1:5d}, Loss: {loss.item():.3f}')
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


if __name__ == "__main__":
    train_inf_data()
