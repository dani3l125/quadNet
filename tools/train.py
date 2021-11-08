import torch
from torch.utils.data import DataLoader
from dataUtils import QuadSet as Dataset, DeviceDataLoader as DDL
from models.QuadNet import QuadNet
import matplotlib.pyplot as plt

r_min = -10000.0
r_max = 10000.0
data_size = int(10e+2)
val_size = int(2 * 10e+1)
# Hyperparams, optimizer:
num_epochs = 20
lr = 0.01
batch_size = 1024
opt_func = torch.optim.Adam

def evaluate(model, val_loader):
    with torch.no_grad():
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# TODO: add lr scheduler
def fit(epochs, lr, model, train_loader, val_loader, opt_f=torch.optim.SGD):
    history = []
    optimizer = opt_f(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if epoch % 10 == 0:
            # save the model
            torch.save(model.state_dict(), 'QuadNet.pth')
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
    # Initialize dataUtils
    dataset = Dataset.Quadset(data_size)

    train_ds, val_ds = torch.utils.data.random_split(dataset, (data_size - val_size, val_size))

    # Initialize data loaders
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False
                                           )

    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4,
                                         pin_memory=False
                                         )

    # Initialize model
    model = QuadNet(input_size=3,
                    output_size=4,
                    )

    # use GPU only if available
    device = DDL.get_default_device()
    train_dl = DDL.DeviceDataLoader(train_dl, device)
    val_dl = DDL.DeviceDataLoader(val_dl, device)
    DDL.to_device(model, device)

    # Perform training and visualize result
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)

    # save the model
    torch.save(model.state_dict(), 'QuadNet.pth')


if __name__ == "__main__":
    train()
