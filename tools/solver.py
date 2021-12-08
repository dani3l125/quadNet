import torch
from torch.utils.data import DataLoader
import dataUtils.DeviceDataLoader as DDL
from dataUtils.MyDataSet import *
from models.PiNet import PiNet
import matplotlib.pyplot as plt

r_min = -1
r_max = 1
data_size = 2000  # 3 * 10e+6
# Hyperparams, optimizer:
num_epochs = 2000
lr = 0.05
batch_size = 1
degree = 1
save_epoch = 200
opt_func = torch.optim.Adam
schedule_func = torch.optim.lr_scheduler.StepLR

def evaluate(model, sample):
    with torch.no_grad():
        model.eval()
        output = model.validation_step(sample)
    return model.solver_epoch_end(output)


# TODO: add lr scheduler
def solve(epochs, lr, model, name, sample, opt_f=torch.optim.SGD, schedule_func=torch.optim.lr_scheduler.StepLR):
    history = []
    optimizer = opt_f(model.parameters(), lr)
    scheduler = schedule_func(optimizer, 200)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        loss = model.training_step(sample)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # Validation phase
        if epoch % 20 == 0:
            result = evaluate(model, sample)
            result['val_mse'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        if epoch % save_epoch == 0:
            # save the model
            torch.save(model.state_dict(), "experiments/" + name + '_epoch' + str(epoch) + '.pth')
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




# Perform solving and visualize result
    history = []
def solver():
    # Initialize dataUtils
    print("Creating dataset...")
    genData(data_size)
    print("Dataset is created")

    dataset = Quadset(data_size)

    # Initialize data loaders
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=False
                                             )
    # use GPU only if available
    device = DDL.get_default_device()
    dataloader = DDL.DeviceDataLoader(dataloader, device)

    # Initialize models
    file = open(os.path.join("experiments", "cubic_unsup_sol.txt"), "w")

    for sample in dataloader:

        Lmodel = PiNet(degree=1,
                       in_size=4,
                       out_size=3,
                       file=file
                       )
        DDL.to_device(Lmodel, device)

        file.write("\n\nSample:{}".format(sample))
        file.write("\nLinear:")
        solve(num_epochs, lr, Lmodel, 'linear', sample, opt_func, schedule_func)


if __name__ == "__main__":
    solver()
