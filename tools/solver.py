import torch
from torch.utils.data import DataLoader
import dataUtils.DeviceDataLoader as DDL
from dataUtils.MyDataSet import *
from models.PiNet import PiNet
import matplotlib.pyplot as plt

with open("config.yml", "r") as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

# Initialize models
file = open(os.path.join("experiments", "cubic_unsup_sol.txt"), "w")


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
        if epoch % cfg['TRAIN']['SAVE_EPOCH'] == 0:
            # save the model
            torch.save(model.state_dict(), "experiments/" + name + '_epoch' + str(epoch) + '.pth')
        if epoch % 50 == 0:
            output = model(sample)
            print("Epoch{}, found roots: {}".format(epoch, output.data))
            file.write("Epoch{}, found roots: {}".format(epoch, output.data))
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
    genData(cfg['DATASET']['SIZE'])
    print("Dataset is created")

    dataset = Quadset(cfg['DATASET']['SIZE'])

    # Initialize data loaders
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=False
                                             )
    # use GPU only if available
    device = DDL.get_default_device()
    dataloader = DDL.DeviceDataLoader(dataloader, device)


    for sample in dataloader:
        Lmodel = PiNet(degree=cfg['TRAIN']['NET_DEGREE'],
                       in_size=cfg['DATASET']['PROBLEM_DEGREE'] + 1,
                       out_size=cfg['TRAIN']['OUT_SIZE'],
                       file=file
                       )
        DDL.to_device(Lmodel, device)

        file.write("\n\nSample:{}".format(sample))
        file.write("\nLinear:")
        solve(cfg['TRAIN']['EPOCHS'], cfg['TRAIN']['LR'], Lmodel, 'linear', sample, eval(cfg['TRAIN']['OPT_FUNC']),
              eval(cfg['TRAIN']['SCHED_FUNC']))


if __name__ == "__main__":
    solver()
