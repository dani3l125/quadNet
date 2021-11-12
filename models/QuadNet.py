import torch.nn as nn
import torch.nn.functional as F
import torch


class QuadNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        sizeoflayer = 60
        self.network = nn.Sequential(
            nn.Linear(input_size, sizeoflayer),
            nn.ReLU(),
            nn.Linear(sizeoflayer, sizeoflayer),
            nn.ReLU(),
            nn.Linear(sizeoflayer, sizeoflayer),
            nn.ReLU(),
            nn.Linear(sizeoflayer, 10),
            nn.ReLU(),
            nn.Linear(10, output_size),
            # nn.Softmax()  # probabilities output
        )
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.MSELoss()

    def forward(self, input):
        out = self.network(input)
        return out

    def get_losses(self, batch):
        parameters, labels = batch
        out = self(parameters).squeeze()
        return self.loss1(out, labels)

    def training_step(self, batch):
        parameters, labels = batch
        out = self(parameters).squeeze()
        return self.loss1(out, labels)

    def validation_step(self, batch):
        parameters, labels = batch
        out = self(parameters).squeeze()
        loss = self.loss1(out, labels)
        return {'entropy_loss': loss.detach(), 'accuracy': self.accuracy(out, labels)}

    def validation_epoch_end(self, outputs_val, outputs_train):
        batch_losses_val = [x['entropy_loss'] for x in outputs_val]
        batch_losses_train = [x['entropy_loss'] for x in outputs_train]
        batch_acc_val = [x['accuracy'] for x in outputs_val]
        batch_acc_train = [x['accuracy'] for x in outputs_train]
        epoch_loss_val = torch.stack(batch_losses_val).mean()  # Combine losses
        epoch_loss_train = torch.stack(batch_losses_train).mean()  # Combine losses
        epoch_acc_val = torch.stack(batch_acc_val).mean()  # Combine accuracies
        epoch_acc_train = torch.stack(batch_acc_train).mean()  # Combine accuracies
        return {'val_entropy_loss': epoch_loss_val.item(), 'train_entropy_loss': epoch_loss_train.item(),
                'val_acc': 100 * epoch_acc_val.item(), 'train_acc': 100 * epoch_acc_train.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], validation cross entropy: {:.6f}, training cross entropy: {:.6f}, validation accuracy: {:.2f}, training accuracy: {:.2f}".format(
                epoch, result['val_entropy_loss'], result['train_entropy_loss'], result['val_acc'], result['train_acc']))

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        print("predictions: ", preds, "Labels: ", labels)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
