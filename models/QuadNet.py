import torch.nn as nn
import torch.nn.functional as F
import torch

class QuadNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size * 8),
            nn.ReLU(),
            nn.Linear(input_size * 8, output_size * 4),
            nn.ReLU(),
            nn.Linear(output_size*4, output_size)
        )

    def forward(self, input):
        out = self.network(input)
        return out

    def get_losses(self, batch):
        parameters, labels = batch
        labels1, labels2 = labels.split(2, dim=2)
        out1, out2 = self(parameters).split(2, dim=2)  # Generate predictions
        loss1 = F.cross_entropy(out1, labels1)  # Calculate loss
        loss2 = F.mse_loss(out2, labels2)
        return loss1, loss2

    def training_step(self, batch):
        loss1, loss2 = self.get_losses(batch)
        return loss1 + loss2

    def validation_step(self, batch):
        loss1, loss2 = self.get_losses(batch)
        return {'val_entropy_loss': loss1.detach(), 'val_mse': loss2.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_entropy_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_mse'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_entropy_loss': epoch_loss.item(), 'val_mse': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], existing roots cross entropy: {:.4f}, roots mean squared error: {:.4f}".format(epoch, result['val_entropy_loss'], result['val_mse']))
