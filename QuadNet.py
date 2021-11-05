import torch.nn as nn
import torch

class MnistModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.reLU(),
            nn.Linear(input_size*2, input_size * 4),
            nn.reLU(),
            nn.Linear(input_size * 4, output_size * 2),
            nn.reLU(),
            nn.Linear(output_size*2, output_size)
        )

    def forward(self, input):
        out = self.network(input)
        return out.reshape(2, 2)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss1 = nn.functional.cross_entropy(out, labels)  # Calculate loss
        loss2 = nn.functional.
        return loss1 + loss2

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss2 = nn.functional.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model = MnistModel()