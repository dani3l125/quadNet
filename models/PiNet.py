import torch
import torch.nn as nn


class PiNet(nn.Module):
    def __init__(self, degree, in_size, out_size):
        super().__init__()
        self.degrees = range(1, degree+1)
        self.in_size = in_size
        self.loss = nn.MSELoss
        self.out_size = out_size
        self.weights_matrices = [nn.Parameter(torch.rand(out_size, 1, requires_grad=True))]
        cols = 1
        for n in self.degrees:
            cols *= in_size
            self.weights_matrices.append(nn.Parameter(torch.rand(out_size, cols, requires_grad=True)))
        self.weights_matrices = nn.ParameterList(self.weights_matrices)

    def forward(self, input):
        input = input.squeeze().unsqueeze(axis=1).type(torch.FloatTensor)
        z_n = input
        out = self.weights_matrices[0]
        for n in self.degrees:
            out = torch.add(out, self.weights_matrices[n].mm(z_n))
            z_n = torch.kron(z_n, input)
        return out

    def get_losses(self, batch):
        parameters, values = batch
        out = self(parameters)
        return self.loss(out, values)

    def training_step(self, batch):
        parameters, values = batch
        out = self(parameters)
        return self.loss(out, values)

    def validation_step(self, batch):
        parameters, values = batch
        out = self(parameters)
        loss = self.loss(out, values)
        return {'mse': loss.detach(), 'accuracy': self.accuracy(out, values)}

    def validation_epoch_end(self, outputs_val):
        batch_losses_val = [x['mse'] for x in outputs_val]
        batch_acc_val = [x['accuracy'] for x in outputs_val]
        epoch_loss_val = torch.stack(batch_losses_val).mean()  # Combine losses
        epoch_acc_val = torch.stack(batch_acc_val).mean()  # Combine accuracies
        return {'val_mse': epoch_loss_val.item(),
                'val_acc': epoch_acc_val.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], validation mean squared error: {:.6f}, validation accuracy: {:.2f}".format(
                epoch, result['val_mse'], result['val_acc']))

    def accuracy(self, outputs, values):
        print("predictions: ", outputs, "values: ", values)
        return torch.tensor(torch.sum(values[0]-1 <= outputs[0] <= values[0]+1).item() / len(preds))


